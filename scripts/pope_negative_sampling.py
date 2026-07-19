#!/usr/bin/env python3
"""
pope_negative_sampling.py
=========================
POPE-style (Li et al., EMNLP 2023) negative-sampling difficulty axis for the
MedGemma-27B presence/absence harness. The validated pieces — image pipeline,
the three format variants, determinism, parser, and the hysteresis GT rule
(present >= 50 vox; a slice is a true-negative for an organ only if that organ
has < 5 vox on it) — are REUSED unchanged from anatomy_prior_probe; this file
only adds how the ABSENT organs in each query are chosen.

Three negative samplers (positives are fixed by the image and shared; only the
negatives differ):
  random       : absent organ drawn uniformly from organs absent on this slice.
  popular      : highest-frequency absent organ (organ_frequency) — the organ
                 the model most expects to exist.
  adversarial  : absent organ with the highest co-occurrence with the organs
                 actually present on this slice (max over present a of
                 cooccurrence[a][candidate]) — the most "expected given what is
                 visible". THIS sampler is the contribution; verify in the dry
                 run that it selects the most-expected-but-absent organ.

Every negative still must pass the < 5-voxel hard-zero check; if a strategy's
top pick is not a clean true-negative, fall to the next candidate.

THIS FILE IS STEP A ONLY: compute + persist btcv_organ_stats.json (organ
frequency + slice-level co-occurrence over all 30 training cases), print the
frequency ranking and top co-occurring pairs, and DRY-RUN the three samplers on
~8 slices printing each strategy's pick + score. NO model calls. Steps B (wire
into the model loop, --limit_n smoke) and C (full sweep + extended aggregator)
follow after this is eyeballed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the validated harness pieces unchanged.
from anatomy_prior_probe import (  # noqa: E402
    BTCV_LABELS, VOCAB, DEFAULT_FOCUS_ORGANS,
    load_volume, organ_areas,
)

PRESENCE_MIN = 50     # >= this -> present  (validated hysteresis)
HARD_ZERO = 5         # < this  -> clean true-negative
# Long-z structures: distractors only (almost no true-absent slices), never
# focus/positive. Negatives may still land on them when genuinely absent (rare).
POSITIVE_ELIGIBLE = set(DEFAULT_FOCUS_ORGANS)

OUT_DIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
STATS_PATH = OUT_DIR / "btcv_organ_stats.json"


# --------------------------------------------------------------------------- #
# Stats precompute (organ frequency + slice-level co-occurrence)
# --------------------------------------------------------------------------- #

def compute_stats(images_dir: str, labels_dir: str) -> dict:
    labels = sorted(Path(labels_dir).glob("*.nii*"))
    if not labels:
        raise SystemExit(f"no label volumes in {labels_dir}")

    n_slices = 0
    present_count = {o: 0 for o in VOCAB}                       # slices where o present
    cooc = {a: {b: 0 for b in VOCAB} for a in VOCAB}            # slices where a&b present
    per_case_slices = {}

    for lbl_path in labels:
        lbl = load_volume(str(lbl_path))
        areas = organ_areas(lbl)                               # name -> per-z counts
        n_z = lbl.shape[2]
        per_case_slices[lbl_path.name.split(".")[0]] = n_z
        n_slices += n_z
        present_mask = {o: (areas[o] >= PRESENCE_MIN) for o in VOCAB}
        for o in VOCAB:
            present_count[o] += int(present_mask[o].sum())
        for a in VOCAB:
            za = present_mask[a]
            if not za.any():
                continue
            for b in VOCAB:
                cooc[a][b] += int((za & present_mask[b]).sum())

    organ_frequency = {o: present_count[o] / n_slices for o in VOCAB}
    cooccurrence = {a: {b: (cooc[a][b] / present_count[a] if present_count[a] else 0.0)
                        for b in VOCAB} for a in VOCAB}

    return {
        "presence_min_pixels": PRESENCE_MIN,
        "n_cases": len(labels),
        "n_slices_total": n_slices,
        "present_slice_count": present_count,
        "organ_frequency": organ_frequency,
        "cooccurrence": cooccurrence,
        "per_case_slices": per_case_slices,
    }


def print_stats(stats: dict, top_pairs: int = 20) -> None:
    freq = stats["organ_frequency"]
    print(f"\n=== ORGAN FREQUENCY (fraction of {stats['n_slices_total']} slices present, "
          f">= {PRESENCE_MIN} vox) ===")
    for o, f in sorted(freq.items(), key=lambda kv: -kv[1]):
        elig = "" if o in POSITIVE_ELIGIBLE else "  [distractor-only]"
        print(f"  {o:22s} {f:.3f}{elig}")

    print(f"\n=== TOP {top_pairs} CO-OCCURRING PAIRS  P(b present | a present) ===")
    pairs = []
    for a in VOCAB:
        for b in VOCAB:
            if a != b:
                pairs.append((stats["cooccurrence"][a][b], a, b))
    for p, a, b in sorted(pairs, reverse=True)[:top_pairs]:
        print(f"  P({b:20s} | {a:20s}) = {p:.3f}")


# --------------------------------------------------------------------------- #
# Per-slice present / hard-zero-absent sets
# --------------------------------------------------------------------------- #

def slice_sets(areas: dict, z: int):
    present = [o for o in VOCAB if areas[o][z] >= PRESENCE_MIN]
    hardzero_absent = [o for o in VOCAB if areas[o][z] < HARD_ZERO]
    return present, hardzero_absent


# --------------------------------------------------------------------------- #
# The three negative samplers — return a RANKED list of (organ, score, why)
# over the genuinely-absent (hard-zero) candidates, best first.
# --------------------------------------------------------------------------- #

# Near-ubiquitous structures (~50% slice prevalence): excluded as adversarial
# CONDITIONERS because P(c | vessel) ~= P(c), so conditioning on them collapses
# co-occurrence to marginal frequency (= the popular ranking) and throws away the
# image-specific signal. They remain valid CANDIDATES and present-context.
CONDITIONER_EXCLUDE = {"aorta", "inferior_vena_cava", "portal_splenic_vein"}


def rank_random(absent: list, present: list, stats: dict, rng) -> list:
    order = list(absent)
    rng.shuffle(order)
    return [(o, None, "uniform") for o in order]


def rank_popular(absent: list, present: list, stats: dict, rng) -> list:
    freq = stats["organ_frequency"]
    ranked = sorted(absent, key=lambda o: -freq[o])
    return [(o, freq[o], "frequency") for o in ranked]


def rank_adversarial(absent: list, present: list, stats: dict, rng):
    """Rank absent candidates by max co-occurrence with the FOCALLY-LOCALIZED
    present organs (vessels excluded as conditioners). Returns None to signal a
    fallback when the slice has no focal conditioner (vessels-only present)."""
    cooc = stats["cooccurrence"]
    conditioners = [a for a in present if a not in CONDITIONER_EXCLUDE]
    if not conditioners:
        return None
    scored = []
    for c in absent:
        best_a, best_p = None, -1.0
        for a in conditioners:
            p = cooc[a][c]
            if p > best_p:
                best_a, best_p = a, p
        scored.append((c, best_p, f"co-occ with {best_a}"))
    scored.sort(key=lambda t: -t[1])
    return scored


SAMPLERS = {"random": rank_random, "popular": rank_popular}


def pick_negatives(strategy: str, present: list, absent: list, stats: dict, rng, k: int):
    """Return (picks, fellback). picks = top-k genuinely-absent (hard-zero) organs
    by the strategy. Adversarial falls back to POPULAR (flagged) on vessels-only
    slices so the fallback is reported, never a silent fourth strategy."""
    fellback = False
    if strategy == "adversarial":
        ranked = rank_adversarial(absent, present, stats, rng)
        if ranked is None:
            ranked = rank_popular(absent, present, stats, rng)
            fellback = True
    else:
        ranked = SAMPLERS[strategy](absent, present, stats, rng)
    return ranked[:k], fellback


# --------------------------------------------------------------------------- #
# Step A dry run: verify the samplers (esp. adversarial) on a few slices
# --------------------------------------------------------------------------- #

def dry_run_samplers(images_dir: str, labels_dir: str, stats: dict,
                     n_slices: int, seed: int, k_neg: int) -> None:
    rng = np.random.default_rng(seed)
    labels = sorted(Path(labels_dir).glob("*.nii*"))[:2]

    # Pick slices with >= 3 present organs (so adversarial has real signal),
    # spread across z, from the first couple of cases.
    picks = []
    for lbl_path in labels:
        lbl = load_volume(str(lbl_path))
        areas = organ_areas(lbl)
        case = lbl_path.name.split(".")[0]
        cand_z = [z for z in range(lbl.shape[2])
                  if len(slice_sets(areas, z)[0]) >= 3 and slice_sets(areas, z)[1]]
        if not cand_z:
            continue
        sel = np.linspace(0, len(cand_z) - 1, num=max(1, n_slices // len(labels)), dtype=int)
        for zi in sel:
            picks.append((case, areas, int(cand_z[zi])))

    print(f"\n=== DRY RUN: negative samplers on {len(picks)} slices "
          f"(k_neg={k_neg}, present>= {PRESENCE_MIN} vox, hard-zero< {HARD_ZERO} vox) ===")
    for case, areas, z in picks:
        present, absent = slice_sets(areas, z)
        present_elig = [o for o in present if o in POSITIVE_ELIGIBLE]
        print("\n" + "-" * 96)
        print(f"{case} z={z}")
        print(f"  present (>={PRESENCE_MIN}): {present}")
        print(f"  positive-eligible        : {present_elig}")
        print(f"  hard-zero absent (<{HARD_ZERO})  : {absent}")
        for strat in ("random", "popular", "adversarial"):
            picks_s, fell = pick_negatives(strat, present, absent, stats, rng, k_neg)
            cells = []
            for organ, score, why in picks_s:
                sc = "" if score is None else f"={score:.3f}"
                cells.append(f"{organ}{sc} [{why}]")
            flag = "  <ADV_FELLBACK->popular>" if fell else ""
            print(f"    {strat:12s}: " + " ; ".join(cells) + flag)
    print("\n" + "=" * 96)
    print("STEP A — no model calls. Verify adversarial picks the most-expected-but-absent")
    print("organ (high co-occurrence with a present organ), not garbage, before Step B.")


def project_cells(images_dir: str, labels_dir: str, stats: dict, seed: int, k_neg: int) -> None:
    """Project the negatives over the ACTUAL sweep slice set (the strata sampler
    from anatomy_prior_probe, all 30 cases) to confirm the adversarial cell is
    not starved and to report the adversarial->popular fallback rate."""
    from anatomy_prior_probe import Config as ACfg, sample_volume
    cfg = ACfg()
    rng = np.random.default_rng(seed)
    images = sorted(Path(images_dir).glob("*.nii*"))
    labels = sorted(Path(labels_dir).glob("*.nii*"))

    strategies = ("random", "popular", "adversarial")
    per_total = {s: 0 for s in strategies}
    per_organ = {s: defaultdict(int) for s in strategies}
    adv_slices = adv_fell = 0
    n_slices = 0

    for img_path, lbl_path in zip(images, labels):
        lbl = load_volume(str(lbl_path))
        areas = organ_areas(lbl)
        case = img_path.name.split(".")[0]
        for s in sample_volume(case, lbl, cfg, rng):
            present, absent = slice_sets(areas, s.z)
            if not absent:
                continue
            n_slices += 1
            for strat in strategies:
                picks, fell = pick_negatives(strat, present, absent, stats, rng, k_neg)
                if strat == "adversarial":
                    adv_slices += 1
                    adv_fell += int(fell)
                for organ, _score, _why in picks:
                    per_total[strat] += 1
                    per_organ[strat][organ] += 1

    print("\n" + "#" * 96)
    print(f"# PROJECTED CELL SIZES over the full sweep ({n_slices} slice-units, all 30 cases, "
          f"k_neg={k_neg})")
    print("#" * 96)
    print("\nabsent items per strategy (= per strategy x format cell; each asked in all 3 formats):")
    for s in strategies:
        print(f"  {s:12s} {per_total[s]:5d}")
    fr = adv_fell / adv_slices if adv_slices else float("nan")
    print(f"\nadversarial->popular fallback: {adv_fell}/{adv_slices} = {fr:.3f} "
          f"(vessels-only slices)")

    print("\nper-organ absent-item count (for per-organ McNemar stability):")
    print(f"{'organ':22s} {'random':>8s} {'popular':>8s} {'adversarial':>12s}")
    for o in VOCAB:
        print(f"{o:22s} {per_organ['random'][o]:8d} {per_organ['popular'][o]:8d} "
              f"{per_organ['adversarial'][o]:12d}")
    print("\n(want a few hundred per strategy x format cell; per-organ thinner is expected "
          "for rare organs — flag any starved adversarial organ before the sweep.)")


# --------------------------------------------------------------------------- #
# STEP B/C: query construction + model sweep (reuses anatomy_prior_probe wrapper,
# formats, parser unchanged; positives shared across strategies, negatives differ)
# --------------------------------------------------------------------------- #

def build_query_sets(sample, areas, stats, cfg, rng):
    """One queried set per strategy. Positives (truly present, focally-eligible)
    are SHARED across strategies; only the negatives differ. Returns
    {strategy: (queried[(organ, present_bool)], adv_fellback)} plus shared parts."""
    present, absent = slice_sets(areas, sample.z)
    present_elig = [o for o in present if o in POSITIVE_ELIGIBLE]

    positives = []
    if sample.focus_present and sample.focus_organ in POSITIVE_ELIGIBLE:
        positives.append(sample.focus_organ)
    pool = [o for o in present_elig if o not in positives]
    rng.shuffle(pool)
    for o in pool:
        if len(positives) >= cfg.n_query_present:
            break
        positives.append(o)

    out = {}
    for strat in ("random", "popular", "adversarial"):
        negs, fell = pick_negatives(strat, present, absent, stats, rng, cfg.n_query_absent)
        queried = [(o, True) for o in positives] + [(o, False) for o, _, _ in negs]
        rng.shuffle(queried)
        out[strat] = (queried, fell)
    return out, positives, present, absent


def run_sweep_pope(cfg, stats: dict, out_dir: Path) -> None:
    import time
    from anatomy_prior_probe import (MedGemma, build_prompt, parse_response,
                                      FORMATS, render_slice, sample_volume)
    rng = np.random.default_rng(cfg.seed)
    images = sorted(Path(cfg.images_dir).glob("*.nii*"))
    labels = sorted(Path(cfg.labels_dir).glob("*.nii*"))
    if cfg.max_cases:
        images, labels = images[: cfg.max_cases], labels[: cfg.max_cases]

    case_areas, flat = {}, []
    for img_path, lbl_path in zip(images, labels):
        lblv = load_volume(str(lbl_path))
        case = img_path.name.split(".")[0]
        case_areas[case] = organ_areas(lblv)
        for s in sample_volume(case, lblv, cfg, rng):
            flat.append((str(img_path), case, s))
    full_n = len(flat)
    run = flat[: cfg.limit_n] if cfg.limit_n else flat
    n_calls_full = full_n * 3 * len(FORMATS)
    print(f"[sweep] {full_n} slices total; running {len(run)} x 3 strategies x "
          f"{len(FORMATS)} formats = {len(run)*3*len(FORMATS)} calls "
          f"(full sweep would be {n_calls_full})", file=sys.stderr)

    model = MedGemma(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, call_times, volcache = [], [], {}

    for imgp, case, s in run:
        if imgp not in volcache:
            volcache[imgp] = load_volume(imgp)
        rgb = render_slice(volcache[imgp], s.z, cfg)
        qsets, _pos, _pres, _abs = build_query_sets(s, case_areas[case], stats, cfg, rng)
        sid = f"{case}_z{s.z:03d}_{s.stratum}_{s.focus_organ}"
        for strat, (queried, fell) in qsets.items():
            for fmt in FORMATS:
                prompt = build_prompt(fmt, queried)
                t0 = time.perf_counter()
                raw = model.ask(rgb, prompt)
                call_times.append(time.perf_counter() - t0)
                if len(call_times) == 1:
                    print(f"[timing] first call {call_times[0]:.1f}s -> full sweep "
                          f"{n_calls_full} calls ~ {call_times[0]*n_calls_full/3600:.1f} h",
                          file=sys.stderr)
                parsed = parse_response(raw, fmt, queried)
                for organ, gt in queried:
                    p = parsed[organ]
                    correct = (not p["parse_fail"]) and (p["pred_present"] == gt)
                    records.append({
                        "sample_id": sid, "case": case, "z": s.z, "stratum": s.stratum,
                        "focus_organ": s.focus_organ, "neg_strategy": strat,
                        "adv_fellback": bool(fell) if strat == "adversarial" else False,
                        "queried_set": [o for o, _ in queried],
                        "organ": organ, "ground_truth_present": bool(gt),
                        "format": fmt, "raw": raw,
                        "parsed_present": p["pred_present"], "parse_fail": p["parse_fail"],
                        "correct": bool(correct),
                    })

    if call_times:
        med = float(np.median(call_times))
        print(f"[timing] {len(call_times)} calls, median {med:.1f}s -> full sweep "
              f"{n_calls_full} calls ~ {med*n_calls_full/3600:.1f} h", file=sys.stderr)

    tag = f"pope_smoke_limit{cfg.limit_n}" if cfg.limit_n else "pope_results"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps({"config_seed": cfg.seed, "full_n_slices": full_n,
                                    "median_call_s": float(np.median(call_times)) if call_times else None,
                                    "records": records}, indent=2))
    print(f"[done] {len(records)} records -> {out_path}", file=sys.stderr)
    if cfg.limit_n:
        _print_pope_smoke(run, records)
    else:
        analyze_pope(records, out_dir, n_boot=getattr(cfg, "n_boot", 2000), seed=cfg.seed)


def _print_pope_smoke(run, records) -> None:
    idx = {}
    for r in records:
        idx.setdefault((r["sample_id"], r["neg_strategy"], r["format"]), []).append(r)
    pf = defaultdict(lambda: [0, 0])
    for imgp, case, s in run:
        sid = f"{case}_z{s.z:03d}_{s.stratum}_{s.focus_organ}"
        print("\n" + "=" * 100)
        print(f"{sid}  stratum={s.stratum}")
        for strat in ("random", "popular", "adversarial"):
            key0 = (sid, strat, "freetext")
            if key0 not in idx:
                continue
            qgt = ", ".join(f"{r['organ']}={'P' if r['ground_truth_present'] else 'A'}"
                            for r in idx[key0])
            fell = idx[key0][0]["adv_fellback"]
            print(f"\n  [{strat}{' FELLBACK' if fell else ''}]  GT: {qgt}")
            for fmt in ("freetext", "verdict_first", "reasoning_first"):
                rs = idx[(sid, strat, fmt)]
                cells = []
                for r in rs:
                    pf[fmt][1] += 1
                    if r["parse_fail"]:
                        pf[fmt][0] += 1
                        m = "FAIL"
                    else:
                        m = ("P" if r["parsed_present"] else "A") + ("ok" if r["correct"] else "WRONG")
                    cells.append(f"{r['organ']}:{m}")
                print(f"      {fmt:16s} " + " | ".join(cells))
    print("\n" + "=" * 100)
    print("parse-fail per format:", {f: f"{pf[f][0]}/{pf[f][1]}" for f in
                                     ("freetext", "verdict_first", "reasoning_first")})
    print("STEP B smoke — verify per-strategy negatives + parsed/GT alignment before Step C.")


# --------------------------------------------------------------------------- #
# STEP C aggregator: strategy x format specificity + per-organ McNemar
# --------------------------------------------------------------------------- #

NEG_STRATEGIES = ("random", "popular", "adversarial")
FMT3 = ("freetext", "verdict_first", "reasoning_first")


def analyze_pope(records: list, out_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    from anatomy_prior_probe import (_specificity, _sensitivity, _base_rate,
                                     _bootstrap_ci, _mcnemar, _r)
    cases = sorted({r["case"] for r in records})
    organs = sorted({r["organ"] for r in records})

    def sel(strategy=None, fmt=None, organ=None, present=None):
        out = []
        for r in records:
            if strategy is not None and r["neg_strategy"] != strategy:
                continue
            if fmt is not None and r["format"] != fmt:
                continue
            if organ is not None and r["organ"] != organ:
                continue
            if present is not None and r["ground_truth_present"] != present:
                continue
            out.append(r)
        return out

    print("\n" + "#" * 96)
    print(f"# POPE ANALYSIS  cases={len(cases)}  records={len(records)}  "
          f"bootstrap={n_boot} (clustered on case)")
    print("#" * 96)

    # adversarial fallback rate
    adv = [r for r in records if r["neg_strategy"] == "adversarial"]
    adv_items = {(r["sample_id"], r["organ"]) for r in adv}
    adv_fb = {(r["sample_id"], r["organ"]) for r in adv if r["adv_fellback"]}
    print(f"\nadversarial->popular fallback items: {len(adv_fb)}/{len(adv_items)} "
          f"= {len(adv_fb)/max(1,len(adv_items)):.3f}")

    # HEADLINE: specificity per neg_strategy x format (expect random>=popular>=adversarial)
    print("\n=== HEADLINE — SPECIFICITY by neg_strategy x format "
          "(always-present baseline = 0.000; base_rate per cell) ===")
    print(f"{'strategy':12s} {'format':16s} {'spec':>6s}  {'95% CI':>18s}  {'n_neg':>6s}  {'base_rate':>9s}")
    summary = {"specificity": {}, "sensitivity": {}, "mcnemar_vf_vs_rf": {}}
    for strat in NEG_STRATEGIES:
        for fmt in FMT3:
            items = sel(strategy=strat, fmt=fmt)
            sp, n = _specificity(items)
            lo, hi = _bootstrap_ci(items, _specificity, n_boot, seed)
            br, _ = _base_rate(items)
            summary["specificity"][f"{strat}|{fmt}"] = {"spec": sp, "ci": [lo, hi], "n": n}
            print(f"{strat:12s} {fmt:16s} {_r(sp):>6s}  [{_r(lo)}, {_r(hi)}]  {n:6d}  {_r(br):>9s}")
        print()

    # sensitivity per cell (should stay flat ~0.93)
    print("=== sensitivity by neg_strategy x format (should be flat; positives shared) ===")
    print(f"{'strategy':12s} {'format':16s} {'sens':>6s} {'n_pos':>6s}")
    for strat in NEG_STRATEGIES:
        for fmt in FMT3:
            se, n = _sensitivity(sel(strategy=strat, fmt=fmt))
            summary["sensitivity"][f"{strat}|{fmt}"] = {"sens": se, "n": n}
            print(f"{strat:12s} {fmt:16s} {_r(se):>6s} {n:6d}")

    # per-organ specificity x strategy (the difficulty gradient, per organ)
    print("\n=== per-organ specificity x neg_strategy (freetext+verdict+reasoning pooled "
          "for display; n=absent items) ===")
    print(f"{'organ':20s} " + " ".join(f"{s:>13s}" for s in NEG_STRATEGIES))
    for o in organs:
        cells = []
        for strat in NEG_STRATEGIES:
            sp, n = _specificity(sel(strategy=strat, organ=o))
            cells.append(f"{_r(sp)}({n})")
        print(f"{o:20s} " + " ".join(f"{c:>13s}" for c in cells))

    # per-organ McNemar verdict_first vs reasoning_first on ABSENT items, within
    # (organ, neg_strategy), NEVER pooled.
    print("\n=== McNemar verdict_first vs reasoning_first on ABSENT items, "
          "per (organ, neg_strategy), NEVER pooled ===")
    print(f"{'organ':20s} {'strategy':12s} {'b':>4s} {'c':>4s} {'p':>8s}")
    for o in organs:
        for strat in NEG_STRATEGIES:
            ivf = sel(strategy=strat, fmt="verdict_first", organ=o, present=False)
            irf = sel(strategy=strat, fmt="reasoning_first", organ=o, present=False)
            bc, cc, p = _mcnemar(ivf, irf)
            if bc + cc > 0:
                summary["mcnemar_vf_vs_rf"][f"{o}|{strat}"] = {"b": bc, "c": cc, "p": p}
                print(f"{o:20s} {strat:12s} {bc:4d} {cc:4d} {_r(p):>8s}")

    out_path = Path(out_dir) / "pope_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[analysis] summary -> {out_path}")
    return summary


def analyze_pope_file(path: str, n_boot: int, seed: int) -> None:
    data = json.loads(Path(path).read_text())
    analyze_pope(data["records"], Path(path).parent, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", default=str(PROJECT_ROOT / "data/btcv/RawData/Training/img"))
    ap.add_argument("--labels_dir", default=str(PROJECT_ROOT / "data/btcv/RawData/Training/label"))
    ap.add_argument("--n_dry_slices", type=int, default=8)
    ap.add_argument("--k_neg", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true",
                    help="recompute btcv_organ_stats.json even if it exists")
    ap.add_argument("--run", action="store_true",
                    help="Step B/C: run the model sweep (default is Step A: stats + dry-run)")
    ap.add_argument("--limit_n", type=int, default=0,
                    help="Step B smoke: run only the first N slices end-to-end")
    ap.add_argument("--max_cases", type=int, default=0, help="cap cases (<=0 = all)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--analyze", default=None,
                    help="Step-C analysis only: aggregate an existing pope results JSON")
    a = ap.parse_args()

    if a.analyze:
        analyze_pope_file(a.analyze, n_boot=a.n_boot, seed=a.seed)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if STATS_PATH.exists() and not a.recompute:
        print(f"[stats] loading cached {STATS_PATH}")
        stats = json.loads(STATS_PATH.read_text())
    else:
        print("[stats] computing organ frequency + co-occurrence over all cases ...")
        stats = compute_stats(a.images_dir, a.labels_dir)
        STATS_PATH.write_text(json.dumps(stats, indent=2))
        print(f"[stats] saved -> {STATS_PATH}")

    if not a.run:
        print_stats(stats)
        dry_run_samplers(a.images_dir, a.labels_dir, stats, a.n_dry_slices, a.seed, a.k_neg)
        project_cells(a.images_dir, a.labels_dir, stats, a.seed, a.k_neg)
        return

    # Step B/C: build a run config from the validated anatomy harness defaults.
    from anatomy_prior_probe import Config as ACfg
    cfg = ACfg()
    cfg.images_dir, cfg.labels_dir = a.images_dir, a.labels_dir
    cfg.seed = a.seed
    cfg.limit_n = a.limit_n
    cfg.max_cases = None if a.max_cases <= 0 else a.max_cases
    cfg.n_boot = a.n_boot
    run_sweep_pope(cfg, stats, OUT_DIR)


if __name__ == "__main__":
    main()
