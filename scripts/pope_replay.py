#!/usr/bin/env python3
"""
pope_replay.py
==============
Cross-model generalization for the POPE organ-presence result. Replays the EXACT
paired items from the MedGemma-27B run (results/medgemma_pope_negsampling/
pope_results.json) through other models — NO re-sampling — so the comparison is
fully paired on (case, z, organ, neg_strategy) and queried-set order. Same three
format prompts (textually identical, reasoning fields before the verdict map),
same greedy decoding, same present/absent ground truth. Only the model and its
image-token/chat-template wrapping change.

Parsing safeguards (critical):
  parse_fail          : JSON unparseable / no extractable verdict map at all.
  organ_not_addressed : output produced but this organ was not cleanly answered
                        (missing JSON key, unrecognised value, or rambly free-text
                        that neither names the organ nor gives a clean list/none).
Specificity/sensitivity are computed ONLY over cleanly-resolved items
(not parse_fail AND not organ_not_addressed); both rates are reported alongside.

Decide-and-fix rule (applied consistently): a (model, format) cell whose
clean-resolution rate is < 0.70 is excluded from the headline table and reported
in an appendix with its resolution rate.

Build order: medgemma4b (smoke --limit_n 8, then full); llavamed (smoke, STOP for
review); qwen (smoke, then full if clean).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from anatomy_prior_probe import (  # noqa: E402
    FORMATS, VOCAB, build_prompt, render_slice, load_volume, Config,
    ORGAN_SYNONYMS, _extract_json, _presence_value, match_organ, _norm,
)

REF = PROJECT_ROOT / "results/medgemma_pope_negsampling/pope_results.json"
IMG_DIR = PROJECT_ROOT / "data/btcv/RawData/Training/img"
OUT_DIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
MODEL_PATHS = {
    "medgemma4b":  PROJECT_ROOT / "models/medgemma-4b-it",
    "medgemma27b": PROJECT_ROOT / "models/medgemma-27b-it",
    "llavamed":    PROJECT_ROOT / "models/llava-med-v1.5-mistral-7b",
    "qwen":        PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct",
}


# --------------------------------------------------------------------------- #
# Exact-replay item loader (reconstruct the paired query items, no re-sampling)
# --------------------------------------------------------------------------- #

def load_items(ref_path: Path) -> list:
    recs = json.loads(ref_path.read_text())["records"]
    groups: dict = {}
    for r in recs:
        key = (r["sample_id"], r["neg_strategy"])
        g = groups.get(key)
        if g is None:
            g = groups[key] = {
                "sample_id": r["sample_id"], "case": r["case"], "z": r["z"],
                "stratum": r["stratum"], "focus_organ": r["focus_organ"],
                "neg_strategy": r["neg_strategy"], "adv_fellback": r["adv_fellback"],
                "queried_set": r["queried_set"], "gt": {},
            }
        g["gt"][r["organ"]] = r["ground_truth_present"]
    items = []
    for g in groups.values():
        g["queried"] = [(o, g["gt"][o]) for o in g["queried_set"]]
        items.append(g)
    items.sort(key=lambda x: (x["case"], x["z"], x["neg_strategy"]))
    return items


# --------------------------------------------------------------------------- #
# Parsing with parse_fail vs organ_not_addressed separated
# --------------------------------------------------------------------------- #

def _freetext_is_clean(resp: str) -> bool:
    """A clean free-text answer is a short organ list or 'none' — not rambly prose."""
    t = resp.strip().lower()
    if not t:
        return False
    if t == "none" or t.startswith("none"):
        return True
    chunks = [c.strip() for c in re.split(r"[,\n]", t) if c.strip()]
    return bool(chunks) and len(t) <= 160 and all(len(c.split()) <= 4 for c in chunks)


def parse_replay(raw: str, fmt: str, queried: list) -> dict:
    organs = [o for o, _ in queried]
    pf_all = {o: {"pred_present": None, "parse_fail": True, "not_addressed": False} for o in organs}

    if fmt in ("verdict_first", "reasoning_first"):
        obj = _extract_json(raw)
        if obj is None:
            return pf_all
        pmap = obj.get("presence") if fmt == "reasoning_first" and isinstance(obj.get("presence"), dict) else obj
        if not isinstance(pmap, dict):
            return pf_all
        out = {}
        for o in organs:
            val = _presence_value(pmap, o)        # 'present'/'absent'/None(unrecog)/'MISSING'
            if val in ("MISSING", None):
                out[o] = {"pred_present": None, "parse_fail": False, "not_addressed": True}
            else:
                out[o] = {"pred_present": val == "present", "parse_fail": False, "not_addressed": False}
        return out

    # free-text
    if not raw.strip():
        return pf_all
    clean = _freetext_is_clean(raw)
    low = _norm(raw)
    is_none = low == "none" or low.startswith("none")
    out = {}
    for o in organs:
        if match_organ(low, o):
            out[o] = {"pred_present": True, "parse_fail": False, "not_addressed": False}
        elif is_none or clean:
            out[o] = {"pred_present": False, "parse_fail": False, "not_addressed": False}
        else:
            out[o] = {"pred_present": None, "parse_fail": False, "not_addressed": True}
    return out


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

_VOCAB_DISP = [o.replace("_", " ") for o in VOCAB]


def freetext_recitation(raw: str) -> bool:
    """True if a free-text response echoes the prompt's candidate/vocabulary list
    (prompt-echo) rather than describing the few organs actually visible. Only
    image-description 'present' calls are a comparable fabrication signal; a
    recitation 'present' is the model parroting our option list. Heuristic: it
    names a large fraction of the 13-organ vocabulary, or reproduces a long run
    of the vocabulary in the prompt's canonical order verbatim."""
    t = _norm(raw)
    distinct = sum(any(s in t for s in ORGAN_SYNONYMS[o]) for o in VOCAB)
    if distinct >= 8:
        return True
    best_run = 0
    for start in range(len(_VOCAB_DISP)):
        for end in range(start + 6, len(_VOCAB_DISP) + 1):
            if ", ".join(_VOCAB_DISP[start:end]) in t:
                best_run = max(best_run, end - start)
    return best_run >= 6


def build_model(name: str, cfg: Config):
    if name in ("medgemma4b", "medgemma27b"):
        from anatomy_prior_probe import MedGemma
        cfg.model_id = str(MODEL_PATHS[name])
        return MedGemma(cfg)
    if name == "llavamed":
        from pope_replay_models import LlavaMed
        return LlavaMed(cfg, str(MODEL_PATHS["llavamed"]))
    if name == "qwen":
        from pope_replay_models import QwenVL
        return QwenVL(cfg, str(MODEL_PATHS["qwen"]))
    raise ValueError(f"unknown model {name}")


# --------------------------------------------------------------------------- #
# Replay sweep
# --------------------------------------------------------------------------- #

def maybe_subsample(items: list, n: int) -> list:
    """Deterministic stratified subsample of N items (equal per neg_strategy),
    persisted to a keyfile so every condition/model replays the SAME items and
    the with-image numbers can be restricted to the same subset."""
    if not n:
        return items
    keyfile = OUT_DIR / f"noimg_subsample_{n}.json"
    if keyfile.exists():
        keys = {tuple(k) for k in json.loads(keyfile.read_text())}
        return [it for it in items if (it["sample_id"], it["neg_strategy"]) in keys]
    rng = np.random.default_rng(0)
    by = defaultdict(list)
    for it in items:
        by[it["neg_strategy"]].append(it)
    per = n // len(by)
    sel = []
    for strat in sorted(by):
        lst = by[strat]
        idx = rng.choice(len(lst), size=min(per, len(lst)), replace=False)
        sel += [lst[i] for i in idx]
    sel.sort(key=lambda x: (x["case"], x["z"], x["neg_strategy"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keyfile.write_text(json.dumps([[it["sample_id"], it["neg_strategy"]] for it in sel]))
    return sel


def run_replay(model_name: str, cfg: Config) -> None:
    items = maybe_subsample(load_items(REF), getattr(cfg, "subsample", 0))
    run = items[: cfg.limit_n] if cfg.limit_n else items
    n_calls_full = len(items) * len(FORMATS)
    print(f"[replay] {len(items)} paired query items; running {len(run)} x {len(FORMATS)} "
          f"formats = {len(run)*len(FORMATS)} calls (full = {n_calls_full})", file=sys.stderr)

    no_image = getattr(cfg, "no_image", None)   # None | "textonly" | "blank"
    blank_img = np.full((896, 896, 3), 128, np.uint8) if no_image == "blank" else None
    print(f"[replay] image mode = {no_image or 'with_image'}", file=sys.stderr)

    model = build_model(model_name, cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, call_times, volcache = [], [], {}

    for it in run:
        if no_image == "textonly":
            rgb = None
        elif no_image == "blank":
            rgb = blank_img
        else:
            if it["case"] not in volcache:
                volcache[it["case"]] = load_volume(str(IMG_DIR / f"{it['case']}.nii.gz"))
            rgb = render_slice(volcache[it["case"]], it["z"], cfg)
        for fmt in FORMATS:
            prompt = build_prompt(fmt, it["queried"])
            t0 = time.perf_counter()
            raw = model.ask(rgb, prompt)
            call_times.append(time.perf_counter() - t0)
            if len(call_times) == 1:
                print(f"[timing] first call {call_times[0]:.1f}s -> full {n_calls_full} calls "
                      f"~ {call_times[0]*n_calls_full/3600:.1f} h", file=sys.stderr)
            parsed = parse_replay(raw, fmt, it["queried"])
            recite = freetext_recitation(raw) if fmt == "freetext" else None
            for organ, gt in it["queried"]:
                p = parsed[organ]
                clean = (not p["parse_fail"]) and (not p["not_addressed"])
                correct = clean and (p["pred_present"] == gt)
                records.append({
                    "sample_id": it["sample_id"], "case": it["case"], "z": it["z"],
                    "stratum": it["stratum"], "focus_organ": it["focus_organ"],
                    "neg_strategy": it["neg_strategy"], "adv_fellback": it["adv_fellback"],
                    "queried_set": it["queried_set"], "organ": organ,
                    "ground_truth_present": bool(gt), "format": fmt, "model": model_name,
                    "image_mode": no_image or "with_image",
                    "raw": raw, "parsed_present": p["pred_present"],
                    "parse_fail": p["parse_fail"], "organ_not_addressed": p["not_addressed"],
                    "freetext_recitation": recite, "correct": bool(correct),
                })

    if call_times:
        med = float(np.median(call_times))
        print(f"[timing] {len(call_times)} calls, median {med:.1f}s -> full {n_calls_full} "
              f"~ {med*n_calls_full/3600:.1f} h", file=sys.stderr)

    suffix = f"_noimg_{no_image}" if no_image else ""
    tag = (f"pope_results_{model_name}{suffix}"
           + (f"_smoke{cfg.limit_n}" if cfg.limit_n else ""))
    out_path = OUT_DIR / f"{tag}.json"
    out_path.write_text(json.dumps({"model": model_name, "n_items": len(run), "records": records}, indent=2))
    print(f"[done] {len(records)} records -> {out_path}", file=sys.stderr)

    _resolution_report(records, model_name)
    if cfg.limit_n:
        _print_smoke(run, records, model_name)


def _resolution_report(records: list, model_name: str) -> None:
    print(f"\n=== resolution report [{model_name}] (clean = not parse_fail AND not not_addressed) ===")
    print(f"{'format':16s} {'parse_fail':>11s} {'not_addr':>9s} {'clean_rate':>11s} {'n':>7s}")
    for fmt in FORMATS:
        items = [r for r in records if r["format"] == fmt]
        n = len(items)
        if not n:
            continue
        pf = sum(r["parse_fail"] for r in items) / n
        na = sum(r["organ_not_addressed"] for r in items) / n
        clean = sum((not r["parse_fail"]) and (not r["organ_not_addressed"]) for r in items) / n
        flag = "  <0.70 -> APPENDIX ONLY" if clean < 0.70 else ""
        print(f"{fmt:16s} {pf:11.3f} {na:9.3f} {clean:11.3f} {n:7d}{flag}")


def _print_smoke(run, records, model_name) -> None:
    idx = defaultdict(list)
    for r in records:
        idx[(r["sample_id"], r["neg_strategy"], r["format"])].append(r)
    for it in run:
        print("\n" + "=" * 100)
        print(f"{it['sample_id']}  strat={it['neg_strategy']}  GT: " +
              ", ".join(f"{o}={'P' if g else 'A'}" for o, g in it["queried"]))
        for fmt in FORMATS:
            rs = idx[(it["sample_id"], it["neg_strategy"], fmt)]
            raw = rs[0]["raw"].replace("\n", " ") if rs else ""
            print(f"\n  [{fmt}] raw: {raw[:240]}")
            cells = []
            for r in rs:
                if r["parse_fail"]:
                    m = "PARSE_FAIL"
                elif r["organ_not_addressed"]:
                    m = "NOT_ADDR"
                else:
                    m = ("P" if r["parsed_present"] else "A") + ("ok" if r["correct"] else "WRONG")
                cells.append(f"{r['organ']}:{m}")
            print("      " + " | ".join(cells))
    print("\n" + "=" * 100)
    print(f"SMOKE [{model_name}] — verify raw formatting, that all queried organs are addressed, "
          "and that parse_fail/not_addressed are categorised correctly.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS))
    ap.add_argument("--limit_n", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--no_pan_and_scan", action="store_true")
    ap.add_argument("--no_image", choices=["textonly", "blank"], default=None,
                    help="no-image control: remove the image (text-only) or use a blank gray 896 image")
    ap.add_argument("--subsample", type=int, default=0,
                    help="stratified subsample of N paired items (0 = all); shared across conditions")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    cfg = Config()
    cfg.limit_n = a.limit_n
    cfg.max_new_tokens = a.max_new_tokens
    cfg.do_pan_and_scan = not a.no_pan_and_scan
    cfg.no_image = a.no_image
    cfg.subsample = a.subsample
    run_replay(a.model, cfg)


if __name__ == "__main__":
    main()
