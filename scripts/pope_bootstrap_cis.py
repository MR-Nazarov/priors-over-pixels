#!/usr/bin/env python3
"""Case-clustered bootstrap 95% CIs for tab:pope and tab:abstention.
Resample the 30 BTCV cases with replacement (not per-query), recompute the
proportion within each replicate. 10k replicates, percentile (2.5/97.5),
fixed seed. Emits raw [lo,hi] and LaTeX-ready snippets."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pope_abstention import is_clean, is_hedged  # noqa: E402

RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
N_BOOT, SEED = 10000, 0
STRATS = ("random", "popular", "adversarial")
FORMATS = ("freetext", "verdict_first", "reasoning_first")


def load(p):
    recs = json.loads(Path(p).read_text())["records"]
    for r in recs:
        r.setdefault("organ_not_addressed", False)
    return recs


def ci(num_by_case, den_by_case):
    """Point estimate + percentile CI from per-case (numerator, denominator)."""
    num = np.array(num_by_case, float)
    den = np.array(den_by_case, float)
    point = num.sum() / den.sum() if den.sum() else float("nan")
    rng = np.random.default_rng(SEED)
    nC = len(num)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, nC, nC)
        d = den[idx].sum()
        vals[b] = num[idx].sum() / d if d > 0 else np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, lo, hi, int(den.sum())


def per_case(items, keep, hit):
    """Return (numerator_by_case, denominator_by_case) over cases, where keep(r)
    selects denominator items and hit(r) selects numerator items."""
    num, den = defaultdict(int), defaultdict(int)
    cases = sorted({r["case"] for r in items})
    for r in items:
        if keep(r):
            den[r["case"]] += 1
            if hit(r):
                num[r["case"]] += 1
    return [num[c] for c in cases], [den[c] for c in cases]


def fmt(x):
    return "nan" if x != x else ("%.3f" % x).lstrip("0").replace("-0.", "-.")


def cell(point, lo, hi):
    return f"${fmt(point)}$ [${fmt(lo)},{fmt(hi)}$]"


# --------------------------------------------------------------------------- #
def main():
    mg27 = load(RDIR / "pope_results.json")

    print("=" * 70, "\nTAB:POPE  (MedGemma-27B)\n" + "=" * 70)
    spec_tab, sens_tab = {}, {}
    for fmt_ in FORMATS:
        for strat in STRATS:
            items = [r for r in mg27 if r["format"] == fmt_ and r["neg_strategy"] == strat]
            n, d = per_case(items, lambda r: (not r["ground_truth_present"]) and is_clean(r),
                            lambda r: not r["parsed_present"])
            p, lo, hi, nn = ci(n, d)
            spec_tab[(strat, fmt_)] = (p, lo, hi, nn)
            print(f"  spec  {fmt_:15s} {strat:11s} {p:.3f} [{lo:.3f},{hi:.3f}]  n={nn}")
        pit = [r for r in mg27 if r["format"] == fmt_]
        n, d = per_case(pit, lambda r: r["ground_truth_present"] and is_clean(r),
                        lambda r: r["parsed_present"])
        p, lo, hi, nn = ci(n, d)
        sens_tab[fmt_] = (p, lo, hi, nn)
        print(f"  sens  {fmt_:15s} {'(all)':11s} {p:.3f} [{lo:.3f},{hi:.3f}]  n={nn}")

    print("\n--- LaTeX: tab:pope ---")
    print(r"    \multicolumn{4}{l}{\emph{Specificity} [95\% CI]} \\")
    for strat in STRATS:
        cells = " & ".join(cell(*spec_tab[(strat, f)][:3]) for f in FORMATS)
        bold = cells
        if strat == "adversarial":  # bold the verdict-first adversarial point as in the draft
            p, lo, hi, _ = spec_tab[("adversarial", "verdict_first")]
            bold = " & ".join(
                (f"$\\mathbf{{{fmt(spec_tab[(strat,f)][0])}}}$ [${fmt(spec_tab[(strat,f)][1])},{fmt(spec_tab[(strat,f)][2])}$]"
                 if f == "verdict_first" else cell(*spec_tab[(strat, f)][:3])) for f in FORMATS)
        print(f"    {strat:11s} & {bold} \\\\")
    print(r"    \midrule")
    print(r"    \multicolumn{4}{l}{\emph{Sensitivity} [95\% CI]} \\")
    scells = " & ".join(cell(*sens_tab[f][:3]) for f in FORMATS)
    print(f"    (all strategies) & {scells} \\\\")

    # ----- tab:abstention: adversarial reasoning-first -----
    print("\n" + "=" * 70, "\nTAB:ABSTENTION  (adversarial, reasoning-first)\n" + "=" * 70)
    models = {"MedGemma-4B": RDIR / "pope_results_medgemma4b.json",
              "MedGemma-27B": RDIR / "pope_results.json",
              "Qwen2.5-VL-7B": RDIR / "pope_results_qwen.json"}
    abst = {}
    for name, path in models.items():
        recs = [r for r in load(path)
                if r["format"] == "reasoning_first" and r["neg_strategy"] == "adversarial"]
        # binary specificity (all clean negatives)
        n, d = per_case(recs, lambda r: (not r["ground_truth_present"]) and is_clean(r),
                        lambda r: not r["parsed_present"])
        bp, blo, bhi, n_neg = ci(n, d)
        # confident-only specificity (clean, NOT hedged)
        nc, dc = per_case(recs, lambda r: (not r["ground_truth_present"]) and is_clean(r)
                          and not is_hedged(r["raw"]),
                          lambda r: not r["parsed_present"])
        cp, clo, chi, n_conf = ci(nc, dc)
        abst[name] = dict(binary=(bp, blo, bhi, n_neg), conf=(cp, clo, chi, n_conf))
        print(f"  {name:14s} binary {bp:.3f} [{blo:.3f},{bhi:.3f}] n_neg={n_neg} | "
              f"confident {cp:.3f} [{clo:.3f},{chi:.3f}] n_conf={n_conf}")

    print("\n--- LaTeX: tab:abstention confident-spec cells ([CI] fill) ---")
    for name in models:
        cp, clo, chi, n_conf = abst[name]["conf"]
        print(f"    {name:13s}  confident spec cell -> ${fmt(cp)}$ [${fmt(clo)},{fmt(chi)}$]")
    print("\n(binary-spec CIs, if wanted:)")
    for name in models:
        bp, blo, bhi, _ = abst[name]["binary"]
        print(f"    {name:13s}  binary spec CI -> [${fmt(blo)},{fmt(bhi)}$]")


if __name__ == "__main__":
    main()
