#!/usr/bin/env python3
"""Stage 3 — comparable numbers for the base-Gemma3 ablation, reusing the EXACT
existing analysis machinery (pope_abstention's 14-cue is_hedged + clean rule;
pope_bootstrap_cis's case-clustered ci()/per_case(), seed 0, 10k reps). Reports
Gemma3-4B beside MedGemma-4B. Report-only; reads JSONs, writes nothing but stdout.

  tab:crossmodel  -> free-text specificity (point) at random / popular / adversarial
  tab:abstention  -> adversarial reasoning-first: binary spec, abstention rate,
                     confident-answer spec + case-clustered 95% CI, n_conf / n_neg
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pope_abstention import is_clean, is_hedged, binary_spec  # noqa: E402
from pope_bootstrap_cis import ci, per_case, load, fmt        # noqa: E402

RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
PAIRS = {
    "4b":  {"Gemma3-4B (base)":   RDIR / "gemma3_4b_pope.json",
            "MedGemma-4B (med)":  RDIR / "pope_results_medgemma4b.json"},
    "27b": {"Gemma3-27B (base)":  RDIR / "gemma3_27b_pope.json",
            "MedGemma-27B (med)": RDIR / "pope_results.json"},
}
STRATS = ("random", "popular", "adversarial")
MODELS = PAIRS["4b"]   # default; overridden in main()


def main():
    global MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="4b", choices=list(PAIRS))
    MODELS = PAIRS[ap.parse_args().tier]
    recs = {name: load(p) for name, p in MODELS.items()}

    print("=" * 74)
    print("TAB:CROSSMODEL  — free-text specificity (point), by negative strategy")
    print("=" * 74)
    print(f"{'model':20s} {'random':>10s} {'popular':>10s} {'adversarial':>12s}")
    for name, rs in recs.items():
        cells = []
        for strat in STRATS:
            items = [r for r in rs if r["neg_strategy"] == strat and r["format"] == "freetext"]
            sp, n = binary_spec(items)
            cells.append(f"{sp:.4f}")
        print(f"{name:20s} {cells[0]:>10s} {cells[1]:>10s} {cells[2]:>12s}")

    print("\n" + "=" * 74)
    print("TAB:ABSTENTION  — adversarial, reasoning-first (case-clustered 95% CI, seed 0)")
    print("=" * 74)
    print(f"{'model':20s} {'binary_spec[CI]':>22s} {'abstain':>8s} "
          f"{'confident_spec[CI]':>24s} {'n_conf':>7s} {'n_neg':>6s}")
    for name, rs in recs.items():
        adv = [r for r in rs if r["format"] == "reasoning_first" and r["neg_strategy"] == "adversarial"]
        # binary specificity (all clean negatives) + CI
        n, d = per_case(adv, lambda r: (not r["ground_truth_present"]) and is_clean(r),
                        lambda r: not r["parsed_present"])
        bp, blo, bhi, n_neg = ci(n, d)
        # abstention rate among clean adversarial reasoning-first items (pos+neg)
        ans = [r for r in adv if is_clean(r)]
        ar = sum(is_hedged(r["raw"]) for r in ans) / len(ans) if ans else float("nan")
        # confident-only specificity (clean, NOT hedged) + CI
        nc, dc = per_case(adv, lambda r: (not r["ground_truth_present"]) and is_clean(r)
                          and not is_hedged(r["raw"]),
                          lambda r: not r["parsed_present"])
        cp, clo, chi, n_conf = ci(nc, dc)
        print(f"{name:20s} {f'{bp:.3f} [{blo:.3f},{bhi:.3f}]':>22s} {ar:8.3f} "
              f"{f'{cp:.3f} [{clo:.3f},{chi:.3f}]':>24s} {n_conf:7d} {n_neg:6d}")

    print("\n--- LaTeX-ready (Gemma3 row, to drop beside MedGemma-4B) ---")
    base_name = next(n for n in MODELS if "base" in n)
    g = recs[base_name]
    for strat in STRATS:
        sp, _ = binary_spec([r for r in g if r["neg_strategy"] == strat and r["format"] == "freetext"])
        print(f"  crossmodel freetext {strat:12s} -> ${fmt(sp)}$")


if __name__ == "__main__":
    main()
