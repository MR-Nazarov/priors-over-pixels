#!/usr/bin/env python3
"""
pope_abstention.py
==================
Abstention/hedge diagnostic for the cross-model POPE results. Binary scoring
stays primary and UNCHANGED (the presence-map value is the binary answer); this
only ADDS two reported dimensions that explain *how* each model reaches its
specificity, without rescoring into a third category. No model re-runs.

  1. abstention_rate[model][format] : fraction of answered (cleanly-resolved)
     items whose response text contains an uncertainty cue.
  2. confident_only_specificity[model][format][strategy] : specificity computed
     only over CONFIDENT true-negative answers — hedged items excluded from BOTH
     numerator AND denominator (a robustness check, not the headline).

Binary specificity/sensitivity are recomputed here too, only so the delta table
(binary vs confident-only) is self-contained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from anatomy_prior_probe import _norm, _r  # noqa: E402

RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
MODEL_FILES = {
    "medgemma4b":  RDIR / "pope_results_medgemma4b.json",
    "medgemma27b": RDIR / "pope_results.json",
    "llavamed":    RDIR / "pope_results_llavamed.json",
    "qwen":        RDIR / "pope_results_qwen.json",
}
MODEL_ORDER = ["medgemma4b", "medgemma27b", "llavamed", "qwen"]
STRATS = ("random", "popular", "adversarial")
FORMATS = ("freetext", "verdict_first", "reasoning_first")

# uncertainty cues -> "abstention". short ambiguous token "may" matched padded.
HEDGE_PHRASES = ["not clear", "not clearly", "not well", "not distinctly", "not directly",
                 "obscured", "difficult", "cannot confirm", "no definite", "unclear",
                 "possibly", "appears", "inferred"]
HEDGE_PADDED = ["may"]


def is_hedged(raw: str) -> bool:
    t = " " + _norm(raw) + " "
    if any(p in t for p in HEDGE_PHRASES):
        return True
    return any(f" {w} " in t for w in HEDGE_PADDED)


def is_clean(r) -> bool:
    return (not r["parse_fail"]) and (not r.get("organ_not_addressed", False))


def load(model, path):
    recs = json.loads(path.read_text())["records"]
    for r in recs:
        r.setdefault("organ_not_addressed", False)
    return recs


def binary_spec(items):
    neg = [r for r in items if (not r["ground_truth_present"]) and is_clean(r)]
    return (sum(not r["parsed_present"] for r in neg) / len(neg), len(neg)) if neg else (float("nan"), 0)


def confident_spec(items):
    """Specificity over confident (non-hedged) true-negative answers; hedged
    excluded from numerator AND denominator."""
    conf = [r for r in items if (not r["ground_truth_present"]) and is_clean(r)
            and not is_hedged(r["raw"])]
    return (sum(not r["parsed_present"] for r in conf) / len(conf), len(conf)) if conf else (float("nan"), 0)


def main() -> None:
    models = {m: load(m, p) for m, p in MODEL_FILES.items() if p.exists()}
    present = [m for m in MODEL_ORDER if m in models]

    out = {"abstention_rate": {}, "confident_only_specificity": {}, "binary_specificity": {}}

    print("=== ABSTENTION / HEDGE RATE  (fraction of answered items with an uncertainty cue) ===")
    print(f"{'model':12s} {'format':16s} {'abstention':>10s} {'n_answered':>11s}")
    for m in present:
        out["abstention_rate"][m] = {}
        for fmt in FORMATS:
            ans = [r for r in models[m] if r["format"] == fmt and is_clean(r)]
            ar = sum(is_hedged(r["raw"]) for r in ans) / len(ans) if ans else float("nan")
            out["abstention_rate"][m][fmt] = ar
            print(f"{m:12s} {fmt:16s} {_r(ar):>10s} {len(ans):11d}")

    # binary + confident-only specificity per model x format x strategy
    for m in present:
        out["binary_specificity"][m] = {}
        out["confident_only_specificity"][m] = {}
        for fmt in FORMATS:
            out["binary_specificity"][m][fmt] = {}
            out["confident_only_specificity"][m][fmt] = {}
            for strat in STRATS:
                items = [r for r in models[m] if r["neg_strategy"] == strat and r["format"] == fmt]
                bs, bn = binary_spec(items)
                cs, cn = confident_spec(items)
                out["binary_specificity"][m][fmt][strat] = {"spec": bs, "n": bn}
                out["confident_only_specificity"][m][fmt][strat] = {"spec": cs, "n": cn}

    # delta table at adversarial
    print("\n=== ADVERSARIAL: binary spec vs abstention vs confident-only spec ===")
    print(f"{'model':12s} {'format':16s} {'binary_spec':>11s} {'abstain':>8s} "
          f"{'confident_spec':>14s} {'n_conf':>7s} {'n_neg':>6s}")
    for m in present:
        for fmt in FORMATS:
            bs = out["binary_specificity"][m][fmt]["adversarial"]
            cs = out["confident_only_specificity"][m][fmt]["adversarial"]
            ar = out["abstention_rate"][m][fmt]
            print(f"{m:12s} {fmt:16s} {_r(bs['spec']):>11s} {_r(ar):>8s} "
                  f"{_r(cs['spec']):>14s} {cs['n']:7d} {bs['n']:6d}")

    (RDIR / "pope_abstention.json").write_text(json.dumps(out, indent=2))
    print(f"\n[abstention] -> {RDIR / 'pope_abstention.json'}")


if __name__ == "__main__":
    main()
