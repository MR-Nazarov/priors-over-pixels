#!/usr/bin/env python3
"""
pope_noimg_delta.py
===================
No-image control aggregator. Compares specificity / sensitivity / abstention for
the no-image conditions (text-only, blank-gray) against the with-image numbers,
for MedGemma-27B and Qwen2.5-VL. Same paired items, same parser; the ONLY change
is the image. If the present-bias persists with no image, it is prior-driven.

Reads whatever result files exist, so it can be run before all four no-image
sweeps finish. Writes pope_noimg_summary.json + prints the with-vs-without
adversarial delta table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from anatomy_prior_probe import _bootstrap_ci, _r  # noqa: E402
from pope_abstention import is_clean, is_hedged    # noqa: E402

RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
STRATS = ("random", "popular", "adversarial")
FORMATS = ("freetext", "verdict_first", "reasoning_first")

# model -> {condition: results file}
SOURCES = {
    "medgemma27b": {
        "with_image": RDIR / "pope_results.json",
        "textonly":   RDIR / "pope_results_medgemma27b_noimg_textonly.json",
        "blank":      RDIR / "pope_results_medgemma27b_noimg_blank.json",
    },
    "qwen": {
        "with_image": RDIR / "pope_results_qwen.json",
        "textonly":   RDIR / "pope_results_qwen_noimg_textonly.json",
        "blank":      RDIR / "pope_results_qwen_noimg_blank.json",
    },
}
CONDS = ("with_image", "textonly", "blank")


SUBSAMPLE = 1500   # 0 = full; else restrict with-image to the no-image subset for a paired delta
_SUBKEYS = None
if SUBSAMPLE:
    kf = RDIR / f"noimg_subsample_{SUBSAMPLE}.json"
    if kf.exists():
        _SUBKEYS = {tuple(k) for k in json.loads(kf.read_text())}


def load(path: Path):
    recs = json.loads(path.read_text())["records"]
    for r in recs:
        r.setdefault("organ_not_addressed", False)
    # restrict with-image (full) records to the no-image subsample for a paired delta
    if _SUBKEYS is not None:
        recs = [r for r in recs if (r["sample_id"], r["neg_strategy"]) in _SUBKEYS]
    return recs


def specificity(items):
    neg = [r for r in items if (not r["ground_truth_present"]) and is_clean(r)]
    return (sum(not r["parsed_present"] for r in neg) / len(neg), len(neg)) if neg else (float("nan"), 0)


def sensitivity(items):
    pos = [r for r in items if r["ground_truth_present"] and is_clean(r)]
    return (sum(r["parsed_present"] for r in pos) / len(pos), len(pos)) if pos else (float("nan"), 0)


def abstention(items):
    ans = [r for r in items if is_clean(r)]
    return (sum(is_hedged(r["raw"]) for r in ans) / len(ans), len(ans)) if ans else (float("nan"), 0)


def main() -> None:
    summary = {}
    for model, srcs in SOURCES.items():
        present = {c: load(p) for c, p in srcs.items() if p.exists()}
        if not present:
            continue
        summary[model] = {}
        for cond, recs in present.items():
            summary[model][cond] = {"specificity": {}, "sensitivity": {}, "abstention": {}}
            for fmt in FORMATS:
                ab, _ = abstention([r for r in recs if r["format"] == fmt])
                summary[model][cond]["abstention"][fmt] = ab
                for strat in STRATS:
                    items = [r for r in recs if r["neg_strategy"] == strat and r["format"] == fmt]
                    sp, n = specificity(items)
                    lo, hi = _bootstrap_ci(items, specificity, 2000, 0)
                    se, _ = sensitivity(items)
                    summary[model][cond]["specificity"][f"{strat}|{fmt}"] = {"spec": sp, "ci": [lo, hi], "n": n}
                    summary[model][cond]["sensitivity"][f"{strat}|{fmt}"] = {"sens": se}

    (RDIR / "pope_noimg_summary.json").write_text(json.dumps(summary, indent=2))

    # delta table at adversarial
    print("=== NO-IMAGE CONTROL: adversarial specificity WITH image vs WITHOUT ===")
    print(f"{'model':12s} {'format':16s} {'with_img':>9s} {'textonly':>9s} {'blank':>8s}  "
          f"{'sens(wi/to)':>14s}  {'abstain(wi/to)':>16s}")
    for model in SOURCES:
        if model not in summary:
            continue
        for fmt in FORMATS:
            row = {}
            for c in CONDS:
                if c in summary[model]:
                    row[c] = summary[model][c]["specificity"].get(f"adversarial|{fmt}", {}).get("spec", float("nan"))
                else:
                    row[c] = float("nan")
            sens_wi = summary[model].get("with_image", {}).get("sensitivity", {}).get(f"adversarial|{fmt}", {}).get("sens", float("nan"))
            sens_to = summary[model].get("textonly", {}).get("sensitivity", {}).get(f"adversarial|{fmt}", {}).get("sens", float("nan"))
            ab_wi = summary[model].get("with_image", {}).get("abstention", {}).get(fmt, float("nan"))
            ab_to = summary[model].get("textonly", {}).get("abstention", {}).get(fmt, float("nan"))
            print(f"{model:12s} {fmt:16s} {_r(row['with_image']):>9s} {_r(row['textonly']):>9s} "
                  f"{_r(row['blank']):>8s}  {_r(sens_wi)+'/'+_r(sens_to):>14s}  "
                  f"{_r(ab_wi)+'/'+_r(ab_to):>16s}")
    print(f"\n[noimg] -> {RDIR / 'pope_noimg_summary.json'}")


if __name__ == "__main__":
    main()
