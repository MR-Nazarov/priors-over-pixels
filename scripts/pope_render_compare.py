#!/usr/bin/env python3
"""Grayscale (main run) vs triple-window (control) verdict agreement on the POPE
adversarial set, MedGemma-27B. Report-only; modifies nothing.

Matches by item id (sample_id, organ, format). Reports agreement at three levels
(raw byte-for-byte, extracted verdict-field byte-for-byte, parsed verdict),
overall and by format, on adversarial NEGATIVES. Raw/verdict-field are also
reported per unique query (the raw string is shared across a query's organs).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
GRAY = RDIR / "pope_results.json"            # main grayscale run
TRIP = RDIR / "pope_adversarial_triplewindow_27b.json"
FORMATS = ("freetext", "verdict_first", "reasoning_first")


def extract_field(raw):
    blob = re.sub(r"```(?:json)?", "", raw) if "```" in raw else raw
    m = re.search(r"\{.*\}", blob, re.DOTALL)
    return m.group(0) if m else None


def frac(a, b):
    return f"{a}/{b} = {a/b:.4f}" if b else f"{a}/0 = n/a"


def main():
    gray = [r for r in json.loads(GRAY.read_text())["records"] if r["neg_strategy"] == "adversarial"]
    trip = json.loads(TRIP.read_text())["records"]
    G = {(r["sample_id"], r["organ"], r["format"]): r for r in gray}
    T = {(r["sample_id"], r["organ"], r["format"]): r for r in trip}
    shared = sorted(set(G) & set(T))
    neg = [k for k in shared if not G[k]["ground_truth_present"]]
    print(f"grayscale adversarial items: {len(G)} | triple items: {len(T)} | shared: {len(shared)}")
    print(f"id keys differ: {set(G) != set(T)}")
    print(f"adversarial NEGATIVE shared items (n): {len(neg)}\n")

    # --- parsed verdict, per NEGATIVE item, overall + by format ---
    print("=== parsed-verdict agreement on adversarial NEGATIVES ===")
    by_fmt_n = defaultdict(int); by_fmt_a = defaultdict(int); ov_a = 0
    for k in neg:
        same = G[k]["parsed_present"] == T[k]["parsed_present"]
        by_fmt_n[k[2]] += 1; by_fmt_a[k[2]] += int(same); ov_a += int(same)
    print(f"  overall: {frac(ov_a, len(neg))}")
    for f in FORMATS:
        print(f"    {f:16s} {frac(by_fmt_a[f], by_fmt_n[f])}")

    # --- raw + verdict-field, per unique adversarial QUERY (shared raw) ---
    qG = {(r["sample_id"], r["format"]): r["raw"] for r in gray}
    qT = {(r["sample_id"], r["format"]): r["raw"] for r in trip}
    qshared = sorted(set(qG) & set(qT))
    raw_a = vf_a = vf_tot = 0
    raw_bf = defaultdict(lambda: [0, 0]); vf_bf = defaultdict(lambda: [0, 0])
    for k in qshared:
        rg, rt = qG[k], qT[k]
        raw_bf[k[1]][1] += 1
        if rg == rt:
            raw_a += 1; raw_bf[k[1]][0] += 1
        fg, ft = extract_field(rg), extract_field(rt)
        if fg is not None and ft is not None:
            vf_tot += 1; vf_bf[k[1]][1] += 1
            if fg == ft:
                vf_a += 1; vf_bf[k[1]][0] += 1
    print(f"\n=== raw-string agreement, per adversarial QUERY (n={len(qshared)}) ===")
    print(f"  overall: {frac(raw_a, len(qshared))}")
    for f in FORMATS:
        print(f"    {f:16s} {frac(raw_bf[f][0], raw_bf[f][1])}")
    print(f"\n=== extracted verdict-field agreement, per query w/ JSON (n={vf_tot}) ===")
    print(f"  overall: {frac(vf_a, vf_tot)}")
    for f in FORMATS:
        print(f"    {f:16s} {frac(vf_bf[f][0], vf_bf[f][1])}")

    # --- 5 mismatched NEGATIVE items (parsed verdict) ---
    print("\n=== up to 5 mismatched adversarial-negative items (parsed verdict) ===")
    shown = 0
    for k in neg:
        if G[k]["parsed_present"] != T[k]["parsed_present"]:
            def v(x):
                return "present" if x["parsed_present"] is True else ("absent" if x["parsed_present"] is False else "FAIL/NA")
            print(f"  {k}  gray={v(G[k])}  triple={v(T[k])}")
            fg, ft = extract_field(G[k]["raw"]), extract_field(T[k]["raw"])
            if fg and ft:
                print(f"     gray  field: {fg[:110]!r}")
                print(f"     triple field: {ft[:110]!r}")
            shown += 1
            if shown == 5:
                break
    if shown == 0:
        print("  (none — parsed verdicts agree on all shared adversarial negatives)")


if __name__ == "__main__":
    main()
