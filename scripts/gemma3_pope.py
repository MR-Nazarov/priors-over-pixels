#!/usr/bin/env python3
"""Base-Gemma3 controlled ablation for the POPE present-bias study. Runs
google/gemma-3-{4b,27b}-it IDENTICALLY to the MedGemma replay (same items by
sample_id, same grayscale render W400/L40, same prompts, same 3 formats, greedy
do_sample=False, max_new_tokens=320, pan&scan) — only medical fine-tuning differs.

Standalone: imports the validated harness, modifies nothing. Same record schema
as pope_results_*.json so pope_abstention.py / pope_bootstrap_cis.py read it as-is.

  --stage1   : parseability gate — 30 adversarial items x 3 formats, print parse rates
  --stage2   : full protocol — all strategies x 3 formats -> gemma3_<m>_pope.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from anatomy_prior_probe import (MedGemma, build_prompt, render_slice,  # noqa: E402
                                 load_volume, Config, FORMATS)
from pope_replay import load_items, parse_replay, freetext_recitation, REF, IMG_DIR  # noqa: E402

RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
PATHS = {"gemma3_4b": PROJECT_ROOT / "models/gemma-3-4b-it",
         "gemma3_27b": PROJECT_ROOT / "models/gemma-3-27b-it"}


def build_records(model_name, items, model):
    records, times, volcache = [], [], {}
    for it in items:
        if it["case"] not in volcache:
            volcache[it["case"]] = load_volume(str(IMG_DIR / f"{it['case']}.nii.gz"))
        rgb = render_slice(volcache[it["case"]], it["z"], Config())   # grayscale W400/L40
        for fmt in FORMATS:
            prompt = build_prompt(fmt, it["queried"])
            t0 = time.perf_counter()
            raw = model.ask(rgb, prompt)
            times.append(time.perf_counter() - t0)
            if len(times) == 1:
                print(f"[timing] first call {times[0]:.1f}s", file=sys.stderr)
            parsed = parse_replay(raw, fmt, it["queried"])
            recite = freetext_recitation(raw) if fmt == "freetext" else None
            for organ, gt in it["queried"]:
                p = parsed[organ]
                clean = (not p["parse_fail"]) and (not p["not_addressed"])
                records.append({
                    "sample_id": it["sample_id"], "case": it["case"], "z": it["z"],
                    "stratum": it["stratum"], "focus_organ": it["focus_organ"],
                    "neg_strategy": it["neg_strategy"], "adv_fellback": it["adv_fellback"],
                    "queried_set": it["queried_set"], "organ": organ,
                    "ground_truth_present": bool(gt), "format": fmt, "model": model_name,
                    "raw": raw, "parsed_present": p["pred_present"],
                    "parse_fail": p["parse_fail"], "organ_not_addressed": p["not_addressed"],
                    "freetext_recitation": recite, "correct": bool(clean and p["pred_present"] == gt),
                })
    if times:
        print(f"[timing] {len(times)} calls, median {np.median(times):.1f}s", file=sys.stderr)
    return records


def parse_rates(records):
    print("\n=== STAGE 1 parse rates (fraction yielding a scorable verdict) ===")
    ok = True
    for fmt in FORMATS:
        items = [r for r in records if r["format"] == fmt]
        clean = sum((not r["parse_fail"]) and (not r["organ_not_addressed"]) for r in items)
        rate = clean / len(items) if items else float("nan")
        gate = "" if fmt == "freetext" else ("  <0.95 FAIL" if rate < 0.95 else "  >=0.95 ok")
        if fmt != "freetext" and rate < 0.95:
            ok = False
        print(f"  {fmt:16s} {clean}/{len(items)} = {rate:.4f}{gate}")
    print("\nGATE:", "PASS -> proceed to stage 2" if ok
          else "FAIL -> halt (verdict-first or reasoning-first <0.95, the LLaVA-Med mode)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3_4b", choices=list(PATHS))
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    cfg.model_id = str(PATHS[a.model])      # max_new_tokens=320, do_pan_and_scan=True defaults
    all_items = load_items(REF)

    if a.stage1:
        items = [it for it in all_items if it["neg_strategy"] == "adversarial"][:30]
        print(f"[stage1] {len(items)} adversarial items x {len(FORMATS)} formats", file=sys.stderr)
        model = MedGemma(cfg)
        recs = build_records(a.model, items, model)
        (RDIR / f"gemma3_{a.model.split('_')[1]}_pope_smoke.json").write_text(
            json.dumps({"model": a.model, "records": recs}, indent=2))
        parse_rates(recs)
    elif a.stage2:
        print(f"[stage2] {len(all_items)} items x {len(FORMATS)} formats", file=sys.stderr)
        model = MedGemma(cfg)
        recs = build_records(a.model, all_items, model)
        out = RDIR / f"gemma3_{a.model.split('_')[1]}_pope.json"
        out.write_text(json.dumps({"model": a.model, "n_items": len(all_items), "records": recs}, indent=2))
        print(f"[done] {len(recs)} records -> {out}", file=sys.stderr)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
