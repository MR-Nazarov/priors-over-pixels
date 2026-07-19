#!/usr/bin/env python3
"""Rendering control on the REAL POPE adversarial set (MedGemma-27B only).
Replays the exact adversarial items from the main grayscale run under the native
triple-window false-color render, holding everything else identical (same items,
same prompts, same 3 formats, greedy decoding, max_new_tokens=320, pan&scan).
The ONLY change vs the main run is the render function.

Standalone — imports the validated harness pieces, modifies nothing. Writes
results/medgemma_pope_negsampling/pope_adversarial_triplewindow_27b.json.

  --run      : run the triple-window sweep, save outputs
  --compare  : compute grayscale-vs-triple agreement from saved files
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from anatomy_prior_probe import (MedGemma, build_prompt, load_volume,  # noqa: E402
                                 to_radiological, Config, FORMATS)
from pope_replay import load_items, parse_replay, REF, IMG_DIR  # noqa: E402

RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
OUTFILE = RDIR / "pope_adversarial_triplewindow_27b.json"
MODEL_27B = str(PROJECT_ROOT / "models/medgemma-27b-it")

# requested native triple-window false-color: (center-width/2, center+width/2)
# bone/lung W2250/L-100 ; soft-tissue W350/L40 ; brain W80/L40  -> R,G,B
TRIPLE = [(-100 - 2250 / 2, -100 + 2250 / 2),   # (-1225, 1025)
          (40 - 350 / 2,  40 + 350 / 2),        # (-135, 215)
          (40 - 80 / 2,   40 + 80 / 2)]         # (0, 80)


def _chan(sl, lo, hi):
    c = np.clip(sl, lo, hi).astype(np.float32)
    return np.round((c - lo) / (hi - lo) * 255).astype(np.uint8)


def triple_render(vol, z):
    sl = vol[:, :, z].astype(np.float32)
    rgb = np.stack([_chan(sl, lo, hi) for lo, hi in TRIPLE], axis=-1)
    return to_radiological(rgb)   # same orientation transform as the grayscale path


def run():
    items = [it for it in load_items(REF) if it["neg_strategy"] == "adversarial"]
    print(f"[render-control] adversarial query items: {len(items)} "
          f"x {len(FORMATS)} formats = {len(items)*len(FORMATS)} calls "
          f"(triple-window, MedGemma-27B)", file=sys.stderr)

    cfg = Config()
    cfg.model_id = MODEL_27B            # max_new_tokens=320, do_pan_and_scan=True (defaults)
    model = MedGemma(cfg)

    records, times, volcache = [], [], {}
    for it in items:
        if it["case"] not in volcache:
            volcache[it["case"]] = load_volume(str(IMG_DIR / f"{it['case']}.nii.gz"))
        rgb = triple_render(volcache[it["case"]], it["z"])
        for fmt in FORMATS:
            prompt = build_prompt(fmt, it["queried"])
            t0 = time.perf_counter()
            raw = model.ask(rgb, prompt)
            times.append(time.perf_counter() - t0)
            if len(times) == 1:
                est = times[0] * len(items) * len(FORMATS) / 3600
                print(f"[timing] first call {times[0]:.1f}s -> ~{est:.1f} h", file=sys.stderr)
            parsed = parse_replay(raw, fmt, it["queried"])
            for organ, gt in it["queried"]:
                p = parsed[organ]
                clean = (not p["parse_fail"]) and (not p["not_addressed"])
                records.append({
                    "sample_id": it["sample_id"], "case": it["case"], "z": it["z"],
                    "neg_strategy": "adversarial", "organ": organ,
                    "ground_truth_present": bool(gt), "format": fmt,
                    "model": "medgemma27b", "render": "triple_window",
                    "raw": raw, "parsed_present": p["pred_present"],
                    "parse_fail": p["parse_fail"], "organ_not_addressed": p["not_addressed"],
                    "correct": bool(clean and p["pred_present"] == gt),
                })
    OUTFILE.write_text(json.dumps({"model": "medgemma27b", "render": "triple_window",
                                   "triple_windows": TRIPLE, "n_items": len(items),
                                   "records": records}, indent=2))
    if times:
        print(f"[timing] {len(times)} calls, median {np.median(times):.1f}s", file=sys.stderr)
    print(f"[done] {len(records)} records -> {OUTFILE}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    else:
        ap.print_help()
