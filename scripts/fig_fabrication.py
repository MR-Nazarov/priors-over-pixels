#!/usr/bin/env python3
"""fig_fabrication: montage of zero-voxel organ slices that MedGemma-27B reports
as PRESENT across all three output formats. Three slices where the probed organ
is absent on that slice (0 voxels) plus a rightmost panel where the probed organ
is absent from the ENTIRE volume. Verdicts read from pope_results.json (locked
run); slice render matches the harness (level 40 / width 400, radiological).
Print-safe vector PDF."""

import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from anatomy_prior_probe import NAME_TO_ID, window_to_uint8, to_radiological, to_rgb  # noqa

RDIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
IMG = PROJECT_ROOT / "data/btcv/RawData/Training/img"
LBL = PROJECT_ROOT / "data/btcv/RawData/Training/label"
OUT = PROJECT_ROOT / "writing/figures"
OUT.mkdir(parents=True, exist_ok=True)
FMTS = ("freetext", "verdict_first", "reasoning_first")
FLABEL = {"freetext": "free-text", "verdict_first": "verdict", "reasoning_first": "reasoning"}


def load_vol(d, case):
    return np.asanyarray(nib.as_closest_canonical(nib.load(str(d / case))).dataobj)


def verdicts_by_triple(recs):
    """(case,z,organ) -> {fmt: parsed_present} for adversarial absent items."""
    out = {}
    for r in recs:
        if r["neg_strategy"] != "adversarial" or r["ground_truth_present"]:
            continue
        k = (r["case"], r["z"], r["organ"])
        out.setdefault(k, {})
        out[k].setdefault(r["format"], r["parsed_present"])
    return out


def main():
    recs = json.loads((RDIR / "pope_results.json").read_text())["records"]
    vmap = verdicts_by_triple(recs)
    absent_in_vol = {(t["case"], t["z"], t["organ"])
                     for t in json.loads((RDIR / "adv_zdist_triples.json").read_text())
                     if t["organ_absent_in_volume"]}

    lblcache = {}

    def focus_area(case, z, organ):
        if case not in lblcache:
            lblcache[case] = load_vol(LBL, f"{case.replace('img', 'label')}.nii.gz")
        return int((lblcache[case][:, :, z] == NAME_TO_ID[organ]).sum())

    # candidates: all 3 formats say present, organ truly 0 voxels on the slice
    all_present = [(k, v) for k, v in vmap.items()
                   if all(v.get(f) is True for f in FMTS) and focus_area(*k) == 0]

    # pick 3 with distinct organs (deterministic), preferring large organs for impact
    BIG = ["spleen", "liver", "stomach", "left_kidney", "right_kidney", "pancreas"]
    picks, used = [], set()
    for organ in BIG:
        for k, v in sorted(all_present):
            if k[2] == organ and k not in absent_in_vol and organ not in used:
                picks.append((k, v, "absent on slice")); used.add(organ); break
        if len(picks) == 3:
            break

    # rightmost: organ absent from the ENTIRE volume, all-present fabrication if possible
    rightmost = None
    for k, v in sorted(vmap.items()):
        if k in absent_in_vol and focus_area(*k) == 0 and any(v.get(f) is True for f in FMTS):
            rightmost = (k, v, "absent from entire volume")
            break
    if rightmost:
        picks.append(rightmost)

    import matplotlib.gridspec as gridspec
    n = len(picks)
    fig = plt.figure(figsize=(2.5 * n, 3.5))
    gs = gridspec.GridSpec(2, n, height_ratios=[5, 1.35], hspace=0.04, wspace=0.06,
                           left=0.01, right=0.99, top=0.80, bottom=0.02)
    for i, ((case, z, organ), v, note) in enumerate(picks):
        vol = load_vol(IMG, f"{case}.nii.gz")
        rgb = to_rgb(to_radiological(window_to_uint8(vol[:, :, z], 40.0, 400.0)))
        axim = fig.add_subplot(gs[0, i])
        axim.imshow(rgb)
        axim.set_xticks([]); axim.set_yticks([])
        axim.set_title(f"probed: {organ.replace('_', ' ')}\n(GT absent — {note})", fontsize=8)
        axt = fig.add_subplot(gs[1, i])
        axt.axis("off")
        for j, f in enumerate(FMTS):
            y = 0.92 - j * 0.34
            present = v.get(f) is True
            axt.text(0.06, y, f"{FLABEL[f]}:", fontsize=7.2, va="top", ha="left")
            axt.text(0.66, y, "present" if present else "absent", fontsize=7.2, va="top",
                     ha="left", fontweight="bold",
                     color="#b00000" if present else "#1a7f37")
    fig.suptitle("MedGemma-27B reports absent organs as present in all three formats "
                 "(bold red = fabricated)", fontsize=8.8, y=0.965)
    fig.savefig(OUT / "fig_fabrication.pdf")
    print(f"-> {OUT / 'fig_fabrication.pdf'}")
    for (case, z, organ), v, note in picks:
        print(f"  {case} z{z} {organ:16s} [{note}]  ",
              {FLABEL[f]: ('P' if v.get(f) else 'A') for f in FMTS})


if __name__ == "__main__":
    main()
