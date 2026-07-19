#!/usr/bin/env python3
"""Concept-figure slices: find a case where an organ is genuinely absent from the
ENTIRE volume (0 voxels across all slices; gallbladder is the usual candidate),
then render axial slices where several other organs are clearly present. Rendered
exactly as the model sees them (canonical RAS via load_volume, radiological flip,
single soft-tissue window W400/L40 -> grayscale). High-res PNG, no axes/padding.

Report-only; new files only (figures/concept_slice_candidates/).
"""
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from anatomy_prior_probe import (NAME_TO_ID, load_volume, window_to_uint8,  # noqa: E402
                                 to_radiological)

IMG = PROJECT_ROOT / "data/btcv/RawData/Training/img"
LBL = PROJECT_ROOT / "data/btcv/RawData/Training/label"
OUT = PROJECT_ROOT / "figures/concept_slice_candidates"
OUT.mkdir(parents=True, exist_ok=True)

ABSENT_CANDIDATES = ["gallbladder", "left_adrenal_gland", "right_adrenal_gland", "esophagus"]
CONTEXT = ["liver", "stomach", "spleen", "left_kidney", "right_kidney", "pancreas", "aorta"]
MIN_PX = 1536          # >=1500 px on long side
SLICE_MIN_VOX = 80     # an organ counts as "clearly visible" on a slice at >= this


def load_lbl(case):
    # labels are loaded through the SAME canonical pipeline as images (RAS), so
    # label indices line up with the rendered slice.
    p = LBL / f"{case.replace('img', 'label')}.nii.gz"
    return np.asanyarray(nib.as_closest_canonical(nib.load(str(p))).dataobj).round().astype(int)


def find_absent_case():
    """First (organ, case) with 0 voxels of that organ in the whole volume."""
    cases = sorted(p.stem.replace(".nii", "") for p in IMG.glob("*.nii.gz"))
    for organ in ABSENT_CANDIDATES:
        oid = NAME_TO_ID[organ]
        for case in cases:
            lbl = load_lbl(case)
            total = int((lbl == oid).sum())
            if total == 0:
                return organ, case, lbl, cases
    return None


def render(vol, z):
    gray = to_radiological(window_to_uint8(vol[:, :, z], 40.0, 400.0))  # W400/L40, HU[-160,240]
    im = Image.fromarray(gray, mode="L")
    if max(im.size) < MIN_PX:
        s = MIN_PX / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    return im


def main():
    res = find_absent_case()
    if res is None:
        print("No whole-volume-absent organ found among", ABSENT_CANDIDATES)
        return
    organ, case, lbl, _ = res
    oid = NAME_TO_ID[organ]
    nz = lbl.shape[2]
    # double-confirm 0 across every slice
    per_slice_absent = all(int((lbl[:, :, z] == oid).sum()) == 0 for z in range(nz))
    print(f"ABSENT ORGAN: {organ} (id {oid}) in {case} -> "
          f"whole-volume voxels = {int((lbl==oid).sum())}; "
          f"absent on all {nz} slices = {per_slice_absent}\n")

    # score slices by how many CONTEXT organs are clearly visible
    scored = []
    for z in range(nz):
        sl = lbl[:, :, z]
        vis = [o for o in CONTEXT if int((sl == NAME_TO_ID[o]).sum()) >= SLICE_MIN_VOX]
        if len(vis) >= 4:
            scored.append((len(vis), z, vis))
    scored.sort(key=lambda t: (-t[0], t[1]))

    # pick 4 well-separated candidates (avoid near-duplicate adjacent slices)
    picks, used_z = [], []
    for nvis, z, vis in scored:
        if all(abs(z - uz) >= 8 for uz in used_z):
            picks.append((z, vis)); used_z.append(z)
        if len(picks) == 4:
            break

    vol = load_volume(str(IMG / f"{case}.nii.gz"))
    print(f"{'file':40s} {'case':9s} {'slice':>5s}  {'absent(0-vox)':14s} visible_organs")
    for z, vis in picks:
        fname = f"{case}_z{z:03d}.png"
        render(vol, z).save(OUT / fname)
        print(f"{fname:40s} {case:9s} {z:5d}  {organ:14s} "
              + ", ".join(o.replace('_', ' ') for o in vis))
    print(f"\n-> {len(picks)} PNGs in {OUT}  (native 512px upscaled to {MIN_PX}px, LANCZOS)")


if __name__ == "__main__":
    main()
