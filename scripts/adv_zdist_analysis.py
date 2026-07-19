#!/usr/bin/env python3
"""
adv_zdist_analysis.py
=====================
Is MedGemma's fabrication on adversarial absent items uniform across z-distance,
or concentrated at the organ's z-boundary? Pre-empts the reviewer objection that
"zero voxels at this z" can mean "one slice past the edge" rather than
"anatomically nowhere near".

For every adversarial record with ground_truth_present == false (reuse
pope_results.json; NO model re-runs):
  z_dist  = min |z - z'| over slices z' in the SAME volume where the probed organ
            is present (>= 50 vox, the threshold used everywhere). inf + flag if
            the organ never reaches >=50 vox anywhere in the volume.
  z_dist_mm = z_dist * slice spacing (mm) from the NIfTI header (primary unit;
            slice counts aren't comparable across volumes).

PHASE 1 (this run): compute z_dist / z_dist_mm, persist per-triple, and print the
distribution + per-band per-format counts, then STOP. Confirm the far bins are
populated before trusting any specificity-vs-distance curve. Run with --plot for
PHASE 2 (specificity-vs-distance + extremes table + figure).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from anatomy_prior_probe import NAME_TO_ID, organ_areas, load_volume  # noqa: E402

RESULTS = PROJECT_ROOT / "results/medgemma_pope_negsampling/pope_results.json"
LBL_DIR = PROJECT_ROOT / "data/btcv/RawData/Training/label"
OUT_DIR = PROJECT_ROOT / "results/medgemma_pope_negsampling"
PRESENCE_MIN = 50
FORMATS = ("freetext", "verdict_first", "reasoning_first")
BANDS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, np.inf)]


def _z_spacing(case: str) -> float:
    img = nib.as_closest_canonical(nib.load(str(LBL_DIR / f"{case.replace('img', 'label')}.nii.gz")))
    return float(img.header.get_zooms()[2])


def compute_zdist(records: list) -> dict:
    """Return {(case,z,organ): {z_dist, z_dist_mm, organ_absent_in_volume}}."""
    adv = [r for r in records if r["neg_strategy"] == "adversarial"
           and not r["ground_truth_present"]]
    triples = sorted({(r["case"], r["z"], r["organ"]) for r in adv})

    present_z, spacing = {}, {}
    info = {}
    for case, z, organ in triples:
        if case not in present_z:
            lbl = load_volume(str(LBL_DIR / f"{case.replace('img', 'label')}.nii.gz"))
            areas = organ_areas(lbl)
            present_z[case] = {o: np.where(areas[o] >= PRESENCE_MIN)[0] for o in areas}
            spacing[case] = _z_spacing(case)
        pz = present_z[case][organ]
        if pz.size == 0:
            info[(case, z, organ)] = {"z_dist": float("inf"), "z_dist_mm": float("inf"),
                                      "organ_absent_in_volume": True}
        else:
            d = int(np.min(np.abs(pz - z)))
            info[(case, z, organ)] = {"z_dist": d, "z_dist_mm": d * spacing[case],
                                      "organ_absent_in_volume": False}
    return info, adv


def _band(mm: float) -> str:
    for lo, hi in BANDS:
        if lo <= mm < hi:
            return f"{lo}-{hi if hi != np.inf else '+'}"
    return f"{BANDS[-1][0]}-+"


def phase1_histogram(records: list) -> dict:
    info, adv = compute_zdist(records)
    triples = list(info)

    # persist per-triple
    persisted = [{"case": c, "z": z, "organ": o, **info[(c, z, o)]} for (c, z, o) in triples]
    (OUT_DIR / "adv_zdist_triples.json").write_text(json.dumps(persisted, indent=2))

    finite_mm = [info[t]["z_dist_mm"] for t in triples if np.isfinite(info[t]["z_dist_mm"])]
    n_inf = sum(info[t]["organ_absent_in_volume"] for t in triples)

    print(f"unique adversarial-absent triples (case,z,organ): {len(triples)}")
    print(f"  organ_absent_in_volume (z_dist=inf): {n_inf}")
    print(f"  finite z_dist_mm: min={min(finite_mm):.1f} median={np.median(finite_mm):.1f} "
          f"max={max(finite_mm):.1f} mm")

    # fine histogram of finite distances
    print("\n=== z_dist_mm distribution (unique triples, finite only) ===")
    edges = [0, 2, 5, 10, 15, 20, 30, 40, 60, 80, np.inf]
    hist, _ = np.histogram(finite_mm, bins=edges)
    for i, c in enumerate(hist):
        hi = edges[i + 1]
        bar = "#" * int(60 * c / max(hist.max(), 1))
        print(f"  {edges[i]:>4.0f}-{('inf' if hi == np.inf else f'{hi:.0f}'):>4s} mm: {c:5d} {bar}")
    print(f"  (inf, organ-absent): {n_inf:5d}")

    # per-band per-format RECORD counts (what specificity will use)
    band_fmt = defaultdict(lambda: defaultdict(int))
    band_total = defaultdict(int)
    for r in adv:
        mm = info[(r["case"], r["z"], r["organ"])]["z_dist_mm"]
        b = "inf" if not np.isfinite(mm) else _band(mm)
        band_fmt[b][r["format"]] += 1
        band_total[b] += 1

    order = [f"{lo}-{hi if hi != np.inf else '+'}" for lo, hi in BANDS] + ["inf"]
    print("\n=== per-band ABSENT-RECORD counts by format (specificity cell sizes) ===")
    print(f"{'band(mm)':10s} " + " ".join(f"{f:>15s}" for f in FORMATS) + f"  {'total':>7s}")
    for b in order:
        if band_total[b] == 0:
            continue
        cells = " ".join(f"{band_fmt[b][f]:15d}" for f in FORMATS)
        print(f"{b:10s} {cells}  {band_total[b]:7d}")

    near = sum(band_total[b] for b in order if b in ("0-5",))
    far = band_total.get("40-+", 0) + band_total.get("inf", 0)
    print(f"\nnear-boundary (<=5mm) records: {near}   |   far (>=40mm incl inf) records: {far}")
    sparse = far < 150 * len(FORMATS)
    print("FAR BINS:", "STARVED -> frame as 'most adversarial negatives are near-boundary'"
          if sparse else "populated -> specificity-vs-distance curve is trustworthy")
    print(f"\npersisted per-triple z_dist -> {OUT_DIR / 'adv_zdist_triples.json'}")
    print("\nPHASE 1 — eyeball the per-band counts. Re-run with --plot for the "
          "specificity-vs-distance figure once the far bins look adequate.")
    return info


def _band_far(mm: float) -> str:
    """Bands for the curve; inf (organ-absent) folds into the >=40 'far' point."""
    if not np.isfinite(mm) or mm >= 40:
        return "40+"
    for lo, hi in BANDS[:-1]:
        if lo <= mm < hi:
            return f"{lo}-{hi}"
    return "40+"


def _spec_metric(items):
    parsed = [r for r in items if not r["parse_fail"]]
    if not parsed:
        return (float("nan"), 0)
    return (sum(r["correct"] for r in parsed) / len(parsed), len(parsed))


def _logistic_slope(items):
    """Slope (log-odds of correct-rejection per mm) from a logistic fit on finite
    z_dist_mm. Returns (slope, n) or (nan, n)."""
    from sklearn.linear_model import LogisticRegression
    xy = [(r["z_dist_mm"], int(r["correct"])) for r in items
          if not r["parse_fail"] and np.isfinite(r["z_dist_mm"])]
    if len({y for _, y in xy}) < 2 or len(xy) < 10:
        return (float("nan"), len(xy))
    x = np.array([v for v, _ in xy]).reshape(-1, 1)
    y = np.array([v for _, v in xy])
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, y)
    return (float(m.coef_[0, 0]), len(xy))


def _boot_slope_ci(items, n_boot, seed):
    rng = np.random.default_rng(seed)
    by_case = defaultdict(list)
    for r in items:
        by_case[r["case"]].append(r)
    cl = list(by_case)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cl), len(cl))
        samp = [r for i in idx for r in by_case[cl[i]]]
        s, _ = _logistic_slope(samp)
        if np.isfinite(s):
            vals.append(s)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def phase2_plot(records: list, n_boot: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from anatomy_prior_probe import _bootstrap_ci, _r

    info, adv = compute_zdist(records)
    for r in adv:
        r["z_dist_mm"] = info[(r["case"], r["z"], r["organ"])]["z_dist_mm"]
        r["organ_absent_in_volume"] = info[(r["case"], r["z"], r["organ"])]["organ_absent_in_volume"]

    curve_bands = ["0-5", "5-10", "10-20", "20-40", "40+"]
    summary = {"per_bin_specificity": {}, "extremes": {}, "logistic_slope_logodds_per_mm": {},
               "organ_absent_in_volume": {}}

    # per (format, band) specificity + CI + mean distance for x-position
    print("=== specificity by z_dist band x format (case-clustered 95% CI) ===")
    print(f"{'band(mm)':9s} " + " ".join(f"{f[:13]:>22s}" for f in FORMATS))
    curve = {f: {"x": [], "y": [], "lo": [], "hi": []} for f in FORMATS}
    for b in curve_bands:
        cells = []
        for f in FORMATS:
            items = [r for r in adv if r["format"] == f and _band_far(r["z_dist_mm"]) == b]
            sp, n = _spec_metric(items)
            lo, hi = _bootstrap_ci(items, _spec_metric, n_boot, seed)
            summary["per_bin_specificity"].setdefault(b, {})[f] = {"spec": sp, "ci": [lo, hi], "n": n}
            xfin = [r["z_dist_mm"] for r in items if np.isfinite(r["z_dist_mm"])]
            xpos = float(np.mean(xfin)) if xfin else 50.0
            curve[f]["x"].append(xpos); curve[f]["y"].append(sp)
            curve[f]["lo"].append(lo); curve[f]["hi"].append(hi)
            cells.append(f"{_r(sp)}[{_r(lo)},{_r(hi)}]n{n}")
        print(f"{b:9s} " + " ".join(f"{c:>22s}" for c in cells))

    # organ-absent-in-volume subset, separate
    print("\n=== organ-absent-in-volume subset (z_dist=inf), separate ===")
    for f in FORMATS:
        items = [r for r in adv if r["format"] == f and r["organ_absent_in_volume"]]
        sp, n = _spec_metric(items)
        summary["organ_absent_in_volume"][f] = {"spec": sp, "n": n}
        print(f"  {f:16s} spec={_r(sp)} n={n}")

    # near vs far extremes table (40+ is the headline far number)
    print("\n=== EXTREMES: near (<=5mm) vs far (>=40mm, incl inf) ===")
    print(f"{'format':16s} {'near_spec(<=5)':>20s} {'far_spec(>=40)':>20s}")
    for f in FORMATS:
        near = [r for r in adv if r["format"] == f and np.isfinite(r["z_dist_mm"]) and r["z_dist_mm"] <= 5]
        far = [r for r in adv if r["format"] == f and _band_far(r["z_dist_mm"]) == "40+"]
        ns, nn = _spec_metric(near); nl, nh = _bootstrap_ci(near, _spec_metric, n_boot, seed)
        fs, fn = _spec_metric(far); fl, fh = _bootstrap_ci(far, _spec_metric, n_boot, seed)
        summary["extremes"][f] = {"near": {"spec": ns, "ci": [nl, nh], "n": nn},
                                  "far": {"spec": fs, "ci": [fl, fh], "n": fn}}
        print(f"{f:16s} {f'{_r(ns)}[{_r(nl)},{_r(nh)}]n{nn}':>20s} "
              f"{f'{_r(fs)}[{_r(fl)},{_r(fh)}]n{fn}':>20s}")

    # logistic slope of correct-rejection vs distance (the prior-vs-boundary number)
    print("\n=== logistic slope: log-odds(correct-rejection) per mm  (≈0 => prior; "
          ">0 => rises with distance) ===")
    for f in FORMATS:
        items = [r for r in adv if r["format"] == f]
        slope, n = _logistic_slope(items)
        lo, hi = _boot_slope_ci(items, n_boot, seed)
        summary["logistic_slope_logodds_per_mm"][f] = {"slope": slope, "ci": [lo, hi], "n": n}
        per10 = slope * 10
        print(f"  {f:16s} slope={_r(slope)}/mm  ({per10:+.3f}/10mm)  "
              f"95%CI[{_r(lo)},{_r(hi)}]  n={n}")

    # figure
    sl = summary["logistic_slope_logodds_per_mm"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"freetext": "#1f77b4", "verdict_first": "#d62728", "reasoning_first": "#2ca02c"}
    for f in FORMATS:
        c = curve[f]
        ax.plot(c["x"], c["y"], "-o", color=colors[f], label=f)
        ax.fill_between(c["x"], c["lo"], c["hi"], color=colors[f], alpha=0.18)
    ax.axhline(0.0, ls="--", lw=1, color="gray", label="always-present baseline")
    ax.set_xlabel("z-distance to nearest presence (mm)")
    ax.set_ylabel("specificity (correct-rejection rate)")
    ft = sl["freetext"]
    ax.set_title("Adversarial absent-item specificity vs z-distance\n"
                 "(flat = prior-driven fabrication; rising = boundary over-extrapolation)\n"
                 f"free-text slope {ft['slope']:+.3f}/mm, 95% CI "
                 f"[{ft['ci'][0]:+.3f}, {ft['ci'][1]:+.3f}] — includes 0 (flat)",
                 fontsize=10)
    ax.set_ylim(0.0, 0.25)
    ax.legend(fontsize=8, loc="upper left")

    lines = ["logistic slope  (log-odds correct-rejection / mm):"]
    for f in FORMATS:
        s = sl[f]
        zero = "  CI∋0 (flat)" if s["ci"][0] <= 0 <= s["ci"][1] else ""
        lines.append(f"  {f:15s} {s['slope']:+.3f}  CI[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]{zero}")
    ax.text(0.975, 0.035, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    png = OUT_DIR / "adv_specificity_vs_zdist.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    (OUT_DIR / "adv_zdist_analysis.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfigure -> {png}")
    print(f"summary -> {OUT_DIR / 'adv_zdist_analysis.json'}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="PHASE 2: specificity-vs-distance plot")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(RESULTS.read_text())["records"]
    if not args.plot:
        phase1_histogram(records)
        return
    phase2_plot(records, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
