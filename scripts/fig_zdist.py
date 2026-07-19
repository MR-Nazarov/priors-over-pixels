#!/usr/bin/env python3
"""fig_zdist: adversarial specificity vs z-distance-to-nearest-presence, per
output format (MedGemma-27B). Plotted straight from adv_zdist_analysis.json
(per-band specificity + CIs already computed) — NOT re-bootstrapped. Verifies the
stored logistic slopes match the locked values (free-text +0.002, verdict-first
+0.027, reasoning-first +0.024) before drawing. y-axis fixed [0,0.25]. Print-safe
vector PDF."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDIR = Path(__file__).resolve().parent.parent / "results/medgemma_pope_negsampling"
OUT = Path(__file__).resolve().parent.parent / "writing/figures"
OUT.mkdir(parents=True, exist_ok=True)

d = json.loads((RDIR / "adv_zdist_analysis.json").read_text())
slopes = d["logistic_slope_logodds_per_mm"]

LOCKED = {"freetext": 0.002, "verdict_first": 0.027, "reasoning_first": 0.024}
for k, v in LOCKED.items():
    got = round(slopes[k]["slope"] if isinstance(slopes[k], dict) else slopes[k], 3)
    assert got == v, f"slope desync {k}: json {got} != locked {v}"
print("slope verification OK:", {k: round((slopes[k]['slope'] if isinstance(slopes[k], dict) else slopes[k]), 3) for k in LOCKED})

BANDS = ["0-5", "5-10", "10-20", "20-40", "40+"]
XC = [2.5, 7.5, 15, 30, 50]            # representative band centres (mm)
FMT = {  # json key -> (label, linestyle, marker, color, slope)
    "freetext":        ("free-text",       "-",  "o", "#1f77b4", "+0.002"),
    "verdict_first":   ("verdict-first",   "--", "s", "#d62728", "+0.027"),
    "reasoning_first": ("reasoning-first", ":",  "^", "#2ca02c", "+0.024"),
}

fig, ax = plt.subplots(figsize=(4.7, 3.4))
for key, (lab, ls, mk, col, _) in FMT.items():
    y = [d["per_bin_specificity"][b][key]["spec"] for b in BANDS]
    lo = [d["per_bin_specificity"][b][key]["ci"][0] for b in BANDS]
    hi = [d["per_bin_specificity"][b][key]["ci"][1] for b in BANDS]
    ax.plot(XC, y, ls=ls, marker=mk, color=col, label=lab, lw=1.7, ms=6,
            markeredgecolor="black", markeredgewidth=.5, zorder=3)
    ax.fill_between(XC, lo, hi, color=col, alpha=.15, zorder=1)
ax.axhline(0.0, ls="--", color="0.6", lw=1, zorder=0)

ax.set_ylim(0, 0.25)
ax.set_xlim(0, 53)
ax.set_xlabel("z-distance to nearest true presence (mm)")
ax.set_ylabel("specificity (correct-rejection rate)")
ax.set_title("Fabrication is distance-flat, not boundary over-extrapolation\n(MedGemma-27B, adversarial)",
             fontsize=9)

txt = "logistic slope (log-odds/mm):\n" + "\n".join(
    f"  {FMT[k][0]:<14s} {FMT[k][4]}" for k in FMT)
ax.text(0.975, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.2, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=.9))
ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=.9)
fig.tight_layout()
fig.savefig(OUT / "fig_zdist.pdf")
print(f"-> {OUT / 'fig_zdist.pdf'}")
