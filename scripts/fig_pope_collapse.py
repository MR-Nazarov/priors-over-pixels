#!/usr/bin/env python3
"""fig_pope_collapse: specificity vs random->popular->adversarial, one line per
output format, CI bands from tab:pope's PRINTED intervals (hardcoded, not
recomputed), with a flat ~0.93 sensitivity reference overlaid. Print-safe vector
PDF (distinct markers + linestyles, survives grayscale)."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "writing/figures"
OUT.mkdir(parents=True, exist_ok=True)

# locked values from tab:pope (specificity, [lo, hi]); DO NOT recompute
SPEC = {
    "free-text":       [(.188, .175, .201), (.122, .112, .132), (.102, .094, .110)],
    "verdict-first":   [(.166, .156, .177), (.113, .104, .121), (.047, .039, .055)],
    "reasoning-first": [(.271, .254, .286), (.156, .142, .171), (.077, .065, .088)],
}
STYLE = {  # linestyle, marker, color  (grayscale-distinct via ls+marker)
    "free-text":       ("-",  "o", "#1f77b4"),
    "verdict-first":   ("--", "s", "#d62728"),
    "reasoning-first": (":",  "^", "#2ca02c"),
}
STRATS = ["random", "popular", "adversarial"]
x = np.arange(3)

fig, ax = plt.subplots(figsize=(4.3, 3.4))
for fmt, (ls, mk, col) in STYLE.items():
    pts = SPEC[fmt]
    y = [p[0] for p in pts]
    lo = [p[1] for p in pts]
    hi = [p[2] for p in pts]
    ax.plot(x, y, ls=ls, marker=mk, color=col, label=fmt, lw=1.7, ms=6,
            markeredgecolor="black", markeredgewidth=.5, zorder=3)
    ax.fill_between(x, lo, hi, color=col, alpha=.15, zorder=1)

ax.axhline(0.93, ls=(0, (6, 2)), color="0.35", lw=1.3, zorder=2)
ax.text(2.05, 0.945, r"sensitivity $\approx$ 0.93", ha="right", va="bottom",
        fontsize=8, color="0.30")
ax.axhline(0.0, ls="-", color="0.7", lw=.8, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(["random", "popular", "adversarial"])
ax.set_xlabel("POPE negative-sampling strategy")
ax.set_ylabel("specificity / sensitivity")
ax.set_ylim(-0.01, 1.0)
ax.set_xlim(-0.18, 2.18)
ax.legend(title="output format", fontsize=8, title_fontsize=8, loc="center left",
          frameon=True, framealpha=.9)
ax.set_title("Specificity collapse under POPE negatives (MedGemma-27B)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_pope_collapse.pdf")
print(f"-> {OUT / 'fig_pope_collapse.pdf'}")
