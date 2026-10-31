"""
plot_improvements.py — Phase 2 vs Phase 3/4 comparison charts
Generates 4 plots saved to analysis_results/phase_comparison/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "analysis_results/phase_comparison"
os.makedirs(OUT, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
# Labels (short), Task1 F1, Task2.3 F1, ECE, Viol%
runs = [
    # label                        T1      T23     ECE     Viol    phase
    ("Baseline\n(P2)",             0.7002, 0.5623, 0.4387, 55.9,  "p2"),
    ("N1 only\n(P2)",              0.7101, 0.6033, 0.4630, 62.3,  "p2"),
    ("N2 only\n(P2)",              0.6989, 0.5677, 0.4187, 49.7,  "p2"),
    ("N3 only\n(P2)",              0.6775, 0.5383, 0.4341, 53.3,  "p2"),
    ("N1+N2\n(P2)",                0.6987, 0.5503, 0.4064, 44.8,  "p2"),
    ("N1+N3\n(P2)",                0.7129, 0.5555, 0.4375, 58.0,  "p2"),
    ("N1+N2+N3\n(P2)",             0.6961, 0.5584, 0.4482, 59.2,  "p2"),
    ("Baseline\n(P3)",             0.7022, 0.5942, 0.3658, 42.0,  "p3"),
    ("N1 only\n(P3)",              0.7270, 0.5970, 0.3777, 45.7,  "p3"),
    ("N1+N2+N3\n(P3)",             0.7032, 0.5573, 0.3685, 40.4,  "p3"),
    ("N1+N3\n(P4)★",              0.7379, 0.5858, 0.3806, 43.4,  "p4"),
]

labels = [r[0] for r in runs]
t1     = [r[1] for r in runs]
t23    = [r[2] for r in runs]
ece    = [r[3] for r in runs]
viol   = [r[4] for r in runs]
phases = [r[5] for r in runs]

COLOR = {"p2": "#4C72B0", "p3": "#55A868", "p4": "#C44E52"}
BAR_COLORS = [COLOR[p] for p in phases]

x = np.arange(len(labels))
W = 0.65

BASELINE_T1  = 0.7002
BASELINE_T23 = 0.5623
BASELINE_ECE = 0.4387
BASELINE_VIO = 55.9

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})


def bar_chart(ax, values, ylabel, title, baseline, higher_better=True,
              fmt=".4f", ypad=0.005):
    bars = ax.bar(x, values, width=W, color=BAR_COLORS, edgecolor="white", linewidth=0.5)
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2,
               label=f"P2 Baseline ({baseline:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ypad,
                format(val, fmt), ha="center", va="bottom", fontsize=7,
                fontweight="bold")

    # highlight winner (best value)
    best_idx = values.index(max(values) if higher_better else min(values))
    bars[best_idx].set_edgecolor("#FFD700")
    bars[best_idx].set_linewidth(2.5)

    # set y limits with padding
    spread = max(values) - min(values)
    if higher_better:
        ax.set_ylim(min(values) - spread * 0.4, max(values) + spread * 0.5)
    else:
        ax.set_ylim(min(values) - spread * 0.4, max(values) + spread * 0.5)


# ── Plot 1: Task1 F1 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
bar_chart(ax, t1, "F1 Score (Macro)", "Task 1 F1 — Phase 2 vs Phase 3/4",
          BASELINE_T1, higher_better=True, ypad=0.002)
legend_patches = [
    mpatches.Patch(color=COLOR["p2"], label="Phase 2 (old hyperparams)"),
    mpatches.Patch(color=COLOR["p3"], label="Phase 3 (F1 monitor, eps=0.10, λ=0.20)"),
    mpatches.Patch(color=COLOR["p4"], label="Phase 4 winner (N1+N3 improved) ★"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/task1_f1_phase_comparison.png", bbox_inches="tight")
plt.close()
print("Saved task1_f1_phase_comparison.png")

# ── Plot 2: Task2.3 F1 ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
bar_chart(ax, t23, "F1 Score (Macro)", "Task 2.3 F1 — Phase 2 vs Phase 3/4",
          BASELINE_T23, higher_better=True, ypad=0.002)
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/task23_f1_phase_comparison.png", bbox_inches="tight")
plt.close()
print("Saved task23_f1_phase_comparison.png")

# ── Plot 3: ECE (lower = better) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
bar_chart(ax, ece, "ECE (lower = better)",
          "Calibration Error (ECE) — Phase 2 vs Phase 3/4",
          BASELINE_ECE, higher_better=False, ypad=0.003)
ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
# annotate: lower is better
ax.text(0.01, 0.97, "↓ lower is better", transform=ax.transAxes,
        fontsize=8, color="gray", va="top")
plt.tight_layout()
plt.savefig(f"{OUT}/ece_phase_comparison.png", bbox_inches="tight")
plt.close()
print("Saved ece_phase_comparison.png")

# ── Plot 4: Violation % (lower = better) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
bar_chart(ax, viol, "Violation Rate (%)",
          "Hierarchical Violation Rate — Phase 2 vs Phase 3/4",
          BASELINE_VIO, higher_better=False, fmt=".1f", ypad=0.8)
ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
ax.text(0.01, 0.97, "↓ lower is better", transform=ax.transAxes,
        fontsize=8, color="gray", va="top")
plt.tight_layout()
plt.savefig(f"{OUT}/violations_phase_comparison.png", bbox_inches="tight")
plt.close()
print("Saved violations_phase_comparison.png")

# ── Plot 5: Combined 4-metric summary (key runs only) ────────────────────────
key = [
    ("P2 Baseline",   0.7002, 0.5623, 0.4387, 55.9),
    ("P2 N1+N2+N3",   0.6961, 0.5584, 0.4482, 59.2),
    ("P2 Best (N1+N3)",0.7129,0.5555, 0.4375, 58.0),
    ("P3 Baseline",   0.7022, 0.5942, 0.3658, 42.0),
    ("P3 N1+N2+N3",   0.7032, 0.5573, 0.3685, 40.4),
    ("P4 N1+N3 ★",   0.7379, 0.5858, 0.3806, 43.4),
]
kl   = [r[0] for r in key]
kt1  = [r[1] for r in key]
kt23 = [r[2] for r in key]
kece = [r[3] for r in key]
kvio = [r[4] for r in key]

kcolors = ["#4C72B0","#4C72B0","#4C72B0","#55A868","#55A868","#C44E52"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Key Runs — All Metrics: Phase 2 → Phase 4", fontsize=13, fontweight="bold")

xk = np.arange(len(kl))
for ax, vals, ylabel, baseline, hi, fmt, yp in [
    (axes[0], kt1,  "Task1 F1",   0.7002, True,  ".4f", 0.002),
    (axes[1], kt23, "Task2.3 F1", 0.5623, True,  ".4f", 0.002),
    (axes[2], kece, "ECE ↓",      0.4387, False, ".4f", 0.003),
    (axes[3], kvio, "Viol% ↓",    55.9,   False, ".1f", 0.8),
]:
    bars = ax.bar(xk, vals, color=kcolors, edgecolor="white", linewidth=0.5)
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(xk)
    ax.set_xticklabels(kl, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yp,
                format(val, fmt), ha="center", va="bottom", fontsize=7, fontweight="bold")
    best = vals.index(max(vals) if hi else min(vals))
    bars[best].set_edgecolor("#FFD700"); bars[best].set_linewidth(2.5)
    spread = max(vals) - min(vals)
    ax.set_ylim(min(vals) - spread*0.4, max(vals) + spread*0.55)

legend_patches2 = [
    mpatches.Patch(color=COLOR["p2"], label="Phase 2"),
    mpatches.Patch(color=COLOR["p3"], label="Phase 3"),
    mpatches.Patch(color=COLOR["p4"], label="Phase 4 ★"),
    mpatches.Patch(color="none", label="-- P2 Baseline", linewidth=1),
]
fig.legend(handles=legend_patches2, fontsize=8, loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{OUT}/summary_all_metrics.png", bbox_inches="tight")
plt.close()
print("Saved summary_all_metrics.png")

print(f"\nAll plots saved to {OUT}/")
