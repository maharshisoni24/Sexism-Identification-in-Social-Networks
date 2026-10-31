"""
plot_hyperparam_sensitivity.py
One chart per hyperparameter — shows how each value affects Task1 F1,
Task2.3 F1, ECE and Violation Rate.
Saves to analysis_results/hyperparam_sensitivity/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = "analysis_results/hyperparam_sensitivity"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METRICS   = ["Task 1 F1", "Task 2.3 F1", "ECE (↓better)", "Viol % (↓better)"]
HIGHER    = [True, True, False, False]   # True = higher is better
COLORS_M  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def multi_metric_bar(param_name, xlabel, values, labels, note="",
                     filename="out.png"):
    """
    values : list of (T1, T23, ECE, Viol) tuples — one per label
    labels : x-axis tick labels (parameter values tested)
    """
    n      = len(labels)
    x      = np.arange(n)
    n_m    = len(METRICS)
    width  = 0.18
    offsets= np.linspace(-(n_m-1)*width/2, (n_m-1)*width/2, n_m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Hyperparameter Sensitivity — {param_name}", fontsize=13,
                 fontweight="bold")

    # ── Left: F1 scores ──────────────────────────────────────────────────────
    ax = axes[0]
    for mi, (metric, higher, color) in enumerate(
            zip(METRICS[:2], HIGHER[:2], COLORS_M[:2])):
        vals = [v[mi] for v in values]
        bars = ax.bar(x + offsets[mi], vals, width=width,
                      color=color, label=metric, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color)

    # star on best per metric
    for mi in range(2):
        vals = [v[mi] for v in values]
        best = vals.index(max(vals))
        ax.annotate("★ best", xy=(x[best] + offsets[mi], vals[best]+0.006),
                    ha="center", fontsize=8, color=COLORS_M[mi])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{param_name}={l}" for l in labels], fontsize=9)
    ax.set_ylabel("F1 Score (Macro)")
    ax.set_title("F1 Scores")
    ymin = min(v[mi] for v in values for mi in range(2)) - 0.04
    ymax = max(v[mi] for v in values for mi in range(2)) + 0.04
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── Right: ECE & Viol% ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2r = ax2.twinx()

    vals_ece  = [v[2] for v in values]
    vals_viol = [v[3] for v in values]

    bars_e = ax2.bar(x - width/2, vals_ece, width=width,
                     color=COLORS_M[2], label="ECE (↓)", alpha=0.88,
                     edgecolor="white")
    bars_v = ax2r.bar(x + width/2, vals_viol, width=width,
                      color=COLORS_M[3], label="Viol% (↓)", alpha=0.88,
                      edgecolor="white")

    for bar, val in zip(bars_e, vals_ece):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=COLORS_M[2])
    for bar, val in zip(bars_v, vals_viol):
        ax2r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                  f"{val:.1f}%", ha="center", va="bottom",
                  fontsize=8, fontweight="bold", color=COLORS_M[3])

    # star = best (min) per metric
    best_ece  = vals_ece.index(min(vals_ece))
    best_viol = vals_viol.index(min(vals_viol))
    ax2.annotate("★ best",
                 xy=(x[best_ece]-width/2, vals_ece[best_ece]+0.008),
                 ha="center", fontsize=8, color=COLORS_M[2])
    ax2r.annotate("★ best",
                  xy=(x[best_viol]+width/2, vals_viol[best_viol]+1.5),
                  ha="center", fontsize=8, color=COLORS_M[3])

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{param_name}={l}" for l in labels], fontsize=9)
    ax2.set_ylabel("ECE", color=COLORS_M[2])
    ax2r.set_ylabel("Violation Rate %", color=COLORS_M[3])
    ax2.set_title("Calibration & Violations")
    ax2.tick_params(axis="y", labelcolor=COLORS_M[2])
    ax2r.tick_params(axis="y", labelcolor=COLORS_M[3])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2r.spines["top"].set_visible(False)

    if note:
        fig.text(0.5, 0.01, f"Note: {note}", ha="center",
                 fontsize=8, color="gray", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(f"{OUT}/{filename}", bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. eps_max sensitivity  (N1 on α=0.5 γ=2, N2 off)
#    0.00 → p3_n1only  (new criterion, N3 off)
#    0.10 → p4_n1n3    (new criterion, N3 on eps=0.10, effective 0.05)
#    0.25 → Phase2 N1+N3 (old criterion — labelled)
# ─────────────────────────────────────────────────────────────────────────────
multi_metric_bar(
    param_name="eps_max (ε_max)",
    xlabel="eps_max value",
    values=[
        (0.7002, 0.5623, 0.4387, 55.9),   # 0.00 — baseline (N1 off too, for ref)
        (0.7270, 0.5970, 0.3777, 45.7),   # 0.00 — N1 only (eps=0 means N3 off)
        (0.7379, 0.5858, 0.3806, 43.4),   # 0.10 — N1+N3 improved
        (0.7129, 0.5555, 0.4375, 58.0),   # 0.25 — N1+N3 Phase 2
    ],
    labels=["0.00\n(baseline,\nno N1/N3)",
            "0.00\n(N1 on,\nN3 off)",
            "0.10\n(N1+N3,\nPhase 4 ★)",
            "0.25\n(N1+N3,\nPhase 2)"],
    note="N1 fixed at α=0.5 γ=2.0, N2 off. eps_max=0.25 used old checkpoint criterion.",
    filename="sensitivity_eps_max.png",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. lambda_cons sensitivity  (N1 on, N3 off — cleanest isolation)
#    0.0 → p3_n1only
#    0.2 → no perfect run (N1+N2 only with λ=0.2 not run) — use P3 all_improved
#          as proxy (also has N3 eps=0.10) — labelled as approximate
#    0.5 → Phase2 N1+N2
# ─────────────────────────────────────────────────────────────────────────────
multi_metric_bar(
    param_name="lambda_cons (λ)",
    xlabel="lambda_cons value",
    values=[
        (0.7022, 0.5942, 0.3658, 42.0),   # 0.0 — P3 baseline (N1 off ref)
        (0.7270, 0.5970, 0.3777, 45.7),   # 0.0 — P3 N1 only
        (0.7032, 0.5573, 0.3685, 40.4),   # 0.2 — P3 all improved (N1+N2+N3)
        (0.6987, 0.5503, 0.4064, 44.8),   # 0.5 — P2 N1+N2
    ],
    labels=["0.0\n(baseline,\nno N2)",
            "0.0\n(N1 on,\nno N2)",
            "0.2\n(N1+N2+N3,\nPhase 3)",
            "0.5\n(N1+N2,\nPhase 2)"],
    note="N1 fixed at α=0.5 γ=2.0. λ=0.2 run also includes N3 eps=0.10. λ=0.5 used old checkpoint criterion.",
    filename="sensitivity_lambda_cons.png",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. focal_alpha sensitivity  (N2 off, N3 off — cleanest)
#    0.0 → p3_baseline
#    0.5 → p3_n1only
# ─────────────────────────────────────────────────────────────────────────────
multi_metric_bar(
    param_name="focal_alpha (α)",
    xlabel="focal_alpha value",
    values=[
        (0.7022, 0.5942, 0.3658, 42.0),   # 0.0 — N1 off (baseline)
        (0.7270, 0.5970, 0.3777, 45.7),   # 0.5 — N1 on
    ],
    labels=["0.0\n(N1 disabled)", "0.5\n(N1 enabled ★)"],
    note="N2 off (λ=0), N3 off (ε=0). Both runs use new F1 checkpoint criterion. γ=2.0 fixed.",
    filename="sensitivity_focal_alpha.png",
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Combined: which novelty combination is best? (radar-style grouped bars)
# ─────────────────────────────────────────────────────────────────────────────
combos = [
    ("Baseline\n(all off)",   0.7002, 0.5623, 0.4387, 55.9),
    ("N1 only\n(P2)",         0.7101, 0.6033, 0.4630, 62.3),
    ("N2 only\n(P2)",         0.6989, 0.5677, 0.4187, 49.7),
    ("N3 only\n(P2)",         0.6775, 0.5383, 0.4341, 53.3),
    ("N1+N2\n(P2)",           0.6987, 0.5503, 0.4064, 44.8),
    ("N1+N3\n(P2)",           0.7129, 0.5555, 0.4375, 58.0),
    ("N1+N2+N3\n(P2)",        0.6961, 0.5584, 0.4482, 59.2),
    ("N1 only\n(P3 ★)",      0.7270, 0.5970, 0.3777, 45.7),
    ("N1+N2+N3\n(P3)",        0.7032, 0.5573, 0.3685, 40.4),
    ("N1+N3\n(P4 ★★)",      0.7379, 0.5858, 0.3806, 43.4),
]

cl = [r[0] for r in combos]
cv = [(r[1], r[2], r[3], r[4]) for r in combos]

# colour: grey=baseline, blue=P2, green=P3, red=P4
ccolors = [
    "#9E9E9E",                    # baseline
    "#4C72B0","#4C72B0","#4C72B0","#4C72B0","#4C72B0","#4C72B0",  # P2
    "#55A868","#55A868",          # P3
    "#C44E52",                    # P4
]

xc = np.arange(len(cl))
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("All Novelty Combinations — 4 Metrics Comparison\n"
             "(Which combination of N1/N2/N3 gives best results?)",
             fontsize=12, fontweight="bold")

for idx, (ax, mi, ylabel, hi) in enumerate(zip(
        axes,
        [0, 1, 2, 3],
        ["Task 1 F1 ↑", "Task 2.3 F1 ↑", "ECE ↓", "Viol % ↓"],
        [True, True, False, False])):

    vals = [v[mi] for v in cv]
    bars = ax.bar(xc, vals, color=ccolors, edgecolor="white", linewidth=0.5)
    ax.axhline([0.7002, 0.5623, 0.4387, 55.9][mi],
               color="black", linestyle="--", linewidth=1.2, label="P2 Baseline")
    ax.set_xticks(xc)
    ax.set_xticklabels(cl, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(ylabel, fontsize=10)

    for bar, val in zip(bars, vals):
        fmt = ".4f" if mi < 3 else ".1f"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+([0.002,0.002,0.002,0.5][mi]),
                format(val, fmt), ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    best = vals.index(max(vals) if hi else min(vals))
    bars[best].set_edgecolor("#FFD700"); bars[best].set_linewidth(2.5)

    spread = max(vals)-min(vals)
    ax.set_ylim(min(vals)-spread*0.35, max(vals)+spread*0.5)

import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color="#9E9E9E", label="Baseline"),
    mpatches.Patch(color="#4C72B0", label="Phase 2 (old hyperparams)"),
    mpatches.Patch(color="#55A868", label="Phase 3 (F1 monitor, eps=0.10, λ=0.20)"),
    mpatches.Patch(color="#C44E52", label="Phase 4 winner ★★"),
]
fig.legend(handles=legend_patches, fontsize=9, loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{OUT}/all_combinations_comparison.png", bbox_inches="tight")
plt.close()
print("Saved all_combinations_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Line chart — eps_max sweep (clearest "parameter tuning" story)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Parameter Tuning: eps_max (ε_max) Effect on Performance\n"
             "(N1 active α=0.5 γ=2.0, N2 off)", fontsize=12, fontweight="bold")

eps_vals  = [0.00, 0.10, 0.25]
eps_t1    = [0.7270, 0.7379, 0.7129]
eps_t23   = [0.5970, 0.5858, 0.5555]
eps_ece   = [0.3777, 0.3806, 0.4375]
eps_viol  = [45.7,   43.4,   58.0]

ax = axes[0]
ax.plot(eps_vals, eps_t1,  "o-", color="#2196F3", linewidth=2, markersize=8, label="Task 1 F1")
ax.plot(eps_vals, eps_t23, "s-", color="#4CAF50", linewidth=2, markersize=8, label="Task 2.3 F1")
for x_, y_ in zip(eps_vals, eps_t1):
    ax.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color="#2196F3", fontweight="bold")
for x_, y_ in zip(eps_vals, eps_t23):
    ax.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=9, color="#4CAF50", fontweight="bold")
ax.axvline(0.10, color="red", linestyle="--", alpha=0.6, label="Selected value (0.10)")
ax.set_xlabel("eps_max value")
ax.set_ylabel("F1 Score (Macro)")
ax.set_title("F1 vs eps_max")
ax.legend(fontsize=9)
ax.set_xticks(eps_vals)
ax.set_ylim(0.53, 0.77)

ax2 = axes[1]
ax2.plot(eps_vals, eps_ece,  "o-", color="#FF5722", linewidth=2, markersize=8, label="ECE (↓ better)")
ax2r = ax2.twinx()
ax2r.plot(eps_vals, eps_viol, "D-", color="#9C27B0", linewidth=2, markersize=8, label="Viol % (↓ better)")
for x_, y_ in zip(eps_vals, eps_ece):
    ax2.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9, color="#FF5722", fontweight="bold")
for x_, y_ in zip(eps_vals, eps_viol):
    ax2r.annotate(f"{y_:.1f}%", (x_, y_), textcoords="offset points",
                  xytext=(0, 8), ha="center", fontsize=9, color="#9C27B0", fontweight="bold")
ax2.axvline(0.10, color="red", linestyle="--", alpha=0.6, label="Selected (0.10)")
ax2.set_xlabel("eps_max value")
ax2.set_ylabel("ECE", color="#FF5722")
ax2r.set_ylabel("Violation %", color="#9C27B0")
ax2.set_title("ECE & Violations vs eps_max")
ax2.tick_params(axis="y", labelcolor="#FF5722")
ax2r.tick_params(axis="y", labelcolor="#9C27B0")
l1, lb1 = ax2.get_legend_handles_labels()
l2, lb2 = ax2r.get_legend_handles_labels()
ax2.legend(l1+l2, lb1+lb2, fontsize=9)
ax2.set_xticks(eps_vals)
ax2.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/line_eps_max_sweep.png", bbox_inches="tight")
plt.close()
print("Saved line_eps_max_sweep.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Line chart — lambda_cons sweep
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Parameter Tuning: lambda_cons (λ) Effect on Performance\n"
             "(N1 active α=0.5 γ=2.0)", fontsize=12, fontweight="bold")

# using best available runs for each lambda value
lam_vals = [0.0,   0.2,   0.5]
lam_t1   = [0.7270, 0.7032, 0.6987]
lam_t23  = [0.5970, 0.5573, 0.5503]
lam_ece  = [0.3777, 0.3685, 0.4064]
lam_viol = [45.7,   40.4,   44.8]

ax = axes[0]
ax.plot(lam_vals, lam_t1,  "o-", color="#2196F3", linewidth=2, markersize=8, label="Task 1 F1")
ax.plot(lam_vals, lam_t23, "s-", color="#4CAF50", linewidth=2, markersize=8, label="Task 2.3 F1")
for x_, y_ in zip(lam_vals, lam_t1):
    ax.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color="#2196F3", fontweight="bold")
for x_, y_ in zip(lam_vals, lam_t23):
    ax.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=9, color="#4CAF50", fontweight="bold")
ax.axvline(0.0, color="red", linestyle="--", alpha=0.6, label="Selected value (0.0 — N2 off)")
ax.set_xlabel("lambda_cons value")
ax.set_ylabel("F1 Score (Macro)")
ax.set_title("F1 vs lambda_cons")
ax.legend(fontsize=9)
ax.set_xticks(lam_vals)
ax.set_ylim(0.53, 0.77)

ax2 = axes[1]
ax2.plot(lam_vals, lam_ece,  "o-", color="#FF5722", linewidth=2, markersize=8, label="ECE (↓ better)")
ax2r = ax2.twinx()
ax2r.plot(lam_vals, lam_viol, "D-", color="#9C27B0", linewidth=2, markersize=8, label="Viol % (↓ better)")
for x_, y_ in zip(lam_vals, lam_ece):
    ax2.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9, color="#FF5722", fontweight="bold")
for x_, y_ in zip(lam_vals, lam_viol):
    ax2r.annotate(f"{y_:.1f}%", (x_, y_), textcoords="offset points",
                  xytext=(0, 8), ha="center", fontsize=9, color="#9C27B0", fontweight="bold")
ax2.axvline(0.0, color="red", linestyle="--", alpha=0.6, label="Selected (0.0)")
ax2.set_xlabel("lambda_cons value")
ax2.set_ylabel("ECE", color="#FF5722")
ax2r.set_ylabel("Violation %", color="#9C27B0")
ax2.set_title("ECE & Violations vs lambda_cons")
ax2.tick_params(axis="y", labelcolor="#FF5722")
ax2r.tick_params(axis="y", labelcolor="#9C27B0")
l1, lb1 = ax2.get_legend_handles_labels()
l2, lb2 = ax2r.get_legend_handles_labels()
ax2.legend(l1+l2, lb1+lb2, fontsize=9)
ax2.set_xticks(lam_vals)
ax2.spines["top"].set_visible(False)

fig.text(0.5, 0.01,
         "Note: λ=0.2 run also includes N3 (eps=0.10). λ=0.5 run uses old checkpoint criterion.",
         ha="center", fontsize=8, color="gray", style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(f"{OUT}/line_lambda_cons_sweep.png", bbox_inches="tight")
plt.close()
print("Saved line_lambda_cons_sweep.png")

print(f"\nAll 6 plots saved to {OUT}/")
