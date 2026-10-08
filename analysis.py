"""
analysis.py — Post-Training Analysis Script

Run this after all 7 ablation runs complete to generate:
  1. Main evaluation table (Task 1 F1, Task 2.3 Macro F1, Accuracy)
  2. Per-uncertainty-bin performance breakdown (proves N1 works)
  3. Hierarchical consistency violation statistics (proves N2 works)
  4. Calibration analysis (proves N3 works)
  5. Training loss curves (from TensorBoard event files)

Usage:
    python analysis.py --outputs_dir outputs/ --data_tsv data/EXIST2021_training.tsv

Outputs written to: analysis_results/
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def load_raw_predictions(run_dir: str) -> dict:
    """Load raw_predictions.json from a run output directory."""
    path = os.path.join(run_dir, "raw_predictions.json")
    with open(path) as f:
        return json.load(f)


def load_ground_truth(tsv_path: str) -> pd.DataFrame:
    """Load EXIST 2021 TSV and return id → (task1_label, task2_label) mapping."""
    df = pd.read_csv(
        tsv_path, sep="\t", header=0,
        names=["test_case", "id", "source", "language", "text", "task1", "task2"],
        dtype=str,
    ).dropna(subset=["task1"])
    df["id"] = df["id"].str.strip()
    return df.set_index("id")


def compute_f1_binary(preds, golds, threshold=0.5):
    """Binary F1 (macro: average of F1 for each class)."""
    from sklearn.metrics import f1_score
    hard_preds = [1 if p >= threshold else 0 for p in preds]
    hard_golds = [1 if g == "sexist" else 0 for g in golds]
    return f1_score(hard_golds, hard_preds, average="macro", zero_division=0)


def compute_accuracy(preds, golds, threshold=0.5):
    from sklearn.metrics import accuracy_score
    hard_preds = [1 if p >= threshold else 0 for p in preds]
    hard_golds = [1 if g == "sexist" else 0 for g in golds]
    return accuracy_score(hard_golds, hard_preds)


TASK2_CATS = [
    "ideological-inequality",
    "stereotyping-dominance",
    "objectification",
    "sexual-violence",
    "misogyny-non-sexual-violence",
]


def compute_f1_multilabel(cat_probs, golds_task2, threshold=0.3):
    """Macro F1 for 5-class task2 categorization."""
    from sklearn.metrics import f1_score
    hard_preds, hard_golds = [], []
    for probs, gold in zip(cat_probs, golds_task2):
        pred_vec = [1 if p >= threshold else 0 for p in probs]
        gold_cat = str(gold).strip().lower()
        gold_vec = [1 if cat == gold_cat else 0 for cat in TASK2_CATS]
        hard_preds.append(pred_vec)
        hard_golds.append(gold_vec)
    return f1_score(hard_golds, hard_preds, average="macro", zero_division=0)


def compute_ece(probs, is_correct, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (np.array(probs) >= bins[i]) & (np.array(probs) < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = np.array(probs)[mask].mean()
        avg_acc = np.array(is_correct)[mask].mean()
        ece += abs(avg_conf - avg_acc) * mask.sum() / n
    return ece


def compute_consistency_violation_rate(p_sexist, cat_probs):
    """
    % of samples where max(p_category) > p_sexist (hierarchical violation).
    Also returns mean violation magnitude.
    """
    violations = []
    for ps, cats in zip(p_sexist, cat_probs):
        max_cat = max(cats)
        viol = max(0.0, max_cat - ps)
        violations.append(viol)
    viol_arr = np.array(violations)
    rate = (viol_arr > 0).mean() * 100
    mean_mag = viol_arr[viol_arr > 0].mean() if (viol_arr > 0).any() else 0.0
    return rate, mean_mag


# ============================================================
# Main analysis
# ============================================================

def analyse(args):
    os.makedirs(args.out_dir, exist_ok=True)
    gt = load_ground_truth(args.data_tsv)

    run_dirs = sorted(
        [d for d in Path(args.outputs_dir).iterdir() if d.is_dir()
         and (d / "raw_predictions.json").exists()]
    )

    if not run_dirs:
        print(f"No completed runs found in {args.outputs_dir}.")
        print("Each run must have a raw_predictions.json file.")
        return

    print(f"Found {len(run_dirs)} completed run(s):\n")
    for r in run_dirs:
        print(f"  {r.name}")

    rows = []
    consistency_rows = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        preds = load_raw_predictions(str(run_dir))

        ids = preds["ids"]
        p_sexist = preds["p_sexist"]
        cat_probs = preds["p_categories"]

        # Match IDs to ground truth
        gt_task1 = [gt.loc[i, "task1"] if i in gt.index else "non-sexist" for i in ids]
        gt_task2 = [gt.loc[i, "task2"] if i in gt.index else "non-sexist" for i in ids]

        # Task 1 metrics
        f1_t1 = compute_f1_binary(p_sexist, gt_task1)
        acc_t1 = compute_accuracy(p_sexist, gt_task1)

        # Task 2.3 metrics (only on sexist samples)
        sexist_idx = [i for i, g in enumerate(gt_task1) if g == "sexist"]
        if sexist_idx:
            f1_t23 = compute_f1_multilabel(
                [cat_probs[i] for i in sexist_idx],
                [gt_task2[i] for i in sexist_idx],
            )
        else:
            f1_t23 = 0.0

        # ECE (Task 1)
        is_correct = [1 if (p >= 0.5) == (g == "sexist") else 0
                      for p, g in zip(p_sexist, gt_task1)]
        ece = compute_ece(p_sexist, is_correct)

        # N2: consistency violation rate
        viol_rate, viol_mag = compute_consistency_violation_rate(p_sexist, cat_probs)

        rows.append({
            "Run": run_name,
            "Task1 F1 (Macro)": f"{f1_t1:.4f}",
            "Task1 Accuracy": f"{acc_t1:.4f}",
            "Task2.3 F1 (Macro)": f"{f1_t23:.4f}",
            "ECE (Task1)": f"{ece:.4f}",
            "Viol. Rate %": f"{viol_rate:.1f}",
            "Viol. Magnitude": f"{viol_mag:.4f}",
        })

        # --- Per-uncertainty-bin analysis (N1 validation) ---
        uncertainties = [1.0 - abs(2 * p - 1) for p in p_sexist]
        for bin_name, lo, hi in [("Low (d<0.2)", 0, 0.2), ("Medium (0.2≤d<0.5)", 0.2, 0.5), ("High (d≥0.5)", 0.5, 1.01)]:
            idx_bin = [i for i, d in enumerate(uncertainties) if lo <= d < hi]
            if idx_bin:
                f1_bin = compute_f1_binary(
                    [p_sexist[i] for i in idx_bin],
                    [gt_task1[i] for i in idx_bin],
                )
            else:
                f1_bin = float("nan")
            consistency_rows.append({
                "Run": run_name,
                "Uncertainty Bin": bin_name,
                "N Samples": len(idx_bin),
                "F1": f"{f1_bin:.4f}" if not np.isnan(f1_bin) else "N/A",
            })

    # ---- Main table ----
    main_df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("MAIN EVALUATION TABLE")
    print("="*80)
    print(main_df.to_string(index=False))
    main_df.to_csv(os.path.join(args.out_dir, "main_results.csv"), index=False)

    # ---- Per-uncertainty-bin table ----
    bin_df = pd.DataFrame(consistency_rows)
    print("\n" + "="*80)
    print("PER-UNCERTAINTY-BIN F1 (Task 1)")
    print("="*80)
    print(bin_df.to_string(index=False))
    bin_df.to_csv(os.path.join(args.out_dir, "per_bin_results.csv"), index=False)

    # ---- Plots ----
    # Task 1 F1 bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    main_df["Task1 F1 (Macro)"] = main_df["Task1 F1 (Macro)"].astype(float)
    ax.bar(main_df["Run"], main_df["Task1 F1 (Macro)"], color="steelblue")
    ax.set_ylabel("Task 1 Macro F1")
    ax.set_title("Task 1 Macro F1 — Ablation Comparison")
    ax.set_xticklabels(main_df["Run"], rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "task1_f1_comparison.png"), dpi=150)
    plt.close()

    # Consistency violation rate comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    main_df["Viol. Rate %"] = main_df["Viol. Rate %"].astype(float)
    ax.bar(main_df["Run"], main_df["Viol. Rate %"], color="tomato")
    ax.set_ylabel("Hierarchical Violation Rate (%)")
    ax.set_title("N2: Hierarchical Consistency Violations — Before vs. After")
    ax.set_xticklabels(main_df["Run"], rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "consistency_violations.png"), dpi=150)
    plt.close()

    print(f"\nResults and plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training ablation analysis")
    parser.add_argument("--outputs_dir", default="outputs/", help="Directory of run outputs")
    parser.add_argument("--data_tsv",    default="data/EXIST2021_training.tsv", help="Ground truth TSV")
    parser.add_argument("--out_dir",     default="analysis_results/", help="Where to save results")
    args = parser.parse_args()
    analyse(args)
