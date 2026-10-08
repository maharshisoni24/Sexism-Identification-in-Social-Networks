"""
train.py — Training Script for EXIST 2021 with Novelties N1/N2/N3

Adapted from AI Wizards EXIST-2026 train.py.

Changes vs. original:
  - Uses EXIST2021Dataset + EXIST2021DataModule (TSV-based, text-only)
  - CustomLoss receives 4 new CLI args: --focal_alpha, --focal_gamma,
    --lambda_cons, --eps_max to independently enable/disable N1/N2/N3
  - Test case name changed to EXIST2021
  - Version string automatically encodes novelty config for easy tracking
  - predict/eval uses internal val/test split from training data (no external
    JSON test labels available for EXIST 2021 without annotation files)
"""

import json
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                          ModelSummary)
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule_2021 import EXIST2021DataModule
from dataset_2021 import EXIST2021Dataset, TASK_2_3_CLASSES
from model import EXISTModel

torch.set_float32_matmul_precision("high")

TEST_CASE = "EXIST2021"


# ---------------------------------------------------------------------------
# Prediction JSON builders (identical logic to baseline, adapted for EXIST 2021)
# ---------------------------------------------------------------------------

def _build_task1_json_entries(ids, yes_probs, hard: bool):
    """Task 1: binary sexism identification (YES / NO)."""
    entries = []
    for sample_id, p_yes in zip(ids, yes_probs):
        value = "YES" if (hard and p_yes >= 0.5) else (
            {"YES": p_yes, "NO": 1.0 - p_yes} if not hard else "NO"
        )
        entries.append({"id": str(sample_id), "value": value, "test_case": TEST_CASE})
    return entries


def _build_task2_1_json_entries(ids, yes_probs, hard: bool):
    return _build_task1_json_entries(ids, yes_probs, hard)


def _build_task2_2_json_entries(ids, p_2_1, judgemental_probs, hard: bool):
    """Task 2.2: source intention (not in EXIST 2021, included for format compat.)"""
    entries = []
    for sample_id, p_yes, p_judg in zip(ids, p_2_1, judgemental_probs):
        if hard:
            value = "NO" if p_yes < 0.5 else ("JUDGEMENTAL" if p_judg >= 0.5 else "DIRECT")
        else:
            value = {
                "JUDGEMENTAL": float(p_yes * p_judg),
                "DIRECT": float(p_yes * (1.0 - p_judg)),
                "NO": float(1.0 - p_yes),
            }
        entries.append({"id": str(sample_id), "value": value, "test_case": TEST_CASE})
    return entries


def _build_task2_3_json_entries(ids, p_2_1, cat_probs, hard: bool):
    """Task 2.3: 5-class multi-label categorization."""
    entries = []
    for sample_id, p_yes, probs in zip(ids, p_2_1, cat_probs):
        if hard:
            if p_yes < 0.5:
                value = ["NO"]
            else:
                value = [TASK_2_3_CLASSES[i] for i, p in enumerate(probs) if p >= 0.5]
                if not value:
                    best_idx = max(range(len(probs)), key=lambda i: probs[i])
                    value = [TASK_2_3_CLASSES[best_idx]]
        else:
            value = {TASK_2_3_CLASSES[i]: float(p_yes * p) for i, p in enumerate(probs)}
            value["NO"] = float(1.0 - p_yes)
        entries.append({"id": str(sample_id), "value": value, "test_case": TEST_CASE})
    return entries


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _run_eval(trainer, model, datamodule, best_ckpt, version, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, version)
    os.makedirs(run_dir, exist_ok=True)

    print("\n===> Running predictions on the test split...")
    predictions = trainer.predict(
        model=model,
        dataloaders=datamodule.predict_dataloader(),
        ckpt_path=best_ckpt,
    )

    all_ids, all_probs_2_1, all_probs_2_2, all_probs_2_3 = [], [], [], []

    for batch_out in predictions:
        batch_ids = batch_out["id"]
        probs_2_1 = torch.sigmoid(batch_out["logits_2_1"]).squeeze(-1).cpu().tolist()
        probs_2_2 = torch.sigmoid(batch_out["logits_2_2"]).squeeze(-1).cpu().tolist()
        probs_2_3 = torch.sigmoid(batch_out["logits_2_3"]).cpu().tolist()
        all_ids.extend(batch_ids)
        all_probs_2_1.extend(probs_2_1)
        all_probs_2_2.extend(probs_2_2)
        all_probs_2_3.extend(probs_2_3)

    combined = list(zip(all_ids, all_probs_2_1, all_probs_2_2, all_probs_2_3))
    combined.sort(key=lambda x: int(x[0]))
    sorted_ids, sorted_p21, sorted_p22, sorted_p23 = zip(*combined)

    for task_name, probs_list, build_fn in [
        ("task2_1", sorted_p21,
         lambda ids, probs, hard: _build_task2_1_json_entries(ids, probs, hard)),
        ("task2_2", sorted_p22,
         lambda ids, probs, hard: _build_task2_2_json_entries(ids, sorted_p21, probs, hard)),
        ("task2_3", sorted_p23,
         lambda ids, probs, hard: _build_task2_3_json_entries(ids, sorted_p21, probs, hard)),
    ]:
        for label, hard_flag in [("hard", True), ("soft", False)]:
            entries = build_fn(sorted_ids, probs_list, hard=hard_flag)
            out_path = os.path.join(run_dir, f"{task_name}_{label}_1.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"  Saved: {out_path}")

    # Save raw probabilities for analysis notebook
    raw_preds = {
        "ids": list(sorted_ids),
        "p_sexist": list(sorted_p21),
        "p_judgemental": list(sorted_p22),
        "p_categories": [list(p) for p in sorted_p23],
    }
    with open(os.path.join(run_dir, "raw_predictions.json"), "w") as f:
        json.dump(raw_preds, f, indent=2)
    print(f"  Raw predictions saved to: {run_dir}/raw_predictions.json")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def make_version_tag(args):
    """Creates a human-readable version string encoding novelty config."""
    n1 = f"N1a{args.focal_alpha}g{args.focal_gamma}" if args.focal_alpha > 0 else "N1off"
    n2 = f"N2l{args.lambda_cons}" if args.lambda_cons > 0 else "N2off"
    n3 = f"N3e{args.eps_max}" if args.eps_max > 0 else "N3off"
    tag = args.version or datetime.now().strftime("%d%m_%H%M")
    return f"{tag}_{n1}_{n2}_{n3}"


def train(args):
    version = make_version_tag(args)
    print(f"\n{'='*60}")
    print(f"Run: {version}")
    print(f"  N1 focal_alpha={args.focal_alpha}  focal_gamma={args.focal_gamma}")
    print(f"  N2 lambda_cons={args.lambda_cons}")
    print(f"  N3 eps_max={args.eps_max}")
    print(f"{'='*60}\n")

    # TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name=version)

    # Optionally add W&B
    loggers = [tb_logger]
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=version,
            save_dir="wandb",
            entity=args.wandb_entity,
            log_model="all",
            config={
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "lambda_cons": args.lambda_cons,
                "eps_max": args.eps_max,
                "epochs": args.epochs,
                "seed": args.seed,
                "language": args.language,
            },
        )
        loggers.append(wandb_logger)

    print("===> Loading EXIST 2021 datasets")
    train_dataset = EXIST2021Dataset(
        tsv_path=args.train_tsv,
        language=args.language,
    )
    print(f"  Training data: {len(train_dataset)} samples")

    datamodule = EXIST2021DataModule(
        train_dataset=train_dataset,
        test_dataset=None,   # use internal split for eval
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_size=0.10,
        test_size=0.10,
    )

    print("===> Building model")
    model = EXISTModel(
        model_name="gemini",
        n_blocks=args.n_blocks,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        soft_gating=args.soft_gating,
        # Novelty hyperparameters
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        lambda_cons=args.lambda_cons,
        eps_max=args.eps_max,
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{version}",
        filename="{epoch:02d}",
        save_top_k=1,
        save_last=False,
        monitor="val/total_loss",
        mode="min",
    )
    callbacks: list[Callback] = [
        model_checkpoint,
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=3),
        pl.callbacks.EarlyStopping(
            monitor="val/total_loss",
            patience=max(5, args.epochs // 5),
            mode="min",
            verbose=True,
        ),
    ]

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
    )

    print("===> Starting training")
    trainer.fit(model, datamodule)

    best_ckpt = model_checkpoint.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")

    _run_eval(trainer, model, datamodule, best_ckpt, version)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EXIST 2021 Novelty Model")

    # Data
    parser.add_argument("--train_tsv",  default="data/EXIST2021_training.tsv",
                        help="Path to EXIST 2021 training TSV")
    parser.add_argument("--language",   default=None, choices=[None, "en", "es"],
                        help="Filter by language (default: both)")

    # Model
    parser.add_argument("--n_blocks",         type=int,   default=2)
    parser.add_argument("--expansion_factor", type=int,   default=2)
    parser.add_argument("--dropout",          type=float, default=0.2)
    parser.add_argument("--soft_gating",      action="store_true")

    # Training
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--version",      type=str,   default=None,
                        help="Version prefix for logging (default: timestamp)")

    # === Novelty hyperparameters ===
    parser.add_argument(
        "--focal_alpha", type=float, default=0.0,
        help="N1: focal modulation strength (0 = disabled, 0.5 = default)"
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0,
        help="N1: focal sharpness (higher = more focus on uncertain samples)"
    )
    parser.add_argument(
        "--lambda_cons", type=float, default=0.0,
        help="N2: hierarchical consistency penalty weight (0 = disabled, 0.5 = default)"
    )
    parser.add_argument(
        "--eps_max", type=float, default=0.0,
        help="N3: maximum adaptive label smoothing (0 = disabled, 0.25 = default)"
    )

    # Logging
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="EXIST2021-novelty")
    parser.add_argument("--wandb_entity",  default=None)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    train(args)
