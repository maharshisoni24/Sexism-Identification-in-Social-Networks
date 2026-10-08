"""
model.py — EXIST 2021 Lightning Model Wrapper

Adapted from AI Wizards EXIST-2026 model.py.
Changes:
  - CustomLoss receives N1/N2/N3 hyperparameters (focal_alpha, focal_gamma,
    lambda_cons, eps_max) passed from training config.
  - Gemini-only backend (SigLIP removed — EXIST 2021 is text-only).
  - Task 2.2 head kept for structural compatibility with the loss pipeline
    (EXIST 2021 has no source-intention labels; the task is effectively
    disabled via zero targets and hierarchical masking).
  - Metrics adapted: Task 1 Binary F1, Task 2.3 Multilabel F1 (macro).
"""

import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, MultilabelF1Score

from loss import CustomLoss
from models.Gemini import Gemini


class EXISTModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "gemini",
        n_blocks: int = 2,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
        soft_gating: bool = False,
        # --- Novelty hyperparameters ---
        focal_alpha: float = 0.0,
        focal_gamma: float = 2.0,
        lambda_cons: float = 0.0,
        eps_max: float = 0.0,
    ):
        super().__init__()
        self.requires_image = False
        self.requires_text = False
        self.requires_ids = True  # Gemini always needs IDs

        if model_name != "gemini":
            raise ValueError(
                f"exist2021-novelty only supports the 'gemini' backend. Got: {model_name}"
            )

        self.model = Gemini(
            n_blocks=n_blocks,
            expansion_factor=expansion_factor,
            dropout=dropout,
            soft_gating=soft_gating,
            use_demographics=False,
            use_sensorial=False,
            use_subject_ids=False,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        # Instantiate loss with novelty hyperparameters
        self.criterion = CustomLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            lambda_cons=lambda_cons,
            eps_max=eps_max,
        )

        self.save_hyperparameters(ignore=["model"])

        # --- Metrics ---
        # Task 1: binary sexism identification
        self.metrics_2_1 = MetricCollection(
            {"f1": BinaryF1Score(threshold=0.5)},
            postfix="_2_1",
        )

        # Task 2.2: not used in EXIST 2021 (kept for structural compatibility)
        self.metrics_2_2 = MetricCollection(
            {"f1": BinaryF1Score(threshold=0.5)},
            postfix="_2_2",
        )

        # Task 2.3: multi-label sexism categorization
        self.metrics_2_3 = MetricCollection(
            {
                "f1_macro": MultilabelF1Score(
                    num_labels=5, average="macro", threshold=0.3
                ),
            },
            postfix="_2_3",
        )

    def forward(self, ids=None, **kwargs):
        if ids is None:
            raise ValueError("Gemini model requires 'ids' in forward pass")
        return self.model(ids)

    def _step(self, batch, batch_idx):
        ids = batch.get("id", None)

        outputs = self.forward(ids=ids)

        targets = {
            "t_2_1": batch["target_2_1"],
            "t_2_2": batch["target_2_2"],
            "t_2_3": batch["target_2_3"],
        }

        masks = {
            "physio_mask": None,
            "cond_mask": (batch["target_2_1"] > 0).float(),
        }

        return outputs, targets, masks

    def compute_metrics(self, outputs, targets, masks):
        preds_2_1_prob = torch.sigmoid(outputs["logits_2_1"])
        preds_2_2_prob = torch.sigmoid(outputs["logits_2_2"])
        preds_2_3_prob = torch.sigmoid(outputs["logits_2_3"])

        t_2_1 = targets["t_2_1"]
        t_2_2 = targets["t_2_2"]
        t_2_3 = targets["t_2_3"]

        # Task 2.1
        hard_t_2_1 = (t_2_1 >= 0.5).int()
        self.metrics_2_1["f1"].update(preds_2_1_prob, hard_t_2_1)

        # Conditional mask (sexist samples only)
        valid_mask = masks["cond_mask"].squeeze().bool()
        if valid_mask.any():
            valid_preds_2_3 = preds_2_3_prob[valid_mask].reshape(-1, 5)
            valid_targets_2_3 = t_2_3[valid_mask].reshape(-1, 5)
            hard_t_2_3 = (valid_targets_2_3 >= 0.5).int()
            self.metrics_2_3["f1_macro"].update(valid_preds_2_3, hard_t_2_3)

    def training_step(self, batch, batch_idx):
        outputs, targets, masks = self._step(batch, batch_idx)
        loss_dict = self.criterion(outputs, targets, masks)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        outputs, targets, masks = self._step(batch, batch_idx)
        loss_dict = self.criterion(outputs, targets, masks)
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.compute_metrics(outputs, targets, masks)
        return {
            "loss": loss_dict["total_loss"],
            "id": batch["id"],
            "logits_2_1": outputs["logits_2_1"].detach(),
            "logits_2_2": outputs["logits_2_2"].detach(),
            "logits_2_3": outputs["logits_2_3"].detach(),
        }

    def on_validation_epoch_end(self):
        self.log_dict(
            {f"val/{k}": v for k, v in self.metrics_2_1.compute().items()},
            logger=True, prog_bar=True,
        )
        self.metrics_2_1.reset()

        try:
            self.log_dict(
                {f"val/{k}": v for k, v in self.metrics_2_3.compute().items()},
                logger=True, prog_bar=False,
            )
        except Exception:
            pass
        finally:
            self.metrics_2_2.reset()
            self.metrics_2_3.reset()

    def predict_step(self, batch, batch_idx):
        outputs, _, _ = self._step(batch, batch_idx)
        return {
            "id": batch["id"],
            "logits_2_1": outputs["logits_2_1"].float(),
            "logits_2_2": outputs["logits_2_2"].float(),
            "logits_2_3": outputs["logits_2_3"].float(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=total_steps,
            pct_start=self.warmup_ratio,
            max_lr=self.lr,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
        print(f"Optimizer: {optimizer}")
        print(f"Scheduler: {scheduler}")
        print(f"Estimated stepping batches: {total_steps}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "onecycle_lr",
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)
