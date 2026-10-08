"""
datamodule_2021.py — DataModule for EXIST 2021

Wraps EXIST2021Dataset for stratified train/val/test splitting.
Adapted from AI Wizards EXIST-2026 datamodule.py.

Key changes:
  - No image/physiological modalities (EXIST 2021 is text-only)
  - Collate fn handles: id (str), text (str), target_2_1/2_2/2_3 (tensors)
  - Stratification is on task1 hard label (1.0 = sexist)
  - 80 / 10 / 10 split (configurable)
"""

from functools import partial
from typing import Optional

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


# -------------------------------------------------------
# Collate function — text-only, no images/physio
# -------------------------------------------------------

def collate_fn_2021(batch):
    """
    Collates a list of EXIST2021Dataset items into a batch dict.
    Images and physiological features are not present.
    """
    collated = {}

    # IDs (keep as strings for JSON submission compatibility)
    if "id" in batch[0]:
        collated["id"] = [str(item["id"]) for item in batch]

    # Text (list of strings — fed to Gemini embedding lookup via ID)
    if "text" in batch[0]:
        collated["text"] = [item["text"] for item in batch]

    # Targets
    collated["target_2_1"] = torch.stack([item["target_2_1"] for item in batch])
    collated["target_2_2"] = torch.stack([item["target_2_2"] for item in batch])
    collated["target_2_3"] = torch.stack([item["target_2_3"] for item in batch])

    return collated


# -------------------------------------------------------
# DataModule
# -------------------------------------------------------

class EXIST2021DataModule(pl.LightningDataModule):
    """
    Args:
        train_dataset   : EXIST2021Dataset for training split
        test_dataset    : EXIST2021Dataset for held-out test (predict mode)
        batch_size      : dataloader batch size
        num_workers     : dataloader workers
        seed            : random seed for stratified split
        val_size        : fraction of training data used for val (default 0.10)
        test_size       : fraction of training data used for test (default 0.10)
    """

    def __init__(
        self,
        train_dataset=None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        val_size: float = 0.10,
        test_size: float = 0.10,
    ):
        super().__init__()
        self.train_full = train_dataset
        self.test_external = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_size = val_size
        self.test_size = test_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage=None):
        if stage == "predict":
            if self.test_external is not None:
                self.predict_dataset = self.test_external
                return

        if self.train_dataset is None and self.train_full is not None:
            # Stratify on binary task1 label
            strata = []
            for _, row in self.train_full.data.iterrows():
                label = str(row.get("task1", "non-sexist")).strip().lower()
                strata.append(1 if label == "sexist" else 0)

            all_idx = list(range(len(self.train_full)))
            split_temp = self.val_size + self.test_size

            train_idx, temp_idx = train_test_split(
                all_idx,
                test_size=split_temp,
                stratify=strata,
                random_state=self.seed,
            )

            temp_strata = [strata[i] for i in temp_idx]
            relative_test = self.test_size / split_temp
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=relative_test,
                stratify=temp_strata,
                random_state=self.seed,
            )

            self.train_dataset = Subset(self.train_full, train_idx)
            self.val_dataset = Subset(self.train_full, val_idx)
            self.test_dataset = Subset(self.train_full, test_idx)
            self.predict_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_2021,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_2021,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_2021,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_2021,
            pin_memory=False,
        )
