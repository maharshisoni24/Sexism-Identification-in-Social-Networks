"""
dataset_2021.py — EXIST 2021 Dataset Reader

Reads the EXIST 2021 TSV files (bilingual Twitter text, hard categorical labels)
and converts them to the tensor format expected by the baseline model/loss pipeline.

TSV columns:
    test_case   id      source      language    text    task1           task2
    EXIST2021   000017  twitter     en          <txt>   non-sexist      non-sexist

Label mapping:
    task1: "sexist" → 1.0, "non-sexist" → 0.0   (binary float, shape [1])
    task2: one of 5 categories or "non-sexist"    (one-hot float, shape [5])

The loader exposes the same sample dict keys as the EXIST-2026 EXISTDataset so
that the existing EXISTDataModule (collate_fn, dataloaders) works without changes.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Class lists — must match CustomLoss / model head output order
# ---------------------------------------------------------------------------
TASK_1_CLASSES = ["sexist", "non-sexist"]

TASK_2_CATEGORIES = [
    "ideological-inequality",
    "stereotyping-dominance",
    "objectification",
    "sexual-violence",
    "misogyny-non-sexual-violence",
]

# For compatibility with train.py JSON submission builder
TASK_2_3_CLASSES = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]


class EXIST2021Dataset(Dataset):
    """
    Loads EXIST 2021 Twitter TSV data and converts hard labels to float tensors.

    Outputs per sample (dict):
        "id"         : str  — sample ID (e.g. "000017")
        "text"       : str  — raw tweet text
        "target_2_1" : FloatTensor [1]  — 1.0 if sexist, 0.0 otherwise
        "target_2_2" : FloatTensor [1]  — always 0.0 (task 2.2 not in EXIST 2021)
        "target_2_3" : FloatTensor [5]  — one-hot category encoding (sexist samples only)

    Why target_2_2 = 0 always?
        EXIST 2021 has no source-intention (direct/judgemental) labels.
        The baseline requires this key; the loss masks it by the sexist flag, so
        zeroed targets for non-sexist samples are consistent — and since EXIST 2021
        never provides a 2.2 label, we simply zero the entire column.  The Kendall
        uncertainty weight for task 2.2 will learn to down-weight this unused head.

    Args:
        tsv_path    : path to TSV file (training or test)
        language    : None (all), "en", or "es" to filter by language
    """

    def __init__(
        self,
        tsv_path: str,
        language: str = None,
    ):
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            header=0,
            names=["test_case", "id", "source", "language", "text", "task1", "task2"],
            dtype=str,
        )

        # Drop rows with missing critical fields
        df = df.dropna(subset=["text", "task1"])

        # Optional language filter
        if language is not None:
            df = df[df["language"] == language].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.data = df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_target_2_1(self, row) -> torch.Tensor:
        """Binary sexism label: sexist → 1.0, non-sexist → 0.0."""
        label = str(row["task1"]).strip().lower()
        val = 1.0 if label == "sexist" else 0.0
        return torch.tensor([val], dtype=torch.float32)

    def _make_target_2_2(self) -> torch.Tensor:
        """EXIST 2021 has no source-intention label. Return neutral 0.0."""
        return torch.tensor([0.0], dtype=torch.float32)

    def _make_target_2_3(self, row) -> torch.Tensor:
        """
        5-dim one-hot vector for sexism category.
        Non-sexist samples (task1 = non-sexist) get an all-zero vector —
        the hierarchical mask in the loss will zero their contribution.
        """
        t = torch.zeros(5, dtype=torch.float32)
        label = str(row.get("task2", "")).strip().lower()
        if label in TASK_2_CATEGORIES:
            idx = TASK_2_CATEGORIES.index(label)
            t[idx] = 1.0
        return t

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]

        sample = {
            "id": str(row["id"]).strip(),
            "text": str(row["text"]).strip(),
            "target_2_1": self._make_target_2_1(row),
            "target_2_2": self._make_target_2_2(),
            "target_2_3": self._make_target_2_3(row),
        }
        return sample
