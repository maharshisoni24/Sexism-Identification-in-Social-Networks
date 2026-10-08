"""
loss.py — Custom Multi-Task Loss with Three Novel Modifications

Baseline (unchanged):
    - binary_kl_divergence_with_logits(): KL divergence vs soft targets
    - Kendall's homoscedastic uncertainty weighting (learnable log-variances)
    - Hierarchical masking: tasks 2.2/2.3 zeroed for non-sexist samples

Novelties (NEW):
    N1 — compute_disagreement_modulator():
        Confidence-Aware Focal Loss.  Up-weights samples near the decision
        boundary (p ≈ 0.5) using a boundary-symmetric uncertainty signal
        d_i = 1 - |2·p_i - 1|.  Unlike standard focal loss which blindly
        up-weights misclassified samples, this specifically targets inherent
        ambiguity as a proxy for annotator disagreement (LeWiDi paradigm).

    N2 — hierarchical_consistency_penalty():
        Hierarchical Confidence-Consistency Regularization.  Adds a quadratic
        penalty when Task 2 category probability exceeds Task 1 detection
        probability, enforcing the logical constraint that you cannot be more
        confident about *what type* of sexism than whether it is sexist at all.
        Task 1 logits are detached so gradients flow only to Task 2 heads.

    N3 — adaptive_label_smoothing():
        Adaptive Uncertainty-Modulated Label Smoothing.  Dynamically adjusts
        the label smoothing amount ε_i = ε_max · d_i per sample.  Clear
        samples (model confident) get sharp targets; ambiguous samples get
        softened targets, matching real annotator uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Baseline utility (unchanged from AI Wizards EXIST-2026)
# ============================================================

def binary_kl_divergence_with_logits(logits, targets, pos_weight=None):
    """
    Computes BCE with logits and subtracts the entropy of the targets
    to get the exact binary KL divergence.
    KL = BCE(logits, targets) - H(targets)
    Returns non-negative values (ReLU clamps floating-point noise).
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )

    # Clamp to avoid log(0) instability
    p = torch.clamp(targets, 1e-7, 1.0 - 1e-7)

    if pos_weight is not None:
        entropy = -(pos_weight * p * torch.log(p) + (1 - p) * torch.log(1 - p))
    else:
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    return F.relu(bce_loss - entropy)


# ============================================================
# N1 — Confidence-Aware Focal Loss
# ============================================================

def compute_disagreement_modulator(
    predicted_probs: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Computes a per-sample focal modulator based on predicted probability distance
    from the decision boundary (p = 0.5).

    disagreement_i = 1 - |2·p_i - 1|
        → 0 when model is very confident (p ≈ 0 or p ≈ 1)
        → 1 when model is maximally uncertain (p ≈ 0.5)

    modulator_i = (1 - alpha) + alpha · disagreement_i ^ gamma
        → baseline weight = (1 - alpha) for confident samples
        → maximum weight = 1.0 for perfectly uncertain samples

    The modulator is detached from the computation graph so it acts as a
    fixed per-step sample weight, not as a gradient path.

    Args:
        predicted_probs : sigmoid(logits), shape [B, *]
        alpha           : modulation strength (0 = disabled, 1 = full)
        gamma           : sharpness of uncertainty focus

    Returns:
        modulator of same shape as predicted_probs (detached)
    """
    disagreement = 1.0 - torch.abs(2.0 * predicted_probs - 1.0)
    modulator = (1.0 - alpha) + alpha * (disagreement ** gamma)
    return modulator.detach()


# ============================================================
# N2 — Hierarchical Confidence-Consistency Regularization
# ============================================================

def hierarchical_consistency_penalty(
    logits_t1: torch.Tensor,
    logits_t2: torch.Tensor,
) -> torch.Tensor:
    """
    Penalizes probability containment violations between hierarchical tasks:
        violation_i = max(0, sigmoid(logits_t2_i) - sigmoid(logits_t1_i))

    If Task 1 (binary sexist/not) gives 55% confidence but Task 2 category
    gives 90% confidence, violation = 0.35 → penalty = 0.35² = 0.1225.

    Task 1 logits are DETACHED so gradients flow only to Task 2 heads.
    Task 1 training is completely unaffected by this penalty.

    Args:
        logits_t1 : Task 1 logits, shape [B, 1]
        logits_t2 : Task 2 logits, shape [B, C]  (C categories)

    Returns:
        scalar penalty
    """
    p1 = torch.sigmoid(logits_t1).detach()   # [B, 1] — detached
    p2 = torch.sigmoid(logits_t2)             # [B, C]
    # p2 should never exceed p1 for any category
    violation = F.relu(p2 - p1)              # broadcast p1 [B,1] → [B,C]
    return torch.mean(violation ** 2)


# ============================================================
# N3 — Adaptive Uncertainty-Modulated Label Smoothing
# ============================================================

def adaptive_label_smoothing(
    logits: torch.Tensor,
    hard_targets: torch.Tensor,
    eps_max: float = 0.25,
) -> torch.Tensor:
    """
    Dynamically smooths hard targets per sample based on model uncertainty.

    ambiguity_i = 1 - |2·p_i - 1|  (same signal as N1)
    eps_i       = eps_max · ambiguity_i

    Smoothed target:
        If target = 1:  smoothed = 1 - eps_i   (e.g. 0.75 when maximally uncertain)
        If target = 0:  smoothed = eps_i        (e.g. 0.25 when maximally uncertain)
    General: smoothed = target * (1 - 2·eps_i) + eps_i

    For confident predictions (p ≈ 1):  eps_i ≈ 0  → targets remain 1.0/0.0 (sharp)
    For uncertain predictions (p ≈ 0.5): eps_i = eps_max → targets softened

    Args:
        logits       : raw model logits, shape [B, *]
        hard_targets : hard 0/1 targets (or one-hot), same shape as logits
        eps_max      : maximum smoothing amount (0 disables, 0.25 default)

    Returns:
        BCE loss per element (same shape as logits), using smoothed targets
    """
    probs = torch.sigmoid(logits).detach()
    ambiguity = 1.0 - torch.abs(2.0 * probs - 1.0)
    adaptive_eps = eps_max * ambiguity
    soft_targets = hard_targets * (1.0 - 2.0 * adaptive_eps) + adaptive_eps
    return F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")


# ============================================================
# CustomLoss — wires everything together
# ============================================================

class CustomLoss(nn.Module):
    """
    Multi-task loss for EXIST 2021 with three optional novelties.

    Args:
        focal_alpha   : N1 modulation strength (0 = disabled)
        focal_gamma   : N1 sharpness parameter
        lambda_cons   : N2 consistency penalty weight (0 = disabled)
        eps_max       : N3 maximum label smoothing (0 = disabled)
    """

    def __init__(
        self,
        focal_alpha: float = 0.0,
        focal_gamma: float = 2.0,
        lambda_cons: float = 0.0,
        eps_max: float = 0.0,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lambda_cons = lambda_cons
        self.eps_max = eps_max

        # Baseline: Kendall learnable log-variances (initialized to 0)
        self.log_var_1 = nn.Parameter(torch.zeros(1))
        self.log_var_2 = nn.Parameter(torch.zeros(1))
        self.log_var_3 = nn.Parameter(torch.zeros(1))

    def forward(self, outputs, targets, masks=None):
        """
        Args:
            outputs : dict with "logits_2_1" [B,1], "logits_2_2" [B,1], "logits_2_3" [B,5]
            targets : dict with "t_2_1" [B,1], "t_2_2" [B,1], "t_2_3" [B,5]
            masks   : dict with optional "physio_mask" and "cond_mask" [B,1]

        Returns:
            dict: loss_2_1, loss_2_2, loss_2_3, total_loss, sigma_1, sigma_2, sigma_3
                  + n1_modulator_mean, n2_consistency_loss, n3_applied (for logging)
        """
        t_2_1 = targets["t_2_1"].float()
        t_2_2 = targets["t_2_2"].float()
        t_2_3 = targets["t_2_3"].float()

        logits_2_1 = outputs["logits_2_1"]
        logits_2_2 = outputs["logits_2_2"]
        logits_2_3 = outputs["logits_2_3"]

        # ----------------------------------------------------------
        # Step 1: Compute predicted probabilities (for N1 & N3)
        # ----------------------------------------------------------
        p_2_1 = torch.sigmoid(logits_2_1)
        p_2_3 = torch.sigmoid(logits_2_3)

        # ----------------------------------------------------------
        # Step 2: Compute raw per-sample losses
        # ----------------------------------------------------------
        use_n3 = (self.eps_max > 0.0)

        if use_n3:
            # N3: compute BCE with smoothed targets instead of standard KL
            L1_raw = adaptive_label_smoothing(logits_2_1, t_2_1, self.eps_max)
            # Task 2.2 — EXIST 2021 has no source-intention labels; use standard BCE
            L2_raw = F.binary_cross_entropy_with_logits(
                logits_2_2, t_2_2, reduction="none"
            )
            L3_raw = adaptive_label_smoothing(logits_2_3, t_2_3, self.eps_max).mean(
                dim=1, keepdim=True
            )
        else:
            # Baseline: KL divergence against targets
            L1_raw = binary_kl_divergence_with_logits(logits_2_1, t_2_1)
            L2_raw = binary_kl_divergence_with_logits(logits_2_2, t_2_2)
            L3_raw = binary_kl_divergence_with_logits(logits_2_3, t_2_3).mean(
                dim=1, keepdim=True
            )

        # ----------------------------------------------------------
        # Step 3: N1 — Confidence-Aware Focal Modulation
        # ----------------------------------------------------------
        n1_applied = False
        n1_modulator_mean = torch.tensor(1.0)

        if self.focal_alpha > 0.0:
            n1_applied = True
            mod_2_1 = compute_disagreement_modulator(p_2_1, self.focal_alpha, self.focal_gamma)
            mod_2_3 = compute_disagreement_modulator(p_2_3, self.focal_alpha, self.focal_gamma)
            # Average across category dimension for task 2.3
            mod_2_3_mean = mod_2_3.mean(dim=1, keepdim=True)

            L1_raw = mod_2_1 * L1_raw
            L3_raw = mod_2_3_mean * L3_raw
            n1_modulator_mean = mod_2_1.mean().detach()

        # ----------------------------------------------------------
        # Step 4: Kendall's homoscedastic uncertainty weighting
        # ----------------------------------------------------------
        precision_1 = torch.exp(-self.log_var_1)
        precision_2 = torch.exp(-self.log_var_2)
        precision_3 = torch.exp(-self.log_var_3)

        L1_scaled = (precision_1 * L1_raw) + (0.5 * self.log_var_1)
        L2_scaled = (precision_2 * L2_raw) + (0.5 * self.log_var_2)
        L3_scaled = (precision_3 * L3_raw) + (0.5 * self.log_var_3)

        # ----------------------------------------------------------
        # Step 5: Hierarchical masking (baseline unchanged)
        # ----------------------------------------------------------
        is_sexist_mask = (t_2_1 > 0.0).float()

        L2_masked = L2_scaled * is_sexist_mask
        L3_masked = L3_scaled * is_sexist_mask

        valid_count = is_sexist_mask.sum() + 1e-8

        loss_1 = L1_scaled.mean()
        loss_2 = L2_masked.sum() / valid_count
        loss_3 = L3_masked.sum() / valid_count

        total_loss = loss_1 + loss_2 + loss_3

        # ----------------------------------------------------------
        # Step 6: N2 — Hierarchical Consistency Regularization
        # ----------------------------------------------------------
        n2_consistency_loss = torch.tensor(0.0, device=logits_2_1.device)

        if self.lambda_cons > 0.0:
            n2_consistency_loss = hierarchical_consistency_penalty(
                logits_2_1, logits_2_3
            )
            total_loss = total_loss + self.lambda_cons * n2_consistency_loss

        # ----------------------------------------------------------
        # Return
        # ----------------------------------------------------------
        return {
            "loss_2_1": loss_1,
            "loss_2_2": loss_2,
            "loss_2_3": loss_3,
            "total_loss": total_loss,
            "sigma_1": torch.exp(0.5 * self.log_var_1),
            "sigma_2": torch.exp(0.5 * self.log_var_2),
            "sigma_3": torch.exp(0.5 * self.log_var_3),
            # Novelty diagnostics (for TensorBoard logging)
            "n1_modulator_mean": n1_modulator_mean if n1_applied else torch.tensor(1.0),
            "n2_consistency_loss": n2_consistency_loss.detach(),
        }
