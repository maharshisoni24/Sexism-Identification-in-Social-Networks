#!/usr/bin/env bash
# =============================================================================
# run_phase3.sh — Phase 3 improved ablation runs
#
# Changes vs Phase 2:
#   I1: eps_max 0.25 → 0.10  (N3 less aggressive)
#   I2: lambda_cons 0.5 → 0.20 (N2 less dominant)
#   I3: checkpoint/earlystop on val/f1_2_1 (in train.py)
#   I4: half eps when N1+N3 both active (in loss.py)
#   I5: epochs=75, patience=max(15, epochs//4) (in train.py)
#   I6: soft_gating on combined run
#
# Run after implementing I3/I4/I5 in train.py and loss.py.
# =============================================================================

set -euo pipefail

SEED=42
EPOCHS=75
BATCH_SIZE=32
LR=1e-4
N_BLOCKS=2
EXPANSION=2
DROPOUT=0.2
LANG=${1:-""}

LANG_FLAG=""
if [ -n "$LANG" ]; then
  LANG_FLAG="--language $LANG"
fi

DATE_TAG=$(date +%d%m_%H%M)

echo "============================================================"
echo " EXIST 2021 Phase 3 — Improved Ablation Runs"
echo " Date tag: $DATE_TAG | epochs=$EPOCHS"
echo "============================================================"

# ---------------------------------------------------------------------------
# P3-Run 1 — Baseline (re-run with new monitoring on val/f1_2_1)
# I3+I5 only — gives fair comparison baseline with same checkpoint criterion
# ---------------------------------------------------------------------------
echo -e "\n[P3-1/4] Baseline (new checkpoint criterion)"
/home/maharshi/myenv/bin/python3 train.py \
  --version "${DATE_TAG}_p3_baseline" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.0 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.0 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# P3-Run 2 — N1 only with new epochs (N1 was best single novelty)
# Check if more epochs + F1 monitoring helps N1 further
# ---------------------------------------------------------------------------
echo -e "\n[P3-2/4] N1 only (75 epochs, F1 monitoring)"
/home/maharshi/myenv/bin/python3 train.py \
  --version "${DATE_TAG}_p3_n1only" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.0 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# P3-Run 3 — All three, improved hyperparams (I1+I2+I3+I4+I5)
# eps_max=0.10, lambda_cons=0.20 — core fix
# ---------------------------------------------------------------------------
echo -e "\n[P3-3/4] N1+N2+N3 improved (eps=0.10, lambda=0.20)"
/home/maharshi/myenv/bin/python3 train.py \
  --version "${DATE_TAG}_p3_all_improved" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.2 --eps_max 0.10 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# P3-Run 4 — All three + soft_gating (I1+I2+I3+I4+I5+I6)
# soft_gating dampens Task2 features by p_task1 — reduces N2 spurious violations
# ---------------------------------------------------------------------------
echo -e "\n[P3-4/4] N1+N2+N3 improved + soft_gating"
/home/maharshi/myenv/bin/python3 train.py \
  --version "${DATE_TAG}_p3_all_softgate" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.2 --eps_max 0.10 \
  --soft_gating \
  $LANG_FLAG

echo -e "\n============================================================"
echo " Phase 3 runs complete!"
echo " Results in: outputs/"
echo " Analyze:    python analysis.py"
echo "============================================================"
