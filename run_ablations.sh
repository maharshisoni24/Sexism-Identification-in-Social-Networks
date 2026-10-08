#!/usr/bin/env bash
# =============================================================================
# run_ablations.sh — 7-run ablation study for EXIST 2021 Novelty project
#
# Usage:
#   chmod +x run_ablations.sh
#   ./run_ablations.sh [--language en|es]
#
# Each run uses a different combination of N1/N2/N3 novelties.
# Results are saved to outputs/<version>/ and checkpoints/<version>/
# TensorBoard logs go to tb_logs/<version>/
#
# Prerequisites (on training machine):
#   1. Generate embeddings:
#        python generate_embeddings.py --backend gemini --api_key YOUR_KEY
#      OR use local fallback:
#        python generate_embeddings.py --backend local
#   2. Install dependencies:
#        pip install -r requirements.txt
# =============================================================================

set -euo pipefail

SEED=42
EPOCHS=50
BATCH_SIZE=32
LR=1e-4
N_BLOCKS=2
EXPANSION=2
DROPOUT=0.2
LANG=${1:-""}  # pass "en" or "es" to filter language; empty = both

LANG_FLAG=""
if [ -n "$LANG" ]; then
  LANG_FLAG="--language $LANG"
fi

DATE_TAG=$(date +%d%m_%H%M)

echo "============================================================"
echo " EXIST 2021 Ablation Study — 7 runs"
echo " Date tag: $DATE_TAG"
echo " Language: ${LANG:-'all (en+es)'}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run 1 — Baseline (no novelties)
# ---------------------------------------------------------------------------
echo -e "\n[1/7] Baseline (N1=off, N2=off, N3=off)"
python train.py \
  --version "${DATE_TAG}_baseline" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.0 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.0 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# Run 2 — N1 only (Confidence-Aware Focal Loss)
# ---------------------------------------------------------------------------
echo -e "\n[2/7] N1 only (focal_alpha=0.5, focal_gamma=2.0)"
python train.py \
  --version "${DATE_TAG}_n1only" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.0 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# Run 3 — N2 only (Hierarchical Consistency Regularization)
# ---------------------------------------------------------------------------
echo -e "\n[3/7] N2 only (lambda_cons=0.5)"
python train.py \
  --version "${DATE_TAG}_n2only" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.0 --focal_gamma 2.0 --lambda_cons 0.5 --eps_max 0.0 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# Run 4 — N3 only (Adaptive Label Smoothing)
# ---------------------------------------------------------------------------
echo -e "\n[4/7] N3 only (eps_max=0.25)"
python train.py \
  --version "${DATE_TAG}_n3only" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.0 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.25 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# Run 5 — N1 + N2
# ---------------------------------------------------------------------------
echo -e "\n[5/7] N1 + N2"
python train.py \
  --version "${DATE_TAG}_n1n2" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.5 --eps_max 0.0 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# Run 6 — N1 + N3
# ---------------------------------------------------------------------------
echo -e "\n[6/7] N1 + N3"
python train.py \
  --version "${DATE_TAG}_n1n3" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.25 \
  $LANG_FLAG

# ---------------------------------------------------------------------------
# Run 7 — All three combined (full proposed system)
# ---------------------------------------------------------------------------
echo -e "\n[7/7] All three combined (N1+N2+N3)"
python train.py \
  --version "${DATE_TAG}_all" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.5 --eps_max 0.25 \
  $LANG_FLAG

echo -e "\n============================================================"
echo " All 7 ablation runs complete!"
echo " Results in: outputs/"
echo " TensorBoard:  tensorboard --logdir tb_logs/"
echo "============================================================"
