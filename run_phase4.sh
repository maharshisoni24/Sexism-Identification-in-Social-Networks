#!/usr/bin/env bash
# run_phase4.sh — 2 targeted runs based on Phase 3 findings

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
if [ -n "$LANG" ]; then LANG_FLAG="--language $LANG"; fi

DATE_TAG=$(date +%d%m_%H%M)

echo "============================================================"
echo " EXIST 2021 Phase 4 — Targeted Runs"
echo " Date tag: $DATE_TAG | epochs=$EPOCHS"
echo "============================================================"

# P4-Run 1: N1 + N3 improved (no N2, no soft_gating)
# N1+N3 was 0.7129 Task1 in Phase 2. With F1 monitoring + eps=0.10 + I4 halving → should be better.
echo -e "\n[P4-1/2] N1+N3 improved (eps=0.10, F1 monitor)"
/home/maharshi/myenv/bin/python3 train.py \
  --version "${DATE_TAG}_p4_n1n3_improved" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.10 \
  $LANG_FLAG

# P4-Run 2: N1 + soft_gating (no N2, no N3)
# soft_gating alone with N1 gave best Task1 (0.7278) but N2 killed Task2.3.
# Test: drop N2, keep soft_gating — does Task2.3 recover while Task1 stays high?
echo -e "\n[P4-2/2] N1 + soft_gating (no N2, no N3)"
/home/maharshi/myenv/bin/python3 train.py \
  --version "${DATE_TAG}_p4_n1_softgate" \
  --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  --n_blocks $N_BLOCKS --expansion_factor $EXPANSION --dropout $DROPOUT \
  --focal_alpha 0.5 --focal_gamma 2.0 --lambda_cons 0.0 --eps_max 0.0 \
  --soft_gating \
  $LANG_FLAG

echo -e "\n============================================================"
echo " Phase 4 runs complete! Analyze: python analysis.py"
echo "============================================================"
