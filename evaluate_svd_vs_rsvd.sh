#!/bin/bash

export HF_HOME=/projectnb/manoslab/cgokmen/hf_cache
export TRANSFORMERS_CACHE=/projectnb/manoslab/cgokmen/hf_cache
export HF_DATASETS_CACHE=/projectnb/manoslab/cgokmen/hf_cache

# ============================================================================
# Model Evaluation Benchmark Script
# ============================================================================

set -e  # Exit on error

# Configuration (must match compression script)
MODEL="jeffwan/llama-7b-hf"
RATIO=0.1 # Can be overridden: RATIO=0.5 ./evaluate_svd_vs_rsvd.sh
DATASET="wikitext2"
SEED=42
SEQ_LEN=2048
EVAL_BATCH_SIZE=1
GEN_SEQ_LEN=128

# Calculate the kept ratio for filename
KEPT_RATIO=$(printf "%.1f" $(echo "1 - $RATIO" | bc -l))

# Use ratio-specific directory
SAVE_DIR="./benchmark_results_comparison/ratio_${RATIO}"
RSVD_SAVE_DIR="$SAVE_DIR/rsvd"
SVD_SAVE_DIR="$SAVE_DIR/svd"

# Device
DEVICE="${DEVICE:-cuda}"

# Check if compressed models exist
if [ ! -d "$SAVE_DIR" ]; then
    echo "Error: Directory $SAVE_DIR does not exist!"
    echo "Please run compression first with: RATIO=$RATIO ./compress_svd_vs_rsvd.sh"
    exit 1
fi

RSVD_MODEL_PATH="$RSVD_SAVE_DIR/jeffwan_llama_7b_hf_whitening_only_${KEPT_RATIO}.pt"
SVD_MODEL_PATH="$SVD_SAVE_DIR/jeffwan_llama_7b_hf_whitening_only_${KEPT_RATIO}.pt"

if [ ! -f "$RSVD_MODEL_PATH" ]; then
    echo "Error: RSVD model not found: $RSVD_MODEL_PATH"
    echo "Please compress first with: RATIO=$RATIO ./compress_svd_vs_rsvd.sh"
    exit 1
fi

if [ ! -f "$SVD_MODEL_PATH" ]; then
    echo "Error: SVD model not found: $SVD_MODEL_PATH"
    echo "Please compress first with: RATIO=$RATIO ./compress_svd_vs_rsvd.sh"
    exit 1
fi

echo "============================================================================"
echo "Model Quality Evaluation (LOADING EXISTING COMPRESSED MODELS)"
echo "============================================================================"
echo "Compression Ratio: $RATIO (keeping ${KEPT_RATIO} or $(echo "scale=1; (1 - $RATIO) * 100" | bc)%)"
echo "Evaluating models from: $SAVE_DIR"
echo ""
echo "Evaluating:"
echo "  1. Original uncompressed model"
echo "  2. RSVD compressed model: $RSVD_MODEL_PATH"
echo "  3. SVD compressed model: $SVD_MODEL_PATH"
echo ""
echo "Metrics:"
echo "  - Step 4: Perplexity on $DATASET (lower is better)"
echo "  - Step 5: Generation efficiency (tokens/sec, higher is better)"
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Evaluate Original Model (Baseline) - PERPLEXITY
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 1: Evaluating Original Model - Perplexity (Step 4)..."
echo "--------------------------------------------------------------------"

START_ORIG=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 4 \
    --model_path "original" \
    --model_seq_len $SEQ_LEN \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/original_perplexity.log

END_ORIG=$(date +%s)
ORIG_PPL_TIME=$((END_ORIG - START_ORIG))

# Extract perplexity from log
ORIG_PPL=$(grep -i "perplexity" $SAVE_DIR/original_perplexity.log | tail -1)

echo ""
echo "✓ Original model perplexity evaluation completed in $ORIG_PPL_TIME seconds"
echo "  $ORIG_PPL"
echo ""

# ============================================================================
# Step 2: Evaluate Original Model - EFFICIENCY
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 2: Evaluating Original Model - Efficiency (Step 5)..."
echo "--------------------------------------------------------------------"

START_ORIG_EFF=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 5 \
    --model_path "original" \
    --gen_seq_len $GEN_SEQ_LEN \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/original_efficiency.log

END_ORIG_EFF=$(date +%s)
ORIG_EFF_TIME=$((END_ORIG_EFF - START_ORIG_EFF))

# Extract efficiency metrics
ORIG_EFF=$(grep -i "tokens/sec\|throughput" $SAVE_DIR/original_efficiency.log | tail -1)

echo ""
echo "✓ Original model efficiency evaluation completed in $ORIG_EFF_TIME seconds"
echo "  $ORIG_EFF"
echo ""

# ============================================================================
# Step 3: Evaluate RSVD Compressed Model - PERPLEXITY
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 3: Evaluating RSVD Compressed Model - Perplexity (Step 4)..."
echo "--------------------------------------------------------------------"

START_RSVD=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 4 \
    --model_path "$RSVD_MODEL_PATH" \
    --model_seq_len $SEQ_LEN \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/rsvd_perplexity.log

END_RSVD=$(date +%s)
RSVD_PPL_TIME=$((END_RSVD - START_RSVD))

# Extract perplexity
RSVD_PPL=$(grep -i "perplexity" $SAVE_DIR/rsvd_perplexity.log | tail -1)

echo ""
echo "✓ RSVD perplexity evaluation completed in $RSVD_PPL_TIME seconds"
echo "  $RSVD_PPL"
echo ""

# ============================================================================
# Step 4: Evaluate RSVD Compressed Model - EFFICIENCY
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 4: Evaluating RSVD Compressed Model - Efficiency (Step 5)..."
echo "--------------------------------------------------------------------"

START_RSVD_EFF=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 5 \
    --model_path "$RSVD_MODEL_PATH" \
    --gen_seq_len $GEN_SEQ_LEN \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/rsvd_efficiency.log

END_RSVD_EFF=$(date +%s)
RSVD_EFF_TIME=$((END_RSVD_EFF - START_RSVD_EFF))

# Extract efficiency metrics
RSVD_EFF=$(grep -i "tokens/sec\|throughput" $SAVE_DIR/rsvd_efficiency.log | tail -1)

echo ""
echo "✓ RSVD efficiency evaluation completed in $RSVD_EFF_TIME seconds"
echo "  $RSVD_EFF"
echo ""

# ============================================================================
# Step 5: Evaluate SVD Compressed Model - PERPLEXITY
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 5: Evaluating SVD Compressed Model - Perplexity (Step 4)..."
echo "--------------------------------------------------------------------"

START_SVD=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 4 \
    --model_path "$SVD_MODEL_PATH" \
    --model_seq_len $SEQ_LEN \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/svd_perplexity.log

END_SVD=$(date +%s)
SVD_PPL_TIME=$((END_SVD - START_SVD))

# Extract perplexity
SVD_PPL=$(grep -i "perplexity" $SAVE_DIR/svd_perplexity.log | tail -1)

echo ""
echo "✓ SVD perplexity evaluation completed in $SVD_PPL_TIME seconds"
echo "  $SVD_PPL"
echo ""

# ============================================================================
# Step 6: Evaluate SVD Compressed Model - EFFICIENCY
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 6: Evaluating SVD Compressed Model - Efficiency (Step 5)..."
echo "--------------------------------------------------------------------"

START_SVD_EFF=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 5 \
    --model_path "$SVD_MODEL_PATH" \
    --gen_seq_len $GEN_SEQ_LEN \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/svd_efficiency.log

END_SVD_EFF=$(date +%s)
SVD_EFF_TIME=$((END_SVD_EFF - START_SVD_EFF))

# Extract efficiency metrics
SVD_EFF=$(grep -i "tokens/sec\|throughput" $SAVE_DIR/svd_efficiency.log | tail -1)

echo ""
echo "✓ SVD efficiency evaluation completed in $SVD_EFF_TIME seconds"
echo "  $SVD_EFF"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "Evaluation Complete!"
echo "============================================================================"
echo ""
echo "PERPLEXITY Results (lower is better):"
echo "  Original: $ORIG_PPL"
echo "  RSVD:     $RSVD_PPL"
echo "  SVD:      $SVD_PPL"
echo ""
echo "EFFICIENCY Results (higher is better):"
echo "  Original: $ORIG_EFF"
echo "  RSVD:     $RSVD_EFF"
echo "  SVD:      $SVD_EFF"
echo ""
echo "Evaluation Time:"
echo "  Original (PPL + EFF): $((ORIG_PPL_TIME + ORIG_EFF_TIME)) seconds"
echo "  RSVD (PPL + EFF):     $((RSVD_PPL_TIME + RSVD_EFF_TIME)) seconds"
echo "  SVD (PPL + EFF):      $((SVD_PPL_TIME + SVD_EFF_TIME)) seconds"
echo ""
echo "All evaluation logs saved in: $SAVE_DIR"
echo "============================================================================"

cat > $SAVE_DIR/evaluation_summary.txt << EOF
Evaluation Results for Compression Ratio $RATIO
================================================

PERPLEXITY (lower is better):
- Original: $ORIG_PPL
- RSVD:     $RSVD_PPL
- SVD:      $SVD_PPL

EFFICIENCY (higher is better):
- Original: $ORIG_EFF
- RSVD:     $RSVD_EFF
- SVD:      $SVD_EFF

Evaluation Time:
- Original (Perplexity):  $ORIG_PPL_TIME seconds
- Original (Efficiency):  $ORIG_EFF_TIME seconds
- RSVD (Perplexity):      $RSVD_PPL_TIME seconds
- RSVD (Efficiency):      $RSVD_EFF_TIME seconds
- SVD (Perplexity):       $SVD_PPL_TIME seconds
- SVD (Efficiency):       $SVD_EFF_TIME seconds

Log Files:
- Original: original_perplexity.log, original_efficiency.log
- RSVD:     rsvd_perplexity.log, rsvd_efficiency.log
- SVD:      svd_perplexity.log, svd_efficiency.log
EOF

echo "Summary saved to: $SAVE_DIR/evaluation_summary.txt"