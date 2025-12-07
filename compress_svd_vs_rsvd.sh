#!/bin/bash

export HF_HOME=/projectnb/manoslab/cgokmen/hf_cache
export TRANSFORMERS_CACHE=/projectnb/manoslab/cgokmen/hf_cache
export HF_DATASETS_CACHE=/projectnb/manoslab/cgokmen/hf_cache

# ============================================================================
# SVD vs RSVD Compression Benchmark Script
# ============================================================================

set -e  # Exit on error

# Check Python dependencies
echo "Checking Python dependencies..."
python -c "import sentencepiece" 2>/dev/null || {
    echo "Error: sentencepiece not installed. Installing..."
    pip install sentencepiece
}
python -c "import torch" 2>/dev/null || {
    echo "Error: torch not installed. Please install: pip install -r requirements.txt"
    exit 1
}
python -c "import transformers" 2>/dev/null || {
    echo "Error: transformers not installed. Please install: pip install -r requirements.txt"
    exit 1
}
echo "✓ All dependencies installed"
echo ""

# Configuration
MODEL="jeffwan/llama-7b-hf"
RATIO=0.1  
DATASET="wikitext2"
WHITENING_SAMPLES=256
SEED=42
SEQ_LEN=2048

# Calculate the kept ratio for filename
KEPT_RATIO=$(printf "%.1f" $(echo "1 - $RATIO" | bc -l))

# Save to benchmark_results_comparison/ratio_<compression_ratio>
SAVE_DIR="./benchmark_results_comparison/ratio_${RATIO}"
RSVD_SAVE_DIR="$SAVE_DIR/rsvd"
SVD_SAVE_DIR="$SAVE_DIR/svd"

# RSVD Parameters
RSVD_OVERSAMPLES=10
RSVD_ITER=2

# Device (cuda for GPU, cpu for CPU)
DEVICE="${DEVICE:-cuda}"

# Create output directories
mkdir -p $SAVE_DIR
mkdir -p $RSVD_SAVE_DIR
mkdir -p $SVD_SAVE_DIR

echo "============================================================================"
echo "SVD vs RSVD Compression Benchmark (Both with Whitening)"
echo "============================================================================"
echo "Model: $MODEL"
echo "Compression Ratio: $RATIO (keeping ${KEPT_RATIO} or $(echo "scale=1; (1 - $RATIO) * 100" | bc)% of parameters)"
echo "Dataset: $DATASET"
echo "Whitening Samples: $WHITENING_SAMPLES"
echo "RSVD Oversamples: $RSVD_OVERSAMPLES"
echo "RSVD Iterations: $RSVD_ITER"
echo "Device: $DEVICE"
echo "Save Directory: $SAVE_DIR"
echo ""
echo "Tests (in order):"
echo "  1. RSVD (with whitening)"
echo "  2. SVD (with whitening)"
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Compress with RSVD (WITH WHITENING)
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 1: Compressing with RSVD (with whitening)..."
echo "--------------------------------------------------------------------"

START_RSVD=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 1 \
    --ratio $RATIO \
    --dataset $DATASET \
    --seed $SEED \
    --model_seq_len $SEQ_LEN \
    --save_path $RSVD_SAVE_DIR \
    --rsvd_oversamples $RSVD_OVERSAMPLES \
    --rsvd_n_iter $RSVD_ITER \
    --whitening_nsamples $WHITENING_SAMPLES \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/rsvd_compression.log

END_RSVD=$(date +%s)
RSVD_TIME=$((END_RSVD - START_RSVD))

# Check if compression succeeded - use dynamic filename
RSVD_MODEL_PATH="$RSVD_SAVE_DIR/jeffwan_llama_7b_hf_whitening_only_${KEPT_RATIO}.pt"
if [ ! -f "$RSVD_MODEL_PATH" ]; then
    echo ""
    echo "✗ RSVD compression FAILED in $RSVD_TIME seconds"
    echo "  Model file not created: $RSVD_MODEL_PATH"
    echo "  Check the log above for errors"
    echo ""
    exit 1
fi

echo ""
echo "✓ RSVD compression completed in $RSVD_TIME seconds"
echo "  Model saved: $RSVD_MODEL_PATH"
echo ""

# ============================================================================
# Step 2: Compress with Standard SVD (WITH WHITENING)
# ============================================================================
echo "--------------------------------------------------------------------"
echo "Step 2: Compressing with Standard SVD (with whitening)..."
echo "--------------------------------------------------------------------"

START_SVD=$(date +%s)

python SVDLLM.py \
    --model $MODEL \
    --step 1 \
    --ratio $RATIO \
    --dataset $DATASET \
    --seed $SEED \
    --model_seq_len $SEQ_LEN \
    --save_path $SVD_SAVE_DIR \
    --use_standard_svd \
    --whitening_nsamples $WHITENING_SAMPLES \
    --DEV $DEVICE \
    2>&1 | tee $SAVE_DIR/svd_compression.log

END_SVD=$(date +%s)
SVD_TIME=$((END_SVD - START_SVD))

# Check if compression succeeded - use dynamic filename
SVD_MODEL_PATH="$SVD_SAVE_DIR/jeffwan_llama_7b_hf_whitening_only_${KEPT_RATIO}.pt"
if [ ! -f "$SVD_MODEL_PATH" ]; then
    echo ""
    echo "✗ SVD compression FAILED in $SVD_TIME seconds"
    echo "  Model file not created: $SVD_MODEL_PATH"
    echo "  Check the log above for errors"
    echo ""
    exit 1
fi

echo ""
echo "✓ SVD compression completed in $SVD_TIME seconds"
echo "  Model saved: $SVD_MODEL_PATH"
echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo "============================================================================"
echo "Compression Benchmark Complete!"
echo "============================================================================"
echo ""
echo "Compression Ratio: $RATIO"
echo "Parameters Kept: ${KEPT_RATIO} ($(echo "scale=1; (1 - $RATIO) * 100" | bc)%)"
echo ""
echo "Compression Timing:"
echo "  1. RSVD (with whitening):       $RSVD_TIME seconds"
echo "  2. SVD (with whitening):        $SVD_TIME seconds"
echo ""
echo "Speedup Analysis:"
echo "  RSVD vs SVD:                    $(echo "scale=2; $SVD_TIME / $RSVD_TIME" | bc)x speedup"
echo ""
echo "Compressed Models:"
echo "  RSVD:                      $RSVD_MODEL_PATH"
echo "  SVD:                       $SVD_MODEL_PATH"
echo ""
echo "All logs saved in: $SAVE_DIR"
echo ""
echo "To evaluate model quality for this ratio, run:"
echo "  RATIO=$RATIO ./evaluate_svd_vs_rsvd.sh"
echo "============================================================================"

# Save compression summary to file
cat > $SAVE_DIR/compression_summary.txt << EOF
SVD vs RSVD Compression Benchmark Results
==========================================

Configuration:
- Model: $MODEL
- Compression Ratio: $RATIO (keeping ${KEPT_RATIO} or $(echo "scale=1; (1 - $RATIO) * 100" | bc)% of parameters)
- Dataset: $DATASET
- Whitening Samples: $WHITENING_SAMPLES (both methods)

Compression Timing:
1. RSVD (with whitening):       $RSVD_TIME seconds
2. SVD (with whitening):        $SVD_TIME seconds

Speedup Analysis:
- RSVD vs SVD:                  $(echo "scale=2; $SVD_TIME / $RSVD_TIME" | bc)x speedup

Compressed Models:
- RSVD:                      $RSVD_MODEL_PATH
- SVD:                       $SVD_MODEL_PATH

Next Step:
Run RATIO=$RATIO ./evaluate_svd_vs_rsvd.sh to evaluate model quality (perplexity)
EOF

echo ""
echo "Compression summary saved to: $SAVE_DIR/compression_summary.txt"