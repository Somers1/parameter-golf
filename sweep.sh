#!/bin/bash
# Sweep script — tests one variable at a time from a known baseline config.
# Run on 1xH100 or Mac. Each run is 10 minutes (MAX_WALLCLOCK_SECONDS=600).
# Results go to logs/ with auto-generated run IDs.

set -e

# Base config — conservative starting point, close to baseline shape
export VOCAB_SIZE=1024
export MLP_WINDOW=1024
export MLP_OVERLAP=0.2
export MODEL_DIM=512
export MTP_HEADS=0
export Q_LATENT=128
export KV_LATENT=64
export GATE_RANK=32
export NUM_LAYERS=9
export ATTEND_EVERY=1
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export TIED_EMBED_LR=0.05
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=600

# Detect GPU count
if command -v nvidia-smi &> /dev/null; then
    NGPU=$(nvidia-smi -L | wc -l)
    RUN_CMD="torchrun --standalone --nproc_per_node=$NGPU train_gpt_ours.py"
    echo "=== Running on ${NGPU}x GPU ==="
else
    RUN_CMD="python3 train_gpt_mlx.py"
    echo "=== Running on Mac (MLX) ==="
fi

run_test() {
    local name=$1
    shift
    echo ""
    echo "=========================================="
    echo "TEST: $name"
    echo "=========================================="
    # Apply overrides
    for arg in "$@"; do
        export "$arg"
        echo "  $arg"
    done
    $RUN_CMD
    # Reset overrides back to base
    export MLP_WINDOW=1024
    export MLP_OVERLAP=0.2
    export MODEL_DIM=512
    export MTP_HEADS=0
    export Q_LATENT=128
    export KV_LATENT=64
    export GATE_RANK=32
    export NUM_LAYERS=9
    export ATTEND_EVERY=1
    export MATRIX_LR=0.04
    export SCALAR_LR=0.04
}

# ==========================================
# TEST 1: Baseline shape with our architecture
# This is the control — closest to original baseline
# ==========================================
run_test "1_baseline_shape"

# ==========================================
# TEST 2: Remove gate — is it helping or hurting?
# ==========================================
run_test "2_no_gate" GATE_RANK=0

# ==========================================
# TEST 3: Lower learning rate — shared MLP gets 9x gradients
# ==========================================
run_test "3_lower_lr" MATRIX_LR=0.02

# ==========================================
# TEST 4: Increase overlap — more weight sharing
# ==========================================
run_test "4_more_overlap" MLP_OVERLAP=0.5

# ==========================================
# TEST 5: Wider MLP window
# ==========================================
run_test "5_wider_mlp" MLP_WINDOW=2048

# ==========================================
# TEST 6: More layers, skip attention
# ==========================================
run_test "6_deep_sparse_attn" NUM_LAYERS=16 ATTEND_EVERY=2

# ==========================================
# TEST 7: Best combo from above (edit after reviewing results)
# ==========================================
# run_test "7_best_combo" MLP_WINDOW=1024 MLP_OVERLAP=0.5 MATRIX_LR=0.02 NUM_LAYERS=16 ATTEND_EVERY=2

echo ""
echo "=========================================="
echo "SWEEP COMPLETE — check logs/ for results"
echo "=========================================="
echo "Compare val_bpb across runs:"
echo "  grep 'val_bpb' logs/*.txt | tail -20"
