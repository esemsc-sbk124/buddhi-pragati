#!/bin/bash

# Simple script to run all crossword benchmark experiments
# Usage: ./run_all_experiments.sh [--dry-run]

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/buddhi_pragati/experiments"
DRY_RUN=false

# Parse arguments
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - showing commands only"
fi

# Create output directory
if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "🧩 Running Buddhi-Pragati Crossword Benchmark Experiments"
echo "Output directory: $OUTPUT_DIR"
echo

# Function to run or show command
run_cmd() {
    echo "Running: $1"
    if $DRY_RUN; then
        echo "[DRY RUN] $1"
    else
        eval "$1"
    fi
    echo
}

# 1. Master experiment (comprehensive)
echo "📊 Step 1: Master Experiment (all models × all languages × all grid sizes)"
run_cmd "python ${PROJECT_ROOT}/run_crossword_benchmark.py evaluate --experiment-mode --experiment-types 0_master --output-dir $OUTPUT_DIR"

# 2. Focused experiments (parameter variations)
echo "🔬 Step 2: Focused Experiments (parameter variations with priority subsets)"
run_cmd "python ${PROJECT_ROOT}/run_crossword_benchmark.py evaluate --experiment-mode --experiment-types 2_shot_variations 3_batch_sizes 4_chain_of_thought 5_reasoning_effort 6_self_reflection 7_language_variants --output-dir $OUTPUT_DIR"

# 3. Analysis experiments (results analysis)
echo "📈 Step 3: Analysis Experiments (results analysis from master experiment)"
MASTER_RESULTS=$(ls ${OUTPUT_DIR}/experiment_0_master_*.json 2>/dev/null | head -1)
if [[ -n "$MASTER_RESULTS" ]] || $DRY_RUN; then
    if $DRY_RUN; then
        MASTER_RESULTS="${OUTPUT_DIR}/experiment_0_master_*.json"
    fi
    run_cmd "python ${PROJECT_ROOT}/run_crossword_benchmark.py evaluate --experiment-mode --experiment-types 1_language_families 8_model_types 9_reasoning_models 10_performance_normalization --master-results-path $MASTER_RESULTS --output-dir $OUTPUT_DIR"
else
    echo "❌ Master experiment results not found. Cannot run analysis experiments."
    exit 1
fi

echo "✅ All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"