#!/bin/bash

# Simple results validation utility
# Usage: ./check_results.sh [RESULTS_DIR]

RESULTS_DIR="${1:-../../experiments}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/buddhi_pragati/experiments"

echo "📋 Experiment Results Validation"
echo "Results directory: $RESULTS_DIR"
echo

# Check if results directory exists
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "❌ Results directory does not exist: $RESULTS_DIR"
    exit 1
fi

# Expected experiments
EXPECTED_EXPERIMENTS=(
    "0_master"
    "1_language_families"
    "2_shot_variations"
    "3_batch_sizes"
    "4_chain_of_thought"
    "5_reasoning_effort"
    "6_self_reflection"
    "7_language_variants"
    "8_model_types"
    "9_reasoning_models"
    "10_performance_normalization"
)

echo "Checking for experiment results..."
found=0
missing=()

for exp in "${EXPECTED_EXPERIMENTS[@]}"; do
    result_files=(${RESULTS_DIR}/experiment_${exp}_*.json)
    if [[ -f "${result_files[0]}" ]]; then
        echo "✅ $exp: $(basename "${result_files[0]}")"
        ((found++))
    else
        echo "❌ $exp: Not found"
        missing+=("$exp")
    fi
done

echo
echo "Summary:"
echo "  Found: $found / ${#EXPECTED_EXPERIMENTS[@]} experiments"
echo "  Missing: ${#missing[@]} experiments"

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "  Missing experiments: ${missing[*]}"
fi

echo
echo "Total result files: $(ls -1 ${RESULTS_DIR}/*.json 2>/dev/null | wc -l)"

# Show disk usage
if command -v du >/dev/null 2>&1; then
    echo "Disk usage: $(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1)"
fi