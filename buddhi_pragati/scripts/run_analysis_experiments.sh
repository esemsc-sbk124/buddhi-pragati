#!/bin/bash

# Simple script to run analysis experiments (1, 8-10)
# Usage: ./run_analysis_experiments.sh [--dry-run] [--output-dir DIR] [--master-results PATH]

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/buddhi_pragati/experiments"
DRY_RUN=false
MASTER_RESULTS=""
EXPERIMENTS="1_language_families 8_model_types 9_reasoning_models 10_performance_normalization"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            echo "DRY RUN MODE - showing commands only"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --master-results)
            MASTER_RESULTS="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--output-dir DIR] [--master-results PATH]"
            exit 1
            ;;
    esac
done

echo "📈 Running Analysis Experiments (1, 8-10)"
echo "Output directory: $OUTPUT_DIR"

# Find master results if not provided
if [[ -z "$MASTER_RESULTS" ]]; then
    MASTER_RESULTS=$(ls ${OUTPUT_DIR}/experiment_0_master_*.json 2>/dev/null | head -1)
    if [[ -z "$MASTER_RESULTS" ]]; then
        echo "❌ Error: Master experiment results not found"
        echo "Run master experiment first or specify --master-results PATH"
        exit 1
    fi
fi

echo "Master results: $MASTER_RESULTS"
echo "Analysis experiments: $EXPERIMENTS"
echo

# Create output directory
if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build command
CMD="python ${PROJECT_ROOT}/run_crossword_benchmark.py evaluate --experiment-mode --experiment-types $EXPERIMENTS --master-results-path $MASTER_RESULTS --output-dir $OUTPUT_DIR"

echo "Command: $CMD"

if $DRY_RUN; then
    echo "[DRY RUN] Would execute analysis experiments"
else
    echo "Executing analysis experiments..."
    eval "$CMD"
    echo "✅ Analysis experiments completed!"
fi

echo "Results will be saved to: $OUTPUT_DIR/experiment_*_analysis_*.json"