#!/bin/bash

# Simple script to run focused experiments (2-7)
# Usage: ./run_focused_experiments.sh [--dry-run] [--output-dir DIR] [--experiments EXP1,EXP2,...]

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/buddhi_pragati/experiments"
DRY_RUN=false
EXPERIMENTS="2_shot_variations 3_batch_sizes 4_chain_of_thought 5_reasoning_effort 6_self_reflection 7_language_variants"

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
        --experiments)
            EXPERIMENTS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--output-dir DIR] [--experiments EXP1,EXP2,...]"
            echo "Available experiments: 2_shot_variations,3_batch_sizes,4_chain_of_thought,5_reasoning_effort,6_self_reflection,7_language_variants"
            exit 1
            ;;
    esac
done

echo "🔬 Running Focused Experiments (2-7)"
echo "Output directory: $OUTPUT_DIR"
echo "Experiments: $EXPERIMENTS"
echo

# Create output directory
if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build command
CMD="python ${PROJECT_ROOT}/run_crossword_benchmark.py evaluate --experiment-mode --experiment-types $EXPERIMENTS --output-dir $OUTPUT_DIR"

echo "Command: $CMD"

if $DRY_RUN; then
    echo "[DRY RUN] Would execute focused experiments"
else
    echo "Executing focused experiments..."
    eval "$CMD"
    echo "✅ Focused experiments completed!"
fi

echo "Results will be saved to: $OUTPUT_DIR/experiment_*_*.json"