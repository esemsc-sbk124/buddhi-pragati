#!/bin/bash

# Simple script to run the master experiment (0)
# Usage: ./run_master_experiment.sh [--dry-run] [--output-dir DIR]

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/buddhi_pragati/experiments"
DRY_RUN=false

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
        *)
            echo "Usage: $0 [--dry-run] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo "🧩 Running Master Experiment (0)"
echo "Output directory: $OUTPUT_DIR"
echo

# Create output directory
if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build command
CMD="python ${PROJECT_ROOT}/run_crossword_benchmark.py evaluate --experiment-mode --experiment-types 0_master --output-dir $OUTPUT_DIR"

echo "Command: $CMD"

if $DRY_RUN; then
    echo "[DRY RUN] Would execute master experiment"
else
    echo "Executing master experiment (this may take several hours)..."
    eval "$CMD"
    echo "✅ Master experiment completed!"
fi

echo "Results will be saved to: $OUTPUT_DIR/experiment_0_master_*.json"