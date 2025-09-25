#!/bin/bash

# Simple environment validation script
# Usage: ./setup_environment.sh [--check]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHECK_ONLY=false

if [[ "$1" == "--check" ]]; then
    CHECK_ONLY=true
fi

echo "🔧 Environment Setup and Validation"
echo "Project root: $PROJECT_ROOT"
echo

# Check Python
echo "Checking Python..."
if ! command -v python >/dev/null 2>&1; then
    echo "❌ Python not found"
    exit 1
fi
echo "✅ Python: $(python --version)"

# Check buddhi_pragati package
echo "Checking buddhi_pragati package..."
if ! python -c "import buddhi_pragati" 2>/dev/null; then
    echo "❌ buddhi_pragati package not available"
    echo "Install with: cd $PROJECT_ROOT && pip install -e ."
    exit 1
fi
echo "✅ buddhi_pragati package available"

# Check CLI
echo "Checking CLI script..."
if [[ ! -f "${PROJECT_ROOT}/run_crossword_benchmark.py" ]]; then
    echo "❌ run_crossword_benchmark.py not found"
    exit 1
fi
echo "✅ CLI script found"

# Check configuration
echo "Checking configuration..."
if [[ ! -f "${PROJECT_ROOT}/crossword_config.txt" ]]; then
    echo "❌ crossword_config.txt not found"
    exit 1
fi
echo "✅ Configuration file found"

# Create directories
if ! $CHECK_ONLY; then
    echo "Creating directories..."
    mkdir -p "${PROJECT_ROOT}/buddhi_pragati/experiments"
    mkdir -p "${PROJECT_ROOT}/logs"
    echo "✅ Directories created"
fi

echo "✅ Environment validation completed"