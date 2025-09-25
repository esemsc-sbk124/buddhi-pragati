#!/bin/bash

# ==========================================
#    BUDDHI-PRAGATI PUZZLE GENERATION SCRIPT
# ==========================================
#
# Generates crossword puzzles for all supported languages
# and grid sizes, with cache clearing after each language
# to prevent excessive disk usage.
#
# Configuration:
# - Languages: All 19 supported languages
# - Grid sizes: 7, 10, 15, 20, 25
# - Upload: Enabled
#
# Usage: bash run_generate_puzzles.sh
# ==========================================

set -e  # Exit on any error

echo "🚀 BUDDHI-PRAGATI PUZZLE GENERATION"
echo "==================================="
echo "Languages: 19 supported languages"
echo "Grid sizes: 7, 10, 15, 20, 25, 30"
echo "Mode: Upload to HuggingFace (no local storage)"
echo ""

# List of languages
LANGUAGES=(
    "Assamese"
    "Bengali" 
    "Bodo"
    "English"
    "Gujarati"
    "Hindi"
    "Kannada"
    "Kashmiri"
    "Konkani"
    "Malayalam"
    "Marathi"
    "Meitei"
    "Nepali"
    "Odia"
    "Punjabi"
    "Sanskrit"
    "Tamil"
    "Telugu"
    "Urdu"
)

# List of grid sizes
GRID_SIZES=(7 10 15 20 25)

# Track progress
TOTAL_LANGUAGES=${#LANGUAGES[@]}
CURRENT=1

# Process each language
for language in "${LANGUAGES[@]}"; do
    echo "📚 [$CURRENT/$TOTAL_LANGUAGES] Processing $language..."
    echo "   Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Iterate over each grid size
    for grid_size in "${GRID_SIZES[@]}"; do
        echo "   🔄 Running for grid size: $grid_size"
        python ../../run_crossword_benchmark.py generate --language "$language" --grid-sizes "$grid_size" --count 50 --upload
    done

    # Clear cache after processing the language
    echo "   🧹 Clearing cache for $language..."
    rm -rf ~/.cache/huggingface/hub/datasets--selim-b-kh___buddhi_pragati-puzzles* 2>/dev/null || true
    rm -rf ~/.cache/huggingface/datasets/*selim-b-kh___buddhi-pragati-puzzles* 2>/dev/null || true

    echo "   ✅ Finished processing $language at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    ((CURRENT++))
done

echo "🎉 PUZZLE GENERATION COMPLETE!"
echo "==================================="
echo "✅ Processed $TOTAL_LANGUAGES languages"
echo "🌐 All puzzles uploaded to HuggingFace Hub"
echo "🧹 Cache management: Cleared after each language to prevent massive disk usage"
echo ""