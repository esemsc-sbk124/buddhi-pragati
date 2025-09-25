#!/bin/bash

# ==========================================
#    BUDDHI-PRAGATI COMPREHENSIVE DATASET 
#         CREATION SCRIPT
# ==========================================
#
# Generates 5000-entry datasets for all 20 supported languages
# using all 4 available sources with optimized batch processing
#
# Configuration:
# - Target size: 5000 entries per language
# - Batch size: 500 (IndicWikiBio auto-capped at 100)
# - Sources: All 4 (MILU, IndicWikiBio, IndoWordNet, Bhasha-Wiki)
# - Upload: Direct to HuggingFace (no local storage)
# - Total expected entries: 100,000 across all languages
#
# Usage: bash buddhi_pragati/scripts/run_dataset_creation.sh
# ==========================================

set -e  # Exit on any error

echo "🚀 BUDDHI-PRAGATI COMPREHENSIVE DATASET CREATION"
echo "=================================================="
echo "Target: 5000 entries × 10 languages = 95,000 total entries"
echo "Batch size: 500 (IndicWikiBio capped at 100)"
echo "Sources: All 4 (MILU, IndicWikiBio, IndoWordNet, Bhasha-Wiki)"
echo "Mode: Upload to HuggingFace (no local storage)"
echo ""

echo "🧹 CACHE MANAGEMENT"
echo "=================="
echo "Clearing HuggingFace cache to prevent massive disk usage..."
rm -rf ~/.cache/huggingface/hub/datasets--selim-b-kh--buddhi_pragati* 2>/dev/null || true
rm -rf ~/.cache/huggingface/datasets/*buddhi_pragati* 2>/dev/null || true
echo "✅ Cache cleared"
echo ""

# List of all 19 supported languages
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

# Configuration parameters
TARGET_SIZE=6000
SOURCES="MILU IndicWikiBio IndoWordNet Bhasha-Wiki"

echo "📋 CONFIGURATION:"
echo "   Languages: ${#LANGUAGES[@]} languages"
echo "   Target size per language: ${TARGET_SIZE}"
echo "   Sources: ${SOURCES}"
echo "   Upload mode: Enabled"
echo ""

# Track progress
TOTAL_LANGUAGES=${#LANGUAGES[@]}
CURRENT=1

# Process each language
for language in "${LANGUAGES[@]}"; do
    echo "📚 [$CURRENT/$TOTAL_LANGUAGES] Processing $language..."
    echo "   Expected entries: $TARGET_SIZE"
    echo "   Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run dataset creation for this language
    python ../../run_crossword_benchmark.py create-dataset \
        --languages "$language" \
        --target-size $TARGET_SIZE \
        --sources $SOURCES \
        --upload
     
    if [ $? -eq 0 ]; then
        echo "   ✅ $language completed successfully"
    else
        echo "   ❌ $language failed - continuing with next language"
    fi
    
    # Clear cache after each language to prevent massive accumulation
    echo "   🧹 Clearing cache for $language..."
    rm -rf ~/.cache/huggingface/hub/datasets--selim-b-kh--buddhi_pragati* 2>/dev/null || true
    rm -rf ~/.cache/huggingface/datasets/*buddhi_pragati* 2>/dev/null || true
    
    echo "   Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    ((CURRENT++))
done

echo "🎉 DATASET CREATION COMPLETE!"
echo "=============================================="
echo "✅ Processed $TOTAL_LANGUAGES languages"
echo "📊 Expected total entries: $((TARGET_SIZE * TOTAL_LANGUAGES))"
echo "🌐 All datasets uploaded to HuggingFace Hub"
echo "🗄️  Repository: selim-b-kh/buddhi_pragati"
echo "🧹 Cache management: Cleared after each language to prevent massive disk usage"
echo ""
echo "Next steps:"
echo "1. Verify datasets: python run_crossword_benchmark.py manage-dataset list"
echo "2. Check statistics: python run_crossword_benchmark.py manage-dataset stats --config-name <language>"
echo "3. Use datasets for crossword generation!"