# Buddhi-Pragati: Crossword Reasoning Benchmark System

A comprehensive LLM reasoning benchmark system that evaluates language model capabilities through crossword puzzle generation and solving across multiple languages and cultural contexts.

## Project Overview

**Buddhi-Pragati** (Sanskrit for "wisdom-progress") is a modular, model-agnostic benchmark architecture that tests language model reasoning skills through crossword puzzles. The system implements a complete pipeline from dataset creation to evaluation, with special focus on Indian languages and cultural context.

### Key Features

- **Multi-Language Support**: 20+ languages including 11 Indic scripts
- **Cultural Context Assessment**: Sophisticated scoring for Indian cultural relevance
- **Model-Agnostic**: Unified interface supporting OpenAI, Anthropic, HuggingFace, OpenRouter, and SarvamAI
- **Advanced Generation**: Memetic algorithms achieving 75%+ grid density
- **Comprehensive Evaluation**: Multiple experimental conditions and prompting strategies

## Architecture Overview

The system consists of six main modules that work together to create a complete benchmark pipeline:

```
buddhi_pragati/
├── core/           # Base interfaces and data structures
├── data/           # Dataset creation and source processing  
├── models/         # Unified model interface system
├── generate/       # Crossword puzzle generation
├── evaluate/       # LLM evaluation and experiments
└── utils/          # Configuration, scoring, and utilities
```

### Module Interactions

```
[Data Sources] → [data/] → [Dataset] → [generate/] → [Puzzles]
                                                        ↓
[models/] ←→ [evaluate/] ←→ [Puzzles] → [Results]
     ↑              ↑
[utils/]    [core/interfaces]
```

## Module Documentation

### `buddhi_pragati/core/` - Base Interfaces

**Purpose**: Provides minimal abstract interfaces that all puzzle types and evaluators must implement.

**Files**:
- `__init__.py` - Package exports and version information
- `base_puzzle.py` - Abstract `BasePuzzle` interface and concrete `CrosswordPuzzle`/`CrosswordClue` implementations
- `base_evaluator.py` - Abstract `BaseEvaluator` interface for evaluation systems

**Architecture**: Clean separation of concerns with minimal dependencies, enabling easy extension to new puzzle types.

### `buddhi_pragati/data/` - Dataset Creation Pipeline

**Purpose**: Multi-source dataset creation pipeline that processes raw data from Indian language sources into structured crossword datasets.

**Files**:
- `__init__.py` - Package exports
- `data_structure.py` - `DatasetEntry` dataclass defining the standard dataset format  
- `dataset_builder.py` - Main `DatasetBuilder` orchestrator coordinating all source processors
- `dataset_state.json` - Processing state tracking for resumable operations
- `datasets_models.txt` - Model compatibility matrix for different languages
- `local_backup_manager.py` - `LocalBackupManager` for disk-based batch processing and recovery
- `source_processors/` - Individual processor implementations:
  - `__init__.py` - Processor exports
  - `milu_processor.py` - MILU examination dataset MCQ processing
  - `indic_wikibio_processor.py` - IndicWikiBio biographical data processing
  - `indowordnet_processor.py` - IndoWordNet dictionary definition processing  
  - `bhasha_wiki_processor.py` - Bhasha-Wiki named entity recognition processing

**Pipeline Flow**: Raw sources → Source processors → Quality filtering → Context scoring → Deduplication → HuggingFace upload

### `buddhi_pragati/models/` - Unified Model Interface

**Purpose**: Provides a single interface for interacting with multiple LLM providers, handling authentication, parameter mapping, and response parsing.

**Files**:
- `__init__.py` - Interface exports
- `model_interface.py` - `UnifiedModelInterface` main class with provider-specific backends (`OpenAIBackend`, `AnthropicBackend`, `SarvamAIBackend`, `OpenRouterBackend`, `HuggingFaceBackend`)

**Features**: Auto-detection of model providers, reasoning mode support, token tracking, official API client integration.

### `buddhi_pragati/generate/` - Puzzle Generation System

**Purpose**: Advanced crossword puzzle generation using memetic algorithms to achieve high-density grids with cultural context optimization.

**Files**:
- `__init__.py` - Generation module exports
- `puzzle_entry.py` - `CrosswordPuzzleEntry` dataclass for generated puzzles
- `corpus_loader.py` - `CrosswordCorpusLoader` for loading and filtering clue-answer datasets
- `memetic_generator.py` - `MemeticCrosswordGenerator` implementing genetic algorithms for puzzle generation
- `puzzle_builder.py` - `PuzzleBuilder` main orchestrator for batch puzzle generation
- `puzzle_visualisation.py` - Grid visualization and ASCII rendering utilities
- `hf_uploader.py` - `PuzzleHFUploader` for uploading generated puzzles to HuggingFace Hub

**Algorithm**: Population-based genetic algorithm with crossover, mutation, and local search optimization targeting density, intersections, and cultural coherence.

### `buddhi_pragati/evaluate/` - Evaluation and Experiments

**Purpose**: Comprehensive LLM evaluation system supporting multiple experimental conditions, prompting strategies, and performance analysis.

**Files**:
- `__init__.py` - Evaluation module exports
- `evaluator.py` - `CrosswordEvaluator` main evaluation orchestrator
- `dataset_loader.py` - `PuzzleDatasetLoader` for loading puzzles from HuggingFace datasets
- `templates.py` - `CrosswordPromptTemplate` with configurable prompting strategies
- `parser.py` - `CrosswordResponseParser` for extracting answers from LLM responses
- `metrics.py` - `EnhancedCrosswordMetrics` implementing WCR, LCR, ICR scoring
- `model_classifier.py` - `ModelClassifier` for organizing models by capabilities and language support
- `experiment_runner.py` - `NewExperimentRunner` orchestrating 10+ experimental conditions
- `results_manager.py` - `ExperimentResultsManager` for analysis and reporting
- `prompts/` - JSON configuration files for different prompting strategies

**Experiments**: Shot variations, chain-of-thought, reasoning effort levels, self-reflection, language families, model types.

### `buddhi_pragati/utils/` - Configuration and Utilities

**Purpose**: Shared utilities, configuration management, and specialized scoring systems.

**Files**:
- `__init__.py` - Utility exports  
- `config_loader.py` - `ConfigLoader` for centralized configuration from `crossword_config.txt`
- `dataset_manager.py` - `DatasetManager` for HuggingFace dataset management operations
- `indian_context_scorer.py` - `IndianContextScorer` implementing multi-tier cultural context assessment
- `unicode_utils.py` - Unicode processing utilities with script detection and validation
- `create_corpus.py` - Corpus creation utilities for Indian cultural references
- `indian_corpus.json` - Indian cultural reference corpus data
- `seed_corpus.json` - Seed corpus for cultural context scoring

**Configuration System**: Centralized parameter management with CLI defaults, fallback hierarchy, and type-safe validation.

### `buddhi_pragati/scripts/` - Automation and Batch Processing

**Purpose**: Shell scripts for batch operations, environment setup, and cluster computing.

**Files**:
- `README.md` - Script documentation
- `run_all_experiments.sh` - Complete experimental evaluation pipeline
- `run_analysis_experiments.sh` - Analysis-specific experiments (language families, model types)
- `run_dataset_creation.sh` - Batch dataset creation across languages
- `run_focused_experiments.sh` - Subset of core experiments for quick validation
- `run_generate_puzzles.sh` - Batch puzzle generation pipeline  
- `run_master_experiment.sh` - Master baseline experiment
- `submit_experiments.pbs` - PBS cluster submission script with dependency management
- `utils/check_results.sh` - Result validation and integrity checking
- `utils/monitor_progress.sh` - Real-time progress monitoring
- `utils/setup_environment.sh` - Environment initialization and dependency installation
- `logs/` - Script execution logs

## Installation and Setup

### Prerequisites

- Python 3.8-3.11
- Access to language model APIs (OpenAI, Anthropic, etc.)
- HuggingFace token for dataset access

### Installation

```bash
# Clone repository
git clone <repository-url>
cd irp-sbk124

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate buddhi-pragati

# Or install with pip
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install specific dependency groups
pip install -e ".[api]"     # API clients
pip install -e ".[dev]"     # Development tools
pip install -e ".[scoring]" # Enhanced context scoring
```

### Configuration

Create and configure `crossword_config.txt` with your parameters:

```ini
# API Keys (optional - can use environment variables)
DEFAULT_OPENAI_API_KEY=your_openai_key
DEFAULT_HF_TOKEN=your_huggingface_token

# Dataset Parameters  
DEFAULT_LANGUAGE=English
TARGET_DATASET_SIZE_PER_LANGUAGE=1000
DEFAULT_DATASET_SOURCES=MILU,IndicWikiBio,IndoWordNet,Bhasha-Wiki

# Evaluation Parameters
DEFAULT_MODEL=gpt-4o
DEFAULT_EVALUATION_MODELS=gpt-4o,claude-sonnet-4
DEFAULT_REASONING_EFFORT=normal
```

## Usage Guide

### Main Executor: `run_crossword_benchmark.py`

The primary interface for all system operations. Supports six main commands:

#### Dataset Creation

```bash
# Create datasets for multiple languages with default settings
python run_crossword_benchmark.py create-dataset --languages Hindi English Bengali --upload

# Custom dataset creation with specific sources
python run_crossword_benchmark.py create-dataset \
  --languages Tamil Telugu \
  --target-size 1500 \
  --sources MILU IndicWikiBio \
  --output-path ./datasets

# Use configuration defaults (minimal command)
python run_crossword_benchmark.py create-dataset --languages English --upload
```

#### Puzzle Generation

```bash
# Generate crosswords using config defaults
python run_crossword_benchmark.py generate

# Generate for specific language and grid sizes
python run_crossword_benchmark.py generate \
  --language Hindi \
  --grid-sizes 8 10 12 \
  --count 10 \
  --upload-to-hf

# Generate with corpus size limit
python run_crossword_benchmark.py generate \
  --language Bengali \
  --corpus-limit 500
```

#### LLM Evaluation

```bash
# Evaluate using HuggingFace datasets (recommended)
python run_crossword_benchmark.py evaluate \
  --languages Hindi English \
  --grid-sizes 7 15 \
  --count 20 \
  --models gpt-4o claude-sonnet-4

# Enhanced evaluation with reasoning
python run_crossword_benchmark.py evaluate \
  --languages Tamil \
  --reasoning-effort high \
  --self-reflection \
  --chain-of-thought

# Legacy local directory evaluation
python run_crossword_benchmark.py evaluate \
  --puzzle-dir generated_crosswords/hindi/10x10 \
  --model gpt-4o
```

#### Experimental Evaluation

```bash
# Run comprehensive experiments
python run_crossword_benchmark.py evaluate \
  --experiment-mode \
  --experiment-types 0_master 2_shot_variations 4_chain_of_thought \
  --models gpt-4o claude-sonnet-4

# Language family analysis
python run_crossword_benchmark.py evaluate \
  --experiment-mode \
  --experiment-types 1_language_families \
  --master-results-path experiment_0_master_results.json
```

#### Dataset Management

```bash
# List all language configurations
python run_crossword_benchmark.py manage-dataset list

# Inspect dataset structure  
python run_crossword_benchmark.py manage-dataset inspect

# Delete specific language configuration
python run_crossword_benchmark.py manage-dataset delete --config-name hindi

# Create backup before operations
python run_crossword_benchmark.py manage-dataset backup --config-name english
```

### Configuration System

The system uses a hierarchical configuration approach:

1. **CLI Arguments** (highest priority)
2. **crossword_config.txt** (fallback)
3. **Environment Variables** (fallback)
4. **Hard-coded Defaults** (lowest priority)

This allows for flexible deployment across different environments while maintaining reproducibility.

## Root-Level Files

- **`run_crossword_benchmark.py`** - Main CLI interface and system orchestrator
- **`crossword_config.txt`** - Centralized configuration file with all system parameters
- **`environment.yml`** - Conda environment specification with PyTorch and HuggingFace channels
- **`pyproject.toml`** - Python package configuration with optional dependency groups
- **`models_description.txt`** - Documentation of supported models and their capabilities
- **`References.md`** - Academic references and related work
- **`LICENSE`** - Creative Commons Attribution-NonCommercial 4.0 International license

## Development and Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=buddhi_pragati

# Run specific module tests
pytest tests/test_evaluator.py
```

### Code Quality

```bash
# Format code
ruff buddhi_pragati/

# Type checking
mypy buddhi_pragati/
```

### Logging and Monitoring

All operations generate detailed logs with timestamps:

```bash
# Monitor execution in real-time
tail -f logs/crossword_benchmark_YYYYMMDD_HHMMSS.log

# Enable verbose output
python run_crossword_benchmark.py generate --verbose
```

## Expected Performance and Design Choices

### Dataset Creation
- **Processing Speed**: 15-20 entries/second typical throughput (when the `create-dataset` command is ran on a standard CPU in a commercial laptop)
- **Memory Footprint**: Bounded by batch size regardless of dataset size  
- **Scalability**: Supports processing datasets with millions of entries (such as Bhasha-Wiki)

### Puzzle Generation  
- **Generation Speed**: 5-10 puzzles/minute for 10x10 grids
- **Density Achievement**: 75%+ fill rate for most grid sizes
- **Success Rate**: 85%+ successful generation across languages

### LLM Evaluation
- **Concurrent Support**: Multiple models evaluated in parallel
- **Token Tracking**: Comprehensive usage monitoring for cost analysis
- **Caching**: Intelligent result caching to avoid redundant API calls

## Contributing

This is an academic research project under the Creative Commons Attribution-NonCommercial 4.0 International license. The system architecture is designed for extensibility:

- **New Languages**: Add to source processors and model classifier
- **New Puzzle Types**: Implement `BasePuzzle` interface
- **New Models**: Add backend to `UnifiedModelInterface`
- **New Evaluation Metrics**: Extend `EnhancedCrosswordMetrics`

## Academic Context

This system was developed as part of an Imperial College London Individual Research Project (IRP). It represents a comprehensive approach to multilingual reasoning evaluation with specific focus on cultural context and crossword puzzle complexity.

For detailed technical documentation, see the generated Doxygen documentation (when available) or examine the comprehensive docstrings throughout the codebase.