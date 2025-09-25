#!/usr/bin/env python3
"""
Buddhi-Pragati Crossword Benchmark System

Complete pipeline for crossword puzzle generation and evaluation using the MILU dataset.
Supports both classical and genetic algorithms for puzzle generation, and comprehensive
LLM evaluation with CrossWordBench metrics.

Features:
- Generate crosswords from MILU dataset in 11 Indic languages
- Evaluate LLMs on crossword solving with WCR, LCR, ICR metrics
- Classical intersection and genetic algorithm generation
- Multi-model support (OpenAI, OpenRouter, HuggingFace)
- Configuration-driven defaults with CLI/env override
- Batch processing and benchmarking capabilities

Usage Examples:
  # Use config defaults (minimal command)
  python run_crossword_benchmark.py generate
  python run_crossword_benchmark.py evaluate --puzzle-dir generated_crosswords/hindi/10x10

  # Enhanced evaluation from HuggingFace datasets
  python run_crossword_benchmark.py evaluate --languages Hindi English --grid-sizes 7 15 --count 20
  python run_crossword_benchmark.py evaluate --languages Bengali --models gpt-4o claude-sonnet-4 --chain-of-thought
  python run_crossword_benchmark.py evaluate --languages Tamil --reasoning-effort high --self-reflection

  # Legacy local directory evaluation
  python run_crossword_benchmark.py evaluate --puzzle-dir generated_crosswords/hindi/10x10 --model gpt-4o

  # Run experimental evaluation
  python run_crossword_benchmark.py run-experiments 0_master
  python run_crossword_benchmark.py run-experiments 2_shot_variations 4_chain_of_thought --models gpt-4o claude-sonnet-4
  python run_crossword_benchmark.py run-experiments 1_language_families --master-results-path experiment_0_master_*.json
"""

import argparse
import logging
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys
from datetime import datetime

# Core imports
from buddhi_pragati.core.base_puzzle import CrosswordPuzzle, CrosswordClue
from buddhi_pragati.evaluate.evaluator import CrosswordEvaluator
from buddhi_pragati.models.model_interface import UnifiedModelInterface

# Dataset creation imports
from buddhi_pragati.data.dataset_builder import DatasetBuilder

# Dataset management imports
from buddhi_pragati.utils.dataset_manager import DatasetManager

# Generation imports
from buddhi_pragati.generate.puzzle_builder import PuzzleBuilder

# Configuration imports
from buddhi_pragati.utils.config_loader import get_config

# Experimental evaluation imports
from buddhi_pragati.evaluate.dataset_loader import PuzzleDatasetLoader
from buddhi_pragati.evaluate.templates import CrosswordPromptTemplate


def get_config_defaults():
    """Get configuration defaults for CLI arguments."""
    config = get_config()
    return config.get_cli_defaults()


def apply_config_defaults(args):
    """Apply configuration defaults to CLI arguments when not specified."""
    defaults = get_config_defaults()

    # Apply token/key defaults if not provided and config has values
    if hasattr(args, "hf_token") and not args.hf_token:
        config_token = defaults["hf_token"]
        if config_token:  # Only if config has a non-empty value
            args.hf_token = config_token

    if hasattr(args, "api_key") and not args.api_key:
        config_key = defaults["openai_api_key"]
        if config_key:  # Only if config has a non-empty value
            args.api_key = config_key

    return args


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration with file output for traceability."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(console_handler)

    # File handler for detailed tracing
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always detailed in file
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        # Log system info
        logging.info("=== BUDDHI-PRAGATI CROSSWORD BENCHMARK TRACE ===")
        logging.info(f"Start time: {datetime.now().isoformat()}")
        logging.info(f"Log file: {log_file}")
        logging.info(f"Verbose mode: {verbose}")
        logging.info("=" * 60)

    return log_file


def load_puzzle_from_json(puzzle_path: str) -> Optional[CrosswordPuzzle]:
    """Load a CrosswordPuzzle from JSON file."""
    try:
        with open(puzzle_path, "r", encoding="utf-8") as f:
            puzzle_data = json.load(f)

        # Create clues from data
        clues = []
        for clue_data in puzzle_data.get("clues", []):
            clue = CrosswordClue(
                number=clue_data["number"],
                direction=clue_data["direction"],
                length=clue_data["length"],
                clue_text=clue_data["clue_text"],
                start_row=clue_data["start_row"],
                start_col=clue_data["start_col"],
                answer=clue_data.get("answer", ""),
            )
            clues.append(clue)

        # Create puzzle (solution_grid is required)
        solution_grid = puzzle_data.get("solution_grid")
        if solution_grid is None:
            raise ValueError(
                f"Puzzle {puzzle_data['puzzle_id']} is missing solution_grid - this indicates corrupted puzzle data"
            )

        puzzle = CrosswordPuzzle(
            puzzle_id=puzzle_data["puzzle_id"],
            grid=puzzle_data["grid"],
            clues=clues,
            size=tuple(puzzle_data["size"]),
            solution_grid=solution_grid,
            solution_words=puzzle_data.get("solution_words", {}),
        )

        return puzzle

    except Exception as e:
        logging.error(f"Failed to load puzzle from {puzzle_path}: {e}")
        return None


def find_puzzle_files(puzzle_dir: str) -> List[str]:
    """Find all puzzle JSON files in directory."""
    puzzle_dir = Path(puzzle_dir)

    if not puzzle_dir.exists():
        logging.error(f"Puzzle directory does not exist: {puzzle_dir}")
        return []

    # Look for puzzle JSON files
    patterns = ["puzzle_*.json", "*.json"]
    puzzle_files = set()  # Use set to avoid duplicates

    for pattern in patterns:
        files = list(puzzle_dir.glob(pattern))
        # Filter out summary files
        files = [f for f in files if "summary" not in f.name]
        puzzle_files.update([str(f) for f in files])  # Use update instead of extend

    puzzle_files_list = sorted(list(puzzle_files))  # Convert back to sorted list

    if not puzzle_files_list:
        logging.warning(f"No puzzle files found in {puzzle_dir}")

    return puzzle_files_list


def run_evaluation(args) -> Dict[str, Any]:
    """
    Evaluate LLM on crossword puzzles loaded from HuggingFace dataset.

    Returns:
        Dict containing:
        - success: bool indicating if evaluation succeeded
        - results: Dict of model_name -> evaluation results
        - config: Dict of evaluation configuration used
        - puzzles: List of puzzles evaluated
    """
    # Support suppressing output for experimental mode
    suppress_output = getattr(args, "suppress_output", False)

    if not suppress_output:
        print("🤖 ENHANCED LLM CROSSWORD EVALUATION")
        print("=" * 60)

    # Load configuration for defaults
    config = get_config()

    # Initialize model classifier for filtering
    from buddhi_pragati.evaluate.model_classifier import ModelClassifier
    model_classifier = ModelClassifier()

    try:
        # Load puzzles from HuggingFace datasets or local directory
        puzzles = []

        if hasattr(args, "languages") and args.languages:
            # New HuggingFace dataset loading mode
            if not suppress_output:
                print("📥 Loading puzzles from HuggingFace datasets...")

            # Initialize dataset loader
            loader = PuzzleDatasetLoader()

            # Load puzzles for each language and grid size combination
            total_puzzles_loaded = 0
            for language in args.languages:
                grid_sizes = getattr(args, "grid_sizes", [None])
                if not grid_sizes:
                    grid_sizes = [None]  # Load all sizes

                for grid_size in grid_sizes:
                    try:
                        # Calculate puzzles per combination with config defaults
                        total_count = getattr(args, "count", None) or int(
                            config.get("DEFAULT_PUZZLE_COUNT", "10")
                        )
                        count_per_combo = total_count // (
                            len(args.languages) * len(grid_sizes)
                        )
                        if count_per_combo < 1:
                            count_per_combo = 1

                        language_puzzles = loader.load_puzzles(
                            language=language.lower(),  # Convert to lowercase for dataset config
                            grid_size=grid_size,
                            max_puzzles=count_per_combo,
                            random_sample=True,
                        )

                        if language_puzzles:
                            puzzles.extend(language_puzzles)
                            total_puzzles_loaded += len(language_puzzles)
                            if not suppress_output:
                                print(
                                    f"   ✅ Loaded {len(language_puzzles)} puzzles for {language} (grid size: {grid_size or 'all'})"
                                )
                        else:
                            if not suppress_output:
                                print(
                                    f"   ⚠️  No puzzles found for {language} (grid size: {grid_size or 'all'})"
                                )

                    except Exception as e:
                        if not suppress_output:
                            print(f"   ❌ Failed to load {language} puzzles: {e}")

        elif args.puzzle_dir:
            # Legacy local directory mode
            if not suppress_output:
                print(f"📁 Loading puzzles from local directory: {args.puzzle_dir}")
            puzzle_files = find_puzzle_files(args.puzzle_dir)
            if not puzzle_files:
                if not suppress_output:
                    print(f"❌ No puzzles found in {args.puzzle_dir}")
                return {"success": False, "error": f"No puzzles found in {args.puzzle_dir}"}

            max_legacy_puzzles = getattr(args, "max_puzzles", None) or int(
                config.get("DEFAULT_PUZZLE_COUNT", "10")
            )
            for puzzle_file in puzzle_files[:max_legacy_puzzles]:
                puzzle = load_puzzle_from_json(puzzle_file)
                if puzzle:
                    puzzles.append(puzzle)
        else:
            if not suppress_output:
                print(
                    "❌ Error: Either --languages or --puzzle-dir required for evaluation"
                )
            return {"success": False, "error": "Either --languages or --puzzle-dir required for evaluation"}

        if not puzzles:
            if not suppress_output:
                print("❌ No valid puzzles loaded")
            return {"success": False, "error": "No valid puzzles loaded"}

        # Limit total puzzles if specified, with config defaults
        max_puzzles = (
            getattr(args, "max_puzzles", None)
            or getattr(args, "count", None)
            or int(config.get("DEFAULT_PUZZLE_COUNT", "10"))
        )
        if max_puzzles and len(puzzles) > max_puzzles:
            puzzles = puzzles[:max_puzzles]

        # Determine target language(s) for model filtering
        target_languages = []
        if hasattr(args, "languages") and args.languages:
            target_languages = args.languages
        elif args.puzzle_dir:
            # Try to infer language from puzzle directory path or puzzles
            # For now, use DEFAULT_LANGUAGE from config as fallback
            target_languages = [config.get("DEFAULT_LANGUAGE", "English")]

        if not target_languages:
            target_languages = [config.get("DEFAULT_LANGUAGE", "English")]

        # Display configuration
        if not suppress_output:
            print("\n📋 Evaluation Configuration:")
            if hasattr(args, "languages") and args.languages:
                print(f"   Languages: {', '.join(args.languages)}")
                if hasattr(args, "grid_sizes") and args.grid_sizes:
                    print(f"   Grid sizes: {', '.join(map(str, args.grid_sizes))}")
            elif args.puzzle_dir:
                print(f"   Puzzle directory: {args.puzzle_dir}")
            print(f"   Puzzles loaded: {len(puzzles)}")

        # Handle multiple models with config defaults
        if hasattr(args, "models") and args.models:
            requested_models = args.models
        elif hasattr(args, "model") and args.model and args.model != config.get("DEFAULT_MODEL", "gpt-4o"):
            # User specified a model via --model
            requested_models = [args.model]
        else:
            # No models specified, use config defaults
            default_models = config.get(
                "DEFAULT_EVALUATION_MODELS", "gpt-4o,claude-sonnet-4"
            )
            requested_models = [m.strip() for m in default_models.split(",")]

        # Check if transformers is available for HuggingFace models
        try:
            hf_available = True
        except ImportError:
            hf_available = False
            if not suppress_output:
                print("⚠️  HuggingFace transformers not available - will skip HuggingFace models")

        # Apply model filtering based on language support and other criteria
        models_to_test = []

        # For each target language, filter models that support it
        for language in target_languages:
            # Get models that support this language
            language_models = model_classifier.get_models_for_language(language)
            language_model_names = {m.name for m in language_models}

            # Filter requested models to only those that support the language
            for model_name in requested_models:
                if model_name in language_model_names:
                    # Skip HuggingFace models if transformers is not available
                    if not hf_available:
                        # Simple HuggingFace model detection
                        if model_name.startswith(("ai4bharat/", "CohereForAI/", "Cognitive-Lab/", "nickmalhotra/", "smallstepai/", "abhinand/")):
                            logging.warning(f"Skipping HuggingFace model {model_name} - transformers not available")
                            continue

                    if model_name not in models_to_test:
                        models_to_test.append(model_name)
                else:
                    logging.warning(f"Model {model_name} does not support language {language}, skipping")

        # Apply additional filters if specified
        if hasattr(args, "priority_only") and args.priority_only:
            priority_models = model_classifier.get_priority_models()
            priority_names = {m.name for m in priority_models}
            models_to_test = [m for m in models_to_test if m in priority_names]

        if hasattr(args, "reasoning_only") and args.reasoning_only:
            reasoning_models = model_classifier.get_reasoning_models()
            reasoning_names = {m.name for m in reasoning_models}
            models_to_test = [m for m in models_to_test if m in reasoning_names]

        if hasattr(args, "indic_only") and args.indic_only:
            indic_models = model_classifier.get_indic_finetuned_models()
            indic_names = {m.name for m in indic_models}
            models_to_test = [m for m in models_to_test if m in indic_names]

        if not models_to_test:
            if not suppress_output:
                print(f"❌ No models found that support languages: {', '.join(target_languages)}")
            return {"success": False, "error": f"No models support languages: {', '.join(target_languages)}"}

        if not suppress_output:
            print(f"   Models: {', '.join(models_to_test)}")
            print(f"   Model source: {args.model_source}")

        # Get prompt configuration with config defaults
        chain_of_thought = getattr(args, "chain_of_thought", False) or config.get_bool(
            "DEFAULT_CHAIN_OF_THOUGHT", False
        )
        reasoning_effort = getattr(args, "reasoning_effort", None) or config.get(
            "DEFAULT_REASONING_EFFORT", "normal"
        )
        self_reflection = getattr(args, "self_reflection", False) or config.get_bool(
            "DEFAULT_SELF_REFLECTION", False
        )

        # Display prompt configuration
        if not suppress_output:
            print("\n🎯 Prompt Configuration:")
            print(f"   Chain of thought: {chain_of_thought}")
            print(f"   Reasoning effort: {reasoning_effort}")
            print(f"   Self reflection: {self_reflection}")

        # Initialize evaluator
        evaluator = CrosswordEvaluator()

        # Test each model
        all_model_results = {}

        for model_name in models_to_test:
            if not suppress_output:
                print(f"\n🔄 Evaluating with model: {model_name}")
                print("-" * 40)

            # Detect model source based on model name
            def detect_model_source(name):
                """Auto-detect model source from model name patterns."""
                if name.startswith(("gpt-", "o1", "o3")):
                    return "openai"
                elif name.startswith("claude"):
                    return "anthropic"
                elif "sarvam" in name.lower():
                    return "sarvamai"
                elif name.startswith(("deepseek/", "google/gemini", "meta-llama/", "mistralai/", "moonshotai/", "qwen/")):
                    return "openrouter"
                elif name.startswith(("ai4bharat/", "CohereForAI/", "Cognitive-Lab/", "nickmalhotra/", "smallstepai/", "abhinand/")):
                    return "huggingface"
                else:
                    # Fall back to CLI/config default if can't detect
                    return getattr(args, 'model_source', config.get("DEFAULT_MODEL_SOURCE", "openai"))

            # Determine actual model source
            actual_model_source = detect_model_source(model_name)

            # Get API key with fallback hierarchy: CLI → config → env
            api_key = getattr(args, "api_key", None)
            if not api_key:
                # Try config defaults based on actual model source
                if actual_model_source == "openai":
                    api_key = config.get("DEFAULT_OPENAI_API_KEY") or os.getenv(
                        "OPENAI_API_KEY"
                    )
                elif actual_model_source == "anthropic":
                    api_key = config.get("DEFAULT_ANTHROPIC_API_KEY") or os.getenv(
                        "ANTHROPIC_API_KEY"
                    )
                elif actual_model_source in ["sarvam", "sarvamai"]:
                    api_key = config.get("DEFAULT_SARVAM_API_KEY") or os.getenv(
                        "SARVAM_API_KEY"
                    )
                elif actual_model_source == "openrouter":
                    api_key = config.get("DEFAULT_OPENROUTER_API_KEY") or os.getenv(
                        "OPENROUTER_API_KEY"
                    )
                elif actual_model_source == "huggingface":
                    api_key = config.get("DEFAULT_HF_TOKEN") or os.getenv("HF_TOKEN")

            logging.info(
                f"API_KEY_CHECK: model={model_name}, detected_source={actual_model_source}, api_key={'***set***' if api_key else 'None'}"
            )

            if not api_key and actual_model_source in [
                "openai",
                "anthropic",
                "sarvamai",
                "openrouter",
            ]:
                if not suppress_output:
                    print(f"❌ Error: API key required for {actual_model_source} model {model_name}")
                    print("   Set appropriate environment variable or use --api-key")
                    print("   Or configure in crossword_config.txt")
                logging.error(
                    f"EVALUATION_FAILED: No API key provided for {actual_model_source} model {model_name}"
                )
                continue

            # Create model with auto-detected source (UnifiedModelInterface will also auto-detect)
            try:
                model = UnifiedModelInterface(
                    model_name=model_name,
                    source=actual_model_source,
                    api_key=api_key
                )
            except ImportError as e:
                # Handle missing dependencies gracefully - skip this model
                error_msg = str(e)
                if not suppress_output:
                    print(f"⚠️  Skipping {model_name}: {error_msg}")
                logging.warning(f"Model {model_name} skipped due to missing dependencies: {e}")

                # Add error to results for this model
                if model_name not in all_model_results:
                    all_model_results[model_name] = {}

                # Add error for all target languages and grid sizes
                for lang in target_languages:
                    if lang not in all_model_results[model_name]:
                        all_model_results[model_name][lang] = {}
                    for grid_size in getattr(args, 'grid_sizes', [10]):  # default grid size
                        all_model_results[model_name][lang][f"grid_{grid_size}"] = {
                            "error": error_msg
                        }
                continue

            # Create template with prompt configuration
            target_language = getattr(
                args, "languages", [config.get("DEFAULT_LANGUAGE", "English")]
            )[0]
            template = CrosswordPromptTemplate(target_language=target_language)

            # Apply prompt configuration from config defaults
            template.chain_of_thought = chain_of_thought
            template.reasoning_effort = reasoning_effort
            template.self_reflection = self_reflection

            logging.info(
                f"EVALUATION_START: {len(puzzles)} puzzles, Model={model_name}"
            )

            # Choose evaluation method based on configuration
            if self_reflection:
                result = evaluator.evaluate_batch_with_reflection(
                    model, puzzles, template
                )
            elif len(puzzles) == 1:
                result = evaluator.evaluate_single(model, puzzles[0], template)
                # Wrap single result in batch format for consistency
                result = {
                    "model_name": model_name,
                    "total_puzzles": 1,
                    "individual_results": [result],
                    "summary_metrics": evaluator._calculate_summary([result]),
                    "template_config": result.get("template_config", {}),
                }
            else:
                result = evaluator.evaluate_batch(model, puzzles, template)

            all_model_results[model_name] = result

            # Display results for this model
            if not suppress_output:
                print(f"\n📊 RESULTS FOR {model_name}")
                print("=" * 50)

                if "summary_metrics" in result:
                    summary = result["summary_metrics"]
                    print(
                        f"✅ Evaluated {result.get('total_puzzles', len(puzzles))} puzzles"
                    )
                    print("\n🎯 Summary Metrics:")
                    for metric, value in summary.items():
                        if isinstance(value, float):
                            if metric.startswith("average_"):
                                clean_name = metric.replace("average_", "").upper()
                                print(f"   {clean_name}: {value:.3f}")
                            else:
                                print(f"   {metric}: {value:.3f}")
                        else:
                            print(f"   {metric}: {value}")

                    # Display token usage if available
                    if "total_tokens" in summary:
                        print("\n💡 Resource Usage:")
                        print(f"   Total tokens: {summary.get('total_tokens', 'N/A')}")
                        print(
                            f"   Avg time per puzzle: {summary.get('average_time_per_puzzle', 0):.2f}s"
                        )

        # Final summary if multiple models
        if len(models_to_test) > 1 and not suppress_output:
            print(f"\n🏆 COMPARATIVE RESULTS ({len(models_to_test)} models)")
            print("=" * 60)

            for model_name, result in all_model_results.items():
                summary = result.get("summary_metrics", {})
                success_rate = summary.get("success_rate", 0)
                avg_word_acc = summary.get("average_word_accuracy_global", 0)
                print(
                    f"   {model_name:20} | Success: {success_rate:.1%} | Word Acc: {avg_word_acc:.3f}"
                )

        # Return structured results
        return {
            "success": True,
            "results": all_model_results,
            "config": {
                "languages": target_languages,
                "models": models_to_test,
                "chain_of_thought": chain_of_thought,
                "reasoning_effort": reasoning_effort,
                "self_reflection": self_reflection,
                "puzzle_count": len(puzzles)
            },
            "puzzles": puzzles
        }

    except Exception as e:
        if not suppress_output:
            print(f"❌ Evaluation failed: {e}")
        logging.error(f"Evaluation error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_create_dataset(args) -> bool:
    """Create datasets from multiple sources for crossword generation."""
    print("📊 MULTI-SOURCE DATASET CREATION")
    print("=" * 60)

    # Enable verbose logging if requested
    if hasattr(args, "verbose") and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logging.info("Verbose mode enabled for dataset creation")

    # Get HuggingFace token with fallback hierarchy: CLI → config → env
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HuggingFace token required for dataset access")
        print("   Get token from: https://huggingface.co/settings/tokens")
        print("   Set: export HF_TOKEN=your_token_here")
        print("   Or use: --hf-token your_token_here")
        print("   Or configure in crossword_config.txt: DEFAULT_HF_TOKEN=your_token")
        return False

    try:
        # Initialize dataset builder with context scoring mode from CLI/config
        builder = DatasetBuilder(hf_token=hf_token, context_scoring_mode=args.mode)

        print("📋 Configuration:")
        print(f"   Languages: {', '.join(args.languages)}")
        print(f"   Target size per language: {args.target_size}")
        print(f"   Sources: {', '.join(args.sources)}")
        print(f"   Context scoring mode: {args.mode}")
        print(f"   Upload to HuggingFace: {'Yes' if args.upload else 'No'}")
        if hasattr(args, "output_path"):
            print(f"   Output path: {args.output_path}")

        # Build datasets for each language
        print("\n🔄 Creating datasets...")
        logging.info(
            f"DATASET_CREATION_START: Languages={args.languages}, Sources={args.sources}"
        )

        for language in args.languages:
            print(f"\n📚 Processing {language}...")
            logging.info(f"LANGUAGE_START: {language}")

            # Create dataset for this language using specified sources
            dataset = builder.build_dataset(
                language=language,
                total_target=args.target_size,
                sources=args.sources,
                upload_to_hf=args.upload,  # Pass upload flag for incremental upload
            )

            if args.upload:
                # In upload mode, dataset is empty (entries uploaded incrementally)
                print(
                    f"   ✅ Uploaded {args.target_size} entries for {language} (incremental upload)"
                )
                logging.info(
                    f"LANGUAGE_COMPLETE: {language} - {args.target_size} entries uploaded incrementally"
                )
                continue
            else:
                # In accumulation mode, check we got entries back
                if dataset is None or len(dataset) == 0:
                    print(f"   ⚠️  No entries created for {language}")
                    logging.warning(f"LANGUAGE_FAILED: {language} - no entries created")
                    continue

            print(f"   ✅ Created {len(dataset)} entries for {language}")

            # Show sample entries
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   Sample: '{sample.clue}' → '{sample.answer}'")
                print(
                    f"   Source: {sample.source}, Quality: {sample.quality_score:.3f}"
                )

            # Save locally if output path specified
            if hasattr(args, "output_path") and args.output_path:
                import json
                from pathlib import Path
                from dataclasses import asdict

                output_dir = Path(args.output_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{language.lower()}_crossword_dataset.json"

                # Convert entries to JSON format
                json_data = [asdict(entry) for entry in dataset]
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

                print(f"   📁 Saved to: {output_file}")

            # Note: Upload handled incrementally in build_dataset() when args.upload=True
            # This section only runs in accumulation mode (args.upload=False)

            logging.info(f"LANGUAGE_COMPLETE: {language} - {len(dataset)} entries")

        print("\n🎉 DATASET CREATION COMPLETE!")
        print("=" * 60)
        print("✅ All languages processed successfully")

        if args.upload:
            print("🌐 Datasets uploaded to HuggingFace Hub")
            print("   Use these datasets for crossword generation!")

        logging.info("DATASET_CREATION_COMPLETE: All languages processed")
        return True

    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        logging.error(f"Dataset creation error: {e}", exc_info=True)
        return False


def run_crossword_generation(args) -> bool:
    """Generate crossword puzzles from corpus dataset."""
    print("🧩 CROSSWORD PUZZLE GENERATION")
    print("=" * 60)

    # Enable verbose logging if requested
    if hasattr(args, "verbose") and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logging.info("Verbose mode enabled for puzzle generation")

    # Get HuggingFace token with fallback hierarchy: CLI → config → env
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HuggingFace token required for corpus dataset access")
        print("   Get token from: https://huggingface.co/settings/tokens")
        print("   Set: export HF_TOKEN=your_token_here")
        print("   Or use: --hf-token your_token_here")
        print("   Or configure in crossword_config.txt: DEFAULT_HF_TOKEN=your_token")
        return False

    try:
        # Initialize puzzle builder
        builder = PuzzleBuilder(hf_token=hf_token)

        # Check if language is supported
        supported_languages = builder.get_supported_languages()
        if args.language not in supported_languages:
            print(f"❌ Error: Language '{args.language}' not supported")
            print(f"   Available languages: {', '.join(supported_languages)}")
            return False

        print("📋 Configuration:")
        print(f"   Language: {args.language}")
        print(f"   Grid sizes: {args.grid_sizes}")
        print(f"   Puzzles per size: {args.count}")
        print(
            f"   Max corpus size: {args.corpus_limit if args.corpus_limit else 'All available'}"
        )
        print(f"   Upload to HuggingFace: {'Yes' if args.upload_to_hf else 'No'}")
        if args.output_dir:
            print(f"   Output directory: {args.output_dir}")

        # Validate corpus for each grid size
        print("\n🔍 Validating corpus for grid sizes...")
        for grid_size in args.grid_sizes:
            validation = builder.validate_corpus_for_grid_size(args.language, grid_size)
            if not validation["valid"]:
                print(
                    f"⚠️  Warning for {grid_size}x{grid_size}: {validation.get('error', validation['recommendation'])}"
                )
            else:
                print(f"✅ {grid_size}x{grid_size}: {validation['recommendation']}")

        print("\n🔄 Generating puzzles...")
        logging.info(
            f"GENERATION_START: Language={args.language}, GridSizes={args.grid_sizes}"
        )

        all_generated = {}
        total_generated = 0

        for grid_size in args.grid_sizes:
            print(
                f"\n📐 Generating {args.count} puzzles for {grid_size}x{grid_size} grid..."
            )

            puzzles = builder.generate_puzzle_batch(
                language=args.language,
                grid_size=grid_size,
                count=args.count,
                max_corpus_size=args.corpus_limit,
                upload_to_hf=args.upload_to_hf,
                output_dir=args.output_dir,
            )

            all_generated[grid_size] = puzzles
            total_generated += len(puzzles)

            print(
                f"   ✅ Generated {len(puzzles)}/{args.count} puzzles for {grid_size}x{grid_size}"
            )

            # Show sample puzzle info
            if puzzles:
                sample = puzzles[0]
                print(f"   Sample: {sample.id}")
                print(f"   Density: {sample.density:.1%}, Words: {sample.word_count}")
                print(
                    f"   Context score: {sample.context_score:.3f}, Quality: {sample.quality_score:.3f}"
                )

        # Show generation statistics
        stats = builder.get_generation_statistics()
        print("\n📊 GENERATION STATISTICS")
        print("=" * 60)
        print(f"Total attempts: {stats['total_attempts']}")
        print(f"Successful generations: {stats['successful_generations']}")
        print(f"Failed generations: {stats['failed_generations']}")
        print(f"Success rate: {stats['success_rate']}")
        print(f"Average density: {stats['average_density']}")
        print(f"Average word count: {stats['average_word_count']}")

        if stats["source_distribution"]:
            print("\nSource distribution:")
            for source, count in stats["source_distribution"].items():
                print(f"  • {source}: {count:.2f}% average usage")

        if args.upload_to_hf:
            print("\n🌐 HUGGINGFACE UPLOAD")
            print("=" * 60)
            print(f"✅ Uploaded {total_generated} puzzles to HuggingFace")
            print("   Puzzles available in generated puzzles repository")

        if args.output_dir:
            print("\n📁 LOCAL OUTPUT")
            print("=" * 60)
            print(f"✅ Saved {total_generated} puzzles to {args.output_dir}")
            print("   JSON files created for each puzzle")

        print("\n🎉 PUZZLE GENERATION COMPLETE!")
        print("=" * 60)
        print(f"✅ Generated {total_generated} total puzzles")

        # Clear caches to free memory
        builder.clear_caches()

        logging.info(f"GENERATION_COMPLETE: {total_generated} puzzles generated")
        return True

    except Exception as e:
        print(f"❌ Puzzle generation failed: {e}")
        logging.error(f"Generation error: {e}", exc_info=True)
        return False


def run_dataset_management(args) -> bool:
    """Handle dataset management operations (list, delete, inspect, backup)."""
    try:
        # Get HF token from args, environment, or config
        hf_token = (
            args.hf_token
            or os.getenv("HF_TOKEN")
            or get_config().get("DEFAULT_HF_TOKEN")
        )

        if not hf_token:
            print("❌ Error: HuggingFace token required for dataset management")
            print(
                "   Set --hf-token argument, HF_TOKEN env var, or DEFAULT_HF_TOKEN in config"
            )
            return False

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.info("Verbose mode enabled for dataset management")

        # Initialize dataset manager
        manager = DatasetManager(hf_token=hf_token)
        repo_id = args.repo_id or manager.default_repo

        print(f"🗄️  Dataset Management for: {repo_id}")
        print(f"   Operation: {args.operation}")

        if args.operation == "list":
            print(f"\n📋 Listing configurations in {repo_id}...")
            configs = manager.list_dataset_configurations(repo_id)
            if configs:
                print(f"Found {len(configs)} configurations:")
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
            else:
                print("No configurations found")

        elif args.operation == "inspect":
            print("\n🔍 Inspecting dataset structure...")
            manager.print_repository_overview(repo_id)

        elif args.operation == "delete":
            if not args.config_name:
                print("❌ Error: --config-name required for delete operation")
                return False

            print(f"\n🗑️  Deleting configuration: {args.config_name}")
            success = manager.delete_configuration(
                config_name=args.config_name,
                repo_id=repo_id,
                create_backup=not args.no_backup,
            )

            if success:
                print(f"✅ Successfully deleted '{args.config_name}' configuration")
                return True
            else:
                print(f"❌ Failed to delete '{args.config_name}' configuration")
                return False

        elif args.operation == "backup":
            if not args.config_name:
                print("❌ Error: --config-name required for backup operation")
                return False

            print(f"\n💾 Creating backup for: {args.config_name}")
            backup_path = manager.backup_configuration(
                config_name=args.config_name,
                repo_id=repo_id,
                backup_dir=args.backup_dir,
            )
            print(f"✅ Backup created at: {backup_path}")

        elif args.operation == "restore":
            if not args.backup_path:
                print("❌ Error: --backup-path required for restore operation")
                return False

            print(f"\n🔄 Restoring from backup: {args.backup_path}")
            success = manager.restore_configuration(
                backup_path=args.backup_path, repo_id=repo_id
            )

            if success:
                print("✅ Configuration restored successfully")
                return True
            else:
                print("❌ Failed to restore configuration")
                return False

        elif args.operation == "stats":
            if not args.config_name:
                print("❌ Error: --config-name required for stats operation")
                return False

            print(f"\n📊 Getting statistics for: {args.config_name}")
            stats = manager.get_configuration_stats(args.config_name, repo_id)

            if "error" not in stats:
                print(f"Configuration: {stats['config_name']}")
                print(f"Total entries: {stats['total_entries']}")
                print(f"Columns: {', '.join(stats['columns'])}")
                if stats["sources"]:
                    print("Sources distribution:")
                    for source, count in stats["sources"].items():
                        print(f"  • {source}: {count} entries")
                if stats["quality_stats"]["mean_quality"]:
                    print(
                        f"Quality score: {stats['quality_stats']['mean_quality']:.3f} (avg)"
                    )
            else:
                print(f"❌ Error getting stats: {stats['error']}")
                return False

        else:
            print(f"❌ Unknown operation: {args.operation}")
            return False

        logging.info(
            f"DATASET_MANAGEMENT_COMPLETE: {args.operation} operation finished"
        )
        return True

    except Exception as e:
        print(f"❌ Dataset management failed: {e}")
        logging.error(f"Dataset management error: {e}", exc_info=True)
        return False



def main():
    """Main CLI entry point."""
    # Get configuration defaults for argparse
    config_defaults = get_config_defaults()

    parser = argparse.ArgumentParser(
        description="Buddhi-Pragati Crossword Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config defaults (minimal commands)
  %(prog)s generate
  %(prog)s evaluate --puzzle-dir generated_crosswords/hindi/10x10

  # Enhanced evaluation from HuggingFace datasets
  %(prog)s evaluate --languages Hindi English --grid-sizes 7 15 --count 20
  %(prog)s evaluate --languages Bengali --models gpt-4o claude-sonnet-4 --chain-of-thought
  %(prog)s evaluate --languages Tamil --reasoning-effort high --self-reflection

  # Override specific parameters
  %(prog)s generate --language Hindi --grid-sizes 8 10 12 --count 10
  %(prog)s evaluate --puzzle-dir generated_crosswords/hindi/10x10 --model gpt-4o
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate LLM on crossword puzzles"
    )
    eval_parser.add_argument(
        "--puzzle-dir",
        help="Directory containing puzzle JSON files (alternative to --languages)",
    )

    # Enhanced evaluation parameters for HuggingFace dataset loading
    eval_parser.add_argument(
        "--languages",
        nargs="+",
        choices=[
            "Assamese",
            "Bengali",
            "Bodo",
            "English",
            "Gujarati",
            "Hindi",
            "Kannada",
            "Kashmiri",
            "Konkani",
            "Malayalam",
            "Marathi",
            "Meitei",
            "Nepali",
            "Odia",
            "Punjabi",
            "Sanskrit",
            "Tamil",
            "Telugu",
            "Urdu",
        ],
        help="Languages to evaluate (loads from HuggingFace dataset)",
    )
    eval_parser.add_argument(
        "--grid-sizes", nargs="+", type=int, help="Grid sizes to filter (e.g., 7 15 25)"
    )
    eval_parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of puzzles to evaluate per configuration (default: 10)",
    )

    # Model configuration
    eval_parser.add_argument(
        "--model",
        default=config_defaults["model"],
        help=f"Model to evaluate (default: {config_defaults['model']})",
    )
    eval_parser.add_argument(
        "--models", nargs="+", help="Multiple models to test (overrides --model)"
    )
    eval_parser.add_argument(
        "--model-source",
        default=config_defaults["model_source"],
        choices=["openai", "anthropic", "openrouter", "huggingface", "sarvamai"],
        help=f"Model provider (default: {config_defaults['model_source']})",
    )
    eval_parser.add_argument(
        "--api-key",
        help="API key for model provider (fallback: environment variables or config defaults)",
    )

    # Prompt configuration flags
    eval_parser.add_argument(
        "--chain-of-thought",
        action="store_true",
        help="Enable chain-of-thought prompting",
    )
    eval_parser.add_argument(
        "--reasoning-effort",
        choices=["low", "normal", "high"],
        help="Reasoning effort level for compatible models",
    )
    eval_parser.add_argument(
        "--self-reflection",
        action="store_true",
        help="Enable self-reflection evaluation with error correction",
    )

    # Model filtering parameters
    eval_parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Only evaluate priority models",
    )
    eval_parser.add_argument(
        "--reasoning-only",
        action="store_true",
        help="Only evaluate reasoning-capable models",
    )
    eval_parser.add_argument(
        "--indic-only",
        action="store_true",
        help="Only evaluate Indic fine-tuned models",
    )

    # Experimental mode parameters
    eval_parser.add_argument(
        "--experiment-mode",
        action="store_true",
        help="Enable experimental evaluation mode",
    )
    eval_parser.add_argument(
        "--experiment-types",
        nargs="+",
        choices=[
            "0_master",
            "2_shot_variations",
            "3_batch_sizes",
            "4_chain_of_thought",
            "5_reasoning_effort",
            "6_self_reflection",
            "7_language_variants",
            "1_language_families",
            "8_model_types",
            "9_reasoning_models",
            "10_performance_normalization"
        ],
        help="Experiment types to run in experimental mode",
    )
    eval_parser.add_argument(
        "--output-dir",
        help="Directory for experimental results",
    )
    eval_parser.add_argument(
        "--master-results-path",
        help="Path to master experiment results (required for analysis experiments 1,8-10)",
    )
    eval_parser.add_argument(
        "--suppress-output",
        action="store_true",
        help="Suppress console output (for experimental mode)",
    )

    # Legacy and general parameters
    eval_parser.add_argument(
        "--max-puzzles",
        type=int,
        default=50,
        help="Maximum puzzles to evaluate (legacy mode)",
    )

    # Create dataset command
    dataset_parser = subparsers.add_parser(
        "create-dataset", help="Create crossword datasets from multiple sources"
    )
    dataset_parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        choices=[
            "Assamese",
            "Bengali",
            "Bodo",
            "English",
            "Gujarati",
            "Hindi",
            "Kannada",
            "Kashmiri",
            "Konkani",
            "Malayalam",
            "Marathi",
            "Meitei",
            "Nepali",
            "Odia",
            "Punjabi",
            "Sanskrit",
            "Tamil",
            "Telugu",
            "Urdu",
        ],
        help="Languages to create datasets for",
    )
    dataset_parser.add_argument(
        "--target-size",
        type=int,
        default=config_defaults["target_size"],
        help=f"Target number of entries per language (default: {config_defaults['target_size']})",
    )
    dataset_parser.add_argument(
        "--sources",
        nargs="+",
        default=config_defaults["sources"],
        choices=["MILU", "IndicWikiBio", "IndoWordNet", "Bhasha-Wiki"],
        help=f"Data sources to use (default: {config_defaults['sources']})",
    )
    dataset_parser.add_argument(
        "--output-path", help="Local directory to save datasets (optional)"
    )
    dataset_parser.add_argument(
        "--upload", action="store_true", help="Upload datasets to HuggingFace Hub"
    )
    dataset_parser.add_argument(
        "--mode",
        choices=["fast", "complete"],
        default=config_defaults["context_scoring_mode"],
        help=f"Context scoring mode: 'fast' (simple India similarity) or 'complete' (full cultural corpus) (default: {config_defaults['context_scoring_mode']})",
    )
    dataset_parser.add_argument(
        "--hf-token",
        help="HuggingFace token (fallback: HF_TOKEN env var or config default)",
    )
    dataset_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging for debugging"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate crossword puzzles from corpus dataset"
    )
    generate_parser.add_argument(
        "--language",
        default=config_defaults["language"],
        choices=[
            "Assamese",
            "Bengali",
            "Bodo",
            "English",
            "Gujarati",
            "Hindi",
            "Kannada",
            "Kashmiri",
            "Konkani",
            "Malayalam",
            "Marathi",
            "Meitei",
            "Nepali",
            "Odia",
            "Punjabi",
            "Sanskrit",
            "Tamil",
            "Telugu",
            "Urdu",
        ],
        help=f"Language to generate puzzles for (default: {config_defaults['language']})",
    )
    generate_parser.add_argument(
        "--grid-sizes",
        nargs="+",
        type=int,
        default=[10],
        help="Grid sizes to generate (3-30, default: [10])",
    )
    generate_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of puzzles to generate per grid size (default: 5)",
    )
    generate_parser.add_argument(
        "--corpus-limit",
        type=int,
        help="Maximum corpus entries to load (None for all available)",
    )
    generate_parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload generated puzzles to HuggingFace Hub",
    )
    generate_parser.add_argument(
        "--output-dir", help="Directory to save puzzle JSON files (optional)"
    )
    generate_parser.add_argument(
        "--hf-token",
        help="HuggingFace token (fallback: HF_TOKEN env var or config default)",
    )
    generate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging for debugging"
    )

    # Dataset management command
    manage_parser = subparsers.add_parser(
        "manage-dataset",
        help="Manage HuggingFace datasets (list, delete, inspect, backup)",
    )
    manage_parser.add_argument(
        "operation",
        choices=["list", "inspect", "delete", "backup", "restore", "stats"],
        help="Operation to perform",
    )
    manage_parser.add_argument(
        "--repo-id",
        default=config_defaults["repo_id"],
        help=f"HuggingFace repository ID (default: {config_defaults['repo_id']})",
    )
    manage_parser.add_argument(
        "--config-name",
        help="Configuration name (language subset) for delete/backup/stats operations",
    )
    manage_parser.add_argument(
        "--backup-dir",
        default="./dataset_backups",
        help="Directory for backups (default: ./dataset_backups)",
    )
    manage_parser.add_argument(
        "--backup-path", help="Path to backup directory for restore operation"
    )
    manage_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation during delete operation",
    )

    manage_parser.add_argument(
        "--hf-token",
        help="HuggingFace token (fallback: HF_TOKEN env var or config default)",
    )
    manage_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging for debugging"
    )


    args = parser.parse_args()

    # Apply configuration defaults to arguments
    args = apply_config_defaults(args)

    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/crossword_benchmark_{timestamp}.log"

    # Setup logging with file output
    actual_log_file = setup_logging(args.verbose, log_file)

    if actual_log_file:
        print(f"📝 Detailed trace logging to: {actual_log_file}")
        print(f"🔍 Monitor progress: tail -f {actual_log_file}")

    logging.info(f"COMMAND: {' '.join(sys.argv)}")
    logging.info(f"ARGUMENTS: {vars(args)}")

    if not args.command:
        parser.print_help()
        return False

    # Route to appropriate function
    try:
        if args.command == "generate":
            return run_crossword_generation(args)
        elif args.command == "evaluate":
            if hasattr(args, "experiment_mode") and args.experiment_mode:
                # Use experimental evaluation mode
                from buddhi_pragati.evaluate.experiment_runner import NewExperimentRunner
                experiment_runner = NewExperimentRunner(args.output_dir)

                # For now, run a basic experiment if experiment types not specified
                if hasattr(args, "experiment_types") and args.experiment_types:
                    results = {}
                    for experiment_id in args.experiment_types:
                        if experiment_id == "0_master":
                            result = experiment_runner.run_experiment_0_master()
                        elif experiment_id == "2_shot_variations":
                            result = experiment_runner.run_experiment_2_shot_variations()
                        elif experiment_id == "4_chain_of_thought":
                            result = experiment_runner.run_experiment_4_chain_of_thought()
                        else:
                            print(f"❌ Experiment {experiment_id} not yet migrated to unified system")
                            continue
                        results[experiment_id] = result
                    return len(results) > 0
                else:
                    print("❌ No experiment types specified for experimental mode")
                    return False
            else:
                # Standard evaluation mode
                result = run_evaluation(args)
                return result.get("success", False) if isinstance(result, dict) else result
        elif args.command == "create-dataset":
            return run_create_dataset(args)
        elif args.command == "manage-dataset":
            return run_dataset_management(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return False

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        logging.info(f"BENCHMARK_END: {datetime.now().isoformat()} - FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
