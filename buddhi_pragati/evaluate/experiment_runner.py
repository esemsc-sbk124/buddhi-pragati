"""
New experiment runner for the redesigned experimental framework.

This module implements the new experiment structure:
- Experiment 0: Master experiment (all models x all languages (for each model, the set of all languages supported varies) x all grid sizes x all puzzles in each one of these languages)
- Experiments 2-7: Focused parameter experiments using priority subsets
- Results analysis methods: Language families, model types, reasoning analysis

The system uses priority settings from configuration to limit scope for focused experiments
while maintaining comprehensive coverage in the master experiment.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from .evaluator import CrosswordEvaluator
from .dataset_loader import PuzzleDatasetLoader
from .model_classifier import ModelClassifier
from .templates import CrosswordPromptTemplate
from ..models.model_interface import UnifiedModelInterface
from ..utils.config_loader import get_config

# Import run_evaluation function for unified evaluation
import sys
# Add parent directory to sys.path to import from main file
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from run_crossword_benchmark import run_evaluation


class NewExperimentRunner:
    """
    Redesigned experiment runner for the new experimental framework.

    Implements 8 experiments:
    - Experiment 0: Master (comprehensive)
    - Experiments 2-7: Focused (single parameter variation)
    - Analysis methods: Results presentation from master experiment
    """

    def __init__(self, output_dir: str = None):
        self.logger = logging.getLogger("NewExperimentRunner")
        self.config = get_config()

        # Initialize components
        self.evaluator = CrosswordEvaluator()
        self.dataset_loader = PuzzleDatasetLoader()
        self.model_classifier = ModelClassifier()

        # Output directory
        output_dir = output_dir or self.config.get(
            "EXPERIMENT_RESULTS_DIR", "buddhi_pragati/experiments"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load priority settings from config
        self._load_priority_settings()

        # Load default prompt configuration
        self._load_default_prompt_config()

        # Load reasoning tokens configuration
        self.reasoning_tokens = {
            "low": int(self.config.get("LOW_REASONING_TOKENS", "500")),
            "normal": int(self.config.get("NORMAL_REASONING_TOKENS", "1000")),
            "high": int(self.config.get("HIGH_REASONING_TOKENS", "2000")),
        }

    def _load_priority_settings(self):
        """Load priority settings from configuration."""
        self.priority_grid_sizes = [
            int(size.strip())
            for size in self.config.get(
                "DEFAULT_PRIORITARY_GRID_SIZES", "7,15,25"
            ).split(",")
        ]

        self.priority_languages = [
            lang.strip()
            for lang in self.config.get(
                "DEFAULT_PRIORITARY_LANGUAGES",
                "Bengali,English,Gujarati,Hindi,Kannada,Malayalam,Odia,Tamil,Telugu,Urdu",
            ).split(",")
        ]

        self.priority_models = [
            model.strip()
            for model in self.config.get(
                "DEFAULT_PRIORITARY_MODELS", "gpt-4o,claude-sonnet-4"
            ).split(",")
        ]

        self.logger.info(
            f"Priority settings loaded: {len(self.priority_models)} models, "
            f"{len(self.priority_languages)} languages, {len(self.priority_grid_sizes)} grid sizes"
        )

    def _load_default_prompt_config(self):
        """Load default prompting configuration."""
        self.default_config = {
            "shot_type": self.config.get("DEFAULT_SHOT_TYPE", "zero-shot"),
            "batch_size": int(self.config.get("DEFAULT_BATCH_SIZE_EVALUATION", "1")),
            "chain_of_thought": self.config.get_bool(
                "DEFAULT_ENABLE_CHAIN_OF_THOUGHT", False
            ),
            "reasoning_effort": self.config.get(
                "DEFAULT_REASONING_EFFORT_LEVEL", "normal"
            ),
            "self_reflection": self.config.get_bool(
                "DEFAULT_ENABLE_SELF_REFLECTION", False
            ),
        }

    def _get_all_models(self) -> List[str]:
        """Get all available models from model classifier, including OpenRouter models."""
        all_models = self.model_classifier.list_all_models()
        self.logger.info(f"Including all {len(all_models)} models (including OpenRouter)")
        return all_models

    def _get_all_languages(self) -> List[str]:
        """Get all supported languages (from priority + others)."""
        # For now, use priority languages as the comprehensive set
        # This can be expanded based on actual dataset availability
        return self.dataset_loader.get_available_languages()

    def _get_all_grid_sizes(self) -> List[int]:
        """Get all available grid sizes."""
        # Can be expanded based on dataset availability
        return self.config.get_list_of_ints("ALL_GRID_SIZES", "7,10,15,20,25")

    def _save_experiment_results(
        self, experiment_id: str, results: Dict[str, Any]
    ) -> str:
        """Save experiment results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{experiment_id}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results saved to {filepath}")
        return str(filepath)

    def _run_evaluation_with_config(
        self,
        language: str,
        grid_size: int,
        models: List[str],
        max_puzzles: int = None,
        **experiment_config
    ) -> Dict[str, Any]:
        """
        Run evaluation using the unified run_evaluation function.

        Args:
            language: Target language for evaluation
            grid_size: Grid size to evaluate
            models: List of model names to test
            max_puzzles: Maximum puzzles to evaluate
            **experiment_config: Additional experimental configuration (chain_of_thought, etc.)

        Returns:
            Dict with evaluation results from run_evaluation
        """
        # Create argparse-like object for run_evaluation
        class EvalArgs:
            def __init__(self):
                self.languages = [language]
                self.grid_sizes = [grid_size] if grid_size else None
                self.models = models
                self.count = max_puzzles
                self.suppress_output = True
                # model_source will be auto-detected per model in run_evaluation

                # Apply experiment configuration
                for key, value in experiment_config.items():
                    setattr(self, key, value)

                # Set default values for required attributes
                if not hasattr(self, 'chain_of_thought'):
                    self.chain_of_thought = experiment_config.get('chain_of_thought', False)
                if not hasattr(self, 'reasoning_effort'):
                    self.reasoning_effort = experiment_config.get('reasoning_effort', 'normal')
                if not hasattr(self, 'self_reflection'):
                    self.self_reflection = experiment_config.get('self_reflection', False)

        args = EvalArgs()

        # Filter models to only those that support the language
        language_models = self.model_classifier.get_models_for_language(language)
        language_model_names = {m.name for m in language_models}
        filtered_models = [m for m in models if m in language_model_names]

        if not filtered_models:
            self.logger.warning(f"No models support language {language}, skipping")
            return {"success": False, "error": f"No models support language {language}"}

        args.models = filtered_models

        # Call unified evaluation function
        result = run_evaluation(args)

        return result

    def run_experiment_0_master(self) -> Dict[str, Any]:
        """
        Experiment 0: Master experiment - comprehensive evaluation.

        Tests ALL models against ALL languages and ALL grid sizes using
        the default prompting configuration. This serves as the baseline
        for all other experiments and analysis.
        """
        self.logger.info("Starting Experiment 0: Master Experiment")

        all_models = self._get_all_models()
        all_languages = self._get_all_languages()
        all_grid_sizes = self._get_all_grid_sizes()

        results = {
            "experiment_id": "0_master",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "scope": "comprehensive",
                "models": all_models,
                "languages": all_languages,
                "grid_sizes": all_grid_sizes,
                "prompt_config": self.default_config,
            },
            "results": {},
        }

        total_combinations = len(all_models) * len(all_languages) * len(all_grid_sizes)
        self.logger.info(f"Master experiment: {total_combinations} total combinations")

        completed = 0

        for language in all_languages:
            for grid_size in all_grid_sizes:
                try:
                    # Use unified evaluation for all models supporting this language
                    eval_result = self._run_evaluation_with_config(
                        language=language,
                        grid_size=grid_size,
                        models=all_models,
                        max_puzzles=None,
                        **self.default_config
                    )

                    if eval_result.get("success", False):
                        # Distribute results to each model
                        for model_name, model_result in eval_result.get("results", {}).items():
                            if model_name not in results["results"]:
                                results["results"][model_name] = {}
                            if language not in results["results"][model_name]:
                                results["results"][model_name][language] = {}
                            results["results"][model_name][language][f"grid_{grid_size}"] = model_result
                    else:
                        # Handle evaluation failure - mark all models as failed for this combination
                        error_result = {"error": eval_result.get("error", "Evaluation failed")}
                        for model_name in all_models:
                            if model_name not in results["results"]:
                                results["results"][model_name] = {}
                            if language not in results["results"][model_name]:
                                results["results"][model_name][language] = {}
                            results["results"][model_name][language][f"grid_{grid_size}"] = error_result

                    completed += 1
                    progress = (completed / total_combinations) * len(all_models)
                    self.logger.info(
                        f"Progress: {progress:.1f}% ({completed}/{len(all_languages) * len(all_grid_sizes)})"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Error evaluating {language}/grid_{grid_size}: {e}"
                    )
                    error_result = {"error": str(e)}
                    for model_name in all_models:
                        if model_name not in results["results"]:
                            results["results"][model_name] = {}
                        if language not in results["results"][model_name]:
                            results["results"][model_name][language] = {}
                        results["results"][model_name][language][f"grid_{grid_size}"] = error_result

        # Save results
        filepath = self._save_experiment_results("0_master", results)
        results["filepath"] = filepath

        return results

    def run_experiment_2_shot_variations(self) -> Dict[str, Any]:
        """
        Experiment 2: Shot variations (0-shot, 1-shot, few-shot).

        Uses priority subset with default configuration except for shot type.
        """
        self.logger.info("Starting Experiment 2: Shot Variations")

        shot_types = ["zero-shot", "one-shot", "few-shot"]

        results = {
            "experiment_id": "2_shot_variations",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "scope": "priority_subset",
                "models": self.priority_models,
                "languages": self.priority_languages,
                "grid_sizes": self.priority_grid_sizes,
                "parameter": "shot_type",
                "variations": shot_types,
            },
            "results": {},
        }

        for shot_type in shot_types:
            self.logger.info(f"Testing shot type: {shot_type}")
            shot_results = {}

            for language in self.priority_languages:
                for grid_size in self.priority_grid_sizes:
                    try:
                        # Create configuration with modified shot type
                        config = self.default_config.copy()
                        config["shot_type"] = shot_type

                        # Use unified evaluation for all priority models supporting this language
                        eval_result = self._run_evaluation_with_config(
                            language=language,
                            grid_size=grid_size,
                            models=self.priority_models,
                            max_puzzles=20,
                            **config
                        )

                        if eval_result.get("success", False):
                            # Distribute results to each model
                            for model_name, model_result in eval_result.get("results", {}).items():
                                if model_name not in shot_results:
                                    shot_results[model_name] = {}
                                if language not in shot_results[model_name]:
                                    shot_results[model_name][language] = {}
                                shot_results[model_name][language][f"grid_{grid_size}"] = model_result
                        else:
                            # Handle evaluation failure
                            error_result = {"error": eval_result.get("error", "Evaluation failed")}
                            for model_name in self.priority_models:
                                if model_name not in shot_results:
                                    shot_results[model_name] = {}
                                if language not in shot_results[model_name]:
                                    shot_results[model_name][language] = {}
                                shot_results[model_name][language][f"grid_{grid_size}"] = error_result

                    except Exception as e:
                        self.logger.error(
                            f"Error in shot variation {shot_type}: {e}"
                        )
                        error_result = {"error": str(e)}
                        for model_name in self.priority_models:
                            if model_name not in shot_results:
                                shot_results[model_name] = {}
                            if language not in shot_results[model_name]:
                                shot_results[model_name][language] = {}
                            shot_results[model_name][language][f"grid_{grid_size}"] = error_result

            results["results"][shot_type] = shot_results

        filepath = self._save_experiment_results("2_shot_variations", results)
        results["filepath"] = filepath

        return results

    def run_experiment_3_batch_sizes(self) -> Dict[str, Any]:
        """
        Experiment 3: Batch sizes (1 puzzle, 10 puzzles, all puzzles).

        Uses priority subset with default configuration except for batch size.
        """
        self.logger.info("Starting Experiment 3: Batch Sizes")

        batch_sizes = [1, 10, "all"]

        results = {
            "experiment_id": "3_batch_sizes",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "scope": "priority_subset",
                "models": self.priority_models,
                "languages": self.priority_languages,
                "grid_sizes": self.priority_grid_sizes,
                "parameter": "batch_size",
                "variations": batch_sizes,
            },
            "results": {},
        }

        for batch_size in batch_sizes:
            self.logger.info(f"Testing batch size: {batch_size}")
            batch_results = {}

            for model_name in self.priority_models:
                model_results = {}

                for language in self.priority_languages:
                    language_results = {}

                    for grid_size in self.priority_grid_sizes:
                        try:
                            # Load appropriate number of puzzles
                            if batch_size == "all":
                                puzzles = self.dataset_loader.load_puzzles(
                                    language, grid_size, max_puzzles=None
                                )
                            else:
                                puzzles = self.dataset_loader.load_puzzles(
                                    language, grid_size, max_puzzles=max(batch_size, 20)
                                )

                            if not puzzles:
                                language_results[f"grid_{grid_size}"] = {
                                    "error": "No puzzles available"
                                }
                                continue

                            # Create configuration with modified batch size
                            config = self.default_config.copy()
                            config["batch_size"] = (
                                len(puzzles) if batch_size == "all" else batch_size
                            )

                            model = UnifiedModelInterface(model_name)
                            template = CrosswordPromptTemplate(language)
                            template.configure_experiment(**config)

                            if batch_size == 1:
                                # Individual puzzle evaluation
                                individual_results = []
                                for puzzle in puzzles[:20]:  # Limit for comparison
                                    result = self.evaluator.evaluate_single(
                                        model, puzzle, template
                                    )
                                    individual_results.append(result)

                                batch_result = {
                                    "evaluation_mode": "individual",
                                    "individual_results": individual_results,
                                    "summary_metrics": self.evaluator._calculate_summary(
                                        individual_results
                                    ),
                                }
                            else:
                                # Batch evaluation
                                batch_result = self.evaluator.evaluate_batch(
                                    model, puzzles, template
                                )
                                batch_result["evaluation_mode"] = "batch"

                            language_results[f"grid_{grid_size}"] = batch_result

                        except Exception as e:
                            self.logger.error(f"Error in batch size {batch_size}: {e}")
                            language_results[f"grid_{grid_size}"] = {"error": str(e)}

                    model_results[language] = language_results

                batch_results[model_name] = model_results

            results["results"][f"batch_{batch_size}"] = batch_results

        filepath = self._save_experiment_results("3_batch_sizes", results)
        results["filepath"] = filepath

        return results

    def run_experiment_4_chain_of_thought(self) -> Dict[str, Any]:
        """
        Experiment 4: Chain of thought prompting (enabled/disabled).

        Uses priority subset with default configuration except for CoT.
        """
        self.logger.info("Starting Experiment 4: Chain of Thought")

        cot_settings = [False, True]

        results = {
            "experiment_id": "4_chain_of_thought",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "scope": "priority_subset",
                "models": self.priority_models,
                "languages": self.priority_languages,
                "grid_sizes": self.priority_grid_sizes,
                "parameter": "chain_of_thought",
                "variations": cot_settings,
            },
            "results": {},
        }

        for cot_enabled in cot_settings:
            cot_label = "chain_of_thought" if cot_enabled else "direct"
            self.logger.info(f"Testing CoT setting: {cot_label}")
            cot_results = {}

            for language in self.priority_languages:
                for grid_size in self.priority_grid_sizes:
                    try:
                        # Create configuration with modified CoT setting
                        config = self.default_config.copy()
                        config["chain_of_thought"] = cot_enabled

                        # Use unified evaluation for all priority models supporting this language
                        eval_result = self._run_evaluation_with_config(
                            language=language,
                            grid_size=grid_size,
                            models=self.priority_models,
                            max_puzzles=20,
                            **config
                        )

                        if eval_result.get("success", False):
                            # Distribute results to each model
                            for model_name, model_result in eval_result.get("results", {}).items():
                                if model_name not in cot_results:
                                    cot_results[model_name] = {}
                                if language not in cot_results[model_name]:
                                    cot_results[model_name][language] = {}
                                cot_results[model_name][language][f"grid_{grid_size}"] = model_result
                        else:
                            # Handle evaluation failure
                            error_result = {"error": eval_result.get("error", "Evaluation failed")}
                            for model_name in self.priority_models:
                                if model_name not in cot_results:
                                    cot_results[model_name] = {}
                                if language not in cot_results[model_name]:
                                    cot_results[model_name][language] = {}
                                cot_results[model_name][language][f"grid_{grid_size}"] = error_result

                    except Exception as e:
                        self.logger.error(f"Error in CoT {cot_label}: {e}")
                        error_result = {"error": str(e)}
                        for model_name in self.priority_models:
                            if model_name not in cot_results:
                                cot_results[model_name] = {}
                            if language not in cot_results[model_name]:
                                cot_results[model_name][language] = {}
                            cot_results[model_name][language][f"grid_{grid_size}"] = error_result

            results["results"][cot_label] = cot_results

        filepath = self._save_experiment_results("4_chain_of_thought", results)
        results["filepath"] = filepath

        return results

    def run_experiment_5_reasoning_effort(self) -> Dict[str, Any]:
        """
        Experiment 5: Reasoning effort levels (low/normal/high).

        Uses priority subset with default configuration except for reasoning effort.
        """
        self.logger.info("Starting Experiment 5: Reasoning Effort")

        effort_levels = ["low", "normal", "high"]

        results = {
            "experiment_id": "5_reasoning_effort",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "scope": "priority_subset",
                "models": self.priority_models,
                "languages": self.priority_languages,
                "grid_sizes": self.priority_grid_sizes,
                "parameter": "reasoning_effort",
                "variations": effort_levels,
                "token_limits": self.reasoning_tokens,
            },
            "results": {},
        }

        for effort_level in effort_levels:
            self.logger.info(f"Testing reasoning effort: {effort_level}")
            effort_results = {}

            for model_name in self.priority_models:
                model_results = {}

                for language in self.priority_languages:
                    language_results = {}

                    for grid_size in self.priority_grid_sizes:
                        try:
                            puzzles = self.dataset_loader.load_puzzles(
                                language, grid_size, max_puzzles=20
                            )

                            if not puzzles:
                                language_results[f"grid_{grid_size}"] = {
                                    "error": "No puzzles available"
                                }
                                continue

                            # Create configuration with modified reasoning effort
                            config = self.default_config.copy()
                            config["reasoning_effort"] = effort_level

                            # Create model with token limit and reasoning mode based on effort
                            max_tokens = self.reasoning_tokens[effort_level]
                            reasoning_mode = effort_level in ["normal", "high"]

                            model = UnifiedModelInterface(
                                model_name,
                                max_tokens=max_tokens,
                                reasoning_mode=reasoning_mode,
                                reasoning_effort=effort_level,
                            )

                            template = CrosswordPromptTemplate(language)
                            template.configure_experiment(**config)

                            batch_result = self.evaluator.evaluate_batch(
                                model, puzzles, template
                            )
                            language_results[f"grid_{grid_size}"] = batch_result

                        except Exception as e:
                            self.logger.error(
                                f"Error in reasoning effort {effort_level}: {e}"
                            )
                            language_results[f"grid_{grid_size}"] = {"error": str(e)}

                    model_results[language] = language_results

                effort_results[model_name] = model_results

            results["results"][effort_level] = effort_results

        filepath = self._save_experiment_results("5_reasoning_effort", results)
        results["filepath"] = filepath

        return results

    def run_experiment_6_self_reflection(self) -> Dict[str, Any]:
        """
        Experiment 6: Self reflection capabilities (enabled/disabled).

        Uses priority subset with default configuration except for self reflection.
        """
        self.logger.info("Starting Experiment 6: Self Reflection")

        reflection_settings = [False, True]

        results = {
            "experiment_id": "6_self_reflection",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "scope": "priority_subset",
                "models": self.priority_models,
                "languages": self.priority_languages,
                "grid_sizes": self.priority_grid_sizes,
                "parameter": "self_reflection",
                "variations": reflection_settings,
            },
            "results": {},
        }

        for reflection_enabled in reflection_settings:
            reflection_label = "self_reflection" if reflection_enabled else "direct"
            self.logger.info(f"Testing self reflection: {reflection_label}")
            reflection_results = {}

            for model_name in self.priority_models:
                model_results = {}

                for language in self.priority_languages:
                    language_results = {}

                    for grid_size in self.priority_grid_sizes:
                        try:
                            puzzles = self.dataset_loader.load_puzzles(
                                language, grid_size, max_puzzles=10
                            )  # Smaller for reflection

                            if not puzzles:
                                language_results[f"grid_{grid_size}"] = {
                                    "error": "No puzzles available"
                                }
                                continue

                            # Create configuration with modified self reflection setting
                            config = self.default_config.copy()
                            config["self_reflection"] = reflection_enabled

                            model = UnifiedModelInterface(model_name)
                            template = CrosswordPromptTemplate(language)
                            template.configure_experiment(**config)

                            if reflection_enabled:
                                # Use self-reflection evaluation method
                                batch_result = (
                                    self.evaluator.evaluate_batch_with_reflection(
                                        model, puzzles, template
                                    )
                                )
                            else:
                                # Standard evaluation
                                batch_result = self.evaluator.evaluate_batch(
                                    model, puzzles, template
                                )

                            language_results[f"grid_{grid_size}"] = batch_result

                        except Exception as e:
                            self.logger.error(
                                f"Error in self reflection {reflection_label}: {e}"
                            )
                            language_results[f"grid_{grid_size}"] = {"error": str(e)}

                    model_results[language] = language_results

                reflection_results[model_name] = model_results

            results["results"][reflection_label] = reflection_results

        filepath = self._save_experiment_results("6_self_reflection", results)
        results["filepath"] = filepath

        return results

    def run_experiment_7_language_variants(self) -> Dict[str, Any]:
        """
        Experiment 7: Language variants (same-language vs cross-language prompts).

        Uses priority subset with default configuration except for prompt language.
        """
        self.logger.info("Experiment dropped")

        # prompt_modes = ["same_language", "cross_language"]

        # results = {
        #     "experiment_id": "7_language_variants",
        #     "timestamp": datetime.now().isoformat(),
        #     "configuration": {
        #         "scope": "priority_subset",
        #         "models": self.priority_models,
        #         "languages": self.priority_languages,
        #         "grid_sizes": self.priority_grid_sizes,
        #         "parameter": "cross_language_prompting",
        #         "variations": prompt_modes,
        #     },
        #     "results": {},
        # }

        # for prompt_mode in prompt_modes:
        #     self.logger.info(f"Testing prompt mode: {prompt_mode}")
        #     mode_results = {}

        #     for model_name in self.priority_models:
        #         model_results = {}

        #         for language in self.priority_languages:
        #             language_results = {}

        #             for grid_size in self.priority_grid_sizes:
        #                 try:
        #                     puzzles = self.dataset_loader.load_puzzles(
        #                         language, grid_size, max_puzzles=20
        #                     )

        #                     if not puzzles:
        #                         language_results[f"grid_{grid_size}"] = {
        #                             "error": "No puzzles available"
        #                         }
        #                         continue

        #                     # Create configuration with modified prompt language setting
        #                     config = self.default_config.copy()
        #                     config["cross_language"] = prompt_mode == "cross_language"

        #                     model = UnifiedModelInterface(model_name)

        #                     # Set prompt language based on mode
        #                     if prompt_mode == "cross_language":
        #                         template = CrosswordPromptTemplate(
        #                             "English"
        #                         )  # English prompts for non-English puzzles
        #                     else:
        #                         template = CrosswordPromptTemplate(
        #                             language
        #                         )  # Same language prompts

        #                     template.configure_experiment(**config)

        #                     batch_result = self.evaluator.evaluate_batch(
        #                         model, puzzles, template
        #                     )
        #                     language_results[f"grid_{grid_size}"] = batch_result

        #                 except Exception as e:
        #                     self.logger.error(
        #                         f"Error in language variant {prompt_mode}: {e}"
        #                     )
        #                     language_results[f"grid_{grid_size}"] = {"error": str(e)}

        #             model_results[language] = language_results

        #         mode_results[model_name] = model_results

        #     results["results"][prompt_mode] = mode_results

        # filepath = self._save_experiment_results("7_language_variants", results)
        # results["filepath"] = filepath

        return {}

    # Analysis methods for results-only experiments (1, 8-10)

    def analyze_language_families(self, master_results_path: str) -> Dict[str, Any]:
        """
        Analysis 1: Language family analysis from master experiment results.

        Analyzes Dravidian vs Indo-Aryan vs Other language performance.
        """
        self.logger.info("Starting Analysis 1: Language Families")

        with open(master_results_path, "r", encoding="utf-8") as f:
            master_results = json.load(f)

        dravidian_langs = self.model_classifier.get_dravidian_languages()
        indo_aryan_langs = self.model_classifier.get_indo_aryan_languages()

        analysis = {
            "analysis_id": "1_language_families",
            "timestamp": datetime.now().isoformat(),
            "source": master_results_path,
            "language_families": {
                "dravidian": list(dravidian_langs),
                "indo_aryan": list(indo_aryan_langs),
                "other": ["English"],  # Can be expanded
            },
            "results": {},
        }

        # Aggregate results by language family
        for model_name, model_data in master_results.get("results", {}).items():
            family_results = {"dravidian": {}, "indo_aryan": {}, "other": {}}

            for language, language_data in model_data.items():
                if language in dravidian_langs:
                    family = "dravidian"
                elif language in indo_aryan_langs:
                    family = "indo_aryan"
                else:
                    family = "other"

                family_results[family][language] = language_data

            analysis["results"][model_name] = family_results

        filepath = self._save_experiment_results(
            "1_language_families_analysis", analysis
        )
        analysis["filepath"] = filepath

        return analysis

    def analyze_model_types(self, master_results_path: str) -> Dict[str, Any]:
        """
        Analysis 8: Model type comparison from master experiment results.

        Compares Indic fine-tuned vs general multilingual models.
        """
        self.logger.info("Starting Analysis 8: Model Types")

        with open(master_results_path, "r", encoding="utf-8") as f:
            master_results = json.load(f)

        indic_models = [
            m.name for m in self.model_classifier.get_indic_finetuned_models()
        ]
        general_models = [
            m.name for m in self.model_classifier.get_general_multilingual_models()
        ]

        analysis = {
            "analysis_id": "8_model_types",
            "timestamp": datetime.now().isoformat(),
            "source": master_results_path,
            "model_types": {
                "indic_finetuned": indic_models,
                "general_multilingual": general_models,
            },
            "results": {"indic_finetuned": {}, "general_multilingual": {}},
        }

        # Categorize results by model type
        for model_name, model_data in master_results.get("results", {}).items():
            if model_name in indic_models:
                analysis["results"]["indic_finetuned"][model_name] = model_data
            elif model_name in general_models:
                analysis["results"]["general_multilingual"][model_name] = model_data

        filepath = self._save_experiment_results("8_model_types_analysis", analysis)
        analysis["filepath"] = filepath

        return analysis

    def analyze_reasoning_models(self, master_results_path: str) -> Dict[str, Any]:
        """
        Analysis 9: Reasoning model analysis from master experiment results.

        Compares reasoning vs non-reasoning models.
        """
        self.logger.info("Starting Analysis 9: Reasoning Models")

        with open(master_results_path, "r", encoding="utf-8") as f:
            master_results = json.load(f)

        reasoning_models = [
            m.name for m in self.model_classifier.get_reasoning_models()
        ]
        non_reasoning_models = [
            m.name for m in self.model_classifier.get_non_reasoning_models()
        ]

        analysis = {
            "analysis_id": "9_reasoning_models",
            "timestamp": datetime.now().isoformat(),
            "source": master_results_path,
            "model_capabilities": {
                "reasoning": reasoning_models,
                "non_reasoning": non_reasoning_models,
            },
            "results": {"reasoning": {}, "non_reasoning": {}},
        }

        # Categorize results by reasoning capability
        for model_name, model_data in master_results.get("results", {}).items():
            if model_name in reasoning_models:
                analysis["results"]["reasoning"][model_name] = model_data
            elif model_name in non_reasoning_models:
                analysis["results"]["non_reasoning"][model_name] = model_data

        filepath = self._save_experiment_results(
            "9_reasoning_models_analysis", analysis
        )
        analysis["filepath"] = filepath

        return analysis

    def analyze_performance_normalization(
        self, master_results_path: str
    ) -> Dict[str, Any]:
        """
        Analysis 10: Performance normalization from master experiment results.

        Normalizes performance by cost and token usage metrics.
        """
        self.logger.info("Starting Analysis 10: Performance Normalization")

        with open(master_results_path, "r", encoding="utf-8") as f:
            master_results = json.load(f)

        analysis = {
            "analysis_id": "10_performance_normalization",
            "timestamp": datetime.now().isoformat(),
            "source": master_results_path,
            "normalization_metrics": [
                "cost_per_puzzle",
                "tokens_per_puzzle",
                "accuracy_per_dollar",
            ],
            "results": {},
        }

        # Calculate normalization metrics for each model
        for model_name, model_data in master_results.get("results", {}).items():
            normalized_results = {}

            for language, language_data in model_data.items():
                for grid_key, grid_data in language_data.items():
                    if isinstance(grid_data, dict) and "summary_metrics" in grid_data:
                        metrics = grid_data["summary_metrics"]

                        # Calculate normalized metrics if cost tracking data is available
                        if "total_tokens" in grid_data and "total_cost" in grid_data:
                            normalized_metrics = {
                                "base_accuracy": metrics.get(
                                    "average_word_accuracy_global", 0
                                ),
                                "cost_per_puzzle": grid_data["total_cost"]
                                / max(1, len(grid_data.get("puzzles", [1]))),
                                "tokens_per_puzzle": grid_data["total_tokens"]
                                / max(1, len(grid_data.get("puzzles", [1]))),
                                "accuracy_per_dollar": metrics.get(
                                    "average_word_accuracy_global", 0
                                )
                                / max(0.001, grid_data["total_cost"]),
                            }
                        else:
                            normalized_metrics = {
                                "base_accuracy": metrics.get(
                                    "average_word_accuracy_global", 0
                                ),
                                "cost_data_unavailable": True,
                            }

                        if language not in normalized_results:
                            normalized_results[language] = {}
                        normalized_results[language][grid_key] = normalized_metrics

            analysis["results"][model_name] = normalized_results

        filepath = self._save_experiment_results(
            "10_performance_normalization_analysis", analysis
        )
        analysis["filepath"] = filepath

        return analysis
