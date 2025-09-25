"""
Enhanced crossword evaluator for experimental evaluation pipeline.

This module provides comprehensive crossword evaluation with support for
all experimental conditions including token tracking, cost analysis,
and various prompting strategies.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np

from .parser import CrosswordResponseParser
from .templates import CrosswordPromptTemplate
from .metrics import EnhancedCrosswordMetrics
from ..core.base_evaluator import BaseEvaluator
from ..utils.config_loader import get_config


class CrosswordEvaluator(BaseEvaluator):
    """
    Enhanced crossword puzzle evaluator with experimental support.

    Provides comprehensive evaluation capabilities including:
    - Standard single/batch evaluation
    - Token usage tracking
    - Cost analysis
    - Few-shot example support
    - Self-reflection capabilities
    - Multi-iteration evaluation
    """

    def __init__(self):
        """Initialize evaluator with integrated components."""
        self.logger = logging.getLogger("CrosswordEvaluator")
        self.config = get_config()

        # Initialize components
        self.parser = CrosswordResponseParser()
        self.metrics = EnhancedCrosswordMetrics()

        # Configuration flags
        self.token_tracking_enabled = self.config.get_bool(
            "ENABLE_TOKEN_TRACKING", True
        )

        self.logger.info("CrosswordEvaluator initialized with enhanced components")

    def evaluate_single(
        self, model, puzzle, template: Optional[CrosswordPromptTemplate] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single crossword puzzle.

        Args:
            model: Model interface with generate_response method
            puzzle: CrosswordPuzzle object
            template: Optional CrosswordPromptTemplate for custom prompting

        Returns:
            Dict with comprehensive evaluation results
        """
        self.logger.info(f"Evaluating puzzle: {puzzle.puzzle_id}")

        try:
            # Use provided template or create default
            if template is None:
                template = CrosswordPromptTemplate()

            # Format puzzle for model
            prompt = template.format_puzzle_for_model(puzzle)

            # Track evaluation time
            start_time = time.time()
            self.logger.info(f"Generated prompt: {prompt[:500]}..." if len(prompt) > 500 else f"Generated prompt: {prompt}")
            # Get model response
            response = model.generate_response(prompt)

            end_time = time.time()
            evaluation_time = end_time - start_time

            # Enhanced response processing and validation
            processed_response = self._process_and_validate_response(
                response, puzzle.puzzle_id
            )

            # Check if response processing failed
            if processed_response is None:
                return {
                    "puzzle_id": puzzle.puzzle_id,
                    "success": False,
                    "prompt": prompt,
                    "response": response,
                    "parsed_grid": None,
                    "parsed_words": {},
                    "metrics": {
                        "success": False,
                        "error": "Response processing failed",
                        "word_accuracy_global": 0.0,
                        "letter_accuracy": 0.0,
                        "intersection_accuracy": 0.0,
                    },
                    "evaluation_time": evaluation_time,
                    "token_info": self._extract_token_info(model, response),
                    "template_config": self._get_template_config(template),
                    "error": "Response processing failed",
                }

            # Parse the processed response
            parsed_grid, parsed_words = self.parser.parse_model_response(
                puzzle, processed_response
            )

            # Get solution grid and words
            solution_grid = puzzle.get_solution_grid()
            solution_words = puzzle.get_solution_words()

            # Compute comprehensive metrics
            metrics = self.metrics.compute_metrics(
                parsed_grid, parsed_words, solution_grid, solution_words, puzzle
            )

            # Extract token usage if available
            token_info = self._extract_token_info(model, response)

            self.logger.info(
                f"Evaluation complete. Key metrics: WA={metrics.get('word_accuracy_global', 0):.3f}, LA={metrics.get('letter_accuracy', 0):.3f}, IA={metrics.get('intersection_accuracy', 0):.3f}"
            )

            # Return comprehensive results
            return {
                "puzzle_id": puzzle.puzzle_id,
                "success": metrics.get("success", False),
                "prompt": prompt,
                "response": response,
                "parsed_grid": parsed_grid,
                "parsed_words": parsed_words,
                "metrics": metrics,
                "evaluation_time": evaluation_time,
                "token_info": token_info,
                "template_config": self._get_template_config(template),
                "error": None,
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {
                "puzzle_id": puzzle.puzzle_id,
                "success": False,
                "prompt": "",
                "response": "",
                "parsed_grid": None,
                "parsed_words": {},
                "metrics": {},
                "evaluation_time": 0,
                "token_info": {},
                "template_config": {},
                "error": str(e),
            }

    def evaluate_batch(
        self, model, puzzles: List, template: Optional[CrosswordPromptTemplate] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on multiple puzzles.

        Args:
            model: Model interface
            puzzles: List of CrosswordPuzzle objects
            template: Optional CrosswordPromptTemplate for custom prompting

        Returns:
            Dict with batch evaluation results
        """
        self.logger.info(f"Starting batch evaluation of {len(puzzles)} puzzles")

        results = []
        total_tokens = 0
        total_time = 0

        for i, puzzle in enumerate(puzzles):
            self.logger.info(
                f"Evaluating puzzle {i + 1}/{len(puzzles)}: {puzzle.puzzle_id}"
            )
            result = self.evaluate_single(model, puzzle, template)
            results.append(result)

            # Accumulate resource usage
            total_time += result.get("evaluation_time", 0)
            token_info = result.get("token_info", {})
            total_tokens += token_info.get("total_tokens", 0)

        # Calculate summary metrics
        summary = self._calculate_summary(results)

        # Add resource usage summary
        summary["total_evaluation_time"] = total_time
        summary["total_tokens"] = total_tokens
        summary["average_time_per_puzzle"] = total_time / len(puzzles) if puzzles else 0
        summary["average_tokens_per_puzzle"] = (
            total_tokens / len(puzzles) if puzzles else 0
        )

        return {
            "model_name": getattr(model, "model_name", "unknown"),
            "total_puzzles": len(puzzles),
            "individual_results": results,
            "summary_metrics": summary,
            "template_config": self._get_template_config(template) if template else {},
        }

    def evaluate_batch_with_examples(
        self,
        model,
        puzzles: List,
        template: CrosswordPromptTemplate,
        examples: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate batch with few-shot examples.

        Args:
            model: Model interface
            puzzles: List of puzzles to evaluate
            template: Configured template with shot type
            examples: Few-shot example puzzles

        Returns:
            Batch evaluation results
        """
        self.logger.info(
            f"Batch evaluation with few-shot examples: {len(examples or [])}"
        )

        results = []

        for puzzle in puzzles:
            # Generate prompt with examples
            prompt = template.format_puzzle_for_model(puzzle, examples)

            try:
                start_time = time.time()
                response = model.generate_response(prompt)
                end_time = time.time()

                # Parse and evaluate
                parsed_grid, parsed_words = self.parser.parse_model_response(
                    puzzle, response
                )
                metrics = self.metrics.compute_metrics(
                    parsed_grid,
                    parsed_words,
                    puzzle.get_solution_grid(),
                    puzzle.get_solution_words(),
                    puzzle,
                )

                result = {
                    "puzzle_id": puzzle.puzzle_id,
                    "success": metrics.get("success", False),
                    "prompt": prompt,
                    "response": response,
                    "parsed_grid": parsed_grid,
                    "parsed_words": parsed_words,
                    "metrics": metrics,
                    "evaluation_time": end_time - start_time,
                    "token_info": self._extract_token_info(model, response),
                    "error": None,
                }

            except Exception as e:
                result = {
                    "puzzle_id": puzzle.puzzle_id,
                    "success": False,
                    "error": str(e),
                }

            results.append(result)

        summary = self._calculate_summary(results)

        return {
            "model_name": getattr(model, "model_name", "unknown"),
            "total_puzzles": len(puzzles),
            "individual_results": results,
            "summary_metrics": summary,
            "template_config": self._get_template_config(template),
            "few_shot_examples": len(examples or []),
        }

    def evaluate_single_with_prompt(
        self, model, puzzle, custom_prompt: str
    ) -> Dict[str, Any]:
        """
        Evaluate with a custom prompt (for self-reflection).

        Args:
            model: Model interface
            puzzle: CrosswordPuzzle object
            custom_prompt: Pre-formatted prompt string

        Returns:
            Evaluation result dictionary
        """
        try:
            start_time = time.time()
            response = model.generate_response(custom_prompt)
            end_time = time.time()

            # Enhanced response processing
            processed_response = self._process_and_validate_response(
                response, puzzle.puzzle_id
            )

            if processed_response is None:
                return {
                    "puzzle_id": puzzle.puzzle_id,
                    "success": False,
                    "prompt": custom_prompt,
                    "response": response,
                    "parsed_grid": None,
                    "parsed_words": {},
                    "metrics": {"success": False, "error": "Response processing failed"},
                    "evaluation_time": end_time - start_time,
                    "error": "Response processing failed",
                }

            # Parse and evaluate
            parsed_grid, parsed_words = self.parser.parse_model_response(
                puzzle, processed_response
            )
            metrics = self.metrics.compute_metrics(
                parsed_grid,
                parsed_words,
                puzzle.get_solution_grid(),
                puzzle.get_solution_words(),
                puzzle,
            )

            return {
                "puzzle_id": puzzle.puzzle_id,
                "success": metrics.get("success", False),
                "prompt": custom_prompt,
                "response": response,
                "parsed_grid": parsed_grid,
                "parsed_words": parsed_words,
                "metrics": metrics,
                "evaluation_time": end_time - start_time,
                "token_info": self._extract_token_info(model, response),
                "error": None,
            }

        except Exception as e:
            return {"puzzle_id": puzzle.puzzle_id, "success": False, "error": str(e)}

    def evaluate_batch_with_tracking(
        self, model, puzzles: List, template: CrosswordPromptTemplate
    ) -> Dict[str, Any]:
        """
        Evaluate batch with detailed resource tracking for efficiency analysis.

        Args:
            model: Model interface
            puzzles: List of puzzles
            template: Template configuration

        Returns:
            Results with detailed resource usage tracking
        """
        self.logger.info(f"Starting tracked batch evaluation of {len(puzzles)} puzzles")

        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_time = 0

        for puzzle in puzzles:
            result = self.evaluate_single(model, puzzle, template)
            results.append(result)

            # Track detailed token usage
            token_info = result.get("token_info", {})
            total_input_tokens += token_info.get("input_tokens", 0)
            total_output_tokens += token_info.get("output_tokens", 0)
            total_time += result.get("evaluation_time", 0)

        # Calculate summary with detailed tracking
        summary = self._calculate_summary(results)

        # Add detailed resource tracking
        summary.update(
            {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_evaluation_time": total_time,
                "tokens_per_puzzle": {
                    "input": total_input_tokens / len(puzzles) if puzzles else 0,
                    "output": total_output_tokens / len(puzzles) if puzzles else 0,
                    "total": (total_input_tokens + total_output_tokens) / len(puzzles)
                    if puzzles
                    else 0,
                },
                "time_per_puzzle": total_time / len(puzzles) if puzzles else 0,
            }
        )

        return {
            "model_name": getattr(model, "model_name", "unknown"),
            "total_puzzles": len(puzzles),
            "individual_results": results,
            "summary_metrics": summary,
            "template_config": self._get_template_config(template),
        }

    def _calculate_summary(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate summary metrics across all results."""
        if not results:
            return {}

        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {"success_rate": 0.0}

        # Calculate averages for each metric
        summary = {}

        # Success rate
        summary["success_rate"] = len(successful_results) / len(results)

        # Average metrics across successful evaluations
        if successful_results:
            all_metric_names = set()
            for result in successful_results:
                if result.get("metrics"):
                    all_metric_names.update(result["metrics"].keys())

            for metric_name in all_metric_names:
                values = []
                for result in successful_results:
                    if result.get("metrics") and metric_name in result["metrics"]:
                        value = result["metrics"][metric_name]
                        if isinstance(value, (int, float)):
                            values.append(value)

                if values:
                    summary[f"average_{metric_name}"] = sum(values) / len(values)
                    summary[f"std_{metric_name}"] = (
                        np.std(values) if len(values) > 1 else 0.0
                    )

        return summary

    def _extract_token_info(self, model, response: str) -> Dict[str, Any]:
        """Extract token usage information from model if available."""
        if not self.token_tracking_enabled:
            return {}

        token_info = {}

        # Try to get token usage from model
        if hasattr(model, "last_token_usage"):
            token_info = model.last_token_usage
        elif hasattr(model, "get_last_token_usage"):
            token_info = model.get_last_token_usage()

        # Fallback: estimate tokens from response length
        if not token_info:
            # Rough approximation: 1 token ≈ 4 characters
            estimated_output_tokens = len(response) // 4
            token_info = {
                "output_tokens": estimated_output_tokens,
                "total_tokens": estimated_output_tokens,
                "estimated": True,
            }

        return token_info

    def evaluate_batch_with_reflection(
        self, model, puzzles: List, template: CrosswordPromptTemplate
    ) -> Dict[str, Any]:
        """
        Evaluate batch with self-reflection capabilities.

        Args:
            model: Model interface
            puzzles: List of puzzles to evaluate
            template: Template configured with self_reflection=True

        Returns:
            Batch evaluation results with reflection iterations
        """
        self.logger.info(
            f"Starting self-reflection evaluation on {len(puzzles)} puzzles"
        )

        results = []
        max_iterations = int(self.config.get("SELF_REFLECTION_MAX_ITERATIONS", "3"))

        for puzzle in puzzles:
            # Initial attempt
            initial_result = self.evaluate_single(model, puzzle, template)

            if initial_result.get("success", False) or not template.self_reflection:
                # Success or no reflection configured - use initial result
                results.append(initial_result)
                continue

            # Self-reflection iterations
            reflection_results = [initial_result]
            current_response = initial_result.get("response", "")

            for iteration in range(max_iterations):
                try:
                    # Generate error feedback (simplified - could be more sophisticated)
                    errors = self._generate_error_feedback(puzzle, initial_result)

                    if not errors:
                        break  # No clear errors identified

                    # Generate reflection prompt
                    reflection_prompt = template.generate_self_reflection_prompt(
                        puzzle, current_response, errors
                    )

                    # Evaluate with reflection prompt
                    reflection_result = self.evaluate_single_with_prompt(
                        model, puzzle, reflection_prompt
                    )
                    reflection_result["reflection_iteration"] = iteration + 1
                    reflection_result["errors_addressed"] = errors

                    reflection_results.append(reflection_result)

                    # Check if reflection improved the result
                    if reflection_result.get("success", False):
                        break

                    current_response = reflection_result.get("response", "")

                except Exception as e:
                    self.logger.error(
                        f"Error in reflection iteration {iteration + 1}: {e}"
                    )
                    break

            # Use best result from all iterations
            best_result = max(
                reflection_results,
                key=lambda r: r.get("metrics", {}).get("word_accuracy_global", 0),
            )
            best_result["reflection_attempts"] = len(reflection_results)
            best_result["all_iterations"] = reflection_results

            results.append(best_result)

        # Calculate summary
        summary = self._calculate_summary(results)

        return {
            "model_name": getattr(model, "model_name", "unknown"),
            "total_puzzles": len(puzzles),
            "individual_results": results,
            "summary_metrics": summary,
            "template_config": self._get_template_config(template),
            "evaluation_mode": "self_reflection",
            "max_reflection_iterations": max_iterations,
        }

    def _generate_error_feedback(self, puzzle, result: Dict[str, Any]) -> List[str]:
        """Generate error feedback for self-reflection."""
        errors = []

        # Check metrics for specific error types
        metrics = result.get("metrics", {})
        parsed_words = result.get("parsed_words", {})
        expected_words = puzzle.get_solution_words()

        # Word accuracy errors
        word_accuracy = metrics.get("word_accuracy_global", 0)
        if word_accuracy < 1.0:
            incorrect_count = len(expected_words) - (
                len(expected_words) * word_accuracy
            )
            errors.append(f"Approximately {int(incorrect_count)} words are incorrect")

        # Letter accuracy errors
        letter_accuracy = metrics.get("letter_accuracy", 0)
        if letter_accuracy < 0.8:
            errors.append("Many letters in the grid don't match the expected solution")

        # Intersection errors
        intersection_accuracy = metrics.get("intersection_accuracy", 0)
        if intersection_accuracy < 0.7:
            errors.append("Letters at word intersections don't align correctly")

        # Missing or extra words
        if len(parsed_words) != len(expected_words):
            if len(parsed_words) < len(expected_words):
                missing_count = len(expected_words) - len(parsed_words)
                errors.append(
                    f"{missing_count} clue answers are missing from your response"
                )
            else:
                extra_count = len(parsed_words) - len(expected_words)
                errors.append(
                    f"You provided {extra_count} extra answers beyond the expected clues"
                )

        return errors[:3]  # Limit to top 3 errors for clarity

    def _get_template_config(self, template: CrosswordPromptTemplate) -> Dict[str, Any]:
        """Extract template configuration for result tracking."""
        if not template:
            return {}

        return {
            "language": template.target_language,
            "shot_type": template.shot_type,
            "chain_of_thought": template.chain_of_thought,
            "reasoning_effort": template.reasoning_effort,
            "self_reflection": template.self_reflection,
            "batch_size": template.batch_size,
        }

    def _process_and_validate_response(
        self, response: str, puzzle_id: str
    ) -> Optional[str]:
        """
        Enhanced response processing and validation for crossword evaluation.

        Provides additional validation and error handling on top of backend
        normalization to ensure robust evaluation pipeline performance.

        Args:
            response: Raw model response
            puzzle_id: Puzzle identifier for logging

        Returns:
            Processed response string or None if processing fails
        """
        # Check for empty response
        if not response or response.strip() == "":
            self.logger.error(f"Empty response from model for puzzle {puzzle_id}")
            return None

        # Check for extremely short responses (likely incomplete)
        if len(response.strip()) < 20:
            self.logger.warning(
                f"Very short response ({len(response.strip())} chars) for puzzle {puzzle_id}: {response[:50]}"
            )

        # Check for crossword-specific content
        has_grid_marker = any(marker in response.upper() for marker in ["GRID:", "WORDS:", "ACROSS", "DOWN"])
        has_letters = any(char.isalpha() for char in response)

        if not has_grid_marker and not has_letters:
            self.logger.warning(
                f"Response lacks crossword structure markers for puzzle {puzzle_id}"
            )

        # Log response characteristics for debugging
        self.logger.debug(
            f"Response processing for {puzzle_id}: "
            f"length={len(response)}, has_grid_marker={has_grid_marker}, "
            f"has_letters={has_letters}"
        )

        # Return processed response (normalized by backend)
        return response.strip()
