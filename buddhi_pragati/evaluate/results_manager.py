"""
Enhanced results management and analysis for the new crossword evaluation framework.

This module provides utilities for:
- Managing the new experiment structure (0, 2-7 + analyses 1, 8-10)
- Priority subset analysis and filtering
- Cross-experiment comparison and visualization support
- Standardized result processing for plotting libraries
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.config_loader import get_config


class ExperimentResultsManager:
    """
    Manages experiment results including saving, loading, and analysis.

    Provides utilities for:
    - Saving experiment results to JSON
    - Loading and aggregating results
    - Statistical analysis across experiments
    - Generating summary reports
    """

    def __init__(self, results_dir: Optional[str] = None):
        self.logger = logging.getLogger("ExperimentResultsManager")
        self.config = get_config()

        # Use config default if no directory specified
        if results_dir is None:
            results_dir = self.config.get(
                "EXPERIMENT_RESULTS_DIR", "buddhi_pragati/experiments"
            )

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_experiment_results(
        self, experiment_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all results for a specific experiment type.

        Args:
            experiment_id: Experiment identifier to load
            limit: Optional limit on number of results to load (most recent first)

        Returns:
            List of experiment result dictionaries
        """
        pattern = f"{experiment_id}_*.json"
        result_files = sorted(
            self.results_dir.glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if limit:
            result_files = result_files[:limit]

        results = []
        for filepath in result_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    result = json.load(f)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to load {filepath}: {e}")
                continue

        self.logger.info(
            f"Loaded {len(results)} results for experiment {experiment_id}"
        )
        return results

    def aggregate_experiment_metrics(
        self, experiment_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple experiment runs.

        Args:
            experiment_results: List of experiment result dictionaries

        Returns:
            Aggregated metrics with mean, std, min, max across runs
        """
        if not experiment_results:
            return {}

        aggregated = {
            "experiment_count": len(experiment_results),
            "aggregated_metrics": {},
            "model_performance": {},
            "statistical_summary": {},
        }

        # Extract all metric values
        metric_values = {}
        model_scores = {}

        for result in experiment_results:
            if "results" not in result:
                continue

            for model_name, model_result in result["results"].items():
                if model_name not in model_scores:
                    model_scores[model_name] = {}

                # Handle different result structures
                if isinstance(model_result, dict):
                    metrics = self._extract_metrics_from_result(model_result)

                    for metric_name, value in metrics.items():
                        if metric_name not in metric_values:
                            metric_values[metric_name] = []
                        if metric_name not in model_scores[model_name]:
                            model_scores[model_name][metric_name] = []

                        metric_values[metric_name].append(value)
                        model_scores[model_name][metric_name].append(value)

        # Calculate aggregated statistics
        for metric_name, values in metric_values.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregated["aggregated_metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }

        # Calculate per-model statistics
        for model_name, metrics in model_scores.items():
            aggregated["model_performance"][model_name] = {}
            for metric_name, values in metrics.items():
                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated["model_performance"][model_name][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values),
                    }

        return aggregated

    def _extract_metrics_from_result(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from a result dictionary."""
        metrics = {}

        # Check for summary metrics
        if "summary_metrics" in result:
            for key, value in result["summary_metrics"].items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)

        # Check for direct metrics
        if "metrics" in result:
            for key, value in result["metrics"].items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)

        # Handle nested structures (e.g., language family results)
        for key, value in result.items():
            if isinstance(value, dict) and "summary_metrics" in value:
                for metric_key, metric_value in value["summary_metrics"].items():
                    if isinstance(metric_value, (int, float)):
                        metrics[f"{key}_{metric_key}"] = float(metric_value)

        return metrics

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare results across different experiment types.

        Args:
            experiment_ids: List of experiment identifiers to compare

        Returns:
            Comparison analysis dictionary
        """
        comparison = {
            "experiment_ids": experiment_ids,
            "comparison_timestamp": datetime.now().isoformat(),
            "experiment_summaries": {},
            "cross_experiment_analysis": {},
        }

        # Load and summarize each experiment
        all_metrics = {}
        for exp_id in experiment_ids:
            results = self.load_experiment_results(exp_id, limit=5)  # Recent 5 runs
            if results:
                summary = self.aggregate_experiment_metrics(results)
                comparison["experiment_summaries"][exp_id] = summary

                # Collect metrics for cross-experiment analysis
                if "aggregated_metrics" in summary:
                    all_metrics[exp_id] = summary["aggregated_metrics"]

        # Cross-experiment metric comparison
        if all_metrics:
            comparison["cross_experiment_analysis"] = (
                self._compare_metrics_across_experiments(all_metrics)
            )

        return comparison

    def _compare_metrics_across_experiments(
        self, all_metrics: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Compare the same metrics across different experiments."""
        comparison = {}

        # Find common metrics
        all_metric_names = set()
        for exp_metrics in all_metrics.values():
            all_metric_names.update(exp_metrics.keys())

        for metric_name in all_metric_names:
            metric_comparison = {}

            for exp_id, metrics in all_metrics.items():
                if metric_name in metrics and "mean" in metrics[metric_name]:
                    metric_comparison[exp_id] = metrics[metric_name]["mean"]

            if len(metric_comparison) > 1:
                # Calculate relative performance
                values = list(metric_comparison.values())
                best_value = max(values)
                worst_value = min(values)

                comparison[metric_name] = {
                    "experiment_means": metric_comparison,
                    "best_experiment": max(
                        metric_comparison.items(), key=lambda x: x[1]
                    ),
                    "worst_experiment": min(
                        metric_comparison.items(), key=lambda x: x[1]
                    ),
                    "performance_range": best_value - worst_value,
                    "coefficient_of_variation": np.std(values) / np.mean(values)
                    if np.mean(values) > 0
                    else 0,
                }

        return comparison

    def generate_experiment_report(
        self, experiment_id: str, output_format: str = "json"
    ) -> str:
        """
        Generate a comprehensive report for an experiment.

        Args:
            experiment_id: Experiment identifier
            output_format: Output format ('json', 'markdown', 'csv')

        Returns:
            Path to generated report file
        """
        results = self.load_experiment_results(experiment_id)
        if not results:
            raise ValueError(f"No results found for experiment {experiment_id}")

        aggregated = self.aggregate_experiment_metrics(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format == "json":
            report_path = self.results_dir / f"{experiment_id}_report_{timestamp}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(aggregated, f, ensure_ascii=False, indent=2, default=str)

        elif output_format == "markdown":
            report_path = self.results_dir / f"{experiment_id}_report_{timestamp}.md"
            with open(report_path, "w", encoding="utf-8") as f:
                self._write_markdown_report(f, experiment_id, aggregated)

        elif output_format == "csv":
            report_path = self.results_dir / f"{experiment_id}_report_{timestamp}.csv"
            self._write_csv_report(report_path, aggregated)

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        self.logger.info(f"Generated {output_format} report: {report_path}")
        return str(report_path)

    def _write_markdown_report(
        self, file_obj, experiment_id: str, aggregated: Dict[str, Any]
    ):
        """Write a markdown format report."""
        file_obj.write(f"# Experiment Report: {experiment_id}\n\n")
        file_obj.write(f"Generated: {datetime.now().isoformat()}\n")
        file_obj.write(
            f"Experiment runs analyzed: {aggregated.get('experiment_count', 0)}\n\n"
        )

        # Aggregated metrics
        if "aggregated_metrics" in aggregated:
            file_obj.write("## Aggregated Metrics\n\n")
            file_obj.write("| Metric | Mean | Std | Min | Max | Count |\n")
            file_obj.write("|--------|------|-----|-----|-----|-------|\n")

            for metric_name, stats in aggregated["aggregated_metrics"].items():
                file_obj.write(
                    f"| {metric_name} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n"
                )

        # Model performance
        if "model_performance" in aggregated:
            file_obj.write("\n## Model Performance\n\n")
            for model_name, metrics in aggregated["model_performance"].items():
                file_obj.write(f"### {model_name}\n\n")
                file_obj.write("| Metric | Mean | Std | Count |\n")
                file_obj.write("|--------|------|-----|-------|\n")

                for metric_name, stats in metrics.items():
                    file_obj.write(
                        f"| {metric_name} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['count']} |\n"
                    )
                file_obj.write("\n")

    def _write_csv_report(self, filepath: Path, aggregated: Dict[str, Any]):
        """Write a CSV format report."""
        rows = []

        # Model performance data
        if "model_performance" in aggregated:
            for model_name, metrics in aggregated["model_performance"].items():
                for metric_name, stats in metrics.items():
                    rows.append(
                        {
                            "model": model_name,
                            "metric": metric_name,
                            "mean": stats["mean"],
                            "std": stats["std"],
                            "count": stats["count"],
                        }
                    )

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        else:
            # Empty CSV with headers
            pd.DataFrame(columns=["model", "metric", "mean", "std", "count"]).to_csv(
                filepath, index=False
            )

    def find_best_performing_models(
        self, experiment_id: str, metric_name: str = "word_accuracy_global"
    ) -> List[Tuple[str, float]]:
        """
        Find the best performing models for a specific metric.

        Args:
            experiment_id: Experiment identifier
            metric_name: Metric to rank by

        Returns:
            List of (model_name, score) tuples sorted by performance
        """
        results = self.load_experiment_results(experiment_id)
        aggregated = self.aggregate_experiment_metrics(results)

        model_scores = []
        if "model_performance" in aggregated:
            for model_name, metrics in aggregated["model_performance"].items():
                if metric_name in metrics:
                    score = metrics[metric_name]["mean"]
                    model_scores.append((model_name, score))

        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return model_scores

    def list_available_experiments(self) -> Dict[str, int]:
        """
        List all available experiment types and their result counts.

        Returns:
            Dictionary mapping experiment_id to result count
        """
        experiments = {}

        for json_file in self.results_dir.glob("*.json"):
            # Extract experiment ID from filename (before first underscore or second underscore for new format)
            filename = json_file.stem
            if filename.startswith("experiment_"):
                # New format: experiment_0_master_timestamp.json
                parts = filename.split("_")
                if len(parts) >= 3:
                    exp_id = f"{parts[1]}_{parts[2]}"  # e.g., "0_master", "2_shot_variations"
                    experiments[exp_id] = experiments.get(exp_id, 0) + 1
            elif "_" in filename:
                # Old format: experiment_id_timestamp.json
                exp_id = filename.split("_")[0]
                experiments[exp_id] = experiments.get(exp_id, 0) + 1

        return experiments

    def load_master_experiment_results(
        self, limit: Optional[int] = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Load the most recent master experiment (experiment 0) results.

        Args:
            limit: Number of most recent results to load (default: 1 for latest only)

        Returns:
            Master experiment results dictionary or None if not found
        """
        master_results = self.load_experiment_results("0_master", limit=limit)
        return master_results[0] if master_results else None

    def filter_results_by_priority_subset(
        self,
        results: Dict[str, Any],
        priority_models: List[str] = None,
        priority_languages: List[str] = None,
        priority_grid_sizes: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Filter experiment results to only include priority subset.

        Args:
            results: Full experiment results
            priority_models: Models to include (None = all)
            priority_languages: Languages to include (None = all)
            priority_grid_sizes: Grid sizes to include (None = all)

        Returns:
            Filtered results containing only priority subset
        """
        if "results" not in results:
            return results

        filtered_results = results.copy()
        filtered_results["results"] = {}

        # Filter by models
        models_to_include = (
            priority_models if priority_models else results["results"].keys()
        )

        for model_name in models_to_include:
            if model_name not in results["results"]:
                continue

            model_data = results["results"][model_name]
            filtered_model_data = {}

            # Filter by languages
            languages_to_include = (
                priority_languages if priority_languages else model_data.keys()
            )

            for language in languages_to_include:
                if language not in model_data:
                    continue

                language_data = model_data[language]
                filtered_language_data = {}

                # Filter by grid sizes
                if priority_grid_sizes:
                    grid_keys_to_include = [
                        f"grid_{size}" for size in priority_grid_sizes
                    ]
                else:
                    grid_keys_to_include = language_data.keys()

                for grid_key in grid_keys_to_include:
                    if grid_key in language_data:
                        filtered_language_data[grid_key] = language_data[grid_key]

                if filtered_language_data:
                    filtered_model_data[language] = filtered_language_data

            if filtered_model_data:
                filtered_results["results"][model_name] = filtered_model_data

        # Update configuration to reflect filtering
        if "configuration" in filtered_results:
            config = filtered_results["configuration"].copy()
            if priority_models:
                config["filtered_models"] = priority_models
            if priority_languages:
                config["filtered_languages"] = priority_languages
            if priority_grid_sizes:
                config["filtered_grid_sizes"] = priority_grid_sizes
            filtered_results["configuration"] = config

        return filtered_results

    def extract_visualization_data(
        self,
        experiment_results: Dict[str, Any],
        metric_name: str = "average_word_accuracy_global",
    ) -> Dict[str, Any]:
        """
        Extract data in format suitable for matplotlib/seaborn visualization.

        Args:
            experiment_results: Experiment results dictionary
            metric_name: Metric to extract for visualization

        Returns:
            Dictionary with visualization-ready data structures
        """
        viz_data = {
            "experiment_id": experiment_results.get("experiment_id", "unknown"),
            "metric_name": metric_name,
            "data_points": [],
            "heatmap_data": {},
            "summary_stats": {},
        }

        if "results" not in experiment_results:
            return viz_data

        # Extract data points for scatter plots, bar charts
        for model_name, model_data in experiment_results["results"].items():
            if isinstance(model_data, dict):
                for language, language_data in model_data.items():
                    if isinstance(language_data, dict):
                        for grid_key, grid_data in language_data.items():
                            if (
                                isinstance(grid_data, dict)
                                and "summary_metrics" in grid_data
                            ):
                                metrics = grid_data["summary_metrics"]
                                if metric_name in metrics:
                                    viz_data["data_points"].append(
                                        {
                                            "model": model_name,
                                            "language": language,
                                            "grid_size": grid_key.replace("grid_", ""),
                                            "metric_value": metrics[metric_name],
                                            "puzzle_count": grid_data.get(
                                                "puzzle_count", 1
                                            ),
                                        }
                                    )

        # Create heatmap data (model x language matrix)
        models = list(set(dp["model"] for dp in viz_data["data_points"]))
        languages = list(set(dp["language"] for dp in viz_data["data_points"]))

        heatmap_matrix = []
        for model in models:
            model_row = []
            for language in languages:
                # Average across grid sizes for each model-language combination
                model_lang_values = [
                    dp["metric_value"]
                    for dp in viz_data["data_points"]
                    if dp["model"] == model and dp["language"] == language
                ]
                avg_value = np.mean(model_lang_values) if model_lang_values else 0
                model_row.append(avg_value)
            heatmap_matrix.append(model_row)

        viz_data["heatmap_data"] = {
            "matrix": heatmap_matrix,
            "models": models,
            "languages": languages,
        }

        # Summary statistics
        if viz_data["data_points"]:
            values = [dp["metric_value"] for dp in viz_data["data_points"]]
            viz_data["summary_stats"] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }

        return viz_data

    def compare_parameter_experiments(
        self,
        experiment_ids: List[str],
        metric_name: str = "average_word_accuracy_global",
    ) -> Dict[str, Any]:
        """
        Compare results across parameter variation experiments (2-7).

        Args:
            experiment_ids: List of experiment IDs to compare (e.g., ["2_shot_variations", "4_chain_of_thought"])
            metric_name: Metric to use for comparison

        Returns:
            Comparison analysis with parameter sensitivity data
        """
        comparison = {
            "comparison_id": f"parameter_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "experiments": experiment_ids,
            "metric_name": metric_name,
            "parameter_effects": {},
            "model_rankings": {},
            "sensitivity_analysis": {},
        }

        experiment_data = {}

        # Load all experiment results
        for exp_id in experiment_ids:
            results = self.load_experiment_results(exp_id, limit=1)
            if results:
                experiment_data[exp_id] = results[0]

                # Extract parameter effects
                if "results" in results[0]:
                    parameter_effects = {}

                    for parameter_value, param_data in results[0]["results"].items():
                        if isinstance(param_data, dict):
                            # Calculate average performance across models for this parameter value
                            values = []
                            for model_data in param_data.values():
                                if isinstance(model_data, dict):
                                    for lang_data in model_data.values():
                                        if isinstance(lang_data, dict):
                                            for grid_data in lang_data.values():
                                                if (
                                                    isinstance(grid_data, dict)
                                                    and "summary_metrics" in grid_data
                                                ):
                                                    if (
                                                        metric_name
                                                        in grid_data["summary_metrics"]
                                                    ):
                                                        values.append(
                                                            grid_data[
                                                                "summary_metrics"
                                                            ][metric_name]
                                                        )

                            if values:
                                parameter_effects[parameter_value] = {
                                    "mean": np.mean(values),
                                    "std": np.std(values),
                                    "count": len(values),
                                }

                    comparison["parameter_effects"][exp_id] = parameter_effects

        # Calculate model rankings across experiments
        model_scores = {}
        for exp_id, exp_data in experiment_data.items():
            if "results" in exp_data:
                for param_value, param_data in exp_data["results"].items():
                    if isinstance(param_data, dict):
                        for model_name, model_data in param_data.items():
                            if model_name not in model_scores:
                                model_scores[model_name] = []

                            # Calculate average score for this model in this parameter setting
                            values = []
                            if isinstance(model_data, dict):
                                for lang_data in model_data.values():
                                    if isinstance(lang_data, dict):
                                        for grid_data in lang_data.values():
                                            if (
                                                isinstance(grid_data, dict)
                                                and "summary_metrics" in grid_data
                                            ):
                                                if (
                                                    metric_name
                                                    in grid_data["summary_metrics"]
                                                ):
                                                    values.append(
                                                        grid_data["summary_metrics"][
                                                            metric_name
                                                        ]
                                                    )

                            if values:
                                model_scores[model_name].append(np.mean(values))

        # Rank models by average performance across all experiments
        for model_name, scores in model_scores.items():
            comparison["model_rankings"][model_name] = {
                "average_score": np.mean(scores),
                "std_score": np.std(scores),
                "rank_score": np.mean(scores),  # Used for ranking
            }

        # Sort by rank score
        comparison["model_rankings"] = dict(
            sorted(
                comparison["model_rankings"].items(),
                key=lambda x: x[1]["rank_score"],
                reverse=True,
            )
        )

        return comparison

    def generate_experiment_summary_report(
        self, experiment_ids: Optional[List[str]] = None, output_format: str = "json"
    ) -> str:
        """
        Generate comprehensive summary report across multiple experiments.

        Args:
            experiment_ids: List of experiments to include (None = all available)
            output_format: Output format ('json', 'markdown')

        Returns:
            Path to generated report file
        """
        if experiment_ids is None:
            experiment_ids = list(self.list_available_experiments().keys())

        summary = {
            "report_id": f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "experiments_included": experiment_ids,
            "summary_statistics": {},
            "key_findings": {},
            "recommendations": {},
        }

        # Load and analyze each experiment
        for exp_id in experiment_ids:
            results = self.load_experiment_results(exp_id, limit=1)
            if results:
                exp_summary = {
                    "experiment_type": exp_id,
                    "total_combinations": 0,
                    "successful_evaluations": 0,
                    "average_performance": {},
                    "top_performers": {},
                }

                # Count evaluations and calculate statistics
                if "results" in results[0]:
                    all_scores = []
                    model_scores = {}

                    for key1, data1 in results[0]["results"].items():
                        if isinstance(data1, dict):
                            for key2, data2 in data1.items():
                                if isinstance(data2, dict):
                                    for key3, data3 in data2.items():
                                        if isinstance(data3, dict):
                                            exp_summary["total_combinations"] += 1

                                            if "summary_metrics" in data3:
                                                exp_summary[
                                                    "successful_evaluations"
                                                ] += 1
                                                metrics = data3["summary_metrics"]

                                                # Collect word accuracy scores
                                                if (
                                                    "average_word_accuracy_global"
                                                    in metrics
                                                ):
                                                    score = metrics[
                                                        "average_word_accuracy_global"
                                                    ]
                                                    all_scores.append(score)

                                                    # Track model performance
                                                    model_key = (
                                                        key1
                                                        if exp_id == "0_master"
                                                        else key2
                                                    )  # Model name varies by experiment structure
                                                    if model_key not in model_scores:
                                                        model_scores[model_key] = []
                                                    model_scores[model_key].append(
                                                        score
                                                    )

                    # Calculate summary statistics
                    if all_scores:
                        exp_summary["average_performance"] = {
                            "mean_word_accuracy": np.mean(all_scores),
                            "std_word_accuracy": np.std(all_scores),
                            "min_word_accuracy": np.min(all_scores),
                            "max_word_accuracy": np.max(all_scores),
                        }

                    # Identify top performers
                    if model_scores:
                        model_averages = {
                            model: np.mean(scores)
                            for model, scores in model_scores.items()
                        }
                        top_3_models = sorted(
                            model_averages.items(), key=lambda x: x[1], reverse=True
                        )[:3]
                        exp_summary["top_performers"] = dict(top_3_models)

                summary["summary_statistics"][exp_id] = exp_summary

        # Generate key findings and recommendations
        summary["key_findings"] = self._extract_key_findings(
            summary["summary_statistics"]
        )
        summary["recommendations"] = self._generate_recommendations(
            summary["summary_statistics"]
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_format == "json":
            report_path = (
                self.results_dir / f"experiment_summary_report_{timestamp}.json"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        elif output_format == "markdown":
            report_path = self.results_dir / f"experiment_summary_report_{timestamp}.md"
            with open(report_path, "w", encoding="utf-8") as f:
                self._write_markdown_summary_report(f, summary)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        self.logger.info(f"Generated summary report: {report_path}")
        return str(report_path)

    def _extract_key_findings(self, experiment_stats: Dict[str, Any]) -> List[str]:
        """Extract key findings from experiment statistics."""
        findings = []

        # Find best performing experiment overall
        if experiment_stats:
            best_exp = max(
                experiment_stats.items(),
                key=lambda x: x[1]
                .get("average_performance", {})
                .get("mean_word_accuracy", 0),
            )
            findings.append(
                f"Best overall performance: {best_exp[0]} with "
                f"{best_exp[1]['average_performance']['mean_word_accuracy']:.3f} average word accuracy"
            )

        # Find most consistent experiment (lowest std)
        if experiment_stats:
            most_consistent = min(
                experiment_stats.items(),
                key=lambda x: x[1]
                .get("average_performance", {})
                .get("std_word_accuracy", float("inf")),
            )
            findings.append(
                f"Most consistent performance: {most_consistent[0]} with "
                f"{most_consistent[1]['average_performance']['std_word_accuracy']:.3f} standard deviation"
            )

        return findings

    def _generate_recommendations(self, experiment_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment statistics."""
        recommendations = []

        recommendations.append(
            "Run the master experiment (0_master) first to establish comprehensive baseline performance"
        )

        recommendations.append(
            "Focus parameter optimization experiments on top-performing models identified in master experiment"
        )

        recommendations.append(
            "Use priority subset settings for focused experiments to balance thoroughness with computational efficiency"
        )

        return recommendations

    def _write_markdown_summary_report(self, file_obj, summary: Dict[str, Any]):
        """Write a markdown format summary report."""
        file_obj.write("# Experiment Summary Report\n\n")
        file_obj.write(f"Generated: {datetime.now().isoformat()}\n\n")
        file_obj.write(
            f"Experiments Analyzed: {len(summary['experiments_included'])}\n\n"
        )

        # Key findings
        if "key_findings" in summary:
            file_obj.write("## Key Findings\n\n")
            for finding in summary["key_findings"]:
                file_obj.write(f"- {finding}\n")
            file_obj.write("\n")

        # Recommendations
        if "recommendations" in summary:
            file_obj.write("## Recommendations\n\n")
            for rec in summary["recommendations"]:
                file_obj.write(f"- {rec}\n")
            file_obj.write("\n")

        # Detailed statistics
        if "summary_statistics" in summary:
            file_obj.write("## Detailed Results\n\n")
            for exp_id, stats in summary["summary_statistics"].items():
                file_obj.write(f"### {exp_id}\n\n")
                file_obj.write(f"- Total combinations: {stats['total_combinations']}\n")
                file_obj.write(
                    f"- Successful evaluations: {stats['successful_evaluations']}\n"
                )

                if "average_performance" in stats:
                    perf = stats["average_performance"]
                    file_obj.write(
                        f"- Average word accuracy: {perf['mean_word_accuracy']:.3f} ± {perf['std_word_accuracy']:.3f}\n"
                    )

                if "top_performers" in stats:
                    file_obj.write("- Top performers:\n")
                    for model, score in stats["top_performers"].items():
                        file_obj.write(f"  - {model}: {score:.3f}\n")

                file_obj.write("\n")
