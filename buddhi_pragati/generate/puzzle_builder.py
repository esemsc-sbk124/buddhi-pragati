"""
Puzzle Builder - Main Orchestrator for Crossword Puzzle Generation

This module coordinates the complete crossword generation pipeline from corpus loading
to puzzle creation and HuggingFace dataset upload.
"""

import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
import gc
from collections import defaultdict

from .corpus_loader import CrosswordCorpusLoader
from .memetic_generator import MemeticCrosswordGenerator
from .puzzle_entry import CrosswordPuzzleEntry
from .hf_uploader import PuzzleHFUploader
from ..utils.config_loader import get_config
from ..utils.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class UsedPairTracker:
    """
    Tracks usage of clue-answer pairs across puzzle generation to ensure diversity.

    Enforces maximum usage percentage per pair to prevent repetitive puzzles
    that would allow LLM memorization.
    """

    def __init__(self, max_reuse_percentage: float = 0.10):
        """
        Initialize tracker.

        Args:
            max_reuse_percentage: Maximum percentage of puzzles that can use the same pair (0.0-1.0)
        """
        self.max_reuse_percentage = max_reuse_percentage
        self.pair_usage = defaultdict(int)  # (clue, answer) -> usage_count
        self.total_puzzles = 0

        logger.info(
            f"Initialized UsedPairTracker with {max_reuse_percentage:.1%} max reuse"
        )

    def can_use_pair(self, clue: str, answer: str) -> bool:
        """
        Check if a pair can be used without exceeding reuse limit.

        Args:
            clue: Clue text
            answer: Answer text

        Returns:
            True if pair can be used, False if overused
        """
        if self.total_puzzles == 0:
            return True  # Always allow first puzzle

        pair_key = (clue.strip(), answer.strip().upper())
        current_usage = self.pair_usage.get(pair_key, 0)
        max_allowed_usage = max(1, int(self.total_puzzles * self.max_reuse_percentage))

        return current_usage < max_allowed_usage

    def filter_corpus(
        self, corpus_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Filter corpus to remove overused pairs.

        Args:
            corpus_pairs: List of (clue, answer) pairs

        Returns:
            Filtered list excluding overused pairs
        """
        if self.total_puzzles == 0:
            return corpus_pairs  # No filtering needed for first puzzle

        filtered_pairs = []
        for clue, answer in corpus_pairs:
            if self.can_use_pair(clue, answer):
                filtered_pairs.append((clue, answer))

        filtered_count = len(corpus_pairs) - len(filtered_pairs)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} overused pairs from corpus")

        return filtered_pairs

    def record_puzzle_usage(self, used_pairs: List[Tuple[str, str]]):
        """
        Record pairs used in a completed puzzle.

        Args:
            used_pairs: List of (clue, answer) pairs used in the puzzle
        """
        for clue, answer in used_pairs:
            pair_key = (clue.strip(), answer.strip().upper())
            self.pair_usage[pair_key] += 1

        self.total_puzzles += 1

        logger.debug(
            f"Recorded usage for {len(used_pairs)} pairs. Total puzzles: {self.total_puzzles}"
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for monitoring.

        Returns:
            Dictionary with usage statistics
        """
        if not self.pair_usage:
            return {"total_puzzles": 0, "unique_pairs": 0, "reuse_stats": {}}

        usage_counts = list(self.pair_usage.values())
        reuse_stats = {
            "max_usage": max(usage_counts),
            "avg_usage": sum(usage_counts) / len(usage_counts),
            "pairs_used_once": sum(1 for count in usage_counts if count == 1),
            "pairs_reused": sum(1 for count in usage_counts if count > 1),
        }

        return {
            "total_puzzles": self.total_puzzles,
            "unique_pairs": len(self.pair_usage),
            "max_reuse_percentage": self.max_reuse_percentage,
            "reuse_stats": reuse_stats,
        }


class PuzzleBuilder:
    """
    Main orchestrator for crossword puzzle generation pipeline.

    Handles the complete workflow:
    1. Load and prioritize corpus from HF dataset
    2. Generate puzzles using memetic algorithm
    3. Convert to HF dataset format
    4. Upload to HuggingFace Hub (optional)
    """

    def __init__(self, hf_token: str = None):
        """
        Initialize puzzle builder.

        Args:
            hf_token: HuggingFace token for dataset operations
        """
        self.config = get_config()
        self.hf_token = hf_token or self.config.get_string("DEFAULT_HF_TOKEN", "")

        # Initialize components
        self.corpus_loader = CrosswordCorpusLoader(hf_token=self.hf_token)
        self.generators = {}  # Cache generators by grid size
        self.hf_uploader = PuzzleHFUploader(hf_token=self.hf_token)
        self.dataset_manager = DatasetManager(hf_token=self.hf_token)
        self.puzzles_repo = self.config.get_string(
            "HF_GENERATED_PUZZLES_REPO", "selim-b-kh/buddhi-pragati-puzzles"
        )

        # Generation statistics
        self.generation_stats = {
            "total_attempts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_density": 0.0,
            "avg_word_count": 0.0,
            "source_distribution": {},
        }

        # Diversity management
        max_reuse_percentage = self.config.get_float("MAX_PAIR_REUSE_PERCENTAGE", 0.10)
        self.pair_tracker = UsedPairTracker(max_reuse_percentage=max_reuse_percentage)

        logger.info("Initialized PuzzleBuilder")

    def generate_puzzle_batch(
        self,
        language: str,
        grid_size: int,
        count: int,
        max_corpus_size: int = None,
        upload_to_hf: bool = False,
        output_dir: str = None,
    ) -> List[CrosswordPuzzleEntry]:
        """
        Generate a batch of crossword puzzles for a specific language and grid size.

        Args:
            language: Language to generate puzzles for (e.g., "English", "Hindi")
            grid_size: Grid size (3-30)
            count: Number of puzzles to generate
            max_corpus_size: Maximum corpus size to load (None for all)
            upload_to_hf: Whether to upload generated puzzles to HuggingFace
            output_dir: Directory to save puzzle JSON files (None to skip)

        Returns:
            List of CrosswordPuzzleEntry instances
        """
        if not (3 <= grid_size <= 30):
            raise ValueError(
                f"Grid size {grid_size} not supported. Must be between 3 and 30."
            )

        logger.info(
            f"Generating {count} puzzles for {language} ({grid_size}x{grid_size})"
        )

        # Load and prioritize corpus
        prioritized_corpus, context_scores = self.corpus_loader.get_prioritized_corpus(
            language=language, max_entries=max_corpus_size
        )

        if not prioritized_corpus:
            logger.error(f"Failed to load corpus for {language}")
            return []

        # Get generator for this grid size
        generator = self._get_generator(grid_size)

        # Get corpus entries for source lookup
        corpus_entries = self.corpus_loader.load_scored_corpus(language)

        # Create lookup: answer -> source
        answer_to_source = {entry.answer: entry.source for entry in corpus_entries}

        generated_puzzles = []
        successful_count = 0

        for i in range(count):
            puzzle_id = self._generate_puzzle_id(language, grid_size, i + 1)

            logger.info(f"Generating puzzle {i + 1}/{count}: {puzzle_id}")
            self.generation_stats["total_attempts"] += 1

            # Filter corpus to remove overused pairs for diversity
            filtered_corpus = self.pair_tracker.filter_corpus(prioritized_corpus)

            if not filtered_corpus:
                logger.warning(
                    f"❌ No available pairs for puzzle {i + 1} (all pairs overused)"
                )
                self.generation_stats["failed_generations"] += 1
                continue

            try:
                # Generate puzzle using memetic algorithm with filtered corpus
                puzzle = generator.generate_puzzle_with_prioritization(
                    prioritized_corpus=filtered_corpus,
                    context_scores=context_scores,
                    puzzle_id=puzzle_id,
                )

                if puzzle:
                    # Convert to HF dataset format
                    puzzle_entry = self._puzzle_to_entry(
                        puzzle=puzzle,
                        context_scores=context_scores,
                        answer_to_source=answer_to_source,
                        language=language,
                        grid_size=grid_size,
                    )

                    # Record usage for diversity tracking
                    used_pairs = [
                        (clue.clue_text, clue.answer) for clue in puzzle.get_clues()
                    ]
                    self.pair_tracker.record_puzzle_usage(used_pairs)

                    generated_puzzles.append(puzzle_entry)
                    successful_count += 1
                    self.generation_stats["successful_generations"] += 1

                    # Update running statistics
                    self._update_generation_stats(puzzle_entry)

                    logger.info(
                        f"✅ Generated puzzle {i + 1}: {puzzle_entry.density:.1%} density, "
                        f"{puzzle_entry.word_count} words, context score {puzzle_entry.context_score:.3f}"
                    )

                    # Save to file if requested
                    if output_dir:
                        self._save_puzzle_to_file(puzzle_entry, output_dir)

                else:
                    logger.warning(f"❌ Failed to generate puzzle {i + 1}")
                    self.generation_stats["failed_generations"] += 1

            except Exception as e:
                logger.error(f"❌ Error generating puzzle {i + 1}: {e}")
                self.generation_stats["failed_generations"] += 1

            # Memory cleanup
            if i % 5 == 0:
                gc.collect()

        logger.info(
            f"Batch generation complete: {successful_count}/{count} puzzles generated"
        )

        # Upload to HuggingFace if requested
        if upload_to_hf and generated_puzzles:
            logger.info("Uploading generated puzzles to HuggingFace...")
            try:
                # Check and create language-specific configuration if needed
                self._ensure_language_configuration(language, grid_size)

                upload_success = self.hf_uploader.upload_puzzle_batch(
                    puzzles=generated_puzzles, language=language, grid_size=grid_size
                )
                if upload_success:
                    logger.info("✅ Successfully uploaded puzzles to HuggingFace")
                else:
                    logger.error("❌ Failed to upload puzzles to HuggingFace")
            except Exception as e:
                logger.error(f"❌ Upload error: {e}")

        # Clean up corpus loader cache to free memory
        self.corpus_loader.clear_cache()

        return generated_puzzles

    def generate_multi_size_batch(
        self,
        language: str,
        grid_sizes: List[int],
        count_per_size: int,
        upload_to_hf: bool = False,
        output_dir: str = None,
    ) -> Dict[int, List[CrosswordPuzzleEntry]]:
        """
        Generate puzzles across multiple grid sizes for a language.

        Args:
            language: Language to generate puzzles for
            grid_sizes: List of grid sizes to generate
            count_per_size: Number of puzzles per grid size
            upload_to_hf: Whether to upload to HuggingFace
            output_dir: Directory to save puzzles (None to skip)

        Returns:
            Dictionary mapping grid_size -> List[CrosswordPuzzleEntry]
        """
        logger.info(f"Generating multi-size batch for {language}: {grid_sizes}")

        # Pre-check all configurations if uploading
        if upload_to_hf:
            for grid_size in grid_sizes:
                self._ensure_language_configuration(language, grid_size)

        results = {}

        for grid_size in grid_sizes:
            logger.info(f"Processing grid size {grid_size}x{grid_size}")

            puzzles = self.generate_puzzle_batch(
                language=language,
                grid_size=grid_size,
                count=count_per_size,
                upload_to_hf=upload_to_hf,
                output_dir=output_dir,
            )

            results[grid_size] = puzzles
            logger.info(
                f"Completed {grid_size}x{grid_size}: {len(puzzles)} puzzles generated"
            )

        return results

    def _get_generator(self, grid_size: int) -> MemeticCrosswordGenerator:
        """Get or create generator for specific grid size."""
        if grid_size not in self.generators:
            self.generators[grid_size] = MemeticCrosswordGenerator(grid_size)

        return self.generators[grid_size]

    def _generate_puzzle_id(self, language: str, grid_size: int, sequence: int) -> str:
        """Generate unique puzzle identifier."""
        return f"{language.lower()}_{grid_size}x{grid_size}_{sequence:03d}"

    def _puzzle_to_entry(
        self,
        puzzle,
        context_scores: Dict[str, float],
        answer_to_source: Dict[str, str],
        language: str,
        grid_size: int,
    ) -> CrosswordPuzzleEntry:
        """Convert CrosswordPuzzle to CrosswordPuzzleEntry for HF dataset."""

        # Get actual words used in this puzzle
        used_answers = [clue.answer for clue in puzzle.get_clues()]

        # Count actual source usage for this specific puzzle
        source_counts = {}
        for answer in used_answers:
            source = answer_to_source.get(answer, "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        # Define all possible sources to ensure consistent schema across languages
        ALL_SOURCES = ["MILU", "IndicWikiBio", "IndoWordNet", "Bhasha-Wiki"]

        # Calculate source mix as percentages (always include all sources)
        total_words = len(used_answers)
        source_mix_percentages = {}

        if total_words > 0:
            for source in ALL_SOURCES:
                count = source_counts.get(source, 0)
                percentage = round((count / total_words) * 100, 2)
                source_mix_percentages[source] = percentage
        else:
            # If no words used, all percentages are 0.00%
            for source in ALL_SOURCES:
                source_mix_percentages[source] = 0.00

        # Generation metadata
        generation_metadata = {
            "language": language,
            "grid_size": grid_size,
            "generation_algorithm": "memetic",
            "generation_timestamp": datetime.now().isoformat(),
            "corpus_source_repository": self.corpus_loader.repo_id,
            "total_corpus_size": len(context_scores),
        }

        return CrosswordPuzzleEntry.from_crossword_puzzle(
            puzzle=puzzle,
            context_scores={
                clue.answer: context_scores.get(clue.answer, 0.0)
                for clue in puzzle.get_clues()
            },
            source_counts=source_mix_percentages,
            generation_info=generation_metadata,
        )

    def _update_generation_stats(self, puzzle_entry: CrosswordPuzzleEntry):
        """Update running generation statistics."""
        # Update averages
        total_successful = self.generation_stats["successful_generations"]

        # Running average for density
        if total_successful == 1:
            self.generation_stats["avg_density"] = puzzle_entry.density
        else:
            self.generation_stats["avg_density"] = (
                self.generation_stats["avg_density"] * (total_successful - 1)
                + puzzle_entry.density
            ) / total_successful

        # Running average for word count
        if total_successful == 1:
            self.generation_stats["avg_word_count"] = puzzle_entry.word_count
        else:
            self.generation_stats["avg_word_count"] = (
                self.generation_stats["avg_word_count"] * (total_successful - 1)
                + puzzle_entry.word_count
            ) / total_successful

        # Aggregate source distribution as running averages of percentages
        for source, percentage in puzzle_entry.source_mix.items():
            if source not in self.generation_stats["source_distribution"]:
                self.generation_stats["source_distribution"][source] = percentage
            else:
                # Running average of percentages across puzzles
                current_avg = self.generation_stats["source_distribution"][source]
                new_avg = (
                    current_avg * (total_successful - 1) + percentage
                ) / total_successful
                self.generation_stats["source_distribution"][source] = round(new_avg, 2)

    def _save_puzzle_to_file(self, puzzle_entry: CrosswordPuzzleEntry, output_dir: str):
        """Save puzzle entry to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{puzzle_entry.id}.json"
        filepath = output_path / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(puzzle_entry.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved puzzle to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save puzzle to {filepath}: {e}")

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics."""
        success_rate = (
            (
                self.generation_stats["successful_generations"]
                / self.generation_stats["total_attempts"]
            )
            if self.generation_stats["total_attempts"] > 0
            else 0.0
        )

        return {
            "total_attempts": self.generation_stats["total_attempts"],
            "successful_generations": self.generation_stats["successful_generations"],
            "failed_generations": self.generation_stats["failed_generations"],
            "success_rate": f"{success_rate:.1%}",
            "average_density": f"{self.generation_stats['avg_density']:.1%}",
            "average_word_count": f"{self.generation_stats['avg_word_count']:.1f}",
            "source_distribution": self.generation_stats["source_distribution"],
        }

    def clear_caches(self):
        """Clear all caches to free memory."""
        self.corpus_loader.clear_cache()
        self.generators.clear()
        gc.collect()
        logger.info("Cleared PuzzleBuilder caches")

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages from corpus loader."""
        return self.corpus_loader.get_supported_languages()

    def _ensure_language_configuration(self, language: str, grid_size: int):
        """
        Ensure that a language-specific configuration exists in the generated puzzles dataset.

        Args:
            language: Target language
            grid_size: Target grid size
        """
        config_name = f"{language.lower()}"

        try:
            # Check if configuration already exists in the puzzles repository
            existing_configs = self.dataset_manager.list_dataset_configurations(
                self.puzzles_repo
            )

            if config_name in existing_configs:
                logger.debug(
                    f"Configuration {config_name} already exists in {self.puzzles_repo}"
                )
                return

            logger.info(
                f"Configuration {config_name} does not exist yet - will be created during upload"
            )

            # Configuration will be created automatically when first puzzles are uploaded
            # The HuggingFace uploader handles this in upload_puzzle_batch

        except Exception as e:
            logger.warning(
                f"Could not check existing configurations in {self.puzzles_repo}: {e}"
            )
            logger.info(
                f"Proceeding with upload - configuration {config_name} will be created if needed"
            )

    def validate_corpus_for_grid_size(
        self, language: str, grid_size: int
    ) -> Dict[str, Any]:
        """
        Validate that corpus has sufficient entries for given grid size.

        Args:
            language: Language to check
            grid_size: Target grid size

        Returns:
            Dictionary with validation results
        """
        try:
            stats = self.corpus_loader.get_corpus_statistics(language)

            if "error" in stats:
                return {"valid": False, "error": stats["error"]}

            # Check word length suitability for grid size
            if grid_size <= 7:
                suitable_count = stats["answer_length"]["suitable_for_small_grids"]
            else:
                suitable_count = stats["answer_length"]["suitable_for_large_grids"]

            min_required = (
                self.config.get_generation_config()["min_words_per_puzzle"] * 10
            )  # Safety factor heuristically determined

            validation_result = {
                "valid": suitable_count >= min_required,
                "total_entries": stats["total_entries"],
                "suitable_for_grid_size": suitable_count,
                "minimum_required": min_required,
                "recommendation": (
                    f"Corpus has {suitable_count} suitable entries for {grid_size}x{grid_size} grid. "
                    f"{'✅ Sufficient' if suitable_count >= min_required else '⚠️ May be insufficient'}"
                ),
            }

            return validation_result

        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {e}"}
