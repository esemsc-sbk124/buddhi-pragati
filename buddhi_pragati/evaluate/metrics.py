"""
Enhanced crossword puzzle metrics for comprehensive evaluation.

This module implements detailed metrics and additional accuracy measures
required for the evaluation pipeline including word accuracy breakdown by direction
and proper intersection accuracy calculation.
"""

import logging
from typing import Dict, Set, Tuple
import numpy as np


class EnhancedCrosswordMetrics:
    """
    Computes comprehensive crossword puzzle evaluation metrics.

    Provides all metrics (WCR, LCR, ICR) plus enhanced accuracy
    measures including direction-specific word accuracy and proper intersection
    detection for accurate ICR calculation.
    """

    def __init__(self):
        self.logger = logging.getLogger("EnhancedCrosswordMetrics")

    def compute_metrics(
        self,
        parsed_grid,
        parsed_words: Dict[str, str],
        expected_grid,
        expected_words: Dict[str, str],
        puzzle,
    ) -> Dict[str, float]:
        """
        Score how well the model solved the crossword with comprehensive metrics.

        Args:
            parsed_grid: Model's grid solution (list or np.ndarray)
            parsed_words: Model's word answers as dict
            expected_grid: Ground truth grid (list or np.ndarray)
            expected_words: Ground truth words as dict
            puzzle: CrosswordPuzzle object for structure information

        Returns:
            Dict with computed metrics including:
            - Word accuracy (global, across, down)
            - Letter accuracy (correct cells in grid)
            - Intersection accuracy (correct intersection cells)
            - All metrics (WCR, LCR, ICR)
        """
        # Convert to numpy arrays if needed
        if not isinstance(parsed_grid, np.ndarray):
            parsed_grid = np.array(parsed_grid)
        if not isinstance(expected_grid, np.ndarray):
            expected_grid = np.array(expected_grid)

        if parsed_grid.size == 0 or expected_grid.size == 0:
            self.logger.error("Parsed grid or expected grid is empty.")
            return self._empty_metrics()

        if parsed_grid.shape != expected_grid.shape:
            self.logger.error(
                f"Grid shape mismatch: {parsed_grid.shape} vs {expected_grid.shape}"
            )
            return self._empty_metrics()

        # Basic grid accuracy (letter accuracy)
        mask = expected_grid != "#"  # Ignore blocked cells
        correct_cells = np.sum((parsed_grid == expected_grid) & mask)
        total_cells = np.sum(mask)
        letter_accuracy = correct_cells / total_cells if total_cells > 0 else 0.0

        # Word accuracy with direction breakdown
        word_metrics = self._compute_word_accuracy_by_direction(
            parsed_words, expected_words, puzzle
        )

        # Intersection accuracy with proper intersection detection
        intersection_accuracy = self._compute_intersection_accuracy(
            parsed_grid, expected_grid, puzzle
        )

        self.logger.info(f"Letter accuracy: {letter_accuracy:.3f}")
        self.logger.info(
            f"Word accuracy - Global: {word_metrics['word_accuracy_global']:.3f}"
        )
        self.logger.info(
            f"Word accuracy - Across: {word_metrics['word_accuracy_across']:.3f}"
        )
        self.logger.info(
            f"Word accuracy - Down: {word_metrics['word_accuracy_down']:.3f}"
        )
        self.logger.info(f"Intersection accuracy: {intersection_accuracy:.3f}")

        return {
            # Required accuracy metrics
            "letter_accuracy": letter_accuracy,
            "word_accuracy_global": word_metrics["word_accuracy_global"],
            "word_accuracy_across": word_metrics["word_accuracy_across"],
            "word_accuracy_down": word_metrics["word_accuracy_down"],
            "intersection_accuracy": intersection_accuracy,
            # Detailed counts for analysis
            "correct_cells": int(correct_cells),
            "total_cells": int(total_cells),
            "correct_words_across": word_metrics["correct_words_across"],
            "total_words_across": word_metrics["total_words_across"],
            "correct_words_down": word_metrics["correct_words_down"],
            "total_words_down": word_metrics["total_words_down"],
            "correct_intersections": word_metrics["correct_intersections"],
            "total_intersections": word_metrics["total_intersections"],
            # Legacy compatibility
            "grid_accuracy": letter_accuracy,
            "words_accuracy": word_metrics["word_accuracy_global"],
            "success": letter_accuracy == 1.0
            and word_metrics["word_accuracy_global"] == 1.0,
        }

    def _compute_word_accuracy_by_direction(
        self, parsed_words: Dict[str, str], expected_words: Dict[str, str], puzzle
    ) -> Dict[str, float]:
        """
        Compute word accuracy broken down by direction (across/down).

        Args:
            parsed_words: Model's answers
            expected_words: Ground truth answers
            puzzle: CrosswordPuzzle for clue direction information

        Returns:
            Dict with accuracy metrics by direction
        """
        correct_across = 0
        correct_down = 0
        total_across = 0
        total_down = 0
        correct_total = 0

        # Create direction mapping from puzzle clues
        direction_map = {}
        for clue in puzzle.get_clues():
            # Format clue key consistently with parser output
            clue_key = f"{clue.number}{clue.direction}"
            direction_map[clue_key] = clue.direction

        for clue_key, expected_answer in expected_words.items():
            direction = direction_map.get(clue_key, "unknown")

            if direction == "across":
                total_across += 1
                if (
                    clue_key in parsed_words
                    and parsed_words[clue_key] == expected_answer
                ):
                    correct_across += 1
                    correct_total += 1
            elif direction == "down":
                total_down += 1
                if (
                    clue_key in parsed_words
                    and parsed_words[clue_key] == expected_answer
                ):
                    correct_down += 1
                    correct_total += 1
            else:
                # Handle unknown direction (shouldn't happen with proper puzzle structure)
                self.logger.warning(f"Unknown direction for clue: {clue_key}")
                if (
                    clue_key in parsed_words
                    and parsed_words[clue_key] == expected_answer
                ):
                    correct_total += 1

        total_words = len(expected_words)

        return {
            "word_accuracy_global": correct_total / total_words
            if total_words > 0
            else 0.0,
            "word_accuracy_across": correct_across / total_across
            if total_across > 0
            else 0.0,
            "word_accuracy_down": correct_down / total_down if total_down > 0 else 0.0,
            "correct_words_across": correct_across,
            "total_words_across": total_across,
            "correct_words_down": correct_down,
            "total_words_down": total_down,
            "correct_intersections": 0,  # Placeholder
            "total_intersections": 0,  # Placeholder
        }

    def _compute_intersection_accuracy(
        self, parsed_grid: np.ndarray, expected_grid: np.ndarray, puzzle
    ) -> float:
        """
        Compute accuracy specifically for intersection cells.

        Args:
            parsed_grid: Model's grid
            expected_grid: Ground truth grid
            puzzle: CrosswordPuzzle for intersection detection

        Returns:
            Intersection accuracy as float
        """
        intersection_positions = self._find_intersection_positions(puzzle)

        if not intersection_positions:
            self.logger.warning("No intersections found in puzzle")
            return 0.0

        correct_intersections = 0
        for row, col in intersection_positions:
            if (
                row < parsed_grid.shape[0]
                and col < parsed_grid.shape[1]
                and parsed_grid[row, col] == expected_grid[row, col]
            ):
                correct_intersections += 1

        return correct_intersections / len(intersection_positions)

    def _find_intersection_positions(self, puzzle) -> Set[Tuple[int, int]]:
        """
        Find all intersection positions in the crossword grid.

        Args:
            puzzle: CrosswordPuzzle object

        Returns:
            Set of (row, col) tuples representing intersection positions
        """
        # Track all positions occupied by words
        position_occupancy = {}  # (row, col) -> list of clue IDs

        for clue in puzzle.get_clues():
            positions = []
            for i in range(clue.length):
                if clue.direction == "across":
                    pos = (clue.start_row, clue.start_col + i)
                else:  # down
                    pos = (clue.start_row + i, clue.start_col)
                positions.append(pos)

                # Track occupancy
                if pos not in position_occupancy:
                    position_occupancy[pos] = []
                position_occupancy[pos].append(f"{clue.number}{clue.direction}")

        # Find intersections (positions occupied by more than one word)
        intersections = set()
        for pos, occupants in position_occupancy.items():
            if len(occupants) > 1:
                intersections.add(pos)

        self.logger.debug(f"Found {len(intersections)} intersection positions")
        return intersections

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict for error cases."""
        return {
            "letter_accuracy": 0.0,
            "word_accuracy_global": 0.0,
            "word_accuracy_across": 0.0,
            "word_accuracy_down": 0.0,
            "intersection_accuracy": 0.0,
            "correct_cells": 0,
            "total_cells": 0,
            "correct_words_across": 0,
            "total_words_across": 0,
            "correct_words_down": 0,
            "total_words_down": 0,
            "correct_intersections": 0,
            "total_intersections": 0,
            "wcr": 0.0,
            "lcr": 0.0,
            "icr": 0.0,
            "grid_accuracy": 0.0,
            "words_accuracy": 0.0,
            "success": False,
        }
