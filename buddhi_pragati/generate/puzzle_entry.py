"""
CrosswordPuzzleEntry format for HuggingFace dataset storage.

This module defines the data structure for storing generated crossword puzzles
in HuggingFace datasets, different from the evaluation-focused CrosswordPuzzle class.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Union
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrosswordPuzzleEntry:
    """
    HuggingFace dataset entry for generated crossword puzzles.

    This format is optimized for dataset storage and differs from the
    CrosswordPuzzle class used in evaluation pipeline.

    Attributes:
        id: Unique puzzle identifier (e.g., "puzzle_hindi_15x15_001")
        clues_with_answer: Dictionary mapping clue identifiers to clue data
                          Format: {"1across": {"clue": "...", "answer": "...", "start": [row, col]}}
        empty_grid: Grid with numbers and blocked cells for solving
        solved_grid: Complete solution grid with all letters
        context_score: Mean indian_context_score of all placed words (0.0-1.0)
        quality_score: Overall puzzle quality metric (0.0-1.0)
        source_mix: Dictionary of source percentages (e.g., {"MILU": 45.67, "IndicWikiBio": 25.33, "IndoWordNet": 29.00, "Bhasha-Wiki": 0.00})
        grid_size: Grid size as integer for square grids (e.g., 10 for 10x10)
        density: Actual grid fill density achieved (0.0-1.0)
        word_count: Number of words placed in puzzle
        generation_metadata: Additional generation information
    """

    id: str
    clues_with_answer: Dict[str, Dict[str, Union[str, List[int]]]]
    empty_grid: List[List[Union[str, int, None]]]
    solved_grid: List[List[str]]
    context_score: float
    quality_score: float
    source_mix: Dict[str, float]
    grid_size: int
    density: float
    word_count: int
    generation_metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate puzzle entry after initialization."""
        # Validate required fields
        if not self.id:
            raise ValueError("Puzzle ID cannot be empty")

        if not self.clues_with_answer:
            raise ValueError("Puzzle must have at least one clue")

        if not self.empty_grid or not self.solved_grid:
            raise ValueError("Both empty_grid and solved_grid are required")

        # Validate grid dimensions consistency
        if len(self.empty_grid) != len(self.solved_grid):
            raise ValueError("Empty grid and solved grid must have same dimensions")

        # Validate grid size for square grids
        actual_rows = len(self.empty_grid)
        actual_cols = len(self.empty_grid[0]) if self.empty_grid else 0

        if actual_rows != self.grid_size or actual_cols != self.grid_size:
            raise ValueError(f"Grid size must match actual grid dimensions: expected {self.grid_size}x{self.grid_size}, got {actual_rows}x{actual_cols}")

        # Validate score ranges
        if not (0.0 <= self.context_score <= 1.0):
            raise ValueError("Context score must be between 0.0 and 1.0")

        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")

        if not (0.0 <= self.density <= 1.0):
            raise ValueError("Density must be between 0.0 and 1.0")

        # Validate word count matches clues
        if self.word_count != len(self.clues_with_answer):
            raise ValueError("Word count must match number of clues")

    @classmethod
    def from_crossword_puzzle(
        cls,
        puzzle,
        context_scores: Dict[str, float],
        source_counts: Dict[str, int],
        generation_info: Dict[str, Any],
    ):
        """
        Create CrosswordPuzzleEntry from CrosswordPuzzle (evaluation format).

        Args:
            puzzle: CrosswordPuzzle instance from generator
            context_scores: Dictionary mapping word->context_score
            source_counts: Dictionary mapping source->count
            generation_info: Additional generation metadata

        Returns:
            CrosswordPuzzleEntry instance
        """
        # Convert clues to HF format
        clues_with_answer = {}
        for clue in puzzle.get_clues():
            clue_id = f"{clue.number}{clue.direction}"
            clues_with_answer[clue_id] = {
                "clue": clue.clue_text,
                "answer": clue.answer,
                "start": [clue.start_row, clue.start_col],
                "length": clue.length,
                "direction": clue.direction,
            }

        # Calculate mean context score
        if context_scores:
            mean_context = sum(context_scores.values()) / len(context_scores)
        else:
            mean_context = 0.0

        # Calculate density from solved grid
        solved_grid = puzzle.get_solution_grid()
        if solved_grid:
            filled_cells = sum(
                1 for row in solved_grid for cell in row if cell and cell != "#"
            )
            total_cells = len(solved_grid) * len(solved_grid[0])
            density = filled_cells / total_cells if total_cells > 0 else 0.0
        else:
            density = 0.0

        # Calculate quality score (can be enhanced later)
        quality_score = min(1.0, density * 1.2 + mean_context * 0.3)

        # Get grid size - assume square grids
        puzzle_size = puzzle.get_size()
        if isinstance(puzzle_size, (list, tuple)):
            grid_size = puzzle_size[0]  # Use first dimension for square grids
        else:
            grid_size = puzzle_size

        return cls(
            id=puzzle.puzzle_id,
            clues_with_answer=clues_with_answer,
            empty_grid=puzzle.get_grid().tolist()
            if hasattr(puzzle.get_grid(), "tolist")
            else puzzle.get_grid(),
            solved_grid=solved_grid,
            context_score=mean_context,
            quality_score=quality_score,
            source_mix=source_counts,
            grid_size=grid_size,
            density=density,
            word_count=len(puzzle.get_clues()),
            generation_metadata=generation_info,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HuggingFace dataset with consistent schema."""
        data = asdict(self)

        # Convert grid values to strings for HuggingFace compatibility
        # This fixes the mixed type issue where grids contain both integers and strings
        data["empty_grid"] = [[str(cell) for cell in row] for row in self.empty_grid]
        data["solved_grid"] = [[str(cell) for cell in row] for row in self.solved_grid]

        # Fix clues schema inconsistency by converting to list format
        # This ensures consistent schema across all puzzles regardless of clue IDs
        clues_list = []
        for clue_id, clue_info in self.clues_with_answer.items():
            clue_entry = {
                "id": clue_id,
                "clue": clue_info["clue"],
                "answer": clue_info["answer"],
                "direction": clue_info["direction"],
                "start_row": clue_info["start"][0],
                "start_col": clue_info["start"][1],
                "length": clue_info["length"],
            }
            clues_list.append(clue_entry)

        data["clues"] = clues_list
        # Remove the original inconsistent field
        del data["clues_with_answer"]

        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def validate_grid_consistency(self) -> bool:
        """Validate that empty_grid and solved_grid are consistent."""
        try:
            for i, row in enumerate(self.empty_grid):
                for j, cell in enumerate(row):
                    solved_cell = self.solved_grid[i][j]

                    # If empty grid has a number, solved grid should have a letter
                    if isinstance(cell, int) and cell > 0:
                        if not (isinstance(solved_cell, str) and solved_cell.isalpha()):
                            logger.warning(
                                f"Inconsistency at [{i},{j}]: empty has number {cell}, solved has '{solved_cell}'"
                            )
                            return False

                    # If empty grid has blocked cell, solved should too
                    elif cell == "#":
                        if solved_cell != "#":
                            logger.warning(
                                f"Inconsistency at [{i},{j}]: empty blocked but solved has '{solved_cell}'"
                            )
                            return False

            return True

        except (IndexError, KeyError) as e:
            logger.error(f"Grid validation error: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about this puzzle."""
        stats = {
            "puzzle_id": self.id,
            "grid_size": f"{self.grid_size}x{self.grid_size}",
            "word_count": self.word_count,
            "density": f"{self.density:.1%}",
            "context_score": f"{self.context_score:.3f}",
            "quality_score": f"{self.quality_score:.3f}",
            "sources_used": len(self.source_mix),
            "source_distribution": self.source_mix,
        }

        # Add direction analysis
        across_count = sum(
            1
            for clue_data in self.clues_with_answer.values()
            if clue_data.get("direction") == "across"
        )
        down_count = self.word_count - across_count

        stats["direction_balance"] = {
            "across": across_count,
            "down": down_count,
            "balance_ratio": min(across_count, down_count)
            / max(across_count, down_count)
            if max(across_count, down_count) > 0
            else 0,
        }

        return stats
