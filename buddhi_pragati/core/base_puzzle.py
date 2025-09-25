"""
Minimal base puzzle interface for crossword and future puzzle types.

This module provides the essential interface that all puzzle types must implement
without unnecessary complexity.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class BasePuzzle(ABC):
    """
    Base class for all puzzle types.

    Provides the minimal interface needed for evaluation.
    """

    puzzle_id: str
    size: Tuple[int, int]

    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        """Return puzzle dimensions."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert puzzle to dictionary for serialization."""
        pass


class CrosswordClue:
    """
    Crossword clue representation with position and direction information.

    Represents a single clue in a crossword puzzle with all necessary
    positioning and metadata for puzzle construction and evaluation.

    Attributes:
        number: Clue number in the puzzle
        direction: "across" or "down" direction
        length: Length of the answer in characters
        clue_text: Human-readable clue text
        start_row: Starting row position in grid
        start_col: Starting column position in grid
        answer: Expected answer (empty string if unknown)
    """

    def __init__(
        self,
        number: int,
        direction: str,
        length: int,
        clue_text: str,
        start_row: int,
        start_col: int,
        answer: str = "",
    ):
        """
        Initialize a crossword clue.

        Args:
            number: Clue number in the puzzle (1-based)
            direction: Direction of the answer ("across" or "down")
            length: Expected length of the answer in characters
            clue_text: Human-readable clue text
            start_row: Starting row position in grid (0-based)
            start_col: Starting column position in grid (0-based)
            answer: Expected answer string (optional, defaults to empty)
        """
        self.number = number
        self.direction = direction  # "across" or "down"
        self.length = length
        self.clue_text = clue_text
        self.start_row = start_row
        self.start_col = start_col
        self.answer = answer


class CrosswordPuzzle(BasePuzzle):
    """
    Concrete crossword puzzle implementation extending BasePuzzle.

    Represents a complete crossword puzzle with grid, clues, and solutions.
    Provides methods for accessing puzzle components and serialization.

    Attributes:
        puzzle_id: Unique identifier for the puzzle
        size: Grid dimensions as (rows, cols) tuple
        grid: Empty grid with numbers and blocked cells
        clues: List of CrosswordClue objects
        solution_grid: Complete solution grid with all answers
        solution_words: Dictionary mapping clue numbers to answers
    """

    def __init__(
        self,
        puzzle_id: str,
        grid,
        clues: List[CrosswordClue],
        size: Tuple[int, int],
        solution_grid=None,
        solution_words: Dict[str, str] = None,
    ):
        """
        Initialize a crossword puzzle.

        Args:
            puzzle_id: Unique identifier for the puzzle
            grid: Empty grid with numbers and blocked cells for solving
            clues: List of CrosswordClue objects
            size: Grid dimensions as (rows, cols) tuple
            solution_grid: Complete solution grid with all answers (required)
            solution_words: Dictionary mapping clue numbers to answers (optional)

        Raises:
            ValueError: If solution_grid is None
        """
        super().__init__(puzzle_id, size)
        self.grid = grid
        self.clues = clues

        if solution_grid is None:
            raise ValueError(
                f"CrosswordPuzzle {puzzle_id} must be initialized with a valid solution_grid"
            )

        self.solution_grid = solution_grid
        self.solution_words = solution_words or {}

    def get_size(self) -> Tuple[int, int]:
        """
        Get puzzle dimensions.

        Returns:
            Tuple of (rows, cols) representing the grid size
        """
        return self.size

    def get_grid(self):
        """
        Get the empty puzzle grid for solving.

        Returns:
            Grid with numbers and blocked cells, answers removed
        """
        return self.grid

    def get_clues(self) -> List[CrosswordClue]:
        """
        Get all clues for this puzzle.

        Returns:
            List of CrosswordClue objects
        """
        return self.clues

    def get_solution_grid(self):
        """
        Get the complete solution grid.
        Returns:
            Grid with all answers filled in
        """
        return self.solution_grid

    def get_solution_words(self) -> Dict[str, str]:
        """
        Get the solution words mapping.
        Returns:
            Dictionary mapping clue numbers to their answers
        """
        return self.solution_words

    def to_dict(self) -> Dict[str, Any]:
        if self.solution_grid is None:
            raise ValueError(
                f"Puzzle {self.puzzle_id} has None solution_grid - this should not happen after proper generation"
            )

        return {
            "puzzle_id": self.puzzle_id,
            "size": self.size,
            "grid": self.grid.tolist() if hasattr(self.grid, "tolist") else self.grid,
            "clues": [
                {
                    "number": clue.number,
                    "direction": clue.direction,
                    "length": clue.length,
                    "clue_text": clue.clue_text,
                    "start_row": clue.start_row,
                    "start_col": clue.start_col,
                    "answer": clue.answer,
                }
                for clue in self.clues
            ],
            "solution_grid": self.solution_grid.tolist()
            if hasattr(self.solution_grid, "tolist")
            else self.solution_grid,
            "solution_words": self.solution_words,
        }
