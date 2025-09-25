"""
Crossword Puzzle Generation Module

This module implements enhanced crossword puzzle generation with memetic algorithms
to achieve high grid density (75%+) across grid sizes from 3x3 to 30x30.

Architecture:
- puzzle_entry: CrosswordPuzzleEntry dataclass for HF dataset format
- corpus_loader: Load and classify clue-answer corpus from existing HF datasets
- memetic_generator: Enhanced generator using genetic/memetic algorithms
- puzzle_builder: Main orchestrator for batch puzzle generation
- hf_uploader: Upload generated puzzles to HuggingFace Hub

Key Features:
- Memetic algorithm optimization for high density grids
- Indian/non-Indian cultural context classification
- Multi-attempt generation with quality selection
- Support for wide range of grid sizes (3x3 to 30x30)
- Integration with existing evaluation pipeline
"""

from .puzzle_entry import CrosswordPuzzleEntry
from .corpus_loader import CrosswordCorpusLoader
from .memetic_generator import MemeticCrosswordGenerator
from .puzzle_builder import PuzzleBuilder
from .hf_uploader import PuzzleHFUploader


__all__ = [
    "CrosswordPuzzleEntry",
    "CrosswordCorpusLoader",
    "MemeticCrosswordGenerator",
    "PuzzleBuilder",
    "PuzzleHFUploader",
]
