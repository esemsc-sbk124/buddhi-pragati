"""
Core base interfaces for the Buddhi-Pragati benchmark system.

This package contains minimal base classes that define the essential
interfaces for puzzles and evaluators.

Classes:
    BasePuzzle: Abstract base class for all puzzle types
    BaseEvaluator: Abstract base class for all evaluators  
    CrosswordPuzzle: Concrete crossword puzzle implementation
    CrosswordClue: Individual crossword clue representation
"""

from .base_puzzle import BasePuzzle, CrosswordPuzzle, CrosswordClue
from .base_evaluator import BaseEvaluator

__all__ = [
    'BasePuzzle',
    'BaseEvaluator', 
    'CrosswordPuzzle',
    'CrosswordClue'
]