"""
Minimal base evaluator interface for all puzzle types.

This module provides the essential evaluator contract without unnecessary complexity.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseEvaluator(ABC):
    """
    Base class for all puzzle evaluators.
    
    Provides the minimal interface needed for puzzle evaluation.
    """
    
    @abstractmethod
    def evaluate_single(self, model, puzzle) -> Dict[str, Any]:
        """
        Evaluate model on a single puzzle.
        
        Args:
            model: Model interface with generate_response method
            puzzle: Puzzle object to evaluate
            
        Returns:
            Dict with evaluation results including success, metrics, etc.
        """
        pass
    
    @abstractmethod 
    def evaluate_batch(self, model, puzzles: List) -> Dict[str, Any]:
        """
        Evaluate model on multiple puzzles.
        
        Args:
            model: Model interface
            puzzles: List of puzzle objects
            
        Returns:
            Dict with batch evaluation results and summary metrics
        """
        pass