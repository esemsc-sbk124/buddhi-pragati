"""
Crossword puzzle evaluation system for Buddhi-Pragati benchmark.

This module provides comprehensive evaluation capabilities for crossword puzzles
including modular prompt templates, experiment orchestration, model classification,
and detailed metrics calculation.

Key Components:
- CrosswordEvaluator: Main evaluation orchestrator
- ModularTemplates: Composable prompt template system  
- ExperimentRunner: Orchestrates 10 experiment types
- ModelClassifier: Classifies models by capabilities and cost
- EnhancedMetrics: Word, letter, and intersection accuracy
- DatasetLoader: Loads puzzles from HuggingFace datasets
"""

from .evaluator import CrosswordEvaluator
from .metrics import EnhancedCrosswordMetrics
from .parser import CrosswordResponseParser

__all__ = [
    'CrosswordEvaluator',
    'EnhancedCrosswordMetrics', 
    'CrosswordResponseParser'
]