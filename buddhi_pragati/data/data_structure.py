"""
Data structure for standardized crossword clue-answer pairs.
"""

from dataclasses import dataclass


@dataclass
class DatasetEntry:
    """
    Standardized dataset entry for crossword clue-answer pairs.

    Language is now implicit from the dataset subset/configuration, not stored per entry.

    Attributes:
        id: Unique identifier for this entry
        clue: Crossword-style clue text
        answer: Single-word answer (uppercase)
        source: Source dataset name
        source_id: Original ID in source dataset
        context_score: Indian cultural context score (0.0-1.0), defaults to 0.0 for deferred computation
        quality_score: Crossword suitability score (0.0-1.0)
    """

    id: str
    clue: str
    answer: str
    source: str
    source_id: str
    context_score: float = 0.0  # Default to 0.0 for deferred context scoring
    quality_score: float = 0.0
