"""
Source-specific processors for dataset creation pipeline.

This package contains processors for different data sources:
- MILUProcessor: Handles MILU dataset MCQ questions
- BhashaWikiProcessor: Handles bhasha-wiki articles with NER extraction
- IndicWikiBioProcessor: Handles IndicWikiBio biographical entries
- IndoWordNetProcessor: Handles IndoWordNet dictionary definitions
"""

from .milu_processor import MILUProcessor
from .bhasha_wiki_processor import BhashaWikiProcessor
from .indic_wikibio_processor import IndicWikiBioProcessor
from .indowordnet_processor import IndoWordNetProcessor

__all__ = [
    'MILUProcessor',
    'BhashaWikiProcessor', 
    'IndicWikiBioProcessor',
    'IndoWordNetProcessor'
]