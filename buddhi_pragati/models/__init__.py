"""
Unified model interface supporting multiple LLM providers.

This package provides a simple, unified interface for working with different
language model providers (OpenAI, OpenRouter, HuggingFace) through a single API.

Classes:
    BaseModelInterface: Abstract model interface contract
    UnifiedModelInterface: Multi-provider model interface implementation
    OpenAIBackend: OpenAI API backend
    OpenRouterBackend: OpenRouter API backend  
    HuggingFaceBackend: HuggingFace local model backend
"""

from .model_interface import (
    BaseModelInterface,
    UnifiedModelInterface, 
    OpenAIBackend,
    OpenRouterBackend,
    HuggingFaceBackend
)

__all__ = [
    "BaseModelInterface",
    "UnifiedModelInterface",
    "OpenAIBackend", 
    "OpenRouterBackend",
    "HuggingFaceBackend"
]