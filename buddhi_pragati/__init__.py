"""
Buddhi-Pragati: LLM Reasoning Benchmark System

A clean, modular benchmark system for evaluating language model reasoning
across crossword puzzles and other puzzle types.

Main Components:
- core: Base interfaces for puzzles and evaluators
- models: Unified model interface (OpenAI, HuggingFace, OpenRouter)
- puzzles: Puzzle-specific implementations (crossword, logic, math)

Quick Start:
    from buddhi_pragati.puzzles.crossword.evaluator import CrosswordEvaluator
    from buddhi_pragati.models.model_interface import UnifiedModelInterface

    evaluator = CrosswordEvaluator()
    model = UnifiedModelInterface("gpt-4o", source="openai", api_key="...")
    result = evaluator.evaluate_single(model, puzzle)
"""

