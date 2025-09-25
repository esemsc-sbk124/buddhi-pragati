"""
Modular crossword prompt template system with multilingual support.

This module provides composable prompt templates that support all 19 languages
and various experimental conditions like few-shot, chain-of-thought, and
different reasoning effort levels.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

from ..utils.config_loader import get_config


class PromptLoader:
    """
    Loads and manages multilingual prompt templates from JSON files.
    """

    def __init__(self):
        self.logger = logging.getLogger("PromptLoader")
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._prompt_cache = {}

    def load_prompts(self, filename: str) -> Dict:
        """Load prompts from a JSON file with caching."""
        if filename in self._prompt_cache:
            return self._prompt_cache[filename]

        filepath = self.prompts_dir / filename
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            self._prompt_cache[filename] = prompts
            return prompts
        except FileNotFoundError:
            self.logger.error(f"Prompt file not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {filepath}: {e}")
            return {}

    def get_prompt(self, filename: str, key: str) -> str:
        """Get a specific prompt (all prompts are now in English)."""
        prompts = self.load_prompts(filename)
        if key in prompts and "English" in prompts[key]:
            return prompts[key]["English"]

        self.logger.error(f"Prompt not found: {key} in {filename}")
        return f"[Missing prompt: {key}]"


class CrosswordPromptTemplate:
    """
    Composable crossword prompt template system.

    Supports various experimental conditions:
    - Zero-shot, one-shot, few-shot prompting
    - Chain of thought vs direct answering
    - High vs low reasoning effort
    - Self-reflection capabilities
    - Multilingual evaluation with English prompts
    """

    def __init__(self, target_language: str = "English"):
        self.target_language = target_language  # Language being evaluated
        self.loader = PromptLoader()
        self.logger = logging.getLogger("CrosswordPromptTemplate")
        self.config = get_config()

        # Experimental configuration with config defaults
        self.shot_type = "zero-shot"  # zero-shot, one-shot, few-shot
        self.chain_of_thought = self.config.get_bool("DEFAULT_CHAIN_OF_THOUGHT", False)
        self.reasoning_effort = self.config.get(
            "DEFAULT_REASONING_EFFORT", "normal"
        )  # low, normal, high
        self.self_reflection = self.config.get_bool("DEFAULT_SELF_REFLECTION", False)
        self.batch_size = int(self.config.get("DEFAULT_EVALUATION_BATCH_SIZE", "1"))

    def _inject_language(self, prompt_text: str) -> str:
        """Inject target evaluation language into prompt placeholders."""
        return prompt_text.format(language=self.target_language)

    def configure_experiment(
        self,
        shot_type: str = "zero-shot",
        chain_of_thought: bool = False,
        reasoning_effort: str = "normal",
        self_reflection: bool = False,
        batch_size: int = 1,
    ):
        """Configure experimental parameters for prompt generation."""
        self.shot_type = shot_type
        self.chain_of_thought = chain_of_thought
        self.reasoning_effort = reasoning_effort
        self.self_reflection = self_reflection
        self.batch_size = batch_size

    def format_puzzle_for_model(
        self, puzzle, few_shot_examples: Optional[List] = None
    ) -> str:
        """
        Generate a complete prompt for the model based on experimental configuration.

        Args:
            puzzle: CrosswordPuzzle object or list of puzzles for batch evaluation
            few_shot_examples: List of example puzzles for few-shot prompting

        Returns:
            Complete formatted prompt string
        """
        prompt_parts = []

        # Add few-shot examples if configured
        if self.shot_type in ["one-shot", "few-shot"] and few_shot_examples:
            prompt_parts.append(self._format_few_shot_section(few_shot_examples))

        # Add reasoning effort instruction
        if self.reasoning_effort == "high":
            instruction = self.loader.get_prompt(
                "reasoning_effort.json", "high_reasoning_instruction"
            )
            prompt_parts.append(instruction)
        elif self.reasoning_effort == "low":
            instruction = self.loader.get_prompt(
                "reasoning_effort.json", "low_reasoning_instruction"
            )
            prompt_parts.append(instruction)

        # Add chain of thought instruction
        if self.chain_of_thought:
            instruction = self.loader.get_prompt(
                "chain_of_thought.json", "chain_of_thought_instruction"
            )
            prompt_parts.append(instruction)
        else:
            instruction = self.loader.get_prompt(
                "chain_of_thought.json", "no_chain_of_thought_instruction"
            )
            prompt_parts.append(instruction)

        # Add main puzzle(s)
        if isinstance(puzzle, list):
            # Batch evaluation
            prompt_parts.append(self._format_batch_puzzles(puzzle))
        else:
            # Single puzzle
            prompt_parts.append(self._format_single_puzzle(puzzle))

        # Add self-reflection instruction if configured
        if self.self_reflection:
            instruction = self.loader.get_prompt(
                "reasoning_effort.json", "self_reflection_instruction"
            )
            prompt_parts.append(instruction)

        return "\n\n".join(prompt_parts)

    def _format_few_shot_section(self, examples: List) -> str:
        """Format few-shot examples section."""
        header = self.loader.get_prompt("few_shot_examples.json", "few_shot_header")
        example_template = self.loader.get_prompt(
            "few_shot_examples.json", "example_template"
        )

        examples_text = [header]

        num_examples = min(len(examples), 1 if self.shot_type == "one-shot" else 3)

        for i, example in enumerate(examples[:num_examples]):
            example_text = example_template.format(
                example_num=i + 1,
                grid=self._format_grid_with_numbers(example),
                clues=self._format_clues_section(example),
                words=self._format_solution_words(example),
                solution_grid=self._format_solution_grid(example),
            )
            examples_text.append(example_text)

        transition = self.loader.get_prompt(
            "few_shot_examples.json", "transition_to_puzzle"
        )
        examples_text.append(transition)

        return "\n".join(examples_text)

    def _format_single_puzzle(self, puzzle) -> str:
        """Format a single puzzle for evaluation."""
        base_instruction = self.loader.get_prompt(
            "base_prompts.json", "base_instruction"
        )
        # Inject target language into base instruction
        base_instruction = self._inject_language(base_instruction)

        grid_header = self.loader.get_prompt("base_prompts.json", "grid_header")
        clues_header = self.loader.get_prompt("base_prompts.json", "clues_header")
        task_description = self.loader.get_prompt(
            "task_instructions.json", "task_description"
        )
        response_format = self.loader.get_prompt(
            "task_instructions.json", "response_format"
        )
        response_example = self.loader.get_prompt(
            "task_instructions.json", "response_example"
        )

        return f"""{base_instruction}

{grid_header}
{self._format_grid_with_numbers(puzzle)}

{clues_header}
{self._format_clues_section(puzzle)}

{task_description}

{response_format}
{response_example}"""

    def _format_batch_puzzles(self, puzzles: List) -> str:
        """Format multiple puzzles for batch evaluation."""
        puzzle_sections = []

        for i, puzzle in enumerate(puzzles):
            section = f"=== PUZZLE {i + 1} ===\n"
            section += self._format_single_puzzle(puzzle)
            puzzle_sections.append(section)

        return "\n\n".join(puzzle_sections)

    def _format_grid_with_numbers(self, puzzle) -> str:
        """Create a numbered grid showing where each word starts."""
        # Create a copy of the grid to add numbers
        display_grid = [row[:] for row in puzzle.get_grid()]

        # Add number markers where words start
        for clue in puzzle.get_clues():
            row, col = clue.start_row, clue.start_col
            if display_grid[row][col] == "_":
                display_grid[row][col] = str(clue.number)

        # Format for display
        lines = []
        for row in display_grid:
            line = " ".join(f"{cell:>2}" for cell in row)
            lines.append(line)

        return "\n".join(lines)

    def _format_clues_section(self, puzzle) -> str:
        """Format the clues section with position details."""
        instructions = []

        for clue in sorted(puzzle.get_clues(), key=lambda c: (c.direction, c.number)):
            instruction = f"""Clue number {clue.number} ({clue.direction}):
- Clue: {clue.clue_text}
- Length: {clue.length} letters
- Starts at: Row {clue.start_row + 1}, Column {clue.start_col + 1}
- Fill positions: {self._get_position_list(clue)}"""

            instructions.append(instruction)

        return "\n".join(instructions)

    def _format_solution_words(self, puzzle) -> str:
        """Format solution words for few-shot examples."""
        words = []
        for clue in sorted(puzzle.get_clues(), key=lambda c: (c.direction, c.number)):
            words.append(f"{clue.number} {clue.direction}: {clue.answer}")
        return "\n".join(words)

    def _format_solution_grid(self, puzzle) -> str:
        """Format solution grid for few-shot examples."""
        solution_grid = puzzle.get_solution_grid()
        lines = []
        for row in solution_grid:
            line = " ".join(f"{cell:>2}" for cell in row)
            lines.append(line)
        return "\n".join(lines)

    def _get_position_list(self, clue) -> str:
        """Get list of grid positions for a word."""
        positions = []
        row, col = clue.start_row, clue.start_col

        for i in range(clue.length):
            if clue.direction == "across":
                positions.append(f"({row + 1},{col + 1 + i})")
            else:  # down
                positions.append(f"({row + 1 + i},{col + 1})")

        return " → ".join(positions)

    def generate_self_reflection_prompt(
        self, puzzle, previous_response: str, errors: List[str]
    ) -> str:
        """Generate a self-reflection prompt with error feedback."""
        correction_template = self.loader.get_prompt(
            "reasoning_effort.json", "self_reflection_correction"
        )

        errors_text = "\n".join([f"- {error}" for error in errors])

        return (
            correction_template.format(errors=errors_text)
            + "\n\n"
            + self._format_single_puzzle(puzzle)
        )
