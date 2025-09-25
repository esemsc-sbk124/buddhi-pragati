"""
Dataset loader for crossword puzzles from HuggingFace datasets.

This module loads generated crossword puzzles from the HuggingFace repository
and converts them to CrosswordPuzzle objects for evaluation.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import random
import re

try:
    from datasets import load_dataset, get_dataset_config_names
    from huggingface_hub import HfApi

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning(
        "HuggingFace datasets not available. Install with: pip install datasets huggingface_hub"
    )

from ..core.base_puzzle import CrosswordPuzzle, CrosswordClue
from ..utils.config_loader import get_config


class PuzzleDatasetLoader:
    """
    Loads crossword puzzles from HuggingFace generated puzzles repository.

    Supports filtering by language, grid size, and other criteria.
    Provides local caching for repeated experiments.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.logger = logging.getLogger("PuzzleDatasetLoader")

        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets required. Install with: pip install datasets huggingface_hub"
            )

        # Get configuration
        self.config = get_config()
        self.dataset_repo = self.config.get(
            "HF_GENERATED_PUZZLES_REPO", "selim-b-kh/buddhi-pragati-puzzles"
        )

        # HuggingFace token
        self.hf_token = hf_token or self.config.get("DEFAULT_HF_TOKEN")

        # Local cache for loaded datasets
        self._cache = {}
        self._temp_dir = None

    def load_puzzles(
        self,
        language: str,
        grid_size: Optional[int] = None,
        max_puzzles: Optional[int] = None,
        random_sample: bool = True,
    ) -> List[CrosswordPuzzle]:
        """
        Load crossword puzzles for a specific language and optional grid size.

        Args:
            language: Target language for puzzles
            grid_size: Optional grid size filter (e.g., 10 for 10x10 grids)
            max_puzzles: Maximum number of puzzles to load
            random_sample: Whether to randomly sample puzzles if limiting count

        Returns:
            List of CrosswordPuzzle objects
        """
        self.logger.info(
            f"Loading puzzles for language: {language}, grid_size: {grid_size}"
        )

        try:
            # Load dataset for the language
            cache_key = f"{language}_{grid_size or 'all'}"

            if cache_key not in self._cache:
                dataset = load_dataset(
                    self.dataset_repo,
                    name=language,  # Language as config name
                    token=self.hf_token,
                    split="train",  # Assuming train split contains generated puzzles
                )

                # Filter by grid size if specified
                if grid_size is not None:
                    dataset = dataset.filter(
                        lambda example: example.get("grid_size") == grid_size
                    )

                self._cache[cache_key] = dataset
                self.logger.info(f"Loaded {len(dataset)} puzzles from HuggingFace")
            else:
                dataset = self._cache[cache_key]
                self.logger.info(f"Using cached dataset with {len(dataset)} puzzles")

            # Convert to list for sampling/limiting
            puzzle_data = list(dataset)

            # Apply sampling/limiting
            if max_puzzles and len(puzzle_data) > max_puzzles:
                if random_sample:
                    puzzle_data = random.sample(puzzle_data, max_puzzles)
                else:
                    puzzle_data = puzzle_data[:max_puzzles]
                self.logger.info(f"Limited to {max_puzzles} puzzles")

            # Convert to CrosswordPuzzle objects
            puzzles = []
            for i, puzzle_dict in enumerate(puzzle_data):
                try:
                    puzzle = self._convert_to_crossword_puzzle(puzzle_dict, i)
                    if puzzle:
                        puzzles.append(puzzle)
                except Exception as e:
                    self.logger.warning(f"Failed to convert puzzle {i}: {e}")
                    continue

            self.logger.info(f"Successfully converted {len(puzzles)} puzzles")
            return puzzles

        except Exception as e:
            self.logger.error(f"Failed to load puzzles: {e}")
            return []

    @staticmethod
    def extract_number(clue):
        """
        Extract number from clue text.

        Args:
            clue: Clue text that may contain a number

        Returns:
            Extracted number or None if no number found
        """
        # Use a regex to extract the leading digits
        match = re.match(r"(\d+)", clue)
        if match:
            return int(match.group(1))  # Convert the matched digits to an integer
        return None  # Return None if no number is found

    def _convert_to_crossword_puzzle(
        self, puzzle_dict: Dict[str, Any], index: int
    ) -> Optional[CrosswordPuzzle]:
        """
        Convert HuggingFace dataset entry to CrosswordPuzzle object.

        Args:
            puzzle_dict: Dictionary from HuggingFace dataset
            index: Index for puzzle ID generation

        Returns:
            CrosswordPuzzle object or None if conversion fails
        """
        try:
            # Extract basic information using actual HF dataset field names
            puzzle_id = puzzle_dict.get("id", f"hf_puzzle_{index}")
            grid_size = puzzle_dict.get("grid_size", 10)

            # Extract grids from HF dataset
            empty_grid = puzzle_dict.get("empty_grid", [])
            solved_grid = puzzle_dict.get("solved_grid", [])

            # Validate grids exist
            if not empty_grid or not solved_grid:
                self.logger.error(f"Missing grid data for puzzle {puzzle_id}")
                return None

            # Parse clues - already structured, no JSON parsing needed
            clues_data = puzzle_dict.get("clues", [])

            # Create CrosswordClue objects and build solution_words dictionary
            clues = []
            solution_words = {}

            for clue_dict in clues_data:
                # Extract number from clue ID (e.g., "5across" -> 5)
                clue_id = clue_dict.get("id", "")
                number = self.extract_number(clue_id)
                if number is None:
                    # Use sequential numbering if no number found in clue ID
                    number = len(clues) + 1

                clue = CrosswordClue(
                    number=number,
                    direction=clue_dict["direction"],
                    clue_text=clue_dict["clue"],  # HF dataset uses 'clue' field
                    answer=clue_dict["answer"],
                    start_row=clue_dict["start_row"],
                    start_col=clue_dict["start_col"],
                    length=clue_dict["length"],
                )
                clues.append(clue)

                # Build solution_words dictionary for evaluator
                solution_words[clue_id] = clue_dict["answer"]

            # Create CrosswordPuzzle with proper grid and solution data
            puzzle = CrosswordPuzzle(
                puzzle_id=puzzle_id,
                grid=empty_grid,  # The puzzle to solve (empty grid)
                solution_grid=solved_grid,  # The complete solution (required by evaluator)
                clues=clues,
                size=(grid_size, grid_size),
                solution_words=solution_words,  # Solution dictionary (required by evaluator)
            )

            self.logger.debug(
                f"Successfully converted puzzle {puzzle_id} with {len(clues)} clues"
            )
            return puzzle

        except Exception as e:
            self.logger.error(f"Error converting puzzle data: {e}")
            return None

    def get_available_languages(self) -> List[str]:
        """
        Get list of available languages in the dataset.

        Returns:
            List of language names (config names)
        """
        try:
            configs = get_dataset_config_names(self.dataset_repo, token=self.hf_token)
            if configs:
                self.logger.info(f"Found {len(configs)} available languages")
                return configs
            # If configs not available, try to fetch dataset info directly
            api = HfApi()
            dataset_info = api.dataset_info(self.dataset_repo, token=self.hf_token)
            if hasattr(dataset_info, "siblings") and dataset_info.siblings:
                # Extract from file siblings (alternative approach)
                config_names = set()
                for sibling in dataset_info.siblings:
                    # Config files follow pattern: {language}/*.parquet
                    if sibling.rfilename.endswith(".parquet"):
                        parts = sibling.rfilename.split("/")
                        if len(parts) >= 2:
                            config_names.add(
                                parts[0]
                            )  # Extract language name (first part)

                if config_names:
                    available_languages = sorted(list(config_names))
                    self.logger.info(
                        f"Found {len(available_languages)} languages from file structure"
                    )
                    return available_languages

            # Fallback: try common languages if API method fails
            self.logger.warning(
                "Could not extract configs from dataset info, falling back to common languages"
            )
            common_languages = [
                "english",
                "hindi",
                "bengali",
                "tamil",
                "telugu",
                "gujarati",
                "kannada",
                "malayalam",
                "marathi",
                "punjabi",
                "assamese",
                "bodo",
                "kashmiri",
                "konkani",
                "meitei",
                "nepali",
                "odia",
                "sanskrit",
                "urdu",
            ]

            available_languages = []
            for lang in common_languages:
                try:
                    # Quick check if config exists by trying to load dataset info for this config
                    dataset = load_dataset(
                        self.dataset_repo,
                        name=lang,
                        token=self.hf_token,
                        split="train[:1]",  # Load just 1 example
                    )
                    if len(dataset) > 0:
                        available_languages.append(lang)
                except Exception as e1:
                    self.logger.error(f"Language {lang} not available: {e1}")

            return available_languages

        except Exception as e:
            self.logger.error(f"Error getting available languages: {e}")
            return []

    def get_grid_sizes(self, language: str) -> List[int]:
        """
        Get available grid sizes for a specific language.

        Args:
            language: Target language

        Returns:
            List of available grid sizes
        """
        try:
            dataset = load_dataset(
                self.dataset_repo, name=language, token=self.hf_token, split="train"
            )

            # Extract unique grid sizes
            grid_sizes = set()
            for example in dataset:
                if "grid_size" in example:
                    grid_sizes.add(example["grid_size"])

            return sorted(list(grid_sizes))

        except Exception as e:
            self.logger.error(f"Error getting grid sizes for {language}: {e}")
            return []

    def get_puzzle_count(self, language: str, grid_size: Optional[int] = None) -> int:
        """
        Get count of available puzzles for specific criteria.

        Args:
            language: Target language
            grid_size: Optional grid size filter

        Returns:
            Number of available puzzles
        """
        try:
            dataset = load_dataset(
                self.dataset_repo, name=language, token=self.hf_token, split="train"
            )

            if grid_size is not None:
                dataset = dataset.filter(
                    lambda example: example.get("grid_size") == grid_size
                )

            return len(dataset)

        except Exception as e:
            self.logger.error(f"Error getting puzzle count: {e}")
            return 0

    def save_puzzles_locally(self, puzzles: List[CrosswordPuzzle], output_dir: str):
        """
        Save puzzles to local JSON files for offline use.

        Args:
            puzzles: List of CrosswordPuzzle objects
            output_dir: Directory to save puzzle files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for puzzle in puzzles:
            puzzle_data = {
                "puzzle_id": puzzle.puzzle_id,
                "size": puzzle.size,
                "grid": puzzle.get_grid(),
                "solution_grid": puzzle.get_solution_grid(),
                "clues": [
                    {
                        "number": clue.number,
                        "direction": clue.direction,
                        "clue_text": clue.clue_text,
                        "answer": clue.answer,
                        "start_row": clue.start_row,
                        "start_col": clue.start_col,
                        "length": clue.length,
                    }
                    for clue in puzzle.get_clues()
                ],
                "solution_words": puzzle.get_solution_words(),
            }

            filepath = output_path / f"{puzzle.puzzle_id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(puzzle_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved {len(puzzles)} puzzles to {output_dir}")

    def load_puzzles_from_local(self, input_dir: str) -> List[CrosswordPuzzle]:
        """
        Load puzzles from local JSON files.

        Args:
            input_dir: Directory containing puzzle JSON files

        Returns:
            List of CrosswordPuzzle objects
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return []

        puzzles = []
        json_files = list(input_path.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    puzzle_data = json.load(f)

                # Convert back to CrosswordPuzzle
                clues = []
                for clue_data in puzzle_data["clues"]:
                    clue = CrosswordClue(
                        number=clue_data["number"],
                        direction=clue_data["direction"],
                        clue_text=clue_data["clue_text"],
                        answer=clue_data["answer"],
                        start_row=clue_data["start_row"],
                        start_col=clue_data["start_col"],
                        length=clue_data["length"],
                    )
                    clues.append(clue)

                puzzle = CrosswordPuzzle(
                    puzzle_id=puzzle_data["puzzle_id"],
                    size=tuple(puzzle_data["size"]),
                    clues=clues,
                    grid=puzzle_data["grid"],
                )

                puzzles.append(puzzle)

            except Exception as e:
                self.logger.warning(f"Failed to load puzzle from {json_file}: {e}")
                continue

        self.logger.info(f"Loaded {len(puzzles)} puzzles from {input_dir}")
        return puzzles

    def clear_cache(self):
        """Clear the internal dataset cache."""
        self._cache.clear()
        self.logger.info("Dataset cache cleared")
