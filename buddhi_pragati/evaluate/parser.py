"""
Crossword response parser for model output processing.

This module handles parsing of model responses for crossword puzzles
with robust fallback strategies and Unicode support.
"""

import logging
import unicodedata
from typing import Dict, Tuple
import numpy as np
import json
from pathlib import Path


class CrosswordResponseParser:
    """
    Parses model responses for crossword puzzles.

    Handles various response formats and provides fallback strategies
    for robust parsing across different model outputs.
    Uses configurable markers from parser_markers.json.
    """

    def __init__(self):
        self.logger = logging.getLogger("CrosswordResponseParser")
        self.markers = self._load_parser_markers()

    def _load_parser_markers(self) -> Dict[str, str]:
        """Load parsing markers from parser_markers.json."""
        markers_file = Path(__file__).parent / "prompts" / "parser_markers.json"
        try:
            with open(markers_file, "r", encoding="utf-8") as f:
                markers_data = json.load(f)
            # Extract English markers (all prompts are English-only now)
            markers = {}
            for key, value in markers_data.items():
                markers[key] = value["English"]
            self.logger.debug(f"Loaded parser markers: {markers}")
            return markers
        except Exception as e:
            self.logger.error(f"Failed to load parser markers: {e}")
            # Fallback to hardcoded markers
            return {
                "words_marker": "WORDS:",
                "grid_marker": "GRID:",
                "across_marker": "across",
                "down_marker": "down",
            }

    def _normalize_grid_shape(self, parsed_grid, target_rows, target_cols):
        """Normalize grid to exact target dimensions."""
        self.logger.info(
            f"Normalizing grid from {len(parsed_grid)}x{len(parsed_grid[0]) if parsed_grid else 0} to {target_rows}x{target_cols}"
        )

        # Handle empty grid
        if not parsed_grid:
            self.logger.warning("Empty grid detected, creating default grid")
            return [["_" for _ in range(target_cols)] for _ in range(target_rows)]

        # Fix row count
        if len(parsed_grid) < target_rows:
            # Add missing rows
            rows_to_add = target_rows - len(parsed_grid)
            self.logger.info(f"Adding {rows_to_add} missing rows")
            for _ in range(rows_to_add):
                parsed_grid.append(["_" for _ in range(target_cols)])
        elif len(parsed_grid) > target_rows:
            # Remove excess rows
            rows_to_remove = len(parsed_grid) - target_rows
            self.logger.info(f"Removing {rows_to_remove} excess rows")
            parsed_grid = parsed_grid[:target_rows]

        # Fix column count for each row
        for i in range(len(parsed_grid)):
            row = parsed_grid[i]
            if len(row) < target_cols:
                # Add missing columns
                cols_to_add = target_cols - len(row)
                self.logger.debug(f"Row {i}: adding {cols_to_add} missing columns")
                row.extend(["_"] * cols_to_add)
            elif len(row) > target_cols:
                # Remove excess columns
                cols_to_remove = len(row) - target_cols
                self.logger.debug(f"Row {i}: removing {cols_to_remove} excess columns")
                row = row[:target_cols]
                parsed_grid[i] = row

        # Verify final dimensions
        actual_rows = len(parsed_grid)
        actual_cols = len(parsed_grid[0]) if parsed_grid else 0
        if actual_rows != target_rows or actual_cols != target_cols:
            self.logger.error(
                f"Grid normalization failed: got {actual_rows}x{actual_cols}, expected {target_rows}x{target_cols}"
            )
            # Fallback to empty grid
            return [["_" for _ in range(target_cols)] for _ in range(target_rows)]

        self.logger.info(f"Grid normalization successful: {actual_rows}x{actual_cols}")
        return parsed_grid

    def parse_model_response(
        self, puzzle, response: str
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Extract the filled grid from model's response and convert it to a NumPy array.

        Args:
            puzzle: CrosswordPuzzle object for context
            response: Raw model response string

        Returns:
            Tuple of (parsed_grid as np.ndarray, parsed_words as dict)
        """
        lines = response.strip().split("\n")
        parsed_grid = []
        parsed_words = {}

        # Response will be logged in main evaluation flow to avoid duplication

        nbr_rows = puzzle.get_size()[0]
        nbr_cols = puzzle.get_size()[1]

        self.logger.info(f"Parsing puzzle: {puzzle.puzzle_id}")
        self.logger.info(f"Model response: {response}")

        words_marker = self.markers.get("words_marker", "WORDS:")
        grid_marker = self.markers.get("grid_marker", "GRID:")

        for i in range(len(lines)):
            if lines[i].strip().startswith(words_marker):
                # Parse the words section
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith(grid_marker):
                        break
                    elif lines[j].strip() == "":
                        continue
                    parts = lines[j].split(":", 1)
                    if len(parts) == 2:
                        clue, word = parts
                        # Normalize clue key by removing spaces to match expected format
                        normalized_clue = clue.strip().replace(" ", "")
                        # Normalize Unicode word for consistent representation
                        normalized_word = unicodedata.normalize("NFC", word.strip())
                        parsed_words[normalized_clue] = normalized_word
            elif lines[i].strip().startswith(grid_marker):
                # Find the actual grid data (skip empty lines after "GRID:")
                grid_start_idx = i + 1
                while (
                    grid_start_idx < len(lines) and lines[grid_start_idx].strip() == ""
                ):
                    grid_start_idx += 1

                # Parse grid rows with better Unicode handling
                for j in range(
                    grid_start_idx, min(grid_start_idx + nbr_rows, len(lines))
                ):
                    if j >= len(lines):
                        break
                    line = lines[j].strip()
                    if line == "":
                        # Add empty row to maintain structure
                        parsed_grid.append(["_"] * nbr_cols)
                        continue

                    # Split by whitespace and handle Unicode characters
                    row_cells = line.split()
                    # Keep original characters, don't convert to '_' unless actually '_'
                    parsed_row = []
                    for cell in row_cells:
                        # Normalize Unicode to ensure consistent representation
                        normalized_cell = unicodedata.normalize("NFC", cell)
                        parsed_row.append(normalized_cell)
                    parsed_grid.append(parsed_row)

                self.logger.info(f"Parsed {len(parsed_grid)} rows from grid section")
            else:
                continue

        # Normalize grid shape to ensure it matches puzzle dimensions
        parsed_grid = self._normalize_grid_shape(parsed_grid, nbr_rows, nbr_cols)

        # Final validation
        self.logger.info(
            f"Final parsed grid dimensions: {len(parsed_grid)}x{len(parsed_grid[0]) if parsed_grid else 0}"
        )
        self.logger.info(
            f"Normalized {len(parsed_words)} word keys (spaces removed for matching)"
        )
        # Detailed parsed data will be logged in main evaluation flow

        return np.array(parsed_grid), parsed_words
