#!/usr/bin/env python3
"""
Simple Puzzle Visualization for Buddhi-Pragati Generated Crosswords

Usage:
    python puzzle_visualisation.py --language hindi english --grid-sizes 8 10 --number-of-puzzles 5
"""

import argparse
import json
import random
import sys
from datasets import load_dataset
from buddhi_pragati.utils.config_loader import get_config


def load_puzzles(language: str, grid_size: int, num_puzzles: int) -> list:
    """Load random puzzles from HuggingFace dataset."""
    config = get_config()
    hf_token = config.get_string("DEFAULT_HF_TOKEN", "")
    repo = config.get_string(
        "HF_GENERATED_PUZZLES_REPO", "selim-b-kh/buddhi-pragati-puzzles"
    )

    config_name = f"{language.lower()}"

    try:
        dataset_kwargs = {"split": "train"}
        if hf_token:
            dataset_kwargs["token"] = hf_token

        dataset = load_dataset(repo, config_name, **dataset_kwargs)

        if len(dataset) == 0:
            return []

        # Filter dataset to include only entries with IDs matching the pattern
        filtered_dataset = [
            entry
            for entry in dataset
            if entry["id"].startswith(f"{config_name}_{grid_size}x{grid_size}")
        ]

        if len(filtered_dataset) == 0:
            return []

        # Sample random puzzles from the filtered dataset
        sample_size = min(num_puzzles, len(filtered_dataset))
        if sample_size < len(filtered_dataset):
            indices = random.sample(range(len(filtered_dataset)), sample_size)
            return [filtered_dataset[i] for i in sorted(indices)]
        else:
            return list(filtered_dataset)

    except Exception as e:
        print(f"Error loading {config_name}: {e}")
        return []


def display_grid(grid):
    """Display a crossword grid in terminal."""
    for row in grid:
        formatted_row = []
        for cell in row:
            if cell == "#":
                formatted_row.append("███")
            elif str(cell).isdigit():
                formatted_row.append(f" {cell} ")
            else:
                formatted_row.append(" _ ")
        print("".join(formatted_row))


def display_puzzle(puzzle, index):
    """Display a single puzzle with grid and clues."""
    print(f"\n{'=' * 60}")
    print(f"PUZZLE {index}: {puzzle['id']}")
    print(
        f"Density: {puzzle.get('density', 0):.1%} | Words: {puzzle.get('word_count', 0)} | Context: {puzzle.get('context_score', 0):.3f}"
    )
    print(f"{'=' * 60}")

    # Display grid
    print("\nGRID:")
    display_grid(puzzle["empty_grid"])

    # Display clues
    clues_data = puzzle.get("clues", puzzle.get("clues_with_answer", {}))
    if isinstance(clues_data, list):
        # New list format
        across_clues = [(c["id"], c) for c in clues_data if c["direction"] == "across"]
        down_clues = [(c["id"], c) for c in clues_data if c["direction"] == "down"]
    else:
        # Legacy dict format
        across_clues = [
            (k, v) for k, v in clues_data.items() if v["direction"] == "across"
        ]
        down_clues = [(k, v) for k, v in clues_data.items() if v["direction"] == "down"]

    if across_clues:
        print("\nACROSS:")
        for clue_id, clue_info in sorted(across_clues):
            print(f"  {clue_id}: {clue_info['clue']} ({clue_info['answer']})")

    if down_clues:
        print("\nDOWN:")
        for clue_id, clue_info in sorted(down_clues):
            print(f"  {clue_id}: {clue_info['clue']} ({clue_info['answer']})")


def web_format(all_puzzles):
    """Output JSON format for web applications."""
    web_data = {"puzzles": []}

    for language, grid_data in all_puzzles.items():
        for grid_size, puzzles in grid_data.items():
            for puzzle in puzzles:
                web_puzzle = {
                    "id": puzzle["id"],
                    "language": language,
                    "grid_size": grid_size,
                    "empty_grid": puzzle["empty_grid"],
                    "solved_grid": puzzle["solved_grid"],
                    "density": puzzle.get("density", 0),
                    "word_count": puzzle.get("word_count", 0),
                    "context_score": puzzle.get("context_score", 0),
                    "clues": puzzle["clues_with_answer"],
                }
                web_data["puzzles"].append(web_puzzle)

    print(json.dumps(web_data, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Visualize crossword puzzles")
    parser.add_argument(
        "--language", nargs="+", required=True, help="Languages to visualize"
    )
    parser.add_argument(
        "--grid-sizes", nargs="+", type=int, required=True, help="Grid sizes"
    )
    parser.add_argument(
        "--number-of-puzzles",
        type=int,
        default=3,
        help="Number of puzzles per combination",
    )
    parser.add_argument(
        "--format",
        choices=["terminal", "web"],
        default="terminal",
        help="Output format",
    )
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    # Load all puzzles
    all_puzzles = {}
    total_loaded = 0

    for language in args.language:
        all_puzzles[language] = {}
        for grid_size in args.grid_sizes:
            puzzles = load_puzzles(language, grid_size, args.number_of_puzzles)
            if puzzles:
                all_puzzles[language][grid_size] = puzzles
                total_loaded += len(puzzles)
                print(
                    f"Loaded {len(puzzles)} puzzles for {language} {grid_size}x{grid_size}"
                )

    if total_loaded == 0:
        print("No puzzles found!")
        sys.exit(1)

    # Display results
    if args.format == "web":
        web_format(all_puzzles)
    else:
        puzzle_count = 1
        for language, grid_data in all_puzzles.items():
            for grid_size, puzzles in grid_data.items():
                print(f"\n\n{language.upper()} {grid_size}x{grid_size} PUZZLES")
                print("=" * 80)

                for puzzle in puzzles:
                    display_puzzle(puzzle, puzzle_count)
                    puzzle_count += 1


if __name__ == "__main__":
    main()
