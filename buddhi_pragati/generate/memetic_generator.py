"""
Memetic Crossword Generator with Enhanced Density Optimization

This module implements a sophisticated crossword generator using memetic algorithms
(genetic algorithm + local search) to achieve high grid density (75%+) while
prioritizing Indian cultural context and supporting wide range of grid sizes (3x3 to 30x30).
"""

import random
import copy
import logging
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np
from enum import Enum

from ..core.base_puzzle import CrosswordPuzzle, CrosswordClue
from ..utils import is_alphabetic_unicode
from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Word placement directions."""

    ACROSS = "across"
    DOWN = "down"


@dataclass
class WordPlacement:
    """Represents a placed word in the crossword grid."""

    text: str
    clue: str
    row: int
    col: int
    direction: Direction
    number: int = 0
    context_score: float = 0.0

    def __post_init__(self):
        self.text = self.text.upper()

    def get_positions(self) -> List[Tuple[int, int]]:
        """Get all grid positions occupied by this word."""
        positions = []
        for i in range(len(self.text)):
            if self.direction == Direction.ACROSS:
                positions.append((self.row, self.col + i))
            else:
                positions.append((self.row + i, self.col))
        return positions

    def intersects_at(self, other: "WordPlacement") -> List[Tuple[int, int]]:
        """Find intersection points with another word."""
        self_positions = set(self.get_positions())
        other_positions = set(other.get_positions())
        return list(self_positions & other_positions)

    def get_end_position(self) -> Tuple[int, int]:
        """Get the ending position of the word."""
        if self.direction == Direction.ACROSS:
            return (self.row, self.col + len(self.text) - 1)
        else:
            return (self.row + len(self.text) - 1, self.col)


class CrosswordCandidate:
    """Represents a candidate crossword in the population."""

    def __init__(self, size: int):
        self.size = size
        self.grid = [["" for _ in range(size)] for _ in range(size)]
        self.placed_words: List[WordPlacement] = []
        self.used_clues: Set[str] = set()  # Track clue texts to prevent duplicates
        self.fitness_score: float = 0.0
        self.density: float = 0.0
        self.word_count: int = 0
        self.intersection_count: int = 0
        self.cultural_coherence: float = 0.0

    def calculate_fitness(self, weights: Dict[str, float]) -> float:
        """Calculate overall fitness score using weighted components."""
        self.density = self._calculate_density()
        self.word_count = len(self.placed_words)
        self.intersection_count = self._calculate_intersections()
        self.cultural_coherence = self._calculate_cultural_coherence()

        # Normalize components
        normalized_density = min(1.0, self.density / 0.75)  # Target 75% density
        normalized_intersections = (
            min(1.0, self.intersection_count / (self.word_count * 0.5))
            if self.word_count > 0
            else 0
        )
        normalized_coherence = self.cultural_coherence

        self.fitness_score = (
            weights["density_weight"] * normalized_density
            + weights["intersection_weight"] * normalized_intersections
            + weights["cultural_coherence_weight"] * normalized_coherence
        )

        return self.fitness_score

    def _calculate_density(self) -> float:
        """Calculate grid fill density."""
        filled_cells = sum(1 for row in self.grid for cell in row if cell != "")
        total_cells = self.size * self.size
        return filled_cells / total_cells if total_cells > 0 else 0.0

    def _calculate_intersections(self) -> int:
        """Calculate total number of intersections between words."""
        intersection_count = 0
        for i, word1 in enumerate(self.placed_words):
            for word2 in self.placed_words[i + 1 :]:
                if word1.intersects_at(word2):
                    intersection_count += 1
        return intersection_count

    def _calculate_cultural_coherence(self) -> float:
        """Calculate mean cultural coherence from context scores."""
        if not self.placed_words:
            return 0.0

        context_scores = [
            word.context_score for word in self.placed_words if word.context_score > 0
        ]
        return sum(context_scores) / len(context_scores) if context_scores else 0.0

    def clone(self) -> "CrosswordCandidate":
        """Create a deep copy of this candidate."""
        new_candidate = CrosswordCandidate(self.size)
        new_candidate.grid = [row[:] for row in self.grid]
        new_candidate.placed_words = [copy.deepcopy(word) for word in self.placed_words]
        new_candidate.used_clues = self.used_clues.copy()  # Copy the set of used clues
        new_candidate.fitness_score = self.fitness_score
        new_candidate.density = self.density
        new_candidate.word_count = self.word_count
        new_candidate.intersection_count = self.intersection_count
        new_candidate.cultural_coherence = self.cultural_coherence
        return new_candidate


class MemeticCrosswordGenerator:
    """
    Enhanced crossword generator using memetic algorithms for high density optimization.

    Combines genetic algorithm population evolution with local search optimization
    to achieve target grid density while prioritizing Indian cultural context.
    """

    def __init__(self, size: int):
        """
        Initialize memetic generator.

        Args:
            size: Grid size (square grid assumed)
        """
        if not (3 <= size <= 30):
            raise ValueError(
                f"Grid size {size} not supported. Must be between 3 and 30."
            )

        self.size = size
        self.config = get_config()
        self.generation_config = self.config.get_generation_config()

        # Memetic algorithm parameters
        self.population_size = max(10, min(50, size * 2))  # Scale with grid size
        self.max_generations = self.generation_config["max_generation_attempts"]
        self.elite_size = max(2, self.population_size // 5)
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7
        self.local_search_probability = 0.3

        # Fitness weights
        self.fitness_weights = {
            "density_weight": self.generation_config["density_weight"],
            "intersection_weight": self.generation_config["intersection_weight"],
            "cultural_coherence_weight": self.generation_config[
                "cultural_coherence_weight"
            ],
        }

        logger.info(f"Initialized MemeticCrosswordGenerator for {size}x{size} grid")
        logger.info(
            f"Population size: {self.population_size}, Elite size: {self.elite_size}"
        )

    def generate_puzzle_with_prioritization(
        self,
        prioritized_corpus: List[Tuple[str, str]],
        context_scores: Dict[str, float],
        puzzle_id: str,
        indian_threshold: float = None,
    ) -> Optional[CrosswordPuzzle]:
        """
        Generate crossword using memetic algorithm with cultural prioritization.

        Args:
            prioritized_corpus: Corpus sorted by Indian context (high to low)
            context_scores: Dictionary mapping answer -> raw context_score
            puzzle_id: Unique identifier for puzzle
            indian_threshold: Threshold for Indian classification (for reporting)

        Returns:
            CrosswordPuzzle instance or None if generation failed
        """
        if indian_threshold is None:
            indian_threshold = self.generation_config["indian_context_threshold"]

        # Filter valid pairs for this grid size
        valid_pairs = self._filter_valid_pairs(prioritized_corpus)
        if len(valid_pairs) < self.generation_config["min_words_per_puzzle"]:
            logger.warning(
                f"Insufficient valid pairs ({len(valid_pairs)}) for {self.size}x{self.size} grid"
            )
            return None

        logger.info(f"Starting memetic generation with {len(valid_pairs)} valid pairs")

        # Initialize population
        population = self._initialize_population(valid_pairs, context_scores)
        if not population:
            logger.error("Failed to initialize population")
            return None

        best_fitness = 0.0
        generations_without_improvement = 0
        max_stagnation = max(10, self.max_generations // 5)

        # Evolve population
        for generation in range(self.max_generations):
            # Evaluate fitness for all candidates
            for candidate in population:
                candidate.calculate_fitness(self.fitness_weights)

            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness_score, reverse=True)

            current_best = population[0].fitness_score
            if current_best > best_fitness:
                best_fitness = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Check for early termination
            if (
                population[0].density >= self.generation_config["target_grid_density"]
                or generations_without_improvement >= max_stagnation
            ):
                break

            # Log progress
            if generation % 10 == 0:
                logger.debug(
                    f"Generation {generation}: Best fitness={current_best:.3f}, "
                    f"Density={population[0].density:.2%}, Words={population[0].word_count}"
                )

            # Create next generation
            new_population = []

            # Keep elite (best performing candidates)
            new_population.extend(population[: self.elite_size])

            # Generate offspring through crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate and len(population) >= 2:
                    # Select parents using tournament selection
                    parent1 = self._tournament_selection(population, 3)
                    parent2 = self._tournament_selection(population, 3)

                    # Create offspring
                    offspring = self._crossover(
                        parent1, parent2, valid_pairs, context_scores
                    )
                    if offspring:
                        # Apply mutation
                        if random.random() < self.mutation_rate:
                            offspring = self._mutate(
                                offspring, valid_pairs, context_scores
                            )

                        # Apply local search occasionally
                        if random.random() < self.local_search_probability:
                            offspring = self._local_search(
                                offspring, valid_pairs, context_scores
                            )

                        new_population.append(offspring)
                else:
                    # Generate new random candidate
                    new_candidate = self._create_random_candidate(
                        valid_pairs, context_scores
                    )
                    if new_candidate:
                        new_population.append(new_candidate)

            population = new_population

        # Select best candidate
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        best_candidate = population[0]

        # Check if meets minimum requirements
        if (
            best_candidate.density >= self.generation_config["min_acceptable_density"]
            and best_candidate.word_count
            >= self.generation_config["min_words_per_puzzle"]
        ):
            logger.info(
                f"Generated puzzle: Density={best_candidate.density:.2%}, "
                f"Words={best_candidate.word_count}, Fitness={best_candidate.fitness_score:.3f}"
            )
        else:
            logger.warning(
                f"Best candidate did not meet minimum requirements - returning best attempt: "
                f"Density={best_candidate.density:.2%} (min: {self.generation_config['min_acceptable_density']:.2%}), "
                f"Words={best_candidate.word_count} (min: {self.generation_config['min_words_per_puzzle']})"
            )

        # Always return best candidate (even if below thresholds) instead of None
        if best_candidate.word_count < 3:
            logger.error("Best candidate has too few words - generation failed")
            return None
        if best_candidate.density < 0.5:
            logger.error("Best candidate has too low density - generation failed")
            return None

        return self._candidate_to_puzzle(best_candidate, puzzle_id)

    def _filter_valid_pairs(
        self, corpus: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Filter corpus for valid crossword words for this grid size."""
        min_word_length = self.config.get_int("MIN_WORD_LENGTH", 2)
        valid_pairs = []
        for clue, answer in corpus:
            if min_word_length <= len(answer) <= self.size and is_alphabetic_unicode(
                answer
            ):  # Additional check for pure alphabetic
                valid_pairs.append((clue, answer))

        return valid_pairs

    def _initialize_population(
        self, valid_pairs: List[Tuple[str, str]], context_scores: Dict[str, float]
    ) -> List[CrosswordCandidate]:
        """Initialize population with diverse candidates."""
        population = []

        for i in range(self.population_size):
            candidate = self._create_random_candidate(valid_pairs, context_scores)
            if candidate:
                population.append(candidate)

        logger.info(f"Initialized population with {len(population)} candidates")
        return population

    def _create_random_candidate(
        self, valid_pairs: List[Tuple[str, str]], context_scores: Dict[str, float]
    ) -> Optional[CrosswordCandidate]:
        """Create a random candidate using greedy word placement with cultural bias."""
        candidate = CrosswordCandidate(self.size)

        # Shuffle pairs but bias towards higher context scores (Indian cultural priority)
        working_pairs = valid_pairs.copy()

        # Use weighted random selection favoring high context scores
        max_words = min(len(working_pairs), self.size * 2)  # Reasonable limit
        placed_count = 0
        attempts = 0
        max_attempts = max_words * 3

        while placed_count < max_words and attempts < max_attempts and working_pairs:
            attempts += 1

            # Select word with bias towards high context scores
            if placed_count == 0:
                # First word: pick randomly from top context scores
                top_candidates = working_pairs[: min(10, len(working_pairs))]
                clue, answer = random.choice(top_candidates)
            else:
                # Subsequent words: weighted selection
                clue, answer = self._weighted_word_selection(
                    working_pairs, context_scores
                )

            # Try to place word
            if placed_count == 0:
                # Place first word in center
                success = self._place_first_word(
                    candidate, clue, answer, context_scores.get(answer, 0.0)
                )
            else:
                # Try to place intersecting word
                success = self._place_intersecting_word(
                    candidate, clue, answer, context_scores.get(answer, 0.0)
                )

            if success:
                placed_count += 1
                # Remove used word
                working_pairs = [(c, a) for c, a in working_pairs if a != answer]
            else:
                # Remove failed word and try next
                working_pairs = [(c, a) for c, a in working_pairs if a != answer]

        return candidate if placed_count >= 3 else None  # Minimum viable candidate

    def _weighted_word_selection(
        self, pairs: List[Tuple[str, str]], context_scores: Dict[str, float]
    ) -> Tuple[str, str]:
        """Select word with bias towards high context scores."""
        if len(pairs) <= 5:
            return random.choice(pairs)

        # Calculate weights based on context scores
        weights = []
        for clue, answer in pairs:
            context_score = context_scores.get(answer, 0.0)
            # Exponential weighting to strongly favor high context scores
            weight = np.exp(context_score * 3)  # Scale factor 3 for strong bias
            weights.append(weight)

        # Weighted random selection
        weights = np.array(weights)
        probabilities = weights / weights.sum()

        selected_index = np.random.choice(len(pairs), p=probabilities)
        return pairs[selected_index]

    def _place_first_word(
        self,
        candidate: CrosswordCandidate,
        clue: str,
        answer: str,
        context_score: float,
    ) -> bool:
        """Place first word in center of grid."""
        if len(answer) > self.size:
            return False

        # Check for duplicate clue text before placing
        clue_normalized = clue.strip().lower()
        if clue_normalized in candidate.used_clues:
            return False  # Reject placement with duplicate clue

        # Center placement
        row = self.size // 2
        col = (self.size - len(answer)) // 2

        placement = WordPlacement(
            text=answer,
            clue=clue,
            row=row,
            col=col,
            direction=Direction.ACROSS,
            context_score=context_score,
        )

        # Place word on grid and track clue usage
        self._place_word_on_grid(candidate, placement)
        candidate.placed_words.append(placement)

        return True

    def _place_intersecting_word(
        self,
        candidate: CrosswordCandidate,
        clue: str,
        answer: str,
        context_score: float,
    ) -> bool:
        """Try to place word that intersects with existing words."""
        possible_placements = []

        for placed_word in candidate.placed_words:
            for i, char in enumerate(answer):
                for j, placed_char in enumerate(placed_word.text):
                    if char == placed_char:
                        # Calculate position for intersecting placement
                        if placed_word.direction == Direction.ACROSS:
                            new_row = placed_word.row - i
                            new_col = placed_word.col + j
                            new_direction = Direction.DOWN
                        else:
                            new_row = placed_word.row + j
                            new_col = placed_word.col - i
                            new_direction = Direction.ACROSS

                        if self._can_place_word(
                            candidate, answer, new_row, new_col, new_direction
                        ):
                            placement = WordPlacement(
                                text=answer,
                                clue=clue,
                                row=new_row,
                                col=new_col,
                                direction=new_direction,
                                context_score=context_score,
                            )
                            if self._is_valid_placement(candidate, placement):
                                possible_placements.append(placement)

        # Try placements randomly
        random.shuffle(possible_placements)
        for placement in possible_placements:
            if self._is_valid_placement(candidate, placement):
                self._place_word_on_grid(candidate, placement)
                candidate.placed_words.append(placement)
                return True

        return False

    def _can_place_word(
        self,
        candidate: CrosswordCandidate,
        word: str,
        row: int,
        col: int,
        direction: Direction,
    ) -> bool:
        """Check if word fits within grid boundaries."""
        if direction == Direction.ACROSS:
            return 0 <= row < self.size and 0 <= col <= self.size - len(word)
        else:
            return 0 <= row <= self.size - len(word) and 0 <= col < self.size

    def _is_valid_placement(
        self, candidate: CrosswordCandidate, placement: WordPlacement
    ) -> bool:
        """Check if word placement is valid (no conflicts, proper intersections, no duplicate clues)."""
        # Check for duplicate clue text within puzzle
        clue_normalized = placement.clue.strip().lower()
        if clue_normalized in candidate.used_clues:
            return False  # Reject placement with duplicate clue

        positions = placement.get_positions()

        for i, (r, c) in enumerate(positions):
            # Check boundaries
            if not (0 <= r < self.size and 0 <= c < self.size):
                return False

            # Check for conflicts
            if candidate.grid[r][c] != "" and candidate.grid[r][c] != placement.text[i]:
                return False

        # Check intersections with existing words
        for other_word in candidate.placed_words:
            intersections = placement.intersects_at(other_word)
            if intersections:
                if len(intersections) != 1:  # Must have exactly one intersection
                    return False

                # Verify character match at intersection
                int_r, int_c = intersections[0]
                self_pos = positions.index((int_r, int_c))
                other_pos = other_word.get_positions().index((int_r, int_c))

                if placement.text[self_pos] != other_word.text[other_pos]:
                    return False

        return True

    def _place_word_on_grid(
        self, candidate: CrosswordCandidate, placement: WordPlacement
    ):
        """Place word on candidate's grid and track clue usage."""
        for i, char in enumerate(placement.text):
            if placement.direction == Direction.ACROSS:
                candidate.grid[placement.row][placement.col + i] = char
            else:
                candidate.grid[placement.row + i][placement.col] = char

        # Track clue usage to prevent duplicates within puzzle
        candidate.used_clues.add(placement.clue.strip().lower())

    def _tournament_selection(
        self, population: List[CrosswordCandidate], tournament_size: int
    ) -> CrosswordCandidate:
        """Select candidate using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness_score)

    def _crossover(
        self,
        parent1: CrosswordCandidate,
        parent2: CrosswordCandidate,
        valid_pairs: List[Tuple[str, str]],
        context_scores: Dict[str, float],
    ) -> Optional[CrosswordCandidate]:
        """Create offspring by combining successful word placements from parents."""
        offspring = CrosswordCandidate(self.size)

        # Combine words from both parents, favoring those with higher fitness contribution
        all_parent_words = parent1.placed_words + parent2.placed_words

        # Sort by context score to prioritize Indian cultural words
        all_parent_words.sort(key=lambda x: x.context_score, reverse=True)

        # Try to place words from parents
        used_answers = set()
        for word in all_parent_words:
            if word.text not in used_answers:
                # Try to place this word
                if self._can_place_word(
                    offspring, word.text, word.row, word.col, word.direction
                ):
                    temp_placement = WordPlacement(
                        text=word.text,
                        clue=word.clue,
                        row=word.row,
                        col=word.col,
                        direction=word.direction,
                        context_score=word.context_score,
                    )
                    if self._is_valid_placement(offspring, temp_placement):
                        self._place_word_on_grid(offspring, temp_placement)
                        offspring.placed_words.append(temp_placement)
                        used_answers.add(word.text)

        # Fill remaining space with new words if possible
        remaining_pairs = [(c, a) for c, a in valid_pairs if a not in used_answers]
        attempts = 0
        max_fill_attempts = 20

        while attempts < max_fill_attempts and remaining_pairs:
            attempts += 1
            clue, answer = self._weighted_word_selection(
                remaining_pairs, context_scores
            )

            if self._place_intersecting_word(
                offspring, clue, answer, context_scores.get(answer, 0.0)
            ):
                remaining_pairs = [(c, a) for c, a in remaining_pairs if a != answer]
            else:
                remaining_pairs = [(c, a) for c, a in remaining_pairs if a != answer]

        return offspring if len(offspring.placed_words) >= 3 else None

    def _mutate(
        self,
        candidate: CrosswordCandidate,
        valid_pairs: List[Tuple[str, str]],
        context_scores: Dict[str, float],
    ) -> CrosswordCandidate:
        """Apply mutation by replacing or repositioning words."""
        mutated = candidate.clone()

        if not mutated.placed_words:
            return mutated

        mutation_type = random.choice(["replace", "reposition"])

        if mutation_type == "replace" and len(mutated.placed_words) > 1:
            # Replace a random word with a new one
            word_to_replace = random.choice(mutated.placed_words)

            # Remove the word
            self._remove_word(mutated, word_to_replace)

            # Try to place a new word with high context score
            available_pairs = [
                (c, a)
                for c, a in valid_pairs
                if a not in [w.text for w in mutated.placed_words]
            ]

            if available_pairs:
                clue, answer = self._weighted_word_selection(
                    available_pairs, context_scores
                )
                self._place_intersecting_word(
                    mutated, clue, answer, context_scores.get(answer, 0.0)
                )

        elif mutation_type == "reposition":
            # Try to improve word positions by local adjustments
            # This is a simplified repositioning - in practice could be more sophisticated
            pass

        return mutated

    def _remove_word(
        self, candidate: CrosswordCandidate, word_to_remove: WordPlacement
    ):
        """Remove word from candidate grid and word list."""
        # Clear grid positions
        for r, c in word_to_remove.get_positions():
            # Only clear if no other word uses this position
            other_words_at_position = [
                w
                for w in candidate.placed_words
                if w != word_to_remove and (r, c) in w.get_positions()
            ]
            if not other_words_at_position:
                candidate.grid[r][c] = ""

        # Remove from word list
        candidate.placed_words = [
            w for w in candidate.placed_words if w != word_to_remove
        ]

    def _local_search(
        self,
        candidate: CrosswordCandidate,
        valid_pairs: List[Tuple[str, str]],
        context_scores: Dict[str, float],
    ) -> CrosswordCandidate:
        """Apply local search to improve candidate."""
        improved = candidate.clone()

        # Try to add more words to improve density
        available_pairs = [
            (c, a)
            for c, a in valid_pairs
            if a not in [w.text for w in improved.placed_words]
        ]

        attempts = 0
        max_attempts = 10

        while attempts < max_attempts and available_pairs:
            attempts += 1
            clue, answer = self._weighted_word_selection(
                available_pairs, context_scores
            )

            if self._place_intersecting_word(
                improved, clue, answer, context_scores.get(answer, 0.0)
            ):
                available_pairs = [(c, a) for c, a in available_pairs if a != answer]
            else:
                available_pairs = [(c, a) for c, a in available_pairs if a != answer]

        return improved

    def _candidate_to_puzzle(
        self, candidate: CrosswordCandidate, puzzle_id: str
    ) -> CrosswordPuzzle:
        """Convert CrosswordCandidate to CrosswordPuzzle for evaluation compatibility."""
        # Assign word numbers
        self._assign_word_numbers(candidate)

        # Create CrosswordClue objects
        crossword_clues = []
        solution_words = {}

        for word in candidate.placed_words:
            clue = CrosswordClue(
                number=word.number,
                direction=word.direction.value,
                length=len(word.text),
                clue_text=word.clue,
                start_row=word.row,
                start_col=word.col,
                answer=word.text,
            )
            crossword_clues.append(clue)
            solution_words[f"{word.number}{word.direction.value}"] = word.text

        # Create solution grid
        solution_grid = copy.deepcopy(candidate.grid)
        for i in range(self.size):
            for j in range(self.size):
                if solution_grid[i][j] == "":
                    solution_grid[i][j] = "#"

        # Create puzzle grid for solving
        puzzle_grid = np.full((self.size, self.size), None, dtype=object)

        # Mark word start positions with numbers
        for word in candidate.placed_words:
            puzzle_grid[word.row][word.col] = word.number

        # Mark blocked cells and empty cells
        for i in range(self.size):
            for j in range(self.size):
                if candidate.grid[i][j] == "":
                    puzzle_grid[i][j] = "#"
                elif puzzle_grid[i][j] is None:
                    puzzle_grid[i][j] = "_"

        return CrosswordPuzzle(
            puzzle_id=puzzle_id,
            grid=puzzle_grid,
            clues=crossword_clues,
            size=(self.size, self.size),
            solution_grid=solution_grid,
            solution_words=solution_words,
        )

    def _assign_word_numbers(self, candidate: CrosswordCandidate):
        """Assign numbers to words based on reading order."""
        # Get starting positions
        start_positions = []
        for word in candidate.placed_words:
            start_positions.append((word.row, word.col, word))

        # Sort by reading order (top-to-bottom, left-to-right)
        start_positions.sort(key=lambda x: (x[0], x[1]))

        # Assign numbers
        number = 1
        used_positions = set()

        for row, col, word in start_positions:
            if (row, col) not in used_positions:
                word.number = number
                used_positions.add((row, col))
                number += 1

        # Handle shared starting positions
        for word in candidate.placed_words:
            if word.number == 0:
                for other_word in candidate.placed_words:
                    if (
                        other_word.row == word.row
                        and other_word.col == word.col
                        and other_word.number > 0
                    ):
                        word.number = other_word.number
                        break
