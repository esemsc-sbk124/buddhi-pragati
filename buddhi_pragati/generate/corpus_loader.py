"""
Crossword Corpus Loader for HuggingFace Dataset Integration

This module loads clue-answer pairs from the existing Buddhi-Pragati HuggingFace dataset
and provides cultural context scoring with language-specific normalization for classification.
"""

import logging
from typing import List, Tuple, Dict, NamedTuple
from datasets import load_dataset
from ..utils.config_loader import get_config
from ..utils.unicode_utils import is_predominantly_latin_script

logger = logging.getLogger(__name__)


class CorpusEntry(NamedTuple):
    """Extended corpus entry with cultural context information."""

    clue: str
    answer: str
    context_score: float  # Raw context score
    source: str
    source_id: str
    quality_score: float


class CrosswordCorpusLoader:
    """
    Loads clue-answer corpus from HuggingFace datasets with cultural context scoring.

    Uses raw context scores for both classification and puzzle entry generation.
    """

    def __init__(self, repo_id: str = None, hf_token: str = None):
        """
        Initialize corpus loader.

        Args:
            repo_id: HuggingFace repository ID (default from config)
            hf_token: HuggingFace token for dataset access (default from config/env)
        """
        self.config = get_config()
        self.repo_id = repo_id or self.config.get_string(
            "HF_DATASET_REPO", "selim-b-kh/buddhi-pragati"
        )
        self.hf_token = hf_token or self.config.get_string("DEFAULT_HF_TOKEN", "")

        self.loaded_corpus = {}  # language -> List[CorpusEntry]
        self.context_scorers = {}  # Cache context scorers per language

        logger.info(f"Initialized CrosswordCorpusLoader for repository: {self.repo_id}")

    def _is_indic_language(self, language: str) -> bool:
        """
        Check if the language is an Indic language that should filter out Latin script answers.

        Args:
            language: Language name

        Returns:
            True if this is an Indic language that needs script filtering
        """
        # Check if script filtering is enabled globally
        if not self.config.get_bool("ENABLE_SCRIPT_FILTERING", True):
            return False

        # Get configurable list of languages requiring script filtering
        indic_languages_str = self.config.get_string(
            "INDIC_LANGUAGES_FOR_FILTERING",
            "Assamese,Bengali,Bodo,Gujarati,Hindi,Kannada,Kashmiri,Konkani,Malayalam,Marathi,Meitei,Nepali,Odia,Punjabi,Sanskrit,Tamil,Telugu,Urdu",
        )
        indic_languages = {
            lang.strip() for lang in indic_languages_str.split(",") if lang.strip()
        }
        return language in indic_languages

    def load_scored_corpus(
        self, language: str, max_entries: int = None
    ) -> List[CorpusEntry]:
        """
        Load corpus with cultural context scores.

        Args:
            language: Language to load (e.g., "English", "Hindi")
            max_entries: Maximum entries to load (None for all)

        Returns:
            List of CorpusEntry with raw context scores
        """
        try:
            config_name = language.lower()

            logger.info(f"Loading scored corpus for {language} from {self.repo_id}")

            # Load dataset with token if available
            dataset_kwargs = {"split": "train"}
            if self.hf_token:
                dataset_kwargs["token"] = self.hf_token

            dataset = load_dataset(self.repo_id, config_name, **dataset_kwargs)

            # Process entries and create corpus with raw context scores
            corpus_entries = []
            raw_scores = []
            entries_processed = 0

            for entry in dataset:
                if max_entries and entries_processed >= max_entries:
                    break

                clue = entry.get("clue", "").strip()
                answer = entry.get("answer", "").upper().strip()
                source = entry.get("source", "unknown")
                source_id = entry.get("source_id", "")
                quality_score = entry.get("quality_score", 0.5)

                # Basic validation
                if not (clue and answer and len(answer) >= 2):
                    continue

                # Script filtering for Indic languages - reject Latin script answers
                if self._is_indic_language(language) and is_predominantly_latin_script(
                    answer
                ):
                    logger.debug(
                        f"Filtered Latin script answer in {language}: {answer}"
                    )
                    continue

                # Calculate or use existing context score
                if "context_score" in entry:
                    # Use pre-computed score from dataset
                    context_score = float(entry["context_score"])
                else:
                    context_score = 0.0

                corpus_entry = CorpusEntry(
                    clue=clue,
                    answer=answer,
                    context_score=context_score,
                    source=source,
                    source_id=source_id,
                    quality_score=quality_score,
                )

                corpus_entries.append(corpus_entry)
                raw_scores.append(context_score)
                entries_processed += 1

                if entries_processed % 100 == 0:
                    logger.debug(
                        f"Processed {entries_processed} entries for {language}"
                    )

            if not corpus_entries:
                logger.warning(f"No valid entries found for {language}")
                return []

            # Log context score distribution
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            mean_score = sum(raw_scores) / len(raw_scores)

            logger.info(f"Context score distribution for {language}:")
            logger.info(
                f"  Min: {min_score:.4f}, Max: {max_score:.4f}, Mean: {mean_score:.4f}"
            )

            logger.info(f"Loaded {len(corpus_entries)} scored entries for {language}")

            # Cache the loaded corpus
            self.loaded_corpus[language] = corpus_entries

            return corpus_entries

        except Exception as e:
            logger.error(f"Failed to load scored corpus for {language}: {e}")
            return []

    def get_prioritized_corpus(
        self, language: str, indian_threshold: float = None, max_entries: int = None
    ) -> Tuple[List[Tuple[str, str]], Dict[str, float]]:
        """
        Get corpus prioritized by Indian cultural context using raw context scores.

        Args:
            language: Language to load
            indian_threshold: Threshold for prioritizing Indian entries (applied to raw context scores)
            max_entries: Maximum entries to load (None for all)

        Returns:
            Tuple of (prioritized_corpus, context_scores)
            - prioritized_corpus: List of (clue, answer) pairs
            - context_scores: Dictionary mapping answer -> context_score
        """
        if indian_threshold is None:
            indian_threshold = self.config.get_float("INDIAN_CONTEXT_THRESHOLD", 0.4)

        # Load scored corpus
        scored_corpus = self.load_scored_corpus(language, max_entries)
        if not scored_corpus:
            logger.warning(f"No corpus loaded for {language}")
            return [], {}

        # Create return formats
        prioritized_pairs = [(entry.clue, entry.answer) for entry in scored_corpus]
        context_scores = {entry.answer: entry.context_score for entry in scored_corpus}

        repartition = {}
        high_context_count = 0

        # Log classification statistics using raw context scores
        for entry in scored_corpus:
            if entry.context_score >= indian_threshold:
                repartition[entry.source] = repartition.get(entry.source, 0) + 1
                high_context_count += 1

        # Show context score ranges for debugging
        all_scores = [entry.context_score for entry in scored_corpus]

        logger.info(f"Prioritized corpus for {language}:")
        logger.info(f"  Total entries: {len(prioritized_pairs)}")
        logger.info(
            f"  High Indian context (>= {indian_threshold}): {high_context_count} ({high_context_count / len(scored_corpus) * 100:.1f}%)"
        )
        logger.info(
            f"  Context scores - Min: {min(all_scores):.4f}, Max: {max(all_scores):.4f}"
        )

        logger.info(f"  Indian source repartition: {repartition}")
        return prioritized_pairs, context_scores

    def get_source_distribution(self, language: str) -> Dict[str, int]:
        """
        Get distribution of sources in the corpus for tracking source_mix.

        Args:
            language: Language to analyze

        Returns:
            Dictionary mapping source -> count
        """
        if language not in self.loaded_corpus:
            self.load_scored_corpus(language)

        if language not in self.loaded_corpus:
            return {}

        source_counts = {}
        for entry in self.loaded_corpus[language]:
            source_counts[entry.source] = source_counts.get(entry.source, 0) + 1

        return source_counts

    def get_corpus_statistics(self, language: str) -> Dict[str, any]:
        """
        Get detailed statistics about the corpus.

        Args:
            language: Language to analyze

        Returns:
            Dictionary with comprehensive corpus statistics
        """
        if language not in self.loaded_corpus:
            self.load_scored_corpus(language)

        if language not in self.loaded_corpus or not self.loaded_corpus[language]:
            return {"error": "No corpus loaded"}

        corpus = self.loaded_corpus[language]

        # Basic statistics
        stats = {
            "total_entries": len(corpus),
            "language": language,
            "repository": self.repo_id,
        }

        # Context score distribution
        context_scores = [entry.context_score for entry in corpus]

        stats["context_scores"] = {
            "min": min(context_scores),
            "max": max(context_scores),
            "avg": sum(context_scores) / len(context_scores),
            "range": max(context_scores) - min(context_scores),
            "high_context_count": sum(1 for score in context_scores if score >= 0.4),
            "high_context_percentage": sum(
                1 for score in context_scores if score >= 0.4
            )
            / len(context_scores)
            * 100,
        }

        # Answer length distribution (for grid size planning)
        answer_lengths = [len(entry.answer) for entry in corpus]
        stats["answer_length"] = {
            "min": min(answer_lengths),
            "max": max(answer_lengths),
            "avg": sum(answer_lengths) / len(answer_lengths),
            "suitable_for_small_grids": sum(
                1 for length in answer_lengths if 3 <= length <= 7
            ),
            "suitable_for_large_grids": sum(
                1 for length in answer_lengths if 8 <= length <= 15
            ),
        }

        # Source distribution
        stats["source_distribution"] = self.get_source_distribution(language)

        return stats

    def clear_cache(self):
        """Clear cached datasets and scorers to free memory."""
        self.loaded_corpus.clear()
        self.context_scorers.clear()
        logger.info("Cleared corpus loader cache")

    def get_supported_languages(self) -> List[str]:
        """
        Get list of languages available in the repository.

        Returns:
            List of available language configuration names
        """
        try:
            from datasets import get_dataset_config_names

            configs = get_dataset_config_names(
                self.repo_id, token=self.hf_token if self.hf_token else None
            )

            # Filter out invalid configurations
            valid_configs = []
            for config in configs:
                if config not in ["data", "default", "train", "test", "validation"]:
                    valid_configs.append(config.title())  # Convert to proper case

            logger.info(
                f"Found {len(valid_configs)} language configurations: {valid_configs}"
            )
            return sorted(valid_configs)

        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return []
