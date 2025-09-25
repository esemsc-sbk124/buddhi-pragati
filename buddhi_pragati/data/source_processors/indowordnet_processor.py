"""
IndoWordNet Dictionary Processor

Handles processing of IndoWordNet dictionary data using pyiwn library
to extract word-definition pairs for crossword clue-answer creation.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional

from ..dataset_builder import SourceProcessor, DatasetEntry
from ...utils.unicode_utils import answer_in_clue_check

logger = logging.getLogger(__name__)


class IndoWordNetProcessor(SourceProcessor):
    """
    Processor for IndoWordNet dictionary data.

    Uses pyiwn library to access word definitions across Indian languages
    and creates crossword clue-answer pairs.
    """

    # Languages supported by IndoWordNet (18 languages from the notebook)
    SUPPORTED_LANGUAGES = {
        "Assamese",
        "Bengali",
        "Bodo",
        "Gujarati",
        "Hindi",
        "Kannada",
        "Kashmiri",
        "Konkani",
        "Malayalam",
        "Marathi",
        "Meitei",
        "Nepali",
        "Odia",
        "Punjabi",
        "Sanskrit",
        "Tamil",
        "Telugu",
        "Urdu",
        # Note: English not in pyiwn language list
    }

    def __init__(self, config: dict, language: str, context_scorer=None):
        super().__init__(config, language, context_scorer)
        self.iwn = None
        self.synsets = []
        self.current_index = 0
        self.total_synsets = 0  # Track total count without loading all
        self._synset_iterator = None  # Lazy iterator for synsets

        # Initialize pyiwn
        self._initialize_wordnet()

    def get_supported_languages(self) -> Set[str]:
        """Return set of languages supported by IndoWordNet."""
        return self.SUPPORTED_LANGUAGES

    def _initialize_wordnet(self):
        """Initialize pyiwn WordNet interface with memory-efficient lazy loading."""
        try:
            import pyiwn

            # Map our language names to pyiwn Language enum
            lang_map = {
                "Assamese": pyiwn.Language.ASSAMESE,
                "Bengali": pyiwn.Language.BENGALI,
                "Bodo": pyiwn.Language.BODO,
                "Gujarati": pyiwn.Language.GUJARATI,
                "Hindi": pyiwn.Language.HINDI,
                "Kannada": pyiwn.Language.KANNADA,
                "Kashmiri": pyiwn.Language.KASHMIRI,
                "Konkani": pyiwn.Language.KONKANI,
                "Malayalam": pyiwn.Language.MALAYALAM,
                "Marathi": pyiwn.Language.MARATHI,
                "Meitei": pyiwn.Language.MEITEI,
                "Nepali": pyiwn.Language.NEPALI,
                "Odia": pyiwn.Language.ORIYA,  # Note: ORIYA in pyiwn
                "Punjabi": pyiwn.Language.PUNJABI,
                "Sanskrit": pyiwn.Language.SANSKRIT,
                "Tamil": pyiwn.Language.TAMIL,
                "Telugu": pyiwn.Language.TELUGU,
                "Urdu": pyiwn.Language.URDU,
            }

            lang_enum = lang_map.get(self.language)
            if not lang_enum:
                logger.error(f"Unsupported language for IndoWordNet: {self.language}")
                return

            self.iwn = pyiwn.IndoWordNet(lang=lang_enum)
            logger.info(f"Initialized IndoWordNet for {self.language}")

            # Get count without loading all synsets for memory efficiency
            try:
                # First try to get noun synsets count
                noun_synsets = self.iwn.all_synsets(pos=pyiwn.PosTag.NOUN)
                self.total_synsets = len(noun_synsets)
                self._synset_iterator = iter(noun_synsets)
                self._synset_type = "noun"
                logger.info(
                    f"Initialized {self.total_synsets} noun synsets for {self.language} (lazy loading)"
                )
                # Don't store the synsets - let them be garbage collected
                del noun_synsets
            except Exception as e:
                logger.warning(
                    f"Could not access noun synsets, trying all synsets: {e}"
                )
                try:
                    all_synsets = self.iwn.all_synsets()
                    self.total_synsets = len(all_synsets)
                    self._synset_iterator = iter(all_synsets)
                    self._synset_type = "all"
                    logger.info(
                        f"Initialized {self.total_synsets} total synsets for {self.language} (lazy loading)"
                    )
                    # Don't store the synsets - let them be garbage collected
                    del all_synsets
                except Exception as e2:
                    logger.error(f"Could not access any synsets: {e2}")
                    self.total_synsets = 0
                    self._synset_iterator = None

        except ImportError:
            logger.error("pyiwn library not found. Install with: pip install pyiwn")
            self.iwn = None
        except Exception as e:
            logger.error(f"Failed to initialize IndoWordNet: {e}")
            self.iwn = None

    def load_raw_data(
        self, batch_size: int, offset: int = 0
    ) -> Tuple[List[Dict], bool]:
        """Load batch of raw IndoWordNet data using memory-efficient streaming."""
        if not self.iwn or self.total_synsets == 0:
            logger.warning("IndoWordNet not available or no synsets initialized")
            return [], False

        # Reset iterator if we need to seek to a specific offset
        if offset != self.current_index or self._synset_iterator is None:
            self._reset_iterator_to_offset(offset)

        # Collect batch_size entries from iterator
        batch = []
        entries_collected = 0

        while (
            entries_collected < batch_size and self.current_index < self.total_synsets
        ):
            try:
                synset = next(self._synset_iterator)

                # Get definition (gloss) for this synset
                gloss = synset.gloss()
                # Get list of words (lemma names) for this synset
                lemma_names = synset.lemma_names()

                batch.append(
                    {
                        "synset": synset,
                        "gloss": gloss,
                        "lemma_names": lemma_names,
                        "synset_id": str(synset.synset_id()),
                    }
                )

                entries_collected += 1
                self.current_index += 1

            except StopIteration:
                logger.info(
                    f"Reached end of IndoWordNet synsets at position {self.current_index}"
                )
                break
            except Exception as e:
                logger.debug(
                    f"Error processing synset at position {self.current_index}: {e}"
                )
                self.current_index += 1
                continue

        # Determine if there's more data
        has_more = self.current_index < self.total_synsets
        logger.debug(
            f"IndoWordNet batch: collected {len(batch)} entries, position {self.current_index}/{self.total_synsets}, has_more={has_more}"
        )

        return batch, has_more

    def _reset_iterator_to_offset(self, offset: int):
        """Reset synset iterator to specific offset for seeking."""
        try:
            import pyiwn

            # Recreate iterator from the beginning
            if self._synset_type == "noun":
                synsets = self.iwn.all_synsets(pos=pyiwn.PosTag.NOUN)
            else:
                synsets = self.iwn.all_synsets()

            self._synset_iterator = iter(synsets)
            self.current_index = 0

            # Skip to offset position
            while self.current_index < offset:
                try:
                    next(self._synset_iterator)
                    self.current_index += 1
                except StopIteration:
                    logger.warning(
                        f"Reached end of synsets while seeking to offset {offset}"
                    )
                    break

            logger.debug(f"Reset IndoWordNet iterator to offset {offset}")

            # Release synsets from memory
            del synsets

        except Exception as e:
            logger.error(f"Failed to reset IndoWordNet iterator: {e}")
            self._synset_iterator = None

    def process_batch(self, batch_data: List[Dict]) -> List[DatasetEntry]:
        """Process batch of IndoWordNet data into dataset entries."""
        entries = []
        batch_start_answer_in_clue = self.answer_in_clue_rejections
        batch_start_other = self.other_rejections

        for item in batch_data:
            try:
                before_answer_in_clue = self.answer_in_clue_rejections
                item_entries = self._process_single_item(item)
                if item_entries:
                    entries.extend(item_entries)
                else:
                    # If answer_in_clue counter didn't increase, this was rejected for other reasons
                    if self.answer_in_clue_rejections == before_answer_in_clue:
                        self.other_rejections += 1
            except Exception as e:
                logger.debug(f"Error processing IndoWordNet item: {e}")
                self.other_rejections += 1
                continue

        # Log batch statistics
        batch_answer_in_clue = (
            self.answer_in_clue_rejections - batch_start_answer_in_clue
        )
        batch_other = self.other_rejections - batch_start_other
        logger.info(
            f"IndoWordNet batch complete: processed {len(batch_data)}, valid {len(entries)}, "
            f"answer-in-clue rejections {batch_answer_in_clue}, other rejections {batch_other}"
        )

        return entries

    def _process_single_item(self, item: Dict) -> List[DatasetEntry]:
        """Process a single synset into multiple dataset entries."""
        gloss = item.get("gloss", "").strip()
        lemma_names = item.get("lemma_names", [])
        synset_id = item.get("synset_id", "")

        if not gloss or not lemma_names:
            return []

        # Check minimum definition length
        min_length = self.config.get("INDOWORDNET_MIN_DEFINITION_LENGTH", 5)
        if len(gloss) < min_length:
            return []

        # Create entries for each word (lemma) in the synset
        entries = []

        for lemma_idx, word in enumerate(lemma_names):
            try:
                if not word or not word.strip():
                    continue

                word = word.strip()

                # Check if word is single word
                if not self.is_single_word(word):
                    continue

                entry = self._create_entry_from_word_definition(
                    word, gloss, synset_id, lemma_idx
                )
                if entry:
                    entries.append(entry)

            except Exception as e:
                logger.debug(f"Error processing lemma: {e}")
                continue

        return entries

    def _create_entry_from_word_definition(
        self, word: str, gloss: str, synset_id: str, lemma_idx: int
    ) -> Optional[DatasetEntry]:
        """Create a dataset entry from word and definition."""
        # Clean up the gloss to make it crossword-appropriate
        cleaned_gloss = self._clean_definition(gloss, word)
        if not cleaned_gloss:
            return None

        # Check gloss length constraints
        min_len = self.config.get("MIN_CLUE_LENGTH", 10)
        max_len = self.config.get("MAX_CLUE_LENGTH", 500)

        if len(cleaned_gloss) < min_len or len(cleaned_gloss) > max_len:
            return None

        # Check minimum word length for crossword suitability
        min_word_length = self.config.get("MIN_WORD_LENGTH", 2)
        if len(word) < min_word_length:
            return None

        # Universal sanity check - ensure answer not in clue
        if answer_in_clue_check(cleaned_gloss, word):
            logger.debug(
                f"IndoWordNet rejected entry: answer '{word}' found in clue '{cleaned_gloss[:100]}...'"
            )
            self.answer_in_clue_rejections += 1
            return None

        # Calculate quality score
        quality_score = self.calculate_quality_score(cleaned_gloss, word)

        # Create unique entry ID
        entry_id = f"indowordnet_{self.language}_{synset_id}_{lemma_idx}"

        # Note: context_score will be computed later after quality filtering (deferred scoring)
        return DatasetEntry(
            id=entry_id,
            clue=cleaned_gloss,
            answer=word.upper(),
            source="IndoWordNet",
            source_id=f"{synset_id}_{lemma_idx}",
            quality_score=quality_score,
            # context_score defaults to 0.0 and will be computed later
        )

    def _clean_definition(self, gloss: str, word: str) -> Optional[str]:
        """Clean definition to make it suitable as crossword clue."""
        # Remove the word itself from definition to avoid giving away the answer
        cleaned = gloss.replace(word, "").strip()

        # Clean up extra whitespace
        cleaned = " ".join(cleaned.split())

        # Make sure we still have meaningful content
        if len(cleaned) < 5:
            return None

        return cleaned
