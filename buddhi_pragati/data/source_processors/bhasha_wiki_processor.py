"""
Bhasha-Wiki Dataset Processor

Handles processing of Bhasha-Wiki translated Wikipedia articles using
IndicNER for named entity extraction to create crossword clue-answer pairs.
"""

import logging
import re
from typing import List, Dict, Set, Tuple, Optional
from datasets import load_dataset
from transformers import pipeline

from ..dataset_builder import SourceProcessor, DatasetEntry
from ...utils.unicode_utils import (
    contains_latin_script,
    extract_first_sentence,
    has_multiple_words,
    answer_in_clue_check,
)

logger = logging.getLogger(__name__)


class BhashaWikiProcessor(SourceProcessor):
    """
    Processor for Bhasha-Wiki dataset.

    Uses IndicNER to extract named entities from Wikipedia articles,
    then creates clue-answer pairs where the entity is the answer and
    the remaining text serves as the clue.
    """

    # Languages supported by Bhasha-Wiki
    SUPPORTED_LANGUAGES = {
        "Bengali",
        "English",
        "Gujarati",
        "Hindi",
        "Kannada",
        "Tamil",
        "Urdu",
    }

    def __init__(self, config: dict, language: str, context_scorer=None):
        super().__init__(config, language, context_scorer)
        self.dataset = None
        self.ner_pipeline = None

        # Initialize NER model
        self._initialize_ner_model()

    def get_supported_languages(self) -> Set[str]:
        """Return set of languages supported by Bhasha-Wiki."""
        return self.SUPPORTED_LANGUAGES

    def _initialize_ner_model(self):
        """Initialize IndicNER model for named entity recognition."""
        try:
            model_name = self.config.get("NER_MODEL_NAME", "ai4bharat/IndicNER")
            logger.info(f"Loading NER model: {model_name}")

            self.ner_pipeline = pipeline(
                "token-classification",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="max",  # Properly aggregates BIO-tagged tokens into complete entities
            )
            logger.info("NER model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            self.ner_pipeline = None

    def load_raw_data(
        self, batch_size: int, offset: int = 0
    ) -> Tuple[List[Dict], bool]:
        """Load batch of raw Bhasha-Wiki data."""
        if self.dataset is None:
            logger.info(f"Loading Bhasha-Wiki dataset for {self.language}...")
            try:
                # Map language to bhasha-wiki config name (includes date prefix)
                lang_map = {
                    "Bengali": "20231101.bn",
                    "English": "20231101.en",
                    "Gujarati": "20231101.gu",
                    "Hindi": "20231101.hi",
                    "Kannada": "20231101.kn",
                    "Tamil": "20231101.ta",
                    "Urdu": "20231101.ur",
                }

                lang_code = lang_map.get(self.language)
                if not lang_code:
                    logger.error(f"Unsupported language: {self.language}")
                    return [], False

                # Use streaming to avoid downloading all 318 parquet files
                self.dataset = load_dataset(
                    "soketlabs/bhasha-wiki",
                    name=lang_code,
                    split="train",
                    streaming=True,
                )
                logger.info(f"Loaded Bhasha-Wiki {self.language} in streaming mode")

            except Exception as e:
                logger.error(f"Failed to load Bhasha-Wiki for {self.language}: {e}")
                return [], False

        # Extract batch from streaming dataset
        if not self.dataset:
            return [], False

        batch = []
        try:
            # Skip to the offset position
            dataset_iter = iter(self.dataset)
            for _ in range(offset):
                next(dataset_iter)

            # Take batch_size items
            for _ in range(batch_size):
                try:
                    item = next(dataset_iter)
                    batch.append(dict(item))
                except StopIteration:
                    break

        except Exception as e:
            logger.warning(f"Error reading streaming dataset: {e}")
            return [], False

        # More intelligent has_more detection to prevent early termination
        # Only assume has_more=False if we get significantly fewer entries than requested
        has_more = len(batch) >= min(
            batch_size * 0.5, 50
        )  # At least 50% of requested or 50 entries minimum
        return batch, has_more

    def process_batch(self, batch_data: List[Dict]) -> List[DatasetEntry]:
        """Process batch of Bhasha-Wiki data into dataset entries."""
        entries = []
        batch_start_answer_in_clue = self.answer_in_clue_rejections
        batch_start_other = self.other_rejections

        if not self.ner_pipeline:
            logger.warning(
                "NER pipeline not available, skipping Bhasha-Wiki processing"
            )
            return entries

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
                logger.debug(f"Error processing Bhasha-Wiki item: {e}")
                self.other_rejections += 1
                continue

        # Log batch statistics
        batch_answer_in_clue = (
            self.answer_in_clue_rejections - batch_start_answer_in_clue
        )
        batch_other = self.other_rejections - batch_start_other
        logger.info(
            f"Bhasha-Wiki batch complete: processed {len(batch_data)}, valid {len(entries)}, "
            f"answer-in-clue rejections {batch_answer_in_clue}, other rejections {batch_other}"
        )

        return entries

    def _process_single_item(self, item: Dict) -> List[DatasetEntry]:
        """Process a single Bhasha-Wiki item into multiple dataset entries."""
        text = item.get("text", "").strip()
        title = item.get("title", "").strip()

        if not text or not title:
            return []

        # Filter out multi-word titles for all languages
        if has_multiple_words(title):
            return []

        # Check minimum text length
        min_length = self.config.get("BHASHA_WIKI_MIN_TEXT_LENGTH", 50)
        if len(text) < min_length:
            return []

        # Extract first sentence only
        first_sentence = extract_first_sentence(text)
        if not first_sentence or len(first_sentence) < min_length:
            return []

        # Handle different processing for English vs Indic languages
        if self.language == "English":
            return self._process_english_entry(first_sentence, title, item)
        else:
            return self._process_indic_entry(first_sentence, title, item)

    def _process_english_entry(
        self, first_sentence: str, title: str, item: Dict
    ) -> List[DatasetEntry]:
        """Process English Bhasha-wiki entry without NER."""
        # For English, we don't use NER - we use title as answer and remove it from text
        source_id = item.get("id", str(self.processed_count))

        # Remove title from first sentence to create clue
        clue = first_sentence.replace(title, "").strip()
        clue = " ".join(clue.split())  # Clean up extra spaces

        # Check if we have meaningful clue after removing title
        min_clue_length = self.config.get("MIN_CLUE_LENGTH", 10)
        if len(clue) < min_clue_length:
            return []

        # Validate title as answer
        if not self.is_single_word(title):
            return []

        min_word_length = self.config.get("MIN_WORD_LENGTH", 2)
        if len(title) < min_word_length:
            return []

        # Universal sanity check
        if answer_in_clue_check(clue, title):
            logger.debug(
                f"Bhasha-Wiki (English) rejected entry: answer '{title}' found in clue '{clue[:100]}...'"
            )
            self.answer_in_clue_rejections += 1
            return []

        # Calculate quality score
        quality_score = self.calculate_quality_score(clue, title)

        # Create entry
        entry_id = f"bhasha_wiki_{self.language}_{source_id}_0"

        # Note: context_score will be computed later after quality filtering (deferred scoring)
        entry = DatasetEntry(
            id=entry_id,
            clue=clue,
            answer=title.upper(),
            source="Bhasha-Wiki",
            source_id=f"{source_id}_0",
            quality_score=quality_score,
            # context_score defaults to 0.0 and will be computed later
        )

        return [entry]

    def _process_indic_entry(
        self, first_sentence: str, title: str, item: Dict
    ) -> List[DatasetEntry]:
        """Process Indic language Bhasha-wiki entry using NER."""
        # Filter out entries with Latin script entities (NER will miss them)
        if contains_latin_script(first_sentence):
            return []

        # Extract named entities using NER
        entities = self._extract_entities(first_sentence)
        if not entities:
            return []

        # Create entries for each valid entity
        entries = []
        source_id = item.get("id", str(self.processed_count))

        for entity_idx, entity_info in enumerate(entities):
            entry = self._create_entry_from_entity(
                first_sentence, entity_info, source_id, entity_idx
            )
            if entry:
                entries.append(entry)

        return entries

    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text using IndicNER."""
        try:
            # Limit text length to avoid memory issues
            max_length = 512  # BERT token limit
            if len(text) > max_length:
                text = text[:max_length]

            # Run NER
            entities = self.ner_pipeline(text)

            # Filter for person and location entities (most suitable for crosswords)
            valid_labels = {"PER", "LOC", "ORG"}  # Person, Location, Organization
            filtered_entities = []

            max_entities = self.config.get("MAX_ENTITIES_PER_TEXT", 5)

            for entity in entities:
                if entity["entity_group"] in valid_labels:
                    entity_text = entity["word"].strip()

                    # Check if entity is single word (if required)
                    if self.config.get("REQUIRE_SINGLE_WORD_ENTITIES", True):
                        if not self.is_single_word(entity_text):
                            continue

                    filtered_entities.append(
                        {
                            "text": entity_text,
                            "start": entity["start"],
                            "end": entity["end"],
                            "score": entity["score"],
                        }
                    )

                if len(filtered_entities) >= max_entities:
                    break

            return filtered_entities

        except Exception as e:
            logger.debug(f"NER extraction failed: {e}")
            return []

    def _create_entry_from_entity(
        self, text: str, entity_info: Dict, source_id: str, entity_idx: int
    ) -> Optional[DatasetEntry]:
        """Create a dataset entry from text and entity information."""
        entity_text = entity_info["text"]
        start_pos = entity_info["start"]
        end_pos = entity_info["end"]

        # Create clue by removing the entity from the text
        clue = text[:start_pos] + text[end_pos:]
        clue = " ".join(clue.split())  # Clean up whitespace

        # Check clue length constraints
        min_len = self.config.get("MIN_CLUE_LENGTH", 10)
        max_len = self.config.get("MAX_CLUE_LENGTH", 500)

        if len(clue) < min_len or len(clue) > max_len:
            return None

        # Clean up entity text
        answer = self._clean_entity(entity_text)
        if not answer:
            return None

        # Check minimum word length for crossword suitability
        min_word_length = self.config.get("MIN_WORD_LENGTH", 2)
        if len(answer) < min_word_length:
            return None

        # Universal sanity check - ensure answer not in clue
        if answer_in_clue_check(clue, answer):
            logger.debug(
                f"Bhasha-Wiki (Indic) rejected entry: answer '{answer}' found in clue '{clue[:100]}...'"
            )
            self.answer_in_clue_rejections += 1
            return None

        # Calculate quality score
        quality_score = self.calculate_quality_score(clue, answer)

        # Create unique entry ID
        entry_id = f"bhasha_wiki_{self.language}_{source_id}_{entity_idx}"

        # Note: context_score will be computed later after quality filtering (deferred scoring)
        return DatasetEntry(
            id=entry_id,
            clue=clue,
            answer=answer.upper(),
            source="Bhasha-Wiki",
            source_id=f"{source_id}_{entity_idx}",
            quality_score=quality_score,
            # context_score defaults to 0.0 and will be computed later
        )

    def _clean_entity(self, entity: str) -> Optional[str]:
        """Clean and validate entity text."""
        # Remove extra whitespace and punctuation
        cleaned = re.sub(r"[^\w\s]", "", entity).strip()
        cleaned = " ".join(cleaned.split())

        if not cleaned:
            return None

        # For single word requirement, take only first word
        if self.config.get("REQUIRE_SINGLE_WORD_ENTITIES", True):
            words = cleaned.split()
            if words:
                return words[0]
            return None

        return cleaned
