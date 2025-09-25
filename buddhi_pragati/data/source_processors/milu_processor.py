"""
MILU Dataset Processor

Handles processing of MILU (Multi-task Indic Language Understanding) dataset
for crossword clue-answer extraction. Filters out contextual MCQ questions
that require multiple choice options to answer correctly.
"""

import re
import logging
from typing import List, Dict, Set, Tuple
from datasets import load_dataset

from ..dataset_builder import SourceProcessor, DatasetEntry
from ...utils.unicode_utils import is_predominantly_latin_script, answer_in_clue_check

logger = logging.getLogger(__name__)


class MILUProcessor(SourceProcessor):
    """
    Processor for MILU dataset MCQ questions.

    Filters out questions that require context or multiple choice options,
    keeping only questions with single-word answers that work as crossword clues.
    """

    # Language codes supported by MILU
    SUPPORTED_LANGUAGES = {
        "Bengali",
        "English",
        "Gujarati",
        "Hindi",
        "Kannada",
        "Malayalam",
        "Marathi",
        "Odia",
        "Punjabi",
        "Tamil",
        "Telugu",
    }

    def __init__(self, config: dict, language: str, context_scorer=None):
        super().__init__(config, language, context_scorer)
        self.dataset = None
        self.current_index = 0
        self.current_batch_number = 0  # Track batch number for ID generation
        self.valid_count_in_batch = 0  # Track valid entries within current batch

        # Language-specific contextual question patterns
        self.contextual_patterns = self._get_contextual_patterns(language)

    def get_supported_languages(self) -> Set[str]:
        """Return set of languages supported by MILU."""
        return self.SUPPORTED_LANGUAGES

    def set_current_offset(self, offset: int):
        """Set current offset for unique ID generation across batches."""
        self.current_offset = offset

    def _get_contextual_patterns(self, language: str) -> List[re.Pattern]:
        """Get compiled regex patterns for detecting contextual questions."""
        patterns = []

        # English patterns - enhanced for stricter filtering
        if language == "English":
            patterns.extend(
                [
                    r"\b(?:which|what)(?:\s+of)?(?:\s+the)?(?:\s+following|above|given)\b",
                    r"\b[Cc]hoose\s+(?:the\s+)?(?:correct|right|best|appropriate)\b",
                    r"\b[Ss]elect\s+(?:the\s+)?(?:correct|right|best|appropriate|one)\b",
                    r"\ball\s+of\s+the\s+above\b",
                    r"\bnone\s+of\s+the\s+above\b",
                    r"\b(?:option|choice)\s*[A-D]\b",
                    r"\b(?:options|choices)\s+[A-D]\b",
                    r"\b[Ff]rom\s+(?:the\s+)?(?:given|above|following)\s+(?:options|choices)\b",
                    r"\bis\s+(?:the\s+)?correct\s+(?:option|answer|choice)\b",
                    r"\bselect\b.*\bfrom\b",
                    r"\bof\s+the\s+following\b",
                    r"\bmultiple\s+choice\b",
                    r"\b[Cc]hoose\s+from\b",
                    r"\b[Ww]hich\s+one\b",
                    r"\b[Mm]ark\s+the\s+correct\b",
                    r"\b[Ii]dentify\s+the\s+correct\b",
                    r"\b[Ss]elect\b",
                    r"\b(?:[Oo]f\s+the\s+following)\b",
                    r"\b[Ww]hich\s+among\b",
                    r"\b[Cc]hoose\s+the\s+one\b",
                    r"\b[Ss]elect\s+one\b",
                    r"\b[Aa]mong\s+the\s+following\b",
                    r"\b[Tt]he\s+correct\s+option\b",
                    r"\b[Tt]he\s+correct\s+answer\b",
                    r"\b[Oo]ut\s+of\s+the\s+following\b",
                    r"\b[Gg]iven\s+the\s+following\b",
                    r"\b[Gg]iven\s+the\s+options\b",
                    r"\b[Gg]iven\s+below\b",
                    r"\b[Gg]iven\s+options\b"
                ]
            )

        # Hindi patterns (no word boundaries for Devanagari)
        elif language == "Hindi":
            patterns.extend(
                [
                    r"(?:कौन\s*सा|क्या)\s+(?:निम्नलिखित|विकल्प|दिया\s+गया)",
                    r"सही\s+(?:विकल्प|उत्तर)\s+(?:चुनें|चुनिए)",
                    r"निम्नलिखित\s+में\s+से",
                    r"उपरोक्त\s+सभी",
                    r"इनमें\s+से\s+कोई\s+नहीं",
                    r"दिए\s+गए\s+विकल्पों\s+में\s+से",
                ]
            )

        # Bengali patterns
        elif language == "Bengali":
            patterns.extend(
                [
                    r"\b(?:কোনটি|কী)\s+(?:নিম্নলিখিত|বিকল্প|দেওয়া)\b",
                    r"\bসঠিক\s+(?:বিকল্প|উত্তর)\s+(?:নির্বাচন|বেছে)\b",
                    r"\bনিম্নলিখিত\s+থেকে\b",
                    r"\bউপরের\s+সবগুলি\b",
                ]
            )

        # Tamil patterns
        elif language == "Tamil":
            patterns.extend(
                [
                    r"\b(?:எது|என்ன)\s+(?:பின்வருவன|விருப்பம்|கொடுக்கப்பட்ட)\b",
                    r"\bசரியான\s+(?:விடை|தேர்வு)\s+(?:தேர்ந்தெடு|தேர்வு)\b",
                ]
            )

        # Telugu patterns
        elif language == "Telugu":
            patterns.extend(
                [
                    r"\b(?:ఏది|ఏమిటి)\s+(?:క్రింది|ఎంపిక|ఇవ్వబడిన)\b",
                    r"\bసరైన\s+(?:సమాధానం|ఎంపిక)\s+(?:ఎంచుకోండి|తేర్చుకోండి)\b",
                ]
            )

        # Compile patterns
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def load_raw_data(
        self, batch_size: int, offset: int = 0
    ) -> Tuple[List[Dict], bool]:
        """Load batch of raw MILU data using streaming to avoid loading entire dataset."""
        if not hasattr(self, "_stream_initialized"):
            logger.info(f"Initializing MILU streaming dataset for {self.language}...")
            try:
                # Use streaming dataset to avoid loading all entries into memory
                self.dataset = load_dataset(
                    "ai4bharat/MILU",
                    name=self.language,
                    split="test",
                    streaming=True,  # This is the key change - streaming mode
                )
                self._stream_iterator = iter(self.dataset)
                self._current_position = 0
                self._stream_initialized = True
                logger.info(f"Initialized MILU streaming dataset for {self.language}")
            except Exception as e:
                logger.error(
                    f"Failed to load MILU streaming dataset for {self.language}: {e}"
                )
                return [], False

        # Skip to desired offset if needed
        while self._current_position < offset:
            try:
                next(self._stream_iterator)
                self._current_position += 1
            except StopIteration:
                logger.info(
                    f"Reached end of MILU dataset at position {self._current_position}"
                )
                return [], False

        # Collect batch_size entries
        batch = []
        entries_collected = 0

        while entries_collected < batch_size:
            try:
                item = next(self._stream_iterator)
                batch.append(dict(item))
                entries_collected += 1
                self._current_position += 1
            except StopIteration:
                logger.info(
                    f"Reached end of MILU dataset at position {self._current_position}"
                )
                break

        # Return batch and whether more data is available
        # More intelligent has_more detection to prevent early termination
        has_more = entries_collected >= min(
            batch_size * 0.5, 50
        )  # At least 50% of requested or 50 entries minimum
        logger.debug(f"MILU batch: collected {len(batch)} entries, has_more={has_more}")

        return batch, has_more

    def process_batch(self, batch_data: List[Dict]) -> List[DatasetEntry]:
        """Process batch of MILU data into dataset entries."""
        entries = []
        batch_start_answer_in_clue = self.answer_in_clue_rejections
        batch_start_other = self.other_rejections

        # Reset valid count for new batch
        self.valid_count_in_batch = 0
        self.current_batch_number += 1

        for item in batch_data:
            try:
                before_answer_in_clue = self.answer_in_clue_rejections
                entry = self._process_single_item(item)
                if entry:
                    entries.append(entry)
                    self.valid_count_in_batch += 1
                else:
                    # If answer_in_clue counter didn't increase, this was rejected for other reasons
                    if self.answer_in_clue_rejections == before_answer_in_clue:
                        self.other_rejections += 1
            except Exception as e:
                logger.debug(f"Error processing MILU item: {e}")
                self.other_rejections += 1
                continue

        # Log batch statistics
        batch_answer_in_clue = (
            self.answer_in_clue_rejections - batch_start_answer_in_clue
        )
        batch_other = self.other_rejections - batch_start_other
        logger.info(
            f"MILU batch complete: processed {len(batch_data)}, valid {len(entries)}, "
            f"answer-in-clue rejections {batch_answer_in_clue}, other rejections {batch_other}"
        )

        return entries

    def _process_single_item(self, item: Dict) -> DatasetEntry:
        """Process a single MILU item into a dataset entry."""
        question = item.get("question", "").strip()
        if not question:
            return None

        # Check if this is a contextual question that needs options
        if self.config.get("MILU_FILTER_CONTEXTUAL_QUESTIONS", True):
            if self._is_contextual_question(question):
                return None

        # Get the correct answer (target is like 'option1', 'option2', etc.)
        target = item.get("target", "")
        if not target or not isinstance(target, str) or not target.startswith("option"):
            return None

        # Extract answer directly using the target field name
        answer = item.get(target, "").strip()
        if not answer:
            return None

        # For Indic languages, filter out entries where options/answer are in Latin script
        if self.language != "English":
            # Check if answer is predominantly Latin script (should be filtered out for Indic languages)
            if is_predominantly_latin_script(answer):
                return None

            # Also check all options to see if any are Latin script
            for i in range(1, 5):  # MILU has up to 4 options
                option_key = f"option{i}"
                if option_key in item:
                    option_value = item.get(option_key, "").strip()
                    if option_value and is_predominantly_latin_script(option_value):
                        return None

        # Check if answer is single word
        if not self.is_single_word(answer):
            return None

        # Clean up the question to make it more crossword-like
        cleaned_question = self._clean_question(question)
        if not cleaned_question:
            return None

        # Universal sanity check - ensure answer not in clue
        if answer_in_clue_check(cleaned_question, answer):
            logger.debug(
                f"MILU rejected entry: answer '{answer}' found in clue '{cleaned_question[:100]}...'"
            )
            self.answer_in_clue_rejections += 1
            return None

        # Check minimum word length for crossword suitability
        min_word_length = self.config.get("MIN_WORD_LENGTH", 2)
        if len(answer) < min_word_length:
            return None

        # Create dataset entry with new ID format: milu_{language}_{batch_number}_{count_in_batch}
        entry_id = f"milu_{self.language}_{self.current_batch_number}_{self.valid_count_in_batch}"

        quality_score = self.calculate_quality_score(cleaned_question, answer)

        # Note: context_score will be computed later after quality filtering (deferred scoring)
        return DatasetEntry(
            id=entry_id,
            clue=cleaned_question,
            answer=answer.upper(),
            source="MILU",
            source_id="N/A",  # As requested, MILU entries have no source ID
            quality_score=quality_score,
            # context_score defaults to 0.0 and will be computed later
        )

    def _is_contextual_question(self, question: str) -> bool:
        """Check if question requires contextual knowledge or multiple choice options."""
        for pattern in self.contextual_patterns:
            if pattern.search(question):
                return True
        return False

    def _clean_question(self, question: str) -> str:
        """Clean question to make it more suitable as a crossword clue."""
        # Remove MCQ-specific phrases
        cleaned = question

        # Remove common MCQ introductions
        mcq_phrases = [
            r"^\s*(?:Q\.?\s*\d+\.?\s*)?",  # Question numbers
            r"\s*\([A-D]\)\s*$",  # Option markers at end
            r"\s*[A-D][\.\)]\s*$",  # Option markers at end
        ]

        for phrase in mcq_phrases:
            cleaned = re.sub(phrase, "", cleaned, flags=re.IGNORECASE)

        # Clean up whitespace
        cleaned = " ".join(cleaned.split())

        # Check length constraints
        min_len = self.config.get("MIN_CLUE_LENGTH", 10)
        max_len = self.config.get("MAX_CLUE_LENGTH", 500)

        if len(cleaned) < min_len or len(cleaned) > max_len:
            return None

        return cleaned
