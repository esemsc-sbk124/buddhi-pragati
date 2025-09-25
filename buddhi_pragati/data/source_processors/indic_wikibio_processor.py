"""
IndicWikiBio Dataset Processor

Handles processing of IndicWikiBio biographical entries to extract
names from infoboxes and use summaries as clues for crossword pairs.
"""

import logging
import json
import zipfile
import tempfile
import os
from typing import List, Dict, Set, Tuple, Optional
from huggingface_hub import hf_hub_download

from ..dataset_builder import SourceProcessor, DatasetEntry
from ...utils.unicode_utils import answer_in_clue_check

logger = logging.getLogger(__name__)


class IndicWikiBioProcessor(SourceProcessor):
    """
    Processor for IndicWikiBio dataset.

    Extracts names from biographical infoboxes and removes them from
    the summary to create crossword clue-answer pairs.
    """

    # Languages supported by IndicWikiBio
    SUPPORTED_LANGUAGES = {
        "Assamese",
        "Bengali",
        "Hindi",
        "Kannada",
        "Malayalam",
        "Odia",
        "Punjabi",
        "Tamil",
        "Telugu",
    }

    def __init__(self, config: dict, language: str, context_scorer=None):
        super().__init__(config, language, context_scorer)
        self.dataset = None
        self._cached_data = None  # Cache for loaded data
        self._data_file_path = None  # Cache for extracted JSONL file

    def get_supported_languages(self) -> Set[str]:
        """Return set of languages supported by IndicWikiBio."""
        return self.SUPPORTED_LANGUAGES

    def load_raw_data(
        self, batch_size: int, offset: int = 0
    ) -> Tuple[List[Dict], bool]:
        """Load batch of raw IndicWikiBio data directly from zip files."""
        # Map language to IndicWikiBio language code
        lang_map = {
            "Assamese": "as",
            "Bengali": "bn",
            "Hindi": "hi",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Odia": "or",
            "Punjabi": "pa",
            "Tamil": "ta",
            "Telugu": "te",
        }

        lang_code = lang_map.get(self.language)
        if not lang_code:
            logger.error(f"Unsupported language: {self.language}")
            return [], False

        # Load data directly from zip files (no API calls needed)
        try:
            return self._load_batch_from_zip(lang_code, batch_size, offset)
        except Exception as e:
            logger.error(f"Failed to load IndicWikiBio batch for {self.language}: {e}")
            return [], False

    def process_batch(self, batch_data: List[Dict]) -> List[DatasetEntry]:
        """Process batch of IndicWikiBio data into dataset entries."""
        entries = []
        processed_count = 0
        error_count = 0
        batch_start_answer_in_clue = self.answer_in_clue_rejections
        batch_start_other = self.other_rejections

        for i, item in enumerate(batch_data):
            try:
                # Validate item structure
                if not isinstance(item, dict):
                    logger.debug(f"Item {i} is not a dict: {type(item)}")
                    error_count += 1
                    continue

                # Check required fields
                if "summary" not in item or "infobox" not in item:
                    logger.debug(
                        f"Item {i} missing required fields. Available: {list(item.keys())}"
                    )
                    error_count += 1
                    continue

                before_answer_in_clue = self.answer_in_clue_rejections
                item_entries = self._process_single_item(item)
                if item_entries:
                    entries.extend(item_entries)
                else:
                    # If answer_in_clue counter didn't increase, this was rejected for other reasons
                    if self.answer_in_clue_rejections == before_answer_in_clue:
                        self.other_rejections += 1
                processed_count += 1

                if i % 10 == 0:  # Log progress every 10 items
                    logger.debug(
                        f"Processed {i + 1}/{len(batch_data)} items, generated {len(entries)} entries so far"
                    )

            except Exception as e:
                logger.debug(f"Error processing IndicWikiBio item {i}: {e}")
                logger.debug(
                    f"Item keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}"
                )
                error_count += 1
                self.other_rejections += 1
                continue

        # Log batch statistics
        batch_answer_in_clue = (
            self.answer_in_clue_rejections - batch_start_answer_in_clue
        )
        batch_other = self.other_rejections - batch_start_other
        logger.info(
            f"IndicWikiBio batch complete: processed {len(batch_data)}, {len(entries)}, "
            f"answer-in-clue rejections {batch_answer_in_clue}, other rejections {batch_other}"
        )

        return entries

    def _process_single_item(self, item: Dict) -> List[DatasetEntry]:
        """Process a single IndicWikiBio item into multiple dataset entries."""
        summary = item.get("summary", "").strip()
        infobox_str = item.get("infobox", "")

        if not summary or not infobox_str:
            return []

        # Check minimum summary length
        min_length = self.config.get("INDIC_WIKIBIO_MIN_SUMMARY_LENGTH", 30)
        if len(summary) < min_length:
            return []

        # Extract names from infobox string
        names = self._extract_names_from_infobox(infobox_str)
        if not names:
            return []

        # Create entries for each valid name
        entries = []
        source_id = str(item.get("id", self.processed_count))

        for name_idx, name in enumerate(names):
            entry = self._create_entry_from_name(
                summary, names, name, source_id, name_idx
            )
            if entry:
                entries.append(entry)

        return entries

    def _extract_names_from_infobox(self, infobox_str: str) -> List[str]:
        """
        Extract names from tab-separated infobox string data.

        The infobox is a string in format: "name_1:value1\tname_2:value2\tother_field:value3"
        We parse this format to extract name fields and convert to individual words.
        """
        names = []

        if not infobox_str or not isinstance(infobox_str, str):
            return names

        try:
            # Parse tab-separated key:value pairs
            infobox_dict = {}
            pairs = infobox_str.split("\t")

            for pair in pairs:
                if ":" in pair:
                    key, value = pair.split(":", 1)  # Split only on first colon
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        infobox_dict[key] = value

            logger.debug(f"Parsed infobox keys: {list(infobox_dict.keys())}")

            # Extract sequential name fields (name_1, name_2, name_3, ...)
            name_parts = []
            i = 1
            while f"name_{i}" in infobox_dict:
                part = infobox_dict[f"name_{i}"].strip()
                if part and part not in ["<", ">", "br"]:  # Filter out HTML artifacts
                    name_parts.append(part)
                i += 1

            # If we found name parts, extract individual words
            if name_parts:
                for part in name_parts:
                    # Clean part of any remaining HTML-like content
                    cleaned_part = part.replace("<", "").replace(">", "").strip()
                    if self.is_single_word(cleaned_part):
                        names.append(cleaned_part)

            # Also check for other common name fields
            other_name_fields = ["full_name", "birth_name", "real_name"]
            for field in other_name_fields:
                if field in infobox_dict:
                    value = infobox_dict[field].strip()
                    if value and value.lower() not in ["unknown", "न/ए", "अज्ञात"]:
                        # Split multi-word names into individual words
                        words = value.split()
                        for word in words:
                            cleaned_word = (
                                word.replace("<", "").replace(">", "").strip()
                            )
                            if self.is_single_word(cleaned_word):
                                names.append(cleaned_word)

            # Remove duplicates while preserving order
            unique_names = []
            seen = set()
            for name in names:
                if name not in seen and len(name) > 1:  # Ensure meaningful names
                    unique_names.append(name)
                    seen.add(name)

            logger.debug(f"Extracted names: {unique_names}")
            return unique_names[:5]  # Limit to 5 names per entry

        except Exception as e:
            logger.debug(f"Error parsing infobox: {e}")
            return []

    def _create_entry_from_name(
        self,
        summary: str,
        all_names: List[str],
        name: str,
        source_id: str,
        name_idx: int,
    ) -> Optional[DatasetEntry]:
        """Create a dataset entry by removing ALL names from summary."""
        # Start with the original summary
        clue = summary

        # Remove ALL extracted names from the clue
        for extracted_name in all_names:
            clue = clue.replace(extracted_name, "").strip()

        # Clean up extra whitespace
        clue = " ".join(clue.split())

        # Apply phonetic pronunciation filter
        clue = self._remove_phonetic_pronunciations(clue)

        # Check if we still have meaningful clue after removing all names
        if len(clue) < 10:  # Increased threshold since we need substantial content
            return None

        # Check clue length constraints
        min_len = self.config.get("MIN_CLUE_LENGTH", 10)
        max_len = self.config.get("MAX_CLUE_LENGTH", 500)

        if len(clue) < min_len or len(clue) > max_len:
            return None

        # Check minimum word length for crossword suitability
        min_word_length = self.config.get("MIN_WORD_LENGTH", 2)
        if len(name) < min_word_length:
            return None

        # Universal sanity check - ensure answer not in clue
        if answer_in_clue_check(clue, name):
            logger.debug(
                f"IndicWikiBio rejected entry: answer '{name}' found in clue '{clue[:100]}...'"
            )
            self.answer_in_clue_rejections += 1
            return None

        # Calculate quality score
        quality_score = self.calculate_quality_score(clue, name)

        # Create unique entry ID with correct format: indic_wiki_bio_{language}_{source_id}_{name_idx}
        # We need name_idx since one source entry can generate multiple clue-answer pairs
        entry_id = f"indic_wiki_bio_{self.language}_{source_id}_{name_idx}"

        # Note: context_score will be computed later after quality filtering (deferred scoring)
        return DatasetEntry(
            id=entry_id,
            clue=clue,
            answer=name.upper(),
            source="IndicWikiBio",
            source_id=source_id,  # Use actual source_id from dataset
            quality_score=quality_score,
            # context_score defaults to 0.0 and will be computed later
        )

    def _remove_phonetic_pronunciations(self, text: str) -> str:
        """Remove phonetic pronunciations and transliterations from text."""
        import re

        # Patterns to remove phonetic pronunciations
        patterns = [
            # Latin script in parentheses: (लातिन : Franciscus [franˈtʃiskus])
            r"\(लातिन\s*:\s*[^)]*\)",
            r"\([Ll]atin\s*:\s*[^)]*\)",
            # Phonetic transcriptions in brackets: [franˈtʃiskus]
            r"\[[^\]]*ˈ[^\]]*\]",  # Contains phonetic stress marks
            r"\[[^\]]*ʃ[^\]]*\]",  # Contains phonetic symbols
            r"\[[^\]]*θ[^\]]*\]",  # Contains phonetic symbols
            r"\[[^\]]*ð[^\]]*\]",  # Contains phonetic symbols
            # Birth names in parentheses that are transliterations
            r"\(\s*जन्म\s*:\s*[^)]*\)",
            r"\([Bb]irth\s*:\s*[^)]*\)",
            r"\([Bb]orn\s*:\s*[^)]*\)",
            # Generic parenthetical content with mixed scripts
            r"\([^)]*[a-zA-Z][^)]*[a-zA-Z][^)]*\)",  # Contains multiple Latin letters
        ]

        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

        # Clean up extra whitespace and punctuation
        cleaned_text = re.sub(r"\s*;\s*", " ", cleaned_text)  # Remove semicolons
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text

    def _load_batch_from_zip(
        self, lang_code: str, batch_size: int, offset: int
    ) -> Tuple[List[Dict], bool]:
        """Load batch of IndicWikiBio data directly from zip files."""

        # Load all data if not cached
        if self._cached_data is None:
            logger.info(f"Loading IndicWikiBio data for {lang_code} from zip file...")

            try:
                # Download the zip file for this language
                zip_filename = f"data/{lang_code}_WikiBio_v1.0.zip"
                zip_path = hf_hub_download(
                    repo_id="ai4bharat/IndicWikiBio",
                    filename=zip_filename,
                    repo_type="dataset",
                )

                # Extract and load training data
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_ref.extractall(temp_dir)

                        # Look for training file
                        train_file = os.path.join(temp_dir, f"{lang_code}_train.jsonl")
                        if os.path.exists(train_file):
                            self._cached_data = []
                            with open(train_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    if line.strip():
                                        entry = json.loads(line.strip())
                                        self._cached_data.append(entry)

                            logger.info(
                                f"Loaded {len(self._cached_data)} entries from {train_file}"
                            )
                        else:
                            logger.error(f"Training file not found: {train_file}")
                            return [], False

            except Exception as e:
                logger.error(f"Failed to load data from zip for {lang_code}: {e}")
                return [], False

        # Return batch from cached data
        if self._cached_data is None:
            return [], False

        total_entries = len(self._cached_data)
        start_idx = offset
        end_idx = min(offset + batch_size, total_entries)

        batch = self._cached_data[start_idx:end_idx]
        has_more = end_idx < total_entries

        logger.debug(
            f"Loaded {len(batch)} entries from zip file for {lang_code}, "
            f"offset={offset}, has_more={has_more} (total: {total_entries})"
        )

        return batch, has_more
