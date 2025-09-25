"""
Comprehensive Dataset Builder for Buddhi-Pragati Crossword System

This module implements a complete pipeline for creating structured crossword clue-answer
datasets from multiple Indian language sources including MILU, bhasha-wiki, IndicWikiBio,
and IndoWordNet.

Key Features:
- Multi-source data integration with fallback for unsupported languages
- Batch processing for memory efficiency
- Quality control and single-word answer enforcement
- HuggingFace dataset upload capability
- Progress tracking and resumable processing

Architecture:
- DatasetBuilder: Main orchestrator class
- SourceProcessor: Abstract base for source-specific processors
- Language-aware processing with automatic source selection
- Configurable parameters from crossword_config.txt
"""

import os
import logging
import gc
import time
import psutil
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import asdict
from abc import ABC, abstractmethod
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login

from ..utils.config_loader import get_config
from ..utils import is_alphabetic_unicode
from ..utils.indian_context_scorer import IndianContextScorer
from .local_backup_manager import LocalBackupManager
from .data_structure import DatasetEntry

logger = logging.getLogger(__name__)


class SourceProcessor(ABC):
    """
    Abstract base class for source-specific data processors.

    Each source (MILU, bhasha-wiki, etc.) implements this interface to provide
    consistent processing of raw data into standardized DatasetEntry objects.
    """

    def __init__(
        self,
        config: dict,
        language: str,
        context_scorer: Optional[IndianContextScorer] = None,
    ):
        self.config = config
        self.language = language
        self.processed_count = 0
        self.valid_count = 0

        # Rejection tracking
        self.answer_in_clue_rejections = 0
        self.other_rejections = 0

        # Enhanced context scoring
        self._context_scorer = (
            context_scorer  # Use provided scorer first, fallback to lazy-loading
        )
        self._enable_enhanced_scoring = config.get(
            "ENABLE_TIERED_CONTEXT_SCORING", True
        )

    @abstractmethod
    def get_supported_languages(self) -> Set[str]:
        """Return set of languages supported by this source."""
        pass

    @abstractmethod
    def process_batch(self, batch_data: List[Dict]) -> List[DatasetEntry]:
        """Process a batch of raw data into standardized entries."""
        pass

    @abstractmethod
    def load_raw_data(
        self, batch_size: int, offset: int = 0
    ) -> Tuple[List[Dict], bool]:
        """Load raw data batch from source. Returns (data, has_more)."""
        pass

    def is_single_word(self, text: str) -> bool:
        """Check if text is a single word suitable for crossword."""
        if not text or not text.strip():
            return False
        words = text.strip().split()
        return len(words) == 1 and is_alphabetic_unicode(words[0])

    def calculate_quality_score(self, clue: str, answer: str) -> float:
        """Calculate crossword suitability score."""
        score = 1.0

        # Penalize very short or very long clues
        clue_len = len(clue)
        if clue_len < 10:
            score *= 0.5
        elif clue_len > 200:
            score *= 0.7

        # Reward medium-length answers (5-8 chars are ideal for crosswords)
        answer_len = len(answer)
        if 5 <= answer_len <= 8:
            score *= 1.2
        elif answer_len < 3 or answer_len > 12:
            score *= 0.6

        # Penalize answers with repeated characters
        unique_chars = len(set(answer))
        if unique_chars / len(answer) < 0.5:
            score *= 0.8

        return min(score, 1.0)

    def _load_context_scorer(self) -> Optional[IndianContextScorer]:
        """Lazy load the Indian context scorer with proper configuration parsing."""
        if self._context_scorer is None and self._enable_enhanced_scoring:
            try:
                # Parse scoring weights from config
                parsed_config = self._parse_scoring_config()
                # Get mode from parent DatasetBuilder if available
                mode = getattr(self, "context_scoring_mode", "complete")
                self._context_scorer = IndianContextScorer(parsed_config, mode=mode)
                logger.info(
                    f"Loaded IndianContextScorer for language: {self.language} (mode: {mode})"
                )
            except Exception as e:
                logger.warning(f"Failed to load IndianContextScorer: {e}")
                self._context_scorer = None

        return self._context_scorer

    def _parse_scoring_config(self) -> Dict[str, any]:
        """Parse scoring configuration from string format to proper types."""
        config = self.config.copy()

        # Parse tier 1 weights (embedding,keyword)
        if "CONTEXT_SCORING_WEIGHTS_TIER1" in config:
            weights_str = config["CONTEXT_SCORING_WEIGHTS_TIER1"]
            if isinstance(weights_str, str):
                weights = [float(x.strip()) for x in weights_str.split(",")]
                if len(weights) == 2:
                    config["CONTEXT_SCORING_WEIGHTS_TIER1"] = {
                        "embedding": weights[0],
                        "keyword": weights[1],
                    }

        # Parse tier 2 weights (multilingual,indic,keyword)
        if "CONTEXT_SCORING_WEIGHTS_TIER2" in config:
            weights_str = config["CONTEXT_SCORING_WEIGHTS_TIER2"]
            if isinstance(weights_str, str):
                weights = [float(x.strip()) for x in weights_str.split(",")]
                if len(weights) == 3:
                    config["CONTEXT_SCORING_WEIGHTS_TIER2"] = {
                        "multilingual": weights[0],
                        "indic": weights[1],
                        "keyword": weights[2],
                    }

        return config


class DatasetBuilder:
    """
    Main orchestrator for building structured crossword datasets.

    Coordinates multiple source processors to create a unified dataset
    with configurable size and quality controls.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        context_scoring_mode: str = "complete",
        custom_batch_size: Optional[int] = None,
    ):
        from .source_processors import (
            MILUProcessor,
            BhashaWikiProcessor,
            IndicWikiBioProcessor,
            IndoWordNetProcessor,
        )

        self.config = get_config()
        self.dataset_config = self.config.get_dataset_config()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.context_scoring_mode = (
            context_scoring_mode  # Store mode for context scoring
        )
        self.custom_batch_size = custom_batch_size  # CLI override for batch size
        self.stats = defaultdict(lambda: defaultdict(int))

        # Memory tracking
        self.initial_memory = self._get_memory_usage()

        # Shared context scorers to avoid repeated model loading
        self.shared_context_scorers = {}  # language -> IndianContextScorer instance

        # Mapping of source names to their processor classes
        self.SOURCE_PROCESSORS = {
            "MILU": MILUProcessor,
            "Bhasha-Wiki": BhashaWikiProcessor,
            "IndicWikiBio": IndicWikiBioProcessor,
            "IndoWordNet": IndoWordNetProcessor,
        }

        # Initialize HF API if token provided
        if self.hf_token:
            login(token=self.hf_token)
            self.hf_api = HfApi()
        else:
            self.hf_api = None

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                "percent": process.memory_percent(),
            }
        except Exception as e:
            logger.debug(f"Failed to get memory usage: {e}")
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

    def _log_memory_usage(self, context: str):
        """Log current memory usage with context."""
        current_memory = self._get_memory_usage()
        memory_change = current_memory["rss_mb"] - self.initial_memory["rss_mb"]

        logger.info(
            f"MEMORY_USAGE ({context}): "
            f"RSS={current_memory['rss_mb']:.1f}MB "
            f"(+{memory_change:+.1f}MB), "
            f"VMS={current_memory['vms_mb']:.1f}MB, "
            f"Percent={current_memory['percent']:.1f}%"
        )

        return current_memory

    def _get_shared_context_scorer(
        self, language: str
    ) -> Optional[IndianContextScorer]:
        """
        Get or create shared IndianContextScorer for a language.

        This method ensures that sentence transformer models are loaded only once per language,
        rather than once per source processor, significantly reducing model loading overhead.

        Args:
            language: Target language for context scoring

        Returns:
            Shared IndianContextScorer instance for the language, or None if scorer disabled/failed
        """
        # Check if enhanced scoring is enabled
        if not self.dataset_config.get("ENABLE_TIERED_CONTEXT_SCORING", True):
            return None

        # Return existing scorer if already created
        if language in self.shared_context_scorers:
            return self.shared_context_scorers[language]

        try:
            # Parse scoring configuration (replicating SourceProcessor._parse_scoring_config logic)
            config = self.dataset_config.copy()

            # Parse tier 1 weights (embedding,ner,keyword)
            if "CONTEXT_SCORING_WEIGHTS_TIER1" in config:
                weights_str = config["CONTEXT_SCORING_WEIGHTS_TIER1"]
                if isinstance(weights_str, str):
                    weights = [float(x.strip()) for x in weights_str.split(",")]
                    if len(weights) == 3:
                        config["CONTEXT_SCORING_WEIGHTS_TIER1"] = {
                            "embedding": weights[0],
                            "ner": weights[1],
                            "keyword": weights[2],
                        }

            # Parse tier 2 weights (multilingual,indic,ner,keyword)
            if "CONTEXT_SCORING_WEIGHTS_TIER2" in config:
                weights_str = config["CONTEXT_SCORING_WEIGHTS_TIER2"]
                if isinstance(weights_str, str):
                    weights = [float(x.strip()) for x in weights_str.split(",")]
                    if len(weights) == 4:
                        config["CONTEXT_SCORING_WEIGHTS_TIER2"] = {
                            "multilingual": weights[0],
                            "indic": weights[1],
                            "ner": weights[2],
                            "keyword": weights[3],
                        }

            # Create and cache the scorer with mode
            logger.info(
                f"Creating shared IndianContextScorer for language: {language} (mode: {self.context_scoring_mode})"
            )
            scorer = IndianContextScorer(config, mode=self.context_scoring_mode)
            self.shared_context_scorers[language] = scorer
            logger.info(
                f"Successfully created shared IndianContextScorer for language: {language} (mode: {self.context_scoring_mode})"
            )

            return scorer

        except Exception as e:
            logger.warning(
                f"Failed to create shared IndianContextScorer for {language}: {e}"
            )
            self.shared_context_scorers[language] = None  # Cache the failure
            return None

    def _cleanup_shared_scorers(self):
        """Clean up shared context scorers to free memory."""
        if self.shared_context_scorers:
            logger.info(
                f"Cleaning up {len(self.shared_context_scorers)} shared context scorers"
            )
            self.shared_context_scorers.clear()
            gc.collect()

    def _cleanup_after_upload(self):
        """Comprehensive cleanup after upload to free memory."""
        logger.info("Performing post-upload memory cleanup")

        # Clean up shared context scorers
        self._cleanup_shared_scorers()

        # Clear processing statistics to free memory
        self.stats.clear()

        # Force garbage collection
        gc.collect()

        # Log memory usage after cleanup
        self._log_memory_usage("POST_UPLOAD_CLEANUP")

        logger.info("Post-upload memory cleanup complete")

    def _compute_deferred_context_scores(
        self, entries: List[DatasetEntry], language: str
    ) -> List[DatasetEntry]:
        """
        Compute context scores for entries that have passed quality filtering.

        This method implements deferred context scoring to avoid expensive ML model
        computations for entries that will be filtered out due to low quality scores.

        Args:
            entries: List of dataset entries that have passed quality filtering
            language: Target language for context scoring

        Returns:
            List of entries with updated context scores
        """
        if not entries:
            return entries

        # Get shared context scorer for this language
        shared_scorer = self._get_shared_context_scorer(language)
        if not shared_scorer:
            logger.warning(
                f"No context scorer available for {language}, keeping default context scores"
            )
            return entries

        logger.info(
            f"Computing deferred context scores for {len(entries)} entries in {language}"
        )

        updated_entries = []
        successful_scores = 0
        failed_scores = 0

        for i, entry in enumerate(entries):
            try:
                # Compute context score using the shared scorer
                result = shared_scorer.score_context(entry.clue, entry.answer, language)
                context_score = result.get("final_score", 0.0)

                # Ensure valid range and update entry
                context_score = max(0.0, min(1.0, context_score))

                # Create updated entry with computed context score
                updated_entry = DatasetEntry(
                    id=entry.id,
                    clue=entry.clue,
                    answer=entry.answer,
                    source=entry.source,
                    source_id=entry.source_id,
                    context_score=context_score,  # Updated with computed score
                    quality_score=entry.quality_score,
                )
                updated_entries.append(updated_entry)
                successful_scores += 1

                # Log progress for large batches
                if (i + 1) % 50 == 0 or i == len(entries) - 1:
                    logger.debug(
                        f"Computed context scores: {i + 1}/{len(entries)} entries processed"
                    )

            except Exception as e:
                logger.debug(
                    f"Failed to compute context score for entry {entry.id}: {e}"
                )
                # Keep original entry with default context score (0.0)
                updated_entries.append(entry)
                failed_scores += 1

        # Log final statistics
        logger.info(
            f"Deferred context scoring complete: {successful_scores} successful, "
            f"{failed_scores} failed, {len(updated_entries)} total entries"
        )

        return updated_entries

    def get_available_sources_for_language(self, language: str) -> List[str]:
        """Get list of sources that support the given language."""
        available = []
        for source_name, processor_class in self.SOURCE_PROCESSORS.items():
            # Create temporary instance to check language support
            temp_processor = processor_class(self.dataset_config, language)
            if language in temp_processor.get_supported_languages():
                available.append(source_name)
        return available

    def calculate_source_batch_size(self, source_name: str, target_size: int) -> int:
        """
        Calculate optimal batch size for a specific source based on source characteristics and target size.

        Args:
            source_name: Name of the data source (MILU, IndicWikiBio, IndoWordNet, Bhasha-Wiki)
            target_size: Target number of entries for this source

        Returns:
            Optimal batch size for this source
        """
        # Use CLI override if provided
        if self.custom_batch_size is not None:
            logger.info(
                f"Using CLI batch size override: {self.custom_batch_size} for {source_name}"
            )
            return self.custom_batch_size

        # Source-specific batch size calculation based on efficiency and constraints
        if source_name == "MILU":
            # MILU: 1.5x target size (good efficiency, can handle larger batches)
            batch_size = int(1.5 * target_size)
        elif source_name == "IndicWikiBio":
            # IndicWikiBio: 0.5x target size (moderate efficiency, medium batches)
            batch_size = int(0.5 * target_size)
        elif source_name == "IndoWordNet":
            # IndoWordNet: 0.25x target size (lower efficiency, smaller batches)
            batch_size = int(0.25 * target_size)
        elif source_name == "Bhasha-Wiki":
            # Bhasha-Wiki: 5x target size (very low efficiency, large batches needed)
            batch_size = int(5 * target_size)
        else:
            # Fallback to config default for unknown sources
            batch_size = self.dataset_config.get("BATCH_SIZE_PROCESSING", 100)
            logger.warning(
                f"Unknown source {source_name}, using default batch size: {batch_size}"
            )

        # Ensure minimum batch size of 1
        batch_size = max(1, batch_size)

        logger.info(
            f"Calculated batch size for {source_name}: {batch_size} (target: {target_size})"
        )
        return batch_size

    def calculate_target_per_source(
        self,
        language: str,
        total_target: int,
        requested_sources: List[str] = None,
    ) -> Dict[str, int]:
        """Calculate how many entries to generate per source for uniform distribution."""
        available_sources = self.get_available_sources_for_language(language)
        if not available_sources:
            raise ValueError(f"No sources available for language: {language}")

        # Filter by requested sources if specified
        if requested_sources:
            # Only keep sources that are both available and requested
            filtered_sources = [s for s in requested_sources if s in available_sources]
            if not filtered_sources:
                raise ValueError(
                    f"None of the requested sources {requested_sources} are available for language: {language}. Available sources: {available_sources}"
                )
            available_sources = filtered_sources

        logger.info(f"Using sources for {language}: {available_sources}")

        # Distribute target evenly across available sources
        per_source = total_target // len(available_sources)
        remainder = total_target % len(available_sources)

        targets = {}
        for i, source in enumerate(available_sources):
            targets[source] = per_source + (1 if i < remainder else 0)

        logger.info(f"Target distribution for {language}: {targets}")
        return targets

    def process_source_streaming(
        self,
        source_name: str,
        language: str,
        target_count: int,
        start_offset: int = 0,
        backup_manager=None,
    ) -> Tuple[int, object]:
        """Process a single source and accumulate entries to disk via backup manager.

        Args:
            source_name: Name of the data source to process
            language: Target language
            target_count: Number of entries to process
            start_offset: Starting offset for processing
            backup_manager: LocalBackupManager instance for saving batches to disk

        Returns:
            Tuple[int, LocalBackupManager]: (total entries saved, backup manager instance)
        """
        logger.info(
            f"Processing {source_name} for {language}, target: {target_count}, starting from offset: {start_offset}"
        )
        logger.info("Mode: ACCUMULATION")

        # Log initial memory usage for this source
        self._log_memory_usage(f"START {source_name}")

        if source_name not in self.SOURCE_PROCESSORS:
            raise ValueError(f"Unknown source: {source_name}")

        processor_class = self.SOURCE_PROCESSORS[source_name]

        # Get shared context scorer for this language to avoid repeated model loading
        shared_scorer = self._get_shared_context_scorer(language)

        processor = processor_class(
            self.dataset_config, language, context_scorer=shared_scorer
        )

        # Calculate source-specific batch size
        batch_size = self.calculate_source_batch_size(source_name, target_count)
        offset = start_offset
        seen_answers = set()  # For deduplication

        # Track processing for disk-based accumulation
        total_saved = 0
        # Calculate starting batch number to avoid overwriting existing batch files
        existing_batches = []
        if backup_manager:
            language_dir = backup_manager.backup_dir / language.lower()
            if language_dir.exists():
                existing_batches = list(
                    language_dir.glob(f"batch_{source_name}_*.json")
                )

        # Start batch numbering after existing batches for this source
        batch_num = len(
            [b for b in existing_batches if f"batch_{source_name}_" in b.name]
        )

        while total_saved < target_count:
            logger.info(
                f"Processing batch {batch_num + 1} for {source_name} (saved: {total_saved}/{target_count})..."
            )

            # Load batch
            raw_batch, has_more = processor.load_raw_data(batch_size, offset)
            if not raw_batch:
                logger.warning(f"No more data available from {source_name}")
                break

            # Set current offset for unique ID generation (especially for MILU)
            if hasattr(processor, "set_current_offset"):
                processor.set_current_offset(offset)

            # Process batch with immediate quality filtering
            raw_batch_entries = processor.process_batch(raw_batch)

            # Apply deduplication and save to disk via backup manager
            deduplicated_batch = []
            for entry in raw_batch_entries:
                answer_key = entry.answer.upper()
                if answer_key not in seen_answers:
                    seen_answers.add(answer_key)
                    deduplicated_batch.append(entry)

            # Apply MIN_QUALITY_SCORE filtering before context scoring
            min_quality = self.dataset_config.get("MIN_QUALITY_SCORE", 0.5)
            quality_filtered_entries = [
                e for e in deduplicated_batch if e.quality_score >= min_quality
            ]

            # Apply deferred context scoring to this batch
            batch_start_time = time.time()
            processed_batch = self._compute_deferred_context_scores(
                quality_filtered_entries, language
            )
            batch_time = time.time() - batch_start_time

            logger.info(
                f"Context scored batch for {source_name}: "
                f"{len(processed_batch)}/{len(processed_batch)} entries processed "
                f"(processing: {batch_time:.1f}s)"
            )

            # Save to disk via backup manager if provided
            if backup_manager and processed_batch:
                batch_path = backup_manager.save_batch(
                    language, source_name, batch_num, processed_batch
                )
                if batch_path:
                    logger.info(f"Saved {len(processed_batch)} entries to {batch_path}")

            total_saved += len(processed_batch)
            batch_num += 1

            # Stop if we've reached our target
            if total_saved >= target_count:
                break

            # Update processing statistics
            batch_processed = len(raw_batch)
            batch_valid = len(processed_batch)

            if source_name not in self.stats:
                self.stats[source_name] = {"processed": 0, "valid": 0}

            self.stats[source_name]["processed"] += batch_processed
            self.stats[source_name]["valid"] += batch_valid

            # Update offset for next iteration
            offset += len(raw_batch)

            # Check if we've reached the end of this source
            if not has_more:
                logger.info(f"Reached end of {source_name} data")
                break

            offset += batch_size

        # Log final memory usage for this source
        self._log_memory_usage(f"END {source_name}")

        # Calculate and log efficiency metrics
        processed_total = self.stats[source_name]["processed"]
        valid_total = self.stats[source_name]["valid"]
        efficiency = (valid_total / processed_total * 100) if processed_total > 0 else 0

        # Get rejection statistics from processor
        answer_in_clue_rejections = getattr(processor, "answer_in_clue_rejections", 0)
        other_rejections = getattr(processor, "other_rejections", 0)
        answer_in_clue_pct = (
            (answer_in_clue_rejections / processed_total * 100)
            if processed_total > 0
            else 0
        )

        logger.info(
            f"Completed {source_name}: {total_saved}/{target_count} entries saved to disk ({batch_num} batches)"
        )
        logger.info(
            f"SOURCE_EFFICIENCY {source_name}: {efficiency:.1f}% "
            f"({valid_total} valid / {processed_total} processed)"
        )
        logger.info(
            f"SOURCE_REJECTIONS {source_name}: answer-in-clue {answer_in_clue_rejections} ({answer_in_clue_pct:.1f}%), "
            f"other {other_rejections}"
        )

        return total_saved, backup_manager

    def build_dataset(
        self,
        language: str,
        total_target: int = None,
        sources: List[str] = None,
    ) -> LocalBackupManager:
        """
        Build complete dataset for a language with disk-based batch accumulation.

        Args:
            language: Target language
            total_target: Total entries to generate (default from config)
            sources: List of sources to use (default from config)

        Returns:
            LocalBackupManager instance with accumulated batch files
        """
        if total_target is None:
            total_target = self.dataset_config.get(
                "TARGET_DATASET_SIZE_PER_LANGUAGE", 1000
            )

        if sources is None:
            # Get default sources from config
            default_sources_str = self.dataset_config.get(
                "DEFAULT_DATASET_SOURCES",
                "MILU,IndicWikiBio,IndoWordNet,Bhasha-Wiki",
            )
            sources = [s.strip() for s in default_sources_str.split(",")]

        logger.info("=" * 60)
        logger.info(f"BUILDING DATASET FOR {language.upper()}")
        logger.info(f"Target size: {total_target} entries")
        logger.info(f"Requested sources: {sources}")
        logger.info("=" * 60)

        # Calculate initial distribution across sources
        source_targets = self.calculate_target_per_source(
            language, total_target, sources
        )

        # Initialize backup manager for disk-based accumulation
        backup_manager = LocalBackupManager()
        actual_counts = {}
        shortfall_total = 0

        # Collect entries from all sources with disk-based accumulation
        logger.info("📝 Building dataset - processing batches to disk from all sources")

        for source_name, target_count in source_targets.items():
            try:
                actual_count, backup_manager = self.process_source_streaming(
                    source_name, language, target_count, backup_manager=backup_manager
                )
                actual_counts[source_name] = actual_count

                shortfall = target_count - actual_count
                if shortfall > 0:
                    shortfall_total += shortfall
                    logger.warning(
                        f"{source_name} shortfall: {shortfall} entries (got {actual_count}/{target_count})"
                    )

            except Exception as e:
                logger.error(f"Error processing {source_name}: {e}")
                actual_counts[source_name] = 0
                shortfall_total += target_count
                continue

        # Get current total from backup manager for redistribution check
        current_total = sum(actual_counts.values())

        # Implement fallback mechanism for shortfall redistribution
        if shortfall_total > 0 and current_total < total_target:
            logger.info(f"FALLBACK MECHANISM: Need {shortfall_total} more entries")
            self._redistribute_shortfall_disk_based(
                language,
                sources,
                actual_counts,
                source_targets,
                shortfall_total,
                backup_manager,
            )

        # Clean up shared context scorers to free memory
        self._cleanup_shared_scorers()

        # Final statistics from backup manager
        final_stats = backup_manager.get_backup_stats(language)
        logger.info(f"Dataset building complete for {language}")
        logger.info(
            f"Final size: {final_stats['total_entries']} entries saved to disk in {final_stats['batch_count']} batch files"
        )

        return backup_manager

    def _redistribute_shortfall_disk_based(
        self,
        language: str,
        sources: List[str],
        actual_counts: Dict[str, int],
        original_targets: Dict[str, int],
        shortfall_total: int,
        backup_manager: LocalBackupManager,
    ):
        """
        Redistribute shortfall entries to sources using disk-based backup manager.

        When sources can't meet their targets, redistribute the missing entries
        to other sources that have capacity for additional processing.
        """
        available_sources = self.get_available_sources_for_language(language)
        if not available_sources:
            available_sources = sources  # Fallback to requested sources

        # Find sources that met their targets (have capacity for more)
        successful_sources = [
            source
            for source in available_sources
            if actual_counts.get(source, 0) >= original_targets.get(source, 0)
        ]

        # If no fully successful sources, try sources that haven't been processed yet
        if not successful_sources:
            unprocessed_sources = [
                source for source in sources if actual_counts.get(source, 0) == 0
            ]
            if unprocessed_sources:
                logger.info(
                    f"No fully successful sources, trying unprocessed sources: {unprocessed_sources}"
                )
                successful_sources = unprocessed_sources
            else:
                logger.warning(
                    "No sources available for redistribution (excluding Bhasha-Wiki)"
                )
                return

        logger.info(
            f"Redistributing {shortfall_total} entries among: {successful_sources}"
        )

        # Distribute shortfall equally among successful sources
        per_source_extra = max(1, shortfall_total // len(successful_sources))
        remainder = shortfall_total % len(successful_sources)

        redistributed_count = 0
        for i, source_name in enumerate(successful_sources):
            # Add remainder to first few sources
            extra_entries = per_source_extra + (1 if i < remainder else 0)

            if redistributed_count >= shortfall_total:
                break

            logger.info(
                f"Requesting {extra_entries} additional entries from {source_name}"
            )

            try:
                # Process additional entries from this source with offset
                original_target = original_targets[source_name]
                batch_size = self.calculate_source_batch_size(
                    source_name, original_target
                )
                existing_count = actual_counts[source_name]
                estimated_batches_processed = (existing_count // batch_size) + (
                    1 if existing_count % batch_size > 0 else 0
                )
                start_offset = estimated_batches_processed * batch_size

                additional_count, backup_manager = self.process_source_streaming(
                    source_name,
                    language,
                    extra_entries,
                    start_offset=start_offset,
                    backup_manager=backup_manager,
                )

                redistributed_count += additional_count

                logger.info(
                    f"Got {additional_count} additional entries from {source_name}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to get additional entries from {source_name}: {e}"
                )
                continue

        logger.info(
            f"Redistribution complete: {redistributed_count}/{shortfall_total} additional entries"
        )

    def _scan_repository_configurations(self, repo_id: str) -> List[str]:
        """Scan HuggingFace repository to find all existing language configurations."""
        try:
            # First try the proper HuggingFace API for getting dataset configurations
            from datasets import get_dataset_config_names

            logger.debug(f"Getting dataset configurations using HF API for {repo_id}")
            configs = get_dataset_config_names(repo_id, token=self.hf_token)

            # Filter out any invalid configurations that might cause issues
            valid_configs = []
            for config in configs:
                # Skip invalid configuration names (including "default" which causes issues)
                invalid_configs = ["data", "train", "test", "validation", "default"]
                if config not in invalid_configs:
                    # Additional validation: config names should be language-like (alphabetic, 3+ chars)
                    if config.isalpha() and len(config) >= 3:
                        valid_configs.append(config)
                    else:
                        logger.debug(
                            f"Skipping non-language configuration name: {config}"
                        )
                else:
                    logger.debug(f"Skipping invalid configuration name: {config}")

            config_list = sorted(valid_configs)
            logger.debug(
                f"Found {len(config_list)} valid configurations in {repo_id}: {config_list}"
            )
            return config_list

        except Exception as e:
            logger.warning(f"Could not get configurations using HF API: {e}")

            # Fallback to file-based scanning with improved logic
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=self.hf_token)

                # Clear any cached dataset information first
                self._clear_dataset_cache(repo_id)

                # Get all files in the repository
                files = list(api.list_repo_files(repo_id, repo_type="dataset"))

                # Extract configuration names from parquet file paths with better filtering
                configs = set()
                for file in files:
                    if file.endswith(".parquet") and "/" in file:
                        config_name = file.split("/")[0]

                        # Skip common false-positive directory names (including "default" which causes issues)
                        invalid_configs = [
                            "data",
                            ".huggingface",
                            "refs",
                            "train",
                            "test",
                            "validation",
                            "default",
                        ]
                        if config_name not in invalid_configs:
                            # Additional validation: config names should be language-like
                            if config_name.isalpha() and len(config_name) >= 3:
                                configs.add(config_name)
                            else:
                                logger.debug(
                                    f"Skipping non-language configuration name in fallback: {config_name}"
                                )
                        else:
                            logger.debug(
                                f"Skipping invalid configuration name in fallback: {config_name}"
                            )

                config_list = sorted(list(configs))
                logger.debug(
                    f"Found {len(config_list)} configurations via fallback method in {repo_id}: {config_list}"
                )
                return config_list

            except Exception as fallback_e:
                logger.warning(f"Fallback configuration scan also failed: {fallback_e}")
                return []

    def _clear_dataset_cache(self, repo_id: str, config_name: str = None):
        """
        Clear HuggingFace dataset cache for a specific repository.

        Args:
            repo_id: Repository ID (e.g., 'selim-b-kh/buddhi-pragati')
            config_name: Optional language configuration to clear specifically.
                        If None, clears entire repository cache.
        """
        try:
            import shutil
            from pathlib import Path

            repo_name = repo_id.replace("/", "___")

            if config_name:
                logger.info(
                    f"Clearing HuggingFace cache for {repo_id}, config: {config_name}"
                )
            else:
                logger.info(f"Clearing all HuggingFace cache for repository: {repo_id}")

            # Clear HuggingFace datasets cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
            if cache_dir.exists():
                if config_name:
                    # Language-specific cache patterns
                    patterns_to_clear = [
                        f"*{repo_name}*{config_name}*",
                        f"*{repo_id.replace('/', '_')}*{config_name}*",
                        f"{repo_id.split('/')[-1]}*{config_name}*",
                    ]
                else:
                    # Original behavior - clear all
                    patterns_to_clear = [
                        f"*{repo_name}*",
                        f"*{repo_id.replace('/', '_')}*",
                        f"{repo_id.split('/')[-1]}*",
                    ]

                cleared_count = 0
                for pattern in patterns_to_clear:
                    for cache_path in cache_dir.glob(pattern):
                        if cache_path.is_dir():
                            logger.debug(f"Clearing dataset cache: {cache_path}")
                            shutil.rmtree(cache_path, ignore_errors=True)
                            cleared_count += 1

                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} dataset cache directories")

            # Clear hub cache
            hub_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if hub_cache_dir.exists():
                if config_name:
                    # Language-specific hub cache patterns
                    hub_patterns = [
                        f"datasets--{repo_name}*{config_name}*",
                        f"datasets--{repo_id.replace('/', '--')}*{config_name}*",
                    ]
                else:
                    # Original behavior - clear all
                    hub_patterns = [
                        f"datasets--{repo_name}*",
                        f"datasets--{repo_id.replace('/', '--')}*",
                    ]

                cleared_count = 0
                for pattern in hub_patterns:
                    for cache_path in hub_cache_dir.glob(pattern):
                        if cache_path.is_dir():
                            logger.debug(f"Clearing hub cache: {cache_path}")
                            shutil.rmtree(cache_path, ignore_errors=True)
                            cleared_count += 1

                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} hub cache directories")

            # Force clear datasets library internal cache
            try:
                from datasets import config as datasets_config

                if hasattr(datasets_config, "HF_DATASETS_CACHE"):
                    datasets_cache_dir = Path(datasets_config.HF_DATASETS_CACHE)
                    if datasets_cache_dir.exists():
                        if config_name:
                            pattern = f"*{repo_name}*{config_name}*"
                        else:
                            pattern = f"*{repo_name}*"

                        cleared_count = 0
                        for cache_path in datasets_cache_dir.glob(pattern):
                            if cache_path.is_dir():
                                logger.debug(
                                    f"Clearing datasets internal cache: {cache_path}"
                                )
                                shutil.rmtree(cache_path, ignore_errors=True)
                                cleared_count += 1

                        if cleared_count > 0:
                            logger.info(
                                f"Cleared {cleared_count} internal cache directories"
                            )
            except Exception:
                pass

        except Exception as e:
            logger.warning(
                f"Failed to clear cache for {repo_id}{f' (config: {config_name})' if config_name else ''}: {e}"
            )

    def _get_configuration_stats(
        self, repo_id: str, config_name: str, force_refresh: bool = False
    ) -> dict:
        """
        Get statistics for a specific language configuration.

        Args:
            repo_id: Repository ID (e.g., 'selim-b-kh/buddhi-pragati')
            config_name: Language configuration name (e.g., 'english', 'hindi')
            force_refresh: If True, force download fresh data. If False, reuse cache when possible.
        """
        try:
            # Validate configuration name before attempting to load
            if not config_name or not config_name.isalpha() or len(config_name) < 3:
                logger.warning(f"Invalid configuration name format: {config_name}")
                raise ValueError(f"Invalid configuration name: {config_name}")

            # Use intelligent download mode based on force_refresh parameter
            download_mode = (
                "force_redownload" if force_refresh else "reuse_cache_if_exists"
            )

            if force_refresh:
                logger.debug(
                    f"Force refreshing stats for {config_name} (fresh download)"
                )
            else:
                logger.debug(
                    f"Getting stats for {config_name} (reuse cache if available)"
                )

            dataset = load_dataset(
                repo_id,
                config_name,
                split="train",
                token=self.hf_token,
                download_mode=download_mode,
                verification_mode="no_checks",  # Skip verification for performance
            )
            df = dataset.to_pandas()

            # Validate required columns exist
            required_columns = ["context_score", "quality_score", "source"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(
                    f"Dataset {config_name} missing required columns: {missing_columns}"
                )
                raise ValueError(f"Missing columns: {missing_columns}")

            stats = {
                "total_entries": len(df),
                "sources": list(set(df["source"].tolist())),
                "min_quality": float(df["quality_score"].min()),
                "max_quality": float(df["quality_score"].max()),
                "min_context": float(df["context_score"].min()),
                "max_context": float(df["context_score"].max()),
                "median_context": float(df["context_score"].median()),
                "mean_context": float(df["context_score"].mean()),
                "mean_quality": float(df["quality_score"].mean()),
                "median_quality": float(df["quality_score"].median()),
                "num_bytes": dataset.info.dataset_size if dataset.info else 0,
                "download_size": dataset.info.download_size if dataset.info else 0,
            }

            logger.debug(
                f"Stats for {config_name}: {stats['total_entries']} entries, sources: {stats['sources']}"
            )
            return stats

        except Exception as e:
            logger.warning(f"Could not get stats for {config_name}: {e}")
            return {
                "total_entries": 0,
                "sources": [],
                "min_quality": 0.0,
                "max_quality": 0.0,
                "min_context": 0.0,
                "max_context": 0.0,
                "median_context": 0.0,
                "mean_context": 0.0,
                "mean_quality": 0.0,
                "median_quality": 0.0,
                "num_bytes": 0,
                "download_size": 0,
            }

    def _generate_dynamic_yaml_header(
        self, repo_id: str, all_configs: List[str], active_language: str = None
    ) -> str:
        """
        Generate accurate YAML header with real dataset statistics.

        Args:
            repo_id: Repository ID
            all_configs: List of all language configurations
            active_language: Language currently being updated (gets fresh stats, others use cache)
        """
        yaml_lines = ["---", "license: apache-2.0", "configs:"]

        dataset_info_lines = ["dataset_info:"]

        for config in all_configs:
            # Config section
            yaml_lines.append(f"- config_name: {config}")
            yaml_lines.append("  data_files:")
            yaml_lines.append("  - split: train")
            yaml_lines.append(f"    path: {config}/train-*")

            # Use intelligent caching - force refresh only for active language
            force_refresh = active_language and config == active_language.lower()
            stats = self._get_configuration_stats(
                repo_id, config, force_refresh=force_refresh
            )

            # Dataset info section
            dataset_info_lines.extend(
                [
                    f"- config_name: {config}",
                    "  features:",
                    "  - name: id",
                    "    dtype: string",
                    "  - name: clue",
                    "    dtype: string",
                    "  - name: answer",
                    "    dtype: string",
                    "  - name: source",
                    "    dtype: string",
                    "  - name: source_id",
                    "    dtype: string",
                    "  - name: context_score",
                    "    dtype: float64",
                    "  - name: quality_score",
                    "    dtype: float64",
                    "  splits:",
                    "  - name: train",
                    f"    num_bytes: {stats['num_bytes']}",
                    f"    num_examples: {stats['total_entries']}",
                    f"  download_size: {stats['download_size']}",
                    f"  dataset_size: {stats['num_bytes']}",
                ]
            )

        # Combine all sections
        yaml_lines.extend(dataset_info_lines)
        yaml_lines.append("---")

        return "\n".join(yaml_lines)

    def _generate_clean_yaml_header(self) -> str:
        """Generate YAML header for clean repository (no configurations)."""
        return """---
license: apache-2.0
configs: []
dataset_info: []
---"""

    def _generate_clean_readme_content(self) -> str:
        """Generate README content for clean repository (no configurations)."""
        return """# Buddhi-Pragati Crossword Dataset

This dataset will contain crossword clue-answer pairs extracted from multiple Indian language sources.

## Status
Repository is currently clean and ready for new dataset population.

## Planned Features
- Support for 20 languages: Assamese, Bengali, Bodo, English, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Malayalam, Marathi, Meitei, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu
- Multiple sources: MILU, IndicWikiBio, IndoWordNet, Bhasha-Wiki
- Quality scoring and cultural context assessment
- 5000 entries per language target

## Dataset Structure

Each entry will contain:
- `id`: Unique identifier
- `clue`: Crossword clue text
- `answer`: Single-word answer (uppercase)
- `source`: Original data source (MILU, IndicWikiBio, IndoWordNet, Bhasha-Wiki)
- `source_id`: Original ID in source dataset
- `context_score`: Indian cultural context score (0.0-1.0)
- `quality_score`: Crossword suitability score (0.0-1.0)

Automatically generated crossword benchmark dataset."""

    def _generate_dynamic_readme_content(
        self, repo_id: str, all_configs: List[str], active_language: str = None
    ) -> str:
        """
        Generate unified README content with real statistics for all languages.

        Args:
            repo_id: Repository ID
            all_configs: List of all language configurations
            active_language: Language currently being updated (gets fresh stats, others use cache)
        """
        content_lines = [
            "# Buddhi-Pragati Crossword Dataset",
            "",
            "This dataset contains crossword clue-answer pairs extracted from multiple Indian language sources.",
            "",
            "## Available Languages",
            "",
        ]

        total_entries = 0
        all_sources = set()

        for config in sorted(all_configs):
            # Use intelligent caching - force refresh only for active language
            force_refresh = active_language and config == active_language.lower()
            stats = self._get_configuration_stats(
                repo_id, config, force_refresh=force_refresh
            )
            total_entries += stats["total_entries"]
            all_sources.update(stats["sources"])

            content_lines.extend(
                [
                    f"### {config.title()}",
                    f"- **Total Entries**: {stats['total_entries']}",
                    f"- **Sources**: {', '.join(stats['sources'])} \n",
                    "Quality Scores:",
                    f"- **Quality Score Range**: {stats['min_quality']:.2f} - {stats['max_quality']:.2f}",
                    f"- **Average Quality Score**: {stats['mean_quality']:.3f}",
                    f"- **Median Quality Score**: {stats['median_quality']:.3f}\n",
                    "Indian Cultural Context Scores:",
                    f"- **Context Score Range**: {stats['min_context']:.2f} - {stats['max_context']:.2f}",
                    f"- **Average Context Score**: {stats['mean_context']:.3f}",
                    f"- **Median Context Score**: {stats['median_context']:.3f}",
                    "",
                ]
            )

        content_lines.extend(
            [
                "## Dataset Structure",
                "",
                "Each entry contains:",
                "- `id`: Unique identifier",
                "- `clue`: Crossword clue text",
                "- `answer`: Single-word answer (uppercase)",
                "- `source`: Original data source (MILU, IndicWikiBio, IndoWordNet, Bhasha-Wiki)",
                "- `source_id`: Original ID in source dataset",
                "- `context_score`: Indian cultural context score (0.0-1.0)",
                "- `quality_score`: Crossword suitability score (0.0-1.0)",
                "",
                "## Usage",
                "",
                "```python",
                "from datasets import load_dataset",
                "",
                "# Load specific language",
            ]
        )

        for config in sorted(all_configs):
            content_lines.append(f'dataset = load_dataset("{repo_id}", "{config}")')

        content_lines.extend(
            [
                "```",
                "",
                "## Summary Statistics",
                "",
                f"- **Total Languages**: {len(all_configs)}",
                f"- **Total Entries**: {total_entries:,}",
                f"- **All Sources**: {', '.join(sorted(all_sources))}",
                "",
            ]
        )

        return "\n".join(content_lines)

    def _update_dataset_card_incremental(
        self, language: str, repo_id: str, force_full_stats: bool = False
    ):
        """
        Update dataset README.md with current real statistics after incremental upload.

        Args:
            language: Language that was just uploaded
            repo_id: Repository ID
            force_full_stats: If True, collect fresh statistics for all languages.
                            If False, only refresh stats for the uploaded language.
        """
        try:
            if force_full_stats:
                logger.debug(
                    f"Updating dataset card for {repo_id} with full statistics refresh"
                )
            else:
                logger.debug(
                    f"Updating dataset card for {repo_id} after {language} upload (intelligent caching)"
                )

            # Scan all configurations in repository
            all_configs = self._scan_repository_configurations(repo_id)

            if not all_configs:
                logger.info(
                    "No configurations found in repository, updating README to show clean state"
                )
                # Generate clean repository README
                yaml_header = self._generate_clean_yaml_header()
                readme_content = self._generate_clean_readme_content()
                full_readme = yaml_header + "\n" + readme_content
            else:
                # Determine active language for intelligent caching
                active_language = language if not force_full_stats else None

                # Generate dynamic YAML header with intelligent caching
                yaml_header = self._generate_dynamic_yaml_header(
                    repo_id, all_configs, active_language=active_language
                )

                # Generate unified README content with intelligent caching
                readme_content = self._generate_dynamic_readme_content(
                    repo_id, all_configs, active_language=active_language
                )

                # Combine YAML header and content
                full_readme = yaml_header + "\n" + readme_content

            # Update dataset card
            from huggingface_hub import DatasetCard

            card = DatasetCard(full_readme)
            card.push_to_hub(repo_id, token=self.hf_token)

            if all_configs:
                stats_type = (
                    "full statistics"
                    if force_full_stats
                    else f"intelligent caching (refreshed {language})"
                )
                logger.info(
                    f"✅ Updated dataset card for {repo_id} with {len(all_configs)} language configurations ({stats_type})"
                )
            else:
                logger.info(
                    f"✅ Updated dataset card for {repo_id} to show clean repository state"
                )

        except Exception as e:
            logger.error(f"Failed to update dataset card: {e}")
            # Don't raise - this shouldn't block the upload process

    def _validate_dataset_entries(
        self, entries: List[DatasetEntry]
    ) -> List[DatasetEntry]:
        """Validate DatasetEntry types and values before upload to prevent type conversion errors."""
        valid_entries = []
        validation_errors = 0

        for i, entry in enumerate(entries):
            try:
                # Validate and ensure correct types
                validated_entry = DatasetEntry(
                    id=str(entry.id),  # Ensure string
                    clue=str(entry.clue),  # Ensure string
                    answer=str(entry.answer),  # Ensure string
                    source=str(entry.source),  # Ensure string
                    source_id=str(
                        entry.source_id
                    ),  # Ensure string (fixes type conversion issues)
                    context_score=float(entry.context_score),  # Ensure float
                    quality_score=float(entry.quality_score),  # Ensure float
                )

                # Additional validation checks
                if (
                    not validated_entry.id
                    or not validated_entry.clue
                    or not validated_entry.answer
                ):
                    logger.debug(f"Entry {i} has empty required fields, skipping")
                    validation_errors += 1
                    continue

                if (
                    validated_entry.context_score < 0.0
                    or validated_entry.context_score > 1.0
                ):
                    logger.debug(
                        f"Entry {i} has invalid context_score: {validated_entry.context_score}, clamping to [0,1]"
                    )
                    validated_entry.context_score = max(
                        0.0, min(1.0, validated_entry.context_score)
                    )

                if (
                    validated_entry.quality_score < 0.0
                    or validated_entry.quality_score > 1.0
                ):
                    logger.debug(
                        f"Entry {i} has invalid quality_score: {validated_entry.quality_score}, clamping to [0,1]"
                    )
                    validated_entry.quality_score = max(
                        0.0, min(1.0, validated_entry.quality_score)
                    )

                valid_entries.append(validated_entry)

            except (ValueError, TypeError) as e:
                logger.debug(f"Entry {i} validation failed: {e}, skipping")
                validation_errors += 1
                continue

        if validation_errors > 0:
            logger.info(
                f"Validation complete: {len(valid_entries)}/{len(entries)} entries valid, {validation_errors} entries rejected"
            )
        else:
            logger.debug(f"Validation complete: all {len(entries)} entries valid")

        return valid_entries

    def upload_to_huggingface(self, entries: List[DatasetEntry], language: str):
        """Upload processed entries to HuggingFace dataset."""
        if not self.hf_api or not self.hf_token:
            logger.warning("No HuggingFace token provided, skipping upload")
            return None

        repo_id = self.dataset_config.get(
            "HF_DATASET_REPO", "selim-b-kh/Buddhi_pragati"
        )

        logger.info(f"Uploading {len(entries)} entries to {repo_id}")

        try:
            # Validate entries before upload to prevent type conversion errors
            validated_entries = self._validate_dataset_entries(entries)
            if not validated_entries:
                logger.warning(
                    f"No valid entries after validation for {language}, skipping upload"
                )
                return None

            if len(validated_entries) != len(entries):
                logger.info(
                    f"Validation filtered {len(entries) - len(validated_entries)} entries for {language}"
                )

            # Ensure repository exists
            from huggingface_hub import create_repo

            try:
                create_repo(
                    repo_id,
                    repo_type="dataset",
                    token=self.hf_token,
                    exist_ok=True,
                    private=False,
                )
                logger.info(f"Ensured repository {repo_id} exists")
            except Exception as repo_error:
                logger.warning(f"Repository creation/check failed: {repo_error}")

            # Convert validated entries to pandas DataFrame - no local merging needed
            new_data_dicts = [asdict(entry) for entry in validated_entries]
            new_df = pd.DataFrame(new_data_dicts)

            # Create HF Dataset from new entries only
            final_dataset = Dataset.from_pandas(new_df)
            config_name = language.lower()

            logger.info(
                f"Uploading {len(validated_entries)} entries as complete config replacement to HF"
            )

            # HuggingFace performs complete config replacement (not merging)
            # This replaces the entire language configuration with new data
            final_dataset.push_to_hub(
                repo_id,
                config_name=config_name,
                split="train",
                token=self.hf_token,
                # This completely replaces the config, no merging occurs
            )

            # Clear any stale caches before updating metadata
            self._clear_dataset_cache(repo_id, config_name)

            # Update dataset card with proper YAML metadata
            self._update_dataset_card_incremental(language, repo_id)

            logger.info(f"Successfully uploaded {language} dataset to {repo_id}")

            # Explicit memory cleanup after upload
            self._cleanup_after_upload()

            return repo_id

        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace: {e}")
            import traceback

            logger.error(f"Upload error details: {traceback.format_exc()}")
            # Cleanup even on failure to free memory
            self._cleanup_after_upload()
            return None
