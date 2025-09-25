"""
Local Backup Manager for Dataset Creation

This module provides functionality for saving dataset entry batches locally during processing
and loading them back for final HuggingFace upload. This enables memory-efficient processing
while maintaining the ability to upload complete language configurations.

Key Features:
- Batch-level disk persistence during processing
- Language-based organization of backup files
- Automatic deduplication when loading accumulated entries
- Cleanup functionality for post-upload maintenance

Architecture:
- Backup files stored in: configs_backup/{language}/batch_{source}_{batch_num}.json
- JSON serialization using DatasetEntry dataclass fields
- Answer-based deduplication across all batches for a language
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict
from .data_structure import DatasetEntry

logger = logging.getLogger(__name__)


class LocalBackupManager:
    """
    Manages local backup files for dataset entries during batch processing.

    This class handles the disk persistence of dataset entries as they are processed
    in batches, allowing for memory-efficient processing and single-upload behavior
    to HuggingFace datasets.
    """

    def __init__(self, backup_dir: str = None):
        """
        Initialize backup manager with specified directory.

        Args:
            backup_dir: Directory for backup storage. Defaults to buddhi_pragati/data/configs_backup
        """
        if backup_dir is None:
            # Default to configs_backup in the data directory
            backup_dir = Path(__file__).parent / "configs_backup"
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"LocalBackupManager initialized with backup directory: {self.backup_dir}"
        )

    def save_batch(
        self, language: str, source: str, batch_num: int, entries: List[DatasetEntry]
    ) -> str:
        """
        Save a batch of dataset entries to disk.

        Args:
            language: Target language (used as subdirectory)
            source: Source name (MILU, IndicWikiBio, etc.)
            batch_num: Batch number for this source
            entries: List of DatasetEntry objects to save

        Returns:
            Path to the saved batch file
        """
        if not entries:
            logger.debug(
                f"No entries to save for {language} {source} batch {batch_num}"
            )
            return None

        # Create language subdirectory
        language_dir = self.backup_dir / language.lower()
        language_dir.mkdir(parents=True, exist_ok=True)

        # Generate batch filename
        batch_filename = f"batch_{source}_{batch_num:03d}.json"
        batch_file = language_dir / batch_filename

        # Convert entries to JSON-serializable format
        try:
            entries_data = [asdict(entry) for entry in entries]

            # Save to JSON file
            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump(entries_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(entries)} entries to {batch_file}")
            return str(batch_file)

        except Exception as e:
            logger.error(f"Failed to save batch {batch_filename} for {language}: {e}")
            return None

    def load_all_entries(self, language: str) -> List[DatasetEntry]:
        """
        Load and merge all batch files for a language with deduplication.

        Args:
            language: Target language to load entries for

        Returns:
            List of deduplicated DatasetEntry objects from all batches
        """
        language_dir = self.backup_dir / language.lower()

        if not language_dir.exists():
            logger.warning(f"No backup directory found for language: {language}")
            return []

        # Find all batch files
        batch_files = list(language_dir.glob("batch_*.json"))
        if not batch_files:
            logger.warning(f"No batch files found for language: {language}")
            return []

        logger.info(f"Loading {len(batch_files)} batch files for {language}")

        all_entries = []
        seen_answers = set()  # For deduplication
        total_loaded = 0
        duplicates_removed = 0

        for batch_file in sorted(batch_files):  # Sort for consistent ordering
            try:
                with open(batch_file, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)

                batch_entries = []
                for entry_data in batch_data:
                    # Reconstruct DatasetEntry from dict
                    entry = DatasetEntry(**entry_data)

                    # Apply deduplication based on answer (case-insensitive)
                    answer_key = entry.answer.upper()
                    if answer_key not in seen_answers:
                        seen_answers.add(answer_key)
                        batch_entries.append(entry)
                    else:
                        duplicates_removed += 1

                all_entries.extend(batch_entries)
                total_loaded += len(batch_data)

                logger.debug(
                    f"Loaded {len(batch_entries)}/{len(batch_data)} entries from {batch_file.name}"
                )

            except Exception as e:
                logger.error(f"Failed to load batch file {batch_file}: {e}")
                continue

        final_count = len(all_entries)
        logger.info(
            f"Loaded {final_count} unique entries for {language} "
            f"(processed {total_loaded} total, removed {duplicates_removed} duplicates)"
        )

        return all_entries

    def cleanup_language(self, language: str) -> bool:
        """
        Delete all backup files for a language after successful upload.

        Args:
            language: Language to clean up backup files for

        Returns:
            True if cleanup successful, False otherwise
        """
        language_dir = self.backup_dir / language.lower()

        if not language_dir.exists():
            logger.debug(f"No backup directory to clean for language: {language}")
            return True

        try:
            # Count files before deletion
            batch_files = list(language_dir.glob("batch_*.json"))
            file_count = len(batch_files)

            # Remove entire language directory
            shutil.rmtree(language_dir, ignore_errors=True)

            logger.info(
                f"Cleaned up {file_count} backup files for language: {language}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup backup files for {language}: {e}")
            return False

    def list_backup_languages(self) -> List[str]:
        """
        List all languages that have backup files.

        Returns:
            List of language names with backup files
        """
        if not self.backup_dir.exists():
            return []

        languages = []
        for item in self.backup_dir.iterdir():
            if item.is_dir() and list(item.glob("batch_*.json")):
                languages.append(item.name.title())  # Capitalize language name

        return sorted(languages)

    def get_backup_stats(self, language: str) -> Dict[str, int]:
        """
        Get statistics about backup files for a language.

        Args:
            language: Language to get statistics for

        Returns:
            Dictionary with batch count, total entries, and file size info
        """
        language_dir = self.backup_dir / language.lower()

        if not language_dir.exists():
            return {"batch_count": 0, "total_entries": 0, "total_size_mb": 0.0}

        batch_files = list(language_dir.glob("batch_*.json"))
        total_entries = 0
        total_size = 0

        for batch_file in batch_files:
            try:
                with open(batch_file, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)
                total_entries += len(batch_data)
                total_size += batch_file.stat().st_size
            except Exception as e:
                logger.debug(f"Could not read {batch_file}: {e}")
                continue

        return {
            "batch_count": len(batch_files),
            "total_entries": total_entries,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
