"""
Simplified comprehensive test suite for buddhi_pragati.data module.
Tests core functionality without complex mocking scenarios.
"""

import pytest
import tempfile
import os
from dataclasses import asdict

# Import modules to test
from buddhi_pragati.data.data_structure import DatasetEntry
from buddhi_pragati.data.local_backup_manager import LocalBackupManager
from buddhi_pragati.utils.unicode_utils import (
    contains_latin_script,
    is_predominantly_latin_script,
    extract_first_sentence,
    has_multiple_words,
    answer_in_clue_check,
)


class TestDatasetEntry:
    """Test DatasetEntry dataclass functionality."""

    def test_dataset_entry_creation(self):
        """Test basic DatasetEntry creation with required fields."""
        entry = DatasetEntry(
            id="test_001",
            clue="What is the capital of India?",
            answer="DELHI",
            source="MILU",
            source_id="milu_english_1_001",
            context_score=0.8,
            quality_score=0.7,
        )

        assert entry.id == "test_001"
        assert entry.clue == "What is the capital of India?"
        assert entry.answer == "DELHI"
        assert entry.source == "MILU"
        assert entry.source_id == "milu_english_1_001"
        assert entry.context_score == 0.8
        assert entry.quality_score == 0.7

    def test_dataset_entry_score_validation(self):
        """Test that scores are within valid ranges."""
        entry = DatasetEntry(
            id="test_002",
            clue="Test clue",
            answer="ANSWER",
            source="TEST",
            source_id="test_001",
            context_score=0.5,
            quality_score=0.9,
        )

        # Scores should be between 0.0 and 1.0
        assert 0.0 <= entry.context_score <= 1.0
        assert 0.0 <= entry.quality_score <= 1.0

    def test_dataset_entry_serialization(self):
        """Test JSON serialization with asdict()."""
        entry = DatasetEntry(
            id="test_003",
            clue="Test serialization",
            answer="SERIAL",
            source="TEST",
            source_id="test_001",
            context_score=0.6,
            quality_score=0.8,
        )

        entry_dict = asdict(entry)

        assert isinstance(entry_dict, dict)
        assert entry_dict["id"] == "test_003"
        assert entry_dict["clue"] == "Test serialization"
        assert entry_dict["answer"] == "SERIAL"
        assert entry_dict["source"] == "TEST"
        assert entry_dict["source_id"] == "test_001"
        assert entry_dict["context_score"] == 0.6
        assert entry_dict["quality_score"] == 0.8


class TestLocalBackupManager:
    """Test local backup functionality."""

    def test_backup_manager_initialization(self):
        """Test LocalBackupManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalBackupManager(temp_dir)
            assert str(manager.backup_dir) == temp_dir
            assert os.path.exists(temp_dir)

    def test_batch_persistence(self):
        """Test JSON serialization/deserialization of DatasetEntry objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalBackupManager(temp_dir)

            # Create test entries
            entries = [
                DatasetEntry(
                    id="test_001",
                    clue="Test clue 1",
                    answer="ANSWER1",
                    source="TEST",
                    source_id="test_001",
                    context_score=0.8,
                    quality_score=0.7,
                ),
                DatasetEntry(
                    id="test_002",
                    clue="Test clue 2",
                    answer="ANSWER2",
                    source="TEST",
                    source_id="test_002",
                    context_score=0.6,
                    quality_score=0.9,
                ),
            ]

            # Save and load entries
            save_result = manager.save_batch("English", "TEST", 1, entries)
            assert save_result is not None  # Should return file path

            loaded_entries = manager.load_all_entries("English")

            assert len(loaded_entries) == 2
            assert loaded_entries[0].id == "test_001"
            assert loaded_entries[1].id == "test_002"

    def test_deduplication_during_loading(self):
        """Test answer-based duplicate removal during loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalBackupManager(temp_dir)

            # Create entries with duplicate answers
            entries = [
                DatasetEntry(
                    id="test_001",
                    clue="First clue",
                    answer="DUPLICATE",
                    source="TEST1",
                    source_id="test_001",
                    context_score=0.8,
                    quality_score=0.7,
                ),
                DatasetEntry(
                    id="test_002",
                    clue="Second clue",
                    answer="DUPLICATE",  # Same answer
                    source="TEST2",
                    source_id="test_002",
                    context_score=0.6,
                    quality_score=0.9,
                ),
            ]

            manager.save_batch("English", "TEST", 1, entries)
            loaded_entries = manager.load_all_entries("English")

            # Should have only one entry after deduplication
            assert len(loaded_entries) == 1

    def test_backup_stats(self):
        """Test backup statistics functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalBackupManager(temp_dir)

            # Test stats for non-existent language
            stats = manager.get_backup_stats("NonExistent")
            assert stats["batch_count"] == 0
            assert stats["total_entries"] == 0
            assert stats["total_size_mb"] == 0.0

            # Create and save some entries
            entries = [
                DatasetEntry(
                    id="test_001",
                    clue="Test clue",
                    answer="ANSWER",
                    source="TEST",
                    source_id="test_001",
                    context_score=0.5,
                    quality_score=0.5,
                )
            ]

            manager.save_batch("English", "TEST", 1, entries)
            stats = manager.get_backup_stats("English")

            assert stats["batch_count"] == 1
            assert stats["total_entries"] == 1
            assert stats["total_size_mb"] >= 0.0  # File exists, even if very small

    def test_language_listing(self):
        """Test listing available backup languages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalBackupManager(temp_dir)

            # Initially no languages
            languages = manager.list_backup_languages()
            assert len(languages) == 0

            # Add some entries
            entries = [
                DatasetEntry(
                    id="test_001",
                    clue="Test clue",
                    answer="ANSWER",
                    source="TEST",
                    source_id="test_001",
                    context_score=0.5,
                    quality_score=0.5,
                )
            ]

            manager.save_batch("English", "TEST", 1, entries)
            manager.save_batch("Hindi", "TEST", 1, entries)

            languages = manager.list_backup_languages()
            assert len(languages) == 2
            assert "English" in languages
            assert "Hindi" in languages

    def test_cleanup_functionality(self):
        """Test cleanup operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalBackupManager(temp_dir)

            # Create some entries
            entries = [
                DatasetEntry(
                    id="test_001",
                    clue="Test clue",
                    answer="ANSWER",
                    source="TEST",
                    source_id="test_001",
                    context_score=0.5,
                    quality_score=0.5,
                )
            ]

            manager.save_batch("English", "TEST", 1, entries)

            # Verify entries exist
            loaded_entries = manager.load_all_entries("English")
            assert len(loaded_entries) == 1

            # Cleanup
            cleanup_result = manager.cleanup_language("English")
            assert cleanup_result is True

            # Verify entries are gone
            loaded_entries = manager.load_all_entries("English")
            assert len(loaded_entries) == 0

    def test_error_resilience(self):
        """Test graceful handling of errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalBackupManager(temp_dir)

            # Test loading from non-existent language
            empty_entries = manager.load_all_entries("NonexistentLanguage")
            assert empty_entries == []

            # Test cleanup of non-existent language
            cleanup_result = manager.cleanup_language("NonexistentLanguage")
            assert cleanup_result is True  # Should not raise errors


class TestUnicodeUtils:
    """Test Unicode utility functions."""

    def test_script_detection(self):
        """Test Latin vs Indic script identification."""
        # Test Latin script
        assert contains_latin_script("Hello World")
        assert contains_latin_script("Delhi is the capital")
        assert is_predominantly_latin_script("This is English text")

        # Test Indic scripts
        assert not is_predominantly_latin_script("यह हिंदी पाठ है")  # This is Hindi text
        assert not is_predominantly_latin_script(
            "এটি বাংলা টেক্সট"
        )  # This is Bengali text

        # Test mixed content
        mixed_text = "Hello यह mixed है text"
        assert contains_latin_script(mixed_text)

    def test_answer_in_clue_validation(self):
        """Test universal leakage prevention across all sources."""
        # Test cases where answer appears in clue (should return True - answer found in clue)
        assert answer_in_clue_check("The capital Delhi is in India", "DELHI")
        assert answer_in_clue_check("Gandhi was a great leader", "GANDHI")

        # Test cases where answer doesn't appear in clue (should return False - no leakage)
        assert not answer_in_clue_check("What is the capital of India?", "DELHI")
        assert not answer_in_clue_check("Who led the independence movement?", "GANDHI")

        # Test case-insensitive matching
        assert answer_in_clue_check("The gandhi memorial", "GANDHI")
        assert answer_in_clue_check("Delhi tourism", "delhi")

    def test_sentence_extraction(self):
        """Test boundary detection across multiple languages."""
        # Test English sentence extraction
        english_text = "This is the first sentence. This is the second sentence."
        first_sentence = extract_first_sentence(english_text)
        assert first_sentence == "This is the first sentence."

        # Test text with no sentence boundary
        no_boundary = "This is just one sentence without ending punctuation"
        result = extract_first_sentence(no_boundary)
        assert result == no_boundary  # Should return original text

        # Test empty text
        empty_result = extract_first_sentence("")
        assert empty_result == ""

    def test_multi_word_detection(self):
        """Test enhanced validation for crossword suitability."""
        # Single words should return False (not multiple words)
        assert not has_multiple_words("DELHI")
        assert not has_multiple_words("GANDHI")
        assert not has_multiple_words("गांधी")

        # Multiple words should return True
        assert has_multiple_words("NEW DELHI")
        assert has_multiple_words("MAHATMA GANDHI")
        assert has_multiple_words("महात्मा गांधी")

        # Edge cases
        assert not has_multiple_words("")
        assert not has_multiple_words("   ")  # Only spaces
        assert not has_multiple_words("WORD   ")  # Trailing spaces


if __name__ == "__main__":
    pytest.main([__file__])
