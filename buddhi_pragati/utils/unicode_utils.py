"""
Unicode utility functions for Buddhi-Pragati crossword benchmark.

This module provides Unicode-aware text processing functions.
"""

import unicodedata
import re
from typing import Optional


def is_alphabetic_unicode(text: str) -> bool:
    """
    Check if text contains only Unicode letters (any script).

    This is a Unicode-aware replacement for str.isalpha() that works
    with all scripts including Devanagari, Bengali, Tamil, Telugu, etc.

    For Indic scripts, this includes combining marks (vowel signs, etc.)
    which are essential parts of words but have categories Mc/Mn instead of L.

    Args:
        text: Text to check

    Returns:
        True if text contains only Unicode letters and combining marks, False otherwise
    """
    if not text:
        return False

    # Check each character is either a Unicode letter or combining mark
    for char in text:
        category = unicodedata.category(char)
        # Accept:
        # - L* (Letter categories: Lu, Ll, Lt, Lm, Lo)
        # - Mc (Mark, spacing combining - vowel signs in Indic scripts)
        # - Mn (Mark, nonspacing - diacritics, anusvara, etc.)
        if not (category.startswith("L") or category in ("Mc", "Mn")):
            return False

    return True


def clean_unicode_text(text: Optional[str]) -> Optional[str]:
    """
    Clean and normalize Unicode text for crossword processing.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text or None if input is invalid
    """
    if not text or not isinstance(text, str):
        return None

    # Normalize Unicode to NFC form (canonical composition)
    # This ensures consistent representation of accented characters
    cleaned = unicodedata.normalize("NFC", text.strip())

    # Remove any leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned if cleaned else None


def get_script_name(text: str) -> str:
    """
    Get the primary script name for a text string.

    Args:
        text: Text to analyze

    Returns:
        Script name (e.g., 'Latin', 'Devanagari', 'Bengali', etc.)
    """
    if not text:
        return "Unknown"

    # Get script of first letter character
    for char in text:
        if unicodedata.category(char).startswith("L"):
            return (
                unicodedata.name(char).split()[0]
                if unicodedata.name(char, None)
                else "Unknown"
            )

    return "Unknown"


def contains_latin_script(text: str) -> bool:
    """
    Check if text contains any Latin script characters.

    This is used to detect entries that have Latin script entities
    embedded in Indic script texts (like scientific names).

    Args:
        text: Text to check

    Returns:
        True if text contains any Latin letters, False otherwise
    """
    if not text:
        return False

    for char in text:
        if unicodedata.category(char).startswith("L"):
            char_script = (
                unicodedata.name(char, "").split()[0]
                if unicodedata.name(char, None)
                else ""
            )
            if char_script == "LATIN":
                return True

    return False


def is_predominantly_latin_script(text: str) -> bool:
    """
    Check if text is predominantly in Latin script.

    This is used to filter out answers that are in Latin script
    when we expect Indic script answers.

    Args:
        text: Text to check

    Returns:
        True if more than 50% of letter characters are Latin script
    """
    if not text:
        return False

    total_letters = 0
    latin_letters = 0

    for char in text:
        if unicodedata.category(char).startswith("L"):
            total_letters += 1
            char_script = (
                unicodedata.name(char, "").split()[0]
                if unicodedata.name(char, None)
                else ""
            )
            if char_script == "LATIN":
                latin_letters += 1

    if total_letters == 0:
        return False

    return (latin_letters / total_letters) > 0.5


def extract_first_sentence(text: str) -> str:
    """
    Extract the first sentence from text.

    This function handles multiple languages and uses Unicode-aware
    sentence boundary detection.

    Args:
        text: Input text

    Returns:
        First sentence from the text, or original text if no sentence boundary found
    """
    if not text:
        return ""

    text = text.strip()

    # Common sentence ending punctuation across languages
    sentence_endings = r"[।॥।।؟؍।|.!?]"

    # Find first sentence ending
    match = re.search(sentence_endings, text)
    if match:
        first_sentence = text[: match.end()].strip()
        # Make sure we got a meaningful sentence (at least 10 characters)
        if len(first_sentence) >= 10:
            return first_sentence

    # If no sentence boundary found or sentence too short, return original
    return text


def has_multiple_words(text: str) -> bool:
    """
    Check if text contains multiple words.

    This is Unicode-aware and handles different scripts properly.

    Args:
        text: Text to check

    Returns:
        True if text contains multiple words separated by spaces
    """
    if not text:
        return False

    # Split on whitespace and filter out empty strings
    words = [word.strip() for word in text.split() if word.strip()]
    return len(words) > 1


def answer_in_clue_check(clue: str, answer: str) -> bool:
    """
    Check if answer appears in clue (case-insensitive).

    This is the universal sanity check to ensure no answer leakage.

    Args:
        clue: The clue text
        answer: The answer text

    Returns:
        True if answer is found in clue, False otherwise
    """
    if not clue or not answer:
        return False

    # Case-insensitive check
    return answer.lower() in clue.lower()
