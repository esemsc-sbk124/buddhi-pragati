#!/usr/bin/env python3
"""
Enhanced Indian Cultural Corpus Generation Script

Creates a comprehensive multilingual corpus of Indian cultural concepts using IndoWordNet's
semantic relationships. Expands from seed concepts through synonyms, hyponyms, and meronyms
across 18 supported Indian languages to build a rich cultural reference corpus.

This script generates an expanded cultural corpus that replaces the limited manual corpus
in indian_context_scorer.py with a systematically derived multilingual corpus covering
hundreds of culturally-specific Indian terms across multiple semantic categories.

Usage:
    python create_corpus.py [--output-file indian_corpus.json] [--max-depth 2] [--verbose]

Architecture:
- Seed-based expansion from manually curated Indian concepts
- IndoWordNet semantic traversal with depth limits
- Multilingual synset linking for comprehensive coverage
- Category-organized output matching existing scorer format
- JSON serialization for easy integration with scoring system

Categories Generated:
- festivals_rituals: Religious and cultural celebrations
- food_cuisine: Traditional Indian foods and beverages
- clothing_textiles: Traditional garments and fabrics
- philosophy_concepts: Spiritual and philosophical terms
- mythology_epics: Mythological figures and epic literature
- people_relations: Social roles and relationships
- flora_fauna: Native plants and animals with cultural significance
- arts_music: Traditional arts, music, and performance
- geography: Indian places and geographical features
- history: Historical figures and events
- modern_india: Contemporary Indian institutions and concepts
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_indowordnet():
    """
    Load IndoWordNet instances for all supported languages.

    Returns:
        Dictionary mapping language names to IndoWordNet instances, or None if loading fails
    """
    try:
        from pyiwn import IndoWordNet, Language

        logger.info("Loading IndoWordNet for all supported languages...")
        iwn_instances = {}

        # Get all available languages
        available_languages = [lang.value for lang in Language]
        logger.info(f"Available languages: {available_languages}")

        # Load an instance for each language
        for lang in available_languages:
            try:
                logger.info(f"Loading {lang} language synsets...")
                iwn_instances[lang] = IndoWordNet(lang=Language[lang.upper()])
                logger.info(f"Successfully loaded {lang}")
            except Exception as e:
                logger.warning(f"Failed to load {lang}: {e}")
                continue

        if iwn_instances:
            logger.info(
                f"IndoWordNet loaded successfully for {len(iwn_instances)} languages"
            )
            return iwn_instances
        else:
            logger.error("Failed to load any IndoWordNet languages")
            return None

    except ImportError:
        logger.error("pyiwn library not found. Install with: pip install pyiwn")
        return None
    except Exception as e:
        logger.error(f"Failed to load IndoWordNet: {e}")
        return None


def create_seed_corpus() -> Dict[str, List[str]]:
    """
    Load comprehensive seed corpus from JSON file and return it directly.

    Returns the complete JSON content from the seed_corpus.json file for use
    in corpus expansion. This replaces the previous flattening logic to allow
    direct access to the structured multilingual corpus.

    Returns:
        Dictionary containing the complete seed corpus structure as loaded from JSON
    """
    try:
        # Load comprehensive seed corpus from JSON file
        corpus_file = Path(__file__).parent / "seed_corpus.json"

        if corpus_file.exists():
            logger.info(f"Loading comprehensive seed corpus from: {corpus_file}")
            with open(corpus_file, "r", encoding="utf-8") as f:
                seed_corpus = json.load(f)

            logger.info(f"Loaded seed corpus with {len(seed_corpus)} categories")
            return seed_corpus

    except Exception as e:
        logger.warning(f"Failed to load seed corpus file: {e}")

    # Fallback to basic corpus if file loading fails
    logger.info("Using fallback basic seed corpus")
    return {
        "india_multilingual": [
            "India",
            "Indian",
            "Bharat",
            "Hindustan",
            "Republic of India",
            "भारत",
            "इंडिया",
            "हिंदुस्तान",
            "भारतीय",
            "हिन्दुस्तान",
            "ভারত",
            "ইন্ডিয়া",
            "ভারতীয়",
            "হিন্দুস্তান",
        ],
        "festivals_rituals": [
            "Diwali",
            "Holi",
            "Eid",
            "Onam",
            "Pongal",
            "Durga Puja",
            "Ganesh Chaturthi",
        ],
        "food_cuisine": ["Curry", "Biryani", "Dosa", "Samosa", "Chai", "Roti", "Naan"],
        "geography": ["Himalayas", "Ganga", "Kerala", "Rajasthan", "Mumbai", "Delhi"],
    }


def expand_concepts_with_indowordnet(
    seed_corpus: Dict[str, any], iwn_instances: Dict[str, any], max_depth: int = 2
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Expand seed concepts using IndoWordNet semantic relationships per language.

    For each language-specific category in the seed corpus, uses the corresponding
    IndoWordNet instance to find synsets and traverse hyponyms/meronyms up to max_depth
    levels. Maintains the structured format: dict(category: dict(language: concepts)).

    Args:
        seed_corpus: Dictionary of seed concepts with structure {category: {language: [concepts]}}
        iwn_instances: Dictionary mapping language names to IndoWordNet instances
        max_depth: Maximum depth for semantic traversal

    Returns:
        Dictionary with structure {category: {language: expanded_concept_sets}}
    """
    expanded_corpus = defaultdict(lambda: defaultdict(set))

    logger.info(f"Expanding corpus with max depth {max_depth}")
    logger.info(
        f"Using {len(iwn_instances)} IndoWordNet language instances: {list(iwn_instances.keys())}"
    )

    # Map IndoWordNet language names to our language names
    iwn_lang_mapping = {
        "hindi": "Hindi",
        "bengali": "Bengali",
        "tamil": "Tamil",
        "telugu": "Telugu",
        "gujarati": "Gujarati",
        "kannada": "Kannada",
        "malayalam": "Malayalam",
        "marathi": "Marathi",
        "punjabi": "Punjabi",
        "odia": "Odia",
        "assamese": "Assamese",
        "sanskrit": "Sanskrit",
        "nepali": "Nepali",
        "urdu": "Urdu",
        "english": "English",
    }

    for category, content in seed_corpus.items():
        logger.info(f"Processing category: {category}")

        if isinstance(content, dict):
            # Language-specific category
            for language, concepts in content.items():
                if not isinstance(concepts, list):
                    continue

                logger.info(
                    f"Processing {category}.{language} with {len(concepts)} seed concepts"
                )
                language_concepts = set()

                # Find matching IndoWordNet instance for this language
                iwn_instance = None
                iwn_lang_name = None

                # Try direct match first
                for iwn_lang, iwn_inst in iwn_instances.items():
                    if iwn_lang.lower() == language.lower():
                        iwn_instance = iwn_inst
                        iwn_lang_name = iwn_lang
                        break

                # Try mapping if no direct match
                if iwn_instance is None:
                    for iwn_lang, mapped_lang in iwn_lang_mapping.items():
                        if (
                            mapped_lang.lower() == language.lower()
                            and iwn_lang in iwn_instances
                        ):
                            iwn_instance = iwn_instances[iwn_lang]
                            iwn_lang_name = iwn_lang
                            break

                if iwn_instance is None:
                    logger.warning(
                        f"No IndoWordNet instance found for language: {language}"
                    )
                    # Just add original concepts without expansion
                    for concept in concepts:
                        language_concepts.add(concept)
                    expanded_corpus[category][language] = language_concepts
                    continue

                logger.info(
                    f"Using IndoWordNet instance '{iwn_lang_name}' for language '{language}'"
                )

                # Expand concepts using the language-specific IndoWordNet instance
                for concept in concepts:
                    # Add original concept
                    language_concepts.add(concept)

                    # Extract individual words from concept phrases
                    words = concept.lower().split()

                    for word in words:
                        # Skip common English words
                        if word in {
                            "and",
                            "or",
                            "the",
                            "a",
                            "an",
                            "is",
                            "are",
                            "with",
                            "of",
                            "in",
                            "on",
                            "at",
                            "to",
                            "for",
                            "by",
                            "from",
                        }:
                            continue

                        # Use only the language-specific IndoWordNet instance
                        try:
                            synsets = iwn_instance.synsets(word)

                            for synset in synsets[
                                :3
                            ]:  # Limit to first 3 synsets per word
                                # Add synonyms from synset
                                for lemma in synset.lemma_names():
                                    if (
                                        lemma and len(lemma) > 2
                                    ):  # Valid non-trivial lemmas
                                        language_concepts.add(lemma)

                                # Expand through hyponyms (is-a-kind-of relationships)
                                hyponyms = synset.hyponyms()
                                for hyponym in hyponyms[:5]:  # Limit hyponyms
                                    for lemma in hyponym.lemma_names():
                                        if lemma and len(lemma) > 2:
                                            language_concepts.add(lemma)

                                    # Second level expansion
                                    if max_depth >= 2:
                                        second_level = hyponym.hyponyms()
                                        for second_hyp in second_level[:3]:
                                            for lemma in second_hyp.lemma_names():
                                                if lemma and len(lemma) > 2:
                                                    language_concepts.add(lemma)

                                # Expand through meronyms (is-a-part-of relationships)
                                meronyms = synset.meronyms()
                                for meronym in meronyms[:3]:  # Limit meronyms
                                    for lemma in meronym.lemma_names():
                                        if lemma and len(lemma) > 2:
                                            language_concepts.add(lemma)

                        except Exception as e:
                            # Silently continue on synset errors
                            logger.debug(
                                f"Failed to process word '{word}' in {language}: {e}"
                            )
                            continue

                expanded_corpus[category][language] = language_concepts
                logger.info(
                    f"Category {category}.{language}: expanded from {len(concepts)} to {len(language_concepts)} concepts"
                )
        else:
            # Global category - treat as all_languages
            logger.info(f"Processing global category: {category}")
            if isinstance(content, list):
                expanded_corpus[category]["all_languages"] = set(content)
            else:
                logger.warning(
                    f"Unexpected content type for category {category}: {type(content)}"
                )

    return expanded_corpus


def format_corpus_for_scorer(
    expanded_corpus: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Format expanded corpus to maintain structured format for indian_context_scorer.

    Converts sets to lists while preserving the category -> language -> concepts structure.
    This allows the IndianContextScorer to access language-specific keywords directly.

    Args:
        expanded_corpus: Dictionary with structure {category: {language: concept_sets}}

    Returns:
        Dictionary with structure {category: {language: concept_lists}} (scorer format)
    """
    formatted_corpus = {}

    for category, language_dict in expanded_corpus.items():
        formatted_corpus[category] = {}
        for language, concepts in language_dict.items():
            # Convert set to sorted list for deterministic output
            formatted_corpus[category][language] = sorted(list(concepts))

    return formatted_corpus


def save_corpus(corpus: Dict[str, Dict[str, List[str]]], output_file: str):
    """
    Save corpus to JSON file with proper formatting.

    Args:
        corpus: Formatted corpus dictionary with structure {category: {language: concepts}}
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    logger.info(f"Corpus saved to: {output_path}")

    # Print statistics
    total_concepts = 0
    for category, language_dict in corpus.items():
        for language, concepts in language_dict.items():
            total_concepts += len(concepts)

    logger.info(f"Total categories: {len(corpus)}")
    logger.info(f"Total concepts: {total_concepts}")

    for category, language_dict in corpus.items():
        category_total = sum(len(concepts) for concepts in language_dict.values())
        logger.info(
            f"  {category}: {category_total} concepts across {len(language_dict)} languages"
        )
        for language, concepts in language_dict.items():
            logger.info(f"    {language}: {len(concepts)} concepts")


def create_fallback_corpus() -> Dict[str, List[str]]:
    """
    Create fallback corpus when IndoWordNet is unavailable.

    Uses the original manual corpus with some expansions as a fallback option.

    Returns:
        Fallback corpus dictionary
    """
    logger.warning("Creating fallback corpus (IndoWordNet unavailable)")

    # Use the original seed corpus as fallback
    return create_seed_corpus()


def main():
    """Main corpus generation function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced Indian cultural corpus using IndoWordNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-file",
        default="indian_corpus.json",
        help="Output file path for generated corpus (default: indian_corpus.json)",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth for semantic relationship traversal (default: 2)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Indian cultural corpus generation")

    # Create seed corpus
    seed_corpus = create_seed_corpus()
    logger.info(f"Created seed corpus with {len(seed_corpus)} categories")

    # Load IndoWordNet
    iwn = load_indowordnet()

    if iwn is None:
        # Use fallback corpus
        final_corpus = create_fallback_corpus()
        logger.warning("Using fallback corpus due to IndoWordNet loading failure")
    else:
        # Expand using IndoWordNet
        expanded_corpus = expand_concepts_with_indowordnet(
            seed_corpus, iwn, args.max_depth
        )
        final_corpus = format_corpus_for_scorer(expanded_corpus)
        logger.info("Successfully expanded corpus using IndoWordNet")

    # Save corpus
    save_corpus(final_corpus, args.output_file)

    logger.info("Corpus generation completed successfully")


if __name__ == "__main__":
    main()
