"""
Enhanced Indian Context Scorer Module

Implements sophisticated multi-tier assessment of "Indianness" for clue-answer pairs
across all supported languages using semantic embeddings, named entity recognition,
and cultural keyword matching. Provides universal coverage for all 20 supported
languages with enhanced accuracy for core Indic languages.

Architecture:
- Tier 1: Multilingual sentence transformers (universal coverage for all 20 languages)
- Tier 2: Indic-specific sentence transformers (enhanced accuracy for 11 core languages)
- Tier 3: NER + keyword scoring (cultural context detection and fallback)

Key Features:
- Language-aware model selection and scoring strategies
- Lazy loading and smart caching for optimal performance
- Cross-language consistency validation and adaptive thresholds
- Cultural reference corpus for semantic similarity assessment
- Integration with existing streaming pipeline architecture
"""

import logging
import numpy as np
from typing import Dict, Set, List, Optional, Any
from functools import lru_cache
import unicodedata
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class IndianContextScorer:
    """
    Multi-tier Indian cultural context scorer for clue-answer pairs.

    Provides sophisticated assessment of "Indianness" using semantic embeddings,
    named entity recognition, and cultural keyword matching. Supports all 20
    languages with enhanced accuracy for core Indic languages.
    """

    # Language tier definitions based on model availability
    TIER_1_LANGUAGES = {  # Universal coverage - all 20 supported languages
        "Assamese",
        "Bengali",
        "Bodo",
        "English",
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
    }

    TIER_2_LANGUAGES = {  # Enhanced scoring with Indic-specific models
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

    def __init__(self, config: Dict[str, Any], mode: str = "complete"):
        """
        Initialize the Indian context scorer with configuration and scoring mode.

        Args:
            config: Configuration dictionary with scoring parameters
            mode: Scoring mode - "fast" (multilingual India keywords) or "complete" (full corpus)
        """
        self.config = config
        self.mode = mode
        self._models = {}  # Lazy-loaded models cache
        self._reference_embeddings = {}  # Cached reference embeddings per model

        # Configuration parameters with defaults
        self.primary_model_name = config.get(
            "INDIAN_CONTEXT_PRIMARY_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.indic_model_name = config.get(
            "INDIAN_CONTEXT_INDIC_MODEL", "l3cube-pune/indic-sentence-similarity-sbert"
        )
        self.enable_tiered_scoring = config.get("ENABLE_TIERED_CONTEXT_SCORING", True)

        # Scoring weights for different tiers
        self.tier1_weights = config.get(
            "CONTEXT_SCORING_WEIGHTS_TIER1", {"embedding": 0.85, "keyword": 0.15}
        )
        self.tier2_weights = config.get(
            "CONTEXT_SCORING_WEIGHTS_TIER2",
            {"multilingual": 0.5, "indic": 0.4, "keyword": 0.1},
        )

        # Initialize cultural reference corpus based on mode
        self._cultural_references = self._initialize_cultural_corpus()

        # Load language-specific expanded corpus for keyword matching
        self._expanded_corpus = self._load_expanded_corpus()

        logger.info(
            f"Initialized IndianContextScorer with mode: {self.mode}, tiered scoring: {self.enable_tiered_scoring}"
        )

    def _initialize_cultural_corpus(self) -> Dict[str, List[str]]:
        """
        Initialize cultural reference corpus based on selected mode.

        Returns:
            Dictionary mapping cultural categories to reference texts
            - Fast mode: Comprehensive multilingual India keywords across all 20 languages
            - Complete mode: Full expanded corpus from generated file or fallback
        """
        if self.mode == "fast":
            # Fast mode: comprehensive India keywords across all supported languages
            logger.info("Using fast mode - comprehensive multilingual India keywords")
            return self._get_multilingual_india_keywords()

        elif self.mode == "complete":
            # Complete mode: try to load expanded corpus, fallback to manual corpus
            corpus_file = Path(__file__).parent / "indian_corpus.json"

            if corpus_file.exists():
                try:
                    logger.info(f"Loading expanded cultural corpus from: {corpus_file}")
                    with open(corpus_file, "r", encoding="utf-8") as f:
                        expanded_corpus = json.load(f)
                    logger.info(
                        f"Loaded expanded corpus with {len(expanded_corpus)} categories"
                    )
                    return expanded_corpus
                except Exception as e:
                    logger.warning(
                        f"Failed to load expanded corpus: {e}, using fallback"
                    )
            else:
                logger.info(
                    "Expanded corpus file not found, using manual fallback corpus"
                )

            # Fallback to original manual corpus
            return self._get_manual_cultural_corpus()

        else:
            logger.warning(f"Unknown mode '{self.mode}', defaulting to complete mode")
            return self._get_manual_cultural_corpus()

    def _get_multilingual_india_keywords(self) -> Dict[str, List[str]]:
        """
        Get comprehensive India keywords across all supported languages for fast mode.

        Returns:
            Dictionary containing India-related keywords in multiple languages
        """
        return {
            "india_multilingual": [
                # English variants
                "India",
                "Indian",
                "Bharat",
                "Hindustan",
                "Republic of India",
                # Hindi (Devanagari)
                "भारत",
                "इंडिया",
                "हिंदुस्तान",
                "भारतीय",
                "हिन्दुस्तान",
                # Bengali
                "ভারত",
                "ইন্ডিয়া",
                "ভারতীয়",
                "হিন্দুস্তান",
                # Tamil
                "இந்தியா",
                "பாரதம்",
                "இந்திய",
                "ஹிந்துஸ்தான்",
                # Telugu
                "భారతదేశం",
                "ఇండియా",
                "భారతీయ",
                "హిందూస్థాన్",
                # Gujarati
                "ભારત",
                "ઇન્ડિયા",
                "ભારતીય",
                "હિંદુસ્તાન",
                # Kannada
                "ಭಾರತ",
                "ಇಂಡಿಯಾ",
                "ಭಾರತೀಯ",
                "ಹಿಂದೂಸ್ಥಾನ",
                # Malayalam
                "ഇന്ത്യ",
                "ഭാരതം",
                "ഇന്ത്യൻ",
                "ഹിന്ദുസ്ഥാൻ",
                # Marathi
                "भारत",
                "इंडिया",
                "भारतीय",
                "हिंदुस्थान",
                # Punjabi (Gurmukhi)
                "ਭਾਰਤ",
                "ਇੰਡੀਆ",
                "ਭਾਰਤੀ",
                "ਹਿੰਦੁਸਤਾਨ",
                # Odia
                "ଭାରତ",
                "ଇଣ୍ଡିଆ",
                "ଭାରତୀୟ",
                "ହିନ୍ଦୁସ୍ତାନ",
                # Assamese
                "ভাৰত",
                "ইণ্ডিয়া",
                "ভাৰতীয়",
                "হিন্দুস্তান",
                # Urdu (Arabic script)
                "ہندوستان",
                "انڈیا",
                "بھارت",
                "ہندی",
                # Sanskrit
                "भारतवर्ष",
                "आर्यावर्त",
                "जम्बूद्वीप",
                "हिन्दुस्थान",
                # Nepali
                "भारत",
                "इन्डिया",
                "भारतीय",
                "हिन्दुस्तान",
                # Common regional/cultural variants
                "Hindustan",
                "Hindusthan",
                "Bharatvarsha",
                "Aryavarta",
                "Jambudvipa",
                "Golden Bird",
                "Sone ki Chidiya",
                # Modern context
                "Republic of India",
                "Union of India",
                "Indian Republic",
                "Incredible India",
                "Digital India",
                "Make in India",
            ]
        }

    def _get_manual_cultural_corpus(self) -> Dict[str, List[str]]:
        """
        Get the original manual cultural corpus as fallback.

        Returns:
            Dictionary mapping cultural categories to reference texts
        """
        return {
            "festivals": [
                "Diwali celebration lights festival India",
                "Holi colors spring festival celebration",
                "Eid festivities Muslim community India",
                "Christmas celebration Christian community",
                "Durga Puja Bengali festival goddess",
                "Ganesh Chaturthi Maharashtra festival",
                "Onam Kerala harvest festival",
            ],
            "geography": [
                "Himalayas mountain range India Nepal",
                "Ganga river sacred Hindu religion",
                "Kerala backwaters tourism India",
                "Rajasthan desert palaces culture",
                "Mumbai financial capital India",
                "Delhi national capital India",
                "Chennai Tamil Nadu south India",
            ],
            "history": [
                "Mahatma Gandhi freedom fighter independence",
                "Jawaharlal Nehru first Prime Minister",
                "Subhas Chandra Bose Azad Hind Fauj",
                "British colonial rule India independence",
                "Mughal empire Delhi Akbar Shah Jahan",
                "Chhatrapati Shivaji Maratha empire",
                "Chandragupta Maurya ancient India",
            ],
            "culture": [
                "Bollywood Hindi cinema Mumbai films",
                "Classical dance Bharatanatyam Kathak Odissi",
                "Indian classical music raga tala",
                "Yoga meditation spiritual practice",
                "Ayurveda traditional medicine system",
                "Sanskrit ancient language scriptures",
                "Vedas Upanishads Hindu philosophy",
            ],
            "cuisine": [
                "Indian curry spices masala cooking",
                "Biryani rice dish Hyderabadi Lucknowi",
                "Dosa idli South Indian breakfast",
                "Roti chapati Indian bread staple",
                "Chai tea masala Indian beverage",
                "Sweets mithai Indian desserts",
                "Regional cuisine Bengali Tamil Punjabi",
            ],
            "modern_india": [
                "Information technology Bangalore silicon valley",
                "Indian Space Research Organisation ISRO",
                "Bollywood entertainment film industry",
                "Cricket national sport India passion",
                "Indian Railways largest railway network",
                "Diverse languages multilingual India",
                "Democratic republic largest democracy",
            ],
        }

    def _load_expanded_corpus(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load language-specific expanded corpus from generated indian_corpus.json file.

        Returns:
            Dictionary mapping categories to language-specific keyword dictionaries
        """
        try:
            corpus_file = Path(__file__).parent / "indian_corpus.json"

            if corpus_file.exists():
                logger.info(f"Loading expanded corpus from: {corpus_file}")
                with open(corpus_file, "r", encoding="utf-8") as f:
                    expanded_corpus = json.load(f)

                logger.info(
                    f"Loaded expanded corpus with {len(expanded_corpus)} categories"
                )
                return expanded_corpus

        except Exception as e:
            logger.warning(f"Failed to load expanded corpus: {e}")

        # Fallback to seed corpus if expanded corpus not available
        try:
            logger.info("Attempting to load seed corpus as fallback")
            corpus_file = Path(__file__).parent / "seed_corpus.json"

            if corpus_file.exists():
                with open(corpus_file, "r", encoding="utf-8") as f:
                    seed_corpus = json.load(f)

                # Process seed corpus to match expected format
                structured_corpus = {}
                for category, content in seed_corpus.items():
                    if isinstance(content, dict):
                        # Language-specific categories
                        structured_corpus[category] = content
                    else:
                        # Global categories - make available for all languages
                        structured_corpus[category] = {"all_languages": content}

                logger.info(
                    f"Loaded seed corpus fallback with {len(structured_corpus)} categories"
                )
                return structured_corpus

        except Exception as e:
            logger.warning(f"Failed to load seed corpus fallback: {e}")

        # Final fallback - empty corpus
        logger.warning("Using empty fallback corpus")
        return {}

    def _load_model(self, model_name: str) -> Any:
        """
        Lazy load sentence transformer model with caching.

        Args:
            model_name: Name of the sentence transformer model

        Returns:
            Loaded sentence transformer model
        """
        if model_name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading sentence transformer model: {model_name}")
                self._models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                return None

        return self._models[model_name]

    def _get_reference_embeddings(self, model_name: str) -> Optional[np.ndarray]:
        """
        Get cached reference embeddings for cultural corpus.

        Args:
            model_name: Name of the sentence transformer model

        Returns:
            Numpy array of reference embeddings or None if model unavailable
        """
        if model_name not in self._reference_embeddings:
            model = self._load_model(model_name)
            if model is None:
                return None

            # Flatten all cultural references into single list
            all_references = []
            for category_refs in self._cultural_references.values():
                all_references.extend(category_refs)

            try:
                logger.info(f"Computing reference embeddings with {model_name}")
                embeddings = model.encode(all_references)
                # Compute centroid for similarity comparison
                self._reference_embeddings[model_name] = np.mean(embeddings, axis=0)
                logger.info(f"Cached reference embeddings for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to compute reference embeddings: {e}")
                return None

        return self._reference_embeddings[model_name]

    def _calculate_embedding_score(self, text: str, model_name: str) -> float:
        """
        Calculate semantic similarity score using embeddings.

        Args:
            text: Input text to score
            model_name: Name of the sentence transformer model

        Returns:
            Similarity score between 0.0 and 1.0
        """
        model = self._load_model(model_name)
        reference_embedding = self._get_reference_embeddings(model_name)

        if model is None or reference_embedding is None:
            logger.debug(f"Model {model_name} or reference embeddings unavailable")
            return 0.0

        try:
            text_embedding = model.encode([text])[0]
            # Compute cosine similarity
            similarity = np.dot(text_embedding, reference_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(reference_embedding)
            )
            # Discard negative similarities (dissimilarity), keep only positive similarities
            return max(0.0, float(similarity))
        except Exception as e:
            logger.debug(f"Failed to compute embedding similarity: {e}")
            return 0.0

    def _calculate_keyword_score(self, text: str, language: str = None) -> float:
        """
        Calculate language-aware keyword-based cultural context score.

        Args:
            text: Input text to analyze
            language: Target language for language-specific keywords (optional)

        Returns:
            Keyword score between 0.0 and 1.0
        """
        text_lower = text.lower()
        # Normalize unicode for better matching
        text_normalized = unicodedata.normalize("NFKD", text_lower)

        total_matches = 0
        total_keywords = 0

        # Get language-specific keywords from expanded corpus if available
        if language and self._expanded_corpus:
            keywords_to_check = []

            # Collect language-specific keywords from all categories
            for category, lang_dict in self._expanded_corpus.items():
                if isinstance(lang_dict, dict):
                    # Check for exact language match first
                    if language in lang_dict:
                        keywords_to_check.extend(lang_dict[language])
                    # Check for English as fallback for better coverage
                    elif "English" in lang_dict:
                        keywords_to_check.extend(lang_dict["English"])
                    # Check for global keywords
                    elif "all_languages" in lang_dict:
                        keywords_to_check.extend(lang_dict["all_languages"])

            # Process language-specific keywords
            for keyword_phrase in keywords_to_check:
                if isinstance(keyword_phrase, str):
                    keywords_in_phrase = keyword_phrase.lower().split()
                    total_keywords += len(keywords_in_phrase)

                    for keyword in keywords_in_phrase:
                        keyword_normalized = unicodedata.normalize("NFKD", keyword)
                        if keyword_normalized in text_normalized:
                            total_matches += 1

        # Fallback to cultural references if no seed corpus or language-specific keywords
        if total_keywords == 0:
            # Check matches across cultural references (mode-based corpus)
            for keywords in self._cultural_references.values():
                for keyword_phrase in keywords:
                    keywords_in_phrase = keyword_phrase.lower().split()
                    total_keywords += len(keywords_in_phrase)

                    for keyword in keywords_in_phrase:
                        keyword_normalized = unicodedata.normalize("NFKD", keyword)
                        if keyword_normalized in text_normalized:
                            total_matches += 1

        # Normalize by text length to avoid bias toward longer texts
        word_count = len(text.split())
        if word_count == 0 or total_keywords == 0:
            return 0.0

        # Balance between absolute matches and relative density
        absolute_score = total_matches / total_keywords
        density_score = total_matches / word_count

        return min(1.0, (absolute_score + density_score) / 2.0)

    def _get_scoring_strategy(self, language: str) -> str:
        """
        Determine scoring strategy based on language tier.

        Args:
            language: Target language

        Returns:
            Scoring strategy ('tier1' or 'tier2')
        """
        if not self.enable_tiered_scoring:
            return "tier1"

        return "tier2" if language in self.TIER_2_LANGUAGES else "tier1"

    def score_context(self, clue: str, answer: str, language: str) -> Dict[str, float]:
        """
        Calculate comprehensive Indian context score for clue-answer pair.

        Args:
            clue: Crossword clue text
            answer: Answer word/phrase
            language: Target language for scoring

        Returns:
            Dictionary containing individual and final scores
        """
        pair_text = f"{clue} {answer}"
        strategy = self._get_scoring_strategy(language)

        if strategy == "tier2":
            # Enhanced scoring with dual models (no NER)
            multilingual_score = self._calculate_embedding_score(
                pair_text, self.primary_model_name
            )
            indic_score = self._calculate_embedding_score(
                pair_text, self.indic_model_name
            )
            keyword_score = self._calculate_keyword_score(pair_text, language)

            # Weighted combination for tier 2
            final_score = (
                self.tier2_weights["multilingual"] * multilingual_score
                + self.tier2_weights["indic"] * indic_score
                + self.tier2_weights["keyword"] * keyword_score
            )

            return {
                "final_score": final_score,
                "multilingual_score": multilingual_score,
                "indic_score": indic_score,
                "keyword_score": keyword_score,
                "strategy": "tier2",
            }

        else:
            # Standard scoring for tier 1 (no NER)
            embedding_score = self._calculate_embedding_score(
                pair_text, self.primary_model_name
            )
            keyword_score = self._calculate_keyword_score(pair_text, language)

            # Weighted combination for tier 1
            final_score = (
                self.tier1_weights["embedding"] * embedding_score
                + self.tier1_weights["keyword"] * keyword_score
            )

            return {
                "final_score": final_score,
                "embedding_score": embedding_score,
                "keyword_score": keyword_score,
                "strategy": "tier1",
            }

    def validate_cross_language_consistency(
        self, scores_by_language: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Validate scoring consistency across languages.

        Args:
            scores_by_language: Dictionary mapping language to list of scores

        Returns:
            Dictionary containing consistency metrics
        """
        if not scores_by_language:
            return {"variance": 0.0, "std_dev": 0.0}

        # Calculate mean scores per language
        language_means = {}
        for language, scores in scores_by_language.items():
            if scores:
                language_means[language] = np.mean(scores)

        if not language_means:
            return {"variance": 0.0, "std_dev": 0.0}

        # Calculate variance across language means
        overall_mean = np.mean(list(language_means.values()))
        variance = np.var(list(language_means.values()))
        std_dev = np.std(list(language_means.values()))

        return {
            "variance": float(variance),
            "std_dev": float(std_dev),
            "language_means": language_means,
            "overall_mean": float(overall_mean),
        }

    @lru_cache(maxsize=1)
    def get_supported_languages(self) -> Set[str]:
        """
        Get set of all supported languages.

        Returns:
            Set of supported language names
        """
        return self.TIER_1_LANGUAGES.copy()

    def get_language_tier_info(self, language: str) -> Dict[str, Any]:
        """
        Get tier information for specific language.

        Args:
            language: Target language

        Returns:
            Dictionary with tier information and capabilities
        """
        tier = 2 if language in self.TIER_2_LANGUAGES else 1
        models_available = []

        # Check model availability
        if self._load_model(self.primary_model_name) is not None:
            models_available.append(self.primary_model_name)

        if tier == 2 and self._load_model(self.indic_model_name) is not None:
            models_available.append(self.indic_model_name)

        return {
            "tier": tier,
            "strategy": self._get_scoring_strategy(language),
            "models_available": models_available,
            "enhanced_accuracy": tier == 2,
            "universal_coverage": True,
        }
