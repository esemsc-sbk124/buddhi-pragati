"""
Model classification and cost tracking system for evaluation experiments.

This module classifies models from models_description.txt by reasoning capability,
Indic fine-tuning status, language support, and cost characteristics.
Provides filtering utilities for experiment design.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

from ..utils.config_loader import get_config


@dataclass
class ModelInfo:
    """Information about a model's capabilities and characteristics."""

    name: str
    supported_languages: List[str]
    is_reasoning_model: bool
    model_type: str  # "Indic Fine-tuned" or "General Multilingual"
    is_indic_finetuned: bool
    provider: str  # "OpenAI", "Anthropic", "OpenRouter", "HuggingFace", "SarvamAI"
    is_priority_model: bool = False
    cost_per_input_token: Optional[float] = None
    cost_per_output_token: Optional[float] = None


class ModelClassifier:
    """
    Classifies and filters models for evaluation experiments.

    Provides utilities for:
    - Loading model information from models.txt
    - Filtering by reasoning capability
    - Filtering by Indic fine-tuning status
    - Calculating evaluation costs
    - Language support queries
    """

    def __init__(self, models_file: Optional[str] = None):
        self.logger = logging.getLogger("ModelClassifier")
        self.config = get_config()

        # Default to models_description.txt in project root
        if models_file is None:
            models_file = Path(__file__).parent.parent.parent / "models_description.txt"

        self.models_file = Path(models_file)

        # Configuration flags
        self.cost_tracking_enabled = self.config.get_bool("ENABLE_COST_TRACKING", True)
        self.models: Dict[str, ModelInfo] = {}
        self._load_models()

        # Language family classifications for experiments
        self.dravidian_languages = {"Tamil", "Telugu", "Kannada", "Malayalam"}
        self.indo_aryan_languages = {
            "Hindi",
            "Bengali",
            "Gujarati",
            "Marathi",
            "Punjabi",
            "Odia",
            "Assamese",
            "Urdu",
            "Nepali",
            "Sanskrit",
            "Kashmiri",
            "Konkani",
        }
        self.other_languages = {"English", "Bodo", "Meitei"}

    def _load_models(self):
        """Load model information from models_description.txt file."""
        try:
            with open(self.models_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            current_provider = None

            for line in lines:
                line = line.strip()

                # Skip empty lines and headers
                if not line or line.startswith("=") or "Model Fetcher Results" in line:
                    continue

                # Detect provider sections
                if "HuggingFace Models" in line:
                    current_provider = "HuggingFace"
                    continue
                elif "SarvamAI Models" in line:
                    current_provider = "SarvamAI"
                    continue
                elif "OpenAI Models" in line:
                    current_provider = "OpenAI"
                    continue
                elif "Anthropic Models" in line:
                    current_provider = "Anthropic"
                    continue
                elif "OpenRouter Models" in line:
                    current_provider = "OpenRouter"
                    continue

                # Parse model lines (start with "- ")
                if line.startswith("- ") and current_provider:
                    self._parse_model_line(line[2:], current_provider)

            self.logger.info(
                f"Loaded {len(self.models)} models from {self.models_file}"
            )

        except FileNotFoundError:
            self.logger.error(f"Models file not found: {self.models_file}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def _parse_model_line(self, line: str, provider: str):
        """Parse a single model line from the CSV format."""
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            self.logger.warning(f"Skipping malformed model line: {line}")
            return

        model_name = parts[0]
        languages_text = parts[1]
        reasoning_text = parts[2]
        model_type = parts[3]
        priority_text = parts[4] if len(parts) > 4 else "N"

        # Parse supported languages
        languages = self._parse_language_list(languages_text)

        # Determine characteristics
        is_reasoning = reasoning_text.strip().upper() == "Y"
        is_indic_finetuned = "Indic Fine-tuned" in model_type
        is_priority = priority_text.strip().upper() == "Y"

        self.models[model_name] = ModelInfo(
            name=model_name,
            supported_languages=languages,
            is_reasoning_model=is_reasoning,
            model_type=model_type,
            is_indic_finetuned=is_indic_finetuned,
            provider=provider,
            is_priority_model=is_priority,
        )

    def get_reasoning_models(self) -> List[ModelInfo]:
        """Get all reasoning-capable models."""
        return [model for model in self.models.values() if model.is_reasoning_model]

    def get_non_reasoning_models(self) -> List[ModelInfo]:
        """Get all non-reasoning models."""
        return [model for model in self.models.values() if not model.is_reasoning_model]

    def get_indic_finetuned_models(self) -> List[ModelInfo]:
        """Get all Indic fine-tuned models."""
        return [model for model in self.models.values() if model.is_indic_finetuned]

    def get_general_multilingual_models(self) -> List[ModelInfo]:
        """Get all general multilingual models."""
        return [model for model in self.models.values() if not model.is_indic_finetuned]

    def _normalize_language_name(self, language: str) -> str:
        """
        Normalize language name for case-insensitive comparison.

        Args:
            language: Language name in any case format

        Returns:
            Lowercase language name for comparison
        """
        return language.strip().lower()

    def get_models_for_language(self, language: str) -> List[ModelInfo]:
        """
        Get models that support a specific language.

        Uses case-insensitive matching to handle different language name formats
        from various data sources (HuggingFace configs vs model definitions).

        Args:
            language: Language name (case-insensitive)

        Returns:
            List of ModelInfo objects that support the language
        """
        normalized_language = self._normalize_language_name(language)
        return [
            model
            for model in self.models.values()
            if any(self._normalize_language_name(lang) == normalized_language
                   for lang in model.supported_languages)
        ]

    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Get all models from a specific provider."""
        return [model for model in self.models.values() if model.provider == provider]

    def classify_language_family(self, language: str) -> str:
        """
        Classify language into Dravidian, Indo-Aryan, or Other family.

        Uses case-insensitive matching to handle different language name formats.

        Args:
            language: Language name (case-insensitive)

        Returns:
            Language family classification string
        """
        normalized_language = self._normalize_language_name(language)

        # Check Dravidian family (case-insensitive)
        if any(self._normalize_language_name(lang) == normalized_language
               for lang in self.dravidian_languages):
            return "Dravidian"

        # Check Indo-Aryan family (case-insensitive)
        elif any(self._normalize_language_name(lang) == normalized_language
                 for lang in self.indo_aryan_languages):
            return "Indo-Aryan"

        # Check Other languages (case-insensitive)
        elif any(self._normalize_language_name(lang) == normalized_language
                 for lang in self.other_languages):
            return "Other"

        else:
            # Unknown language - default to Other
            self.logger.warning(f"Unknown language '{language}' classified as Other")
            return "Other"

    def get_dravidian_languages(self) -> Set[str]:
        """Get all Dravidian languages for experiments."""
        return self.dravidian_languages

    def get_indo_aryan_languages(self) -> Set[str]:
        """Get all Indo-Aryan languages for experiments."""
        return self.indo_aryan_languages

    def calculate_evaluation_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        """
        Calculate cost for evaluating with a specific model.

        Returns None if cost tracking is disabled or cost information is not available.
        """
        if not self.cost_tracking_enabled:
            return None

        if model_name not in self.models:
            return None

        model = self.models[model_name]
        if model.cost_per_input_token is None or model.cost_per_output_token is None:
            return None

        return (
            input_tokens * model.cost_per_input_token
            + output_tokens * model.cost_per_output_token
        )

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model."""
        return self.models.get(model_name)

    def list_all_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())

    def _parse_language_list(self, languages_text: str) -> List[str]:
        """Parse comma and space separated language list."""
        # Handle various formats: "Hindi, Tamil, Telugu" or "Hindi Tamil Telugu"
        languages_text = languages_text.replace(",", " ")
        return [lang.strip() for lang in languages_text.split() if lang.strip()]

    def get_priority_models(self) -> List[ModelInfo]:
        """Get all priority models."""
        return [model for model in self.models.values() if model.is_priority_model]

    def get_priority_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """Get priority models of a specific type."""
        return [
            model
            for model in self.models.values()
            if model.is_priority_model and model.model_type == model_type
        ]

    def get_priority_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Get priority models from a specific provider."""
        return [
            model
            for model in self.models.values()
            if model.is_priority_model and model.provider == provider
        ]

    def get_experiment_model_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get model pairs for comparison experiments.

        Returns:
            Dict mapping experiment type to list of (model1, model2) comparison pairs
        """
        reasoning_models = [m.name for m in self.get_reasoning_models()]
        non_reasoning_models = [m.name for m in self.get_non_reasoning_models()]
        indic_models = [m.name for m in self.get_indic_finetuned_models()]
        general_models = [m.name for m in self.get_general_multilingual_models()]

        return {
            "reasoning_vs_non_reasoning": [
                (r, nr) for r in reasoning_models[:3] for nr in non_reasoning_models[:3]
            ],
            "indic_vs_general": [
                (i, g) for i in indic_models[:3] for g in general_models[:3]
            ],
        }
