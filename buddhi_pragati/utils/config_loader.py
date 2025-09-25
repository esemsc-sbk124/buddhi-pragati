"""
Configuration Management System for Buddhi-Pragati Benchmark.

This module provides centralized configuration management for the crossword benchmark system.
It loads parameters from crossword_config.txt with type-safe parsing, hierarchical fallbacks,
and comprehensive default values for all system components.

Key Features:
- Type-safe parameter parsing (string, int, float, bool)
- Hierarchical configuration: CLI args → Config file → Environment → Defaults
- Comprehensive validation and error handling
- Support for complex configuration objects (weights, sources, model lists)
- CLI defaults integration for streamlined command-line usage

Architecture:
- ConfigLoader: Main configuration management class
- Global config singleton via get_config()
- Automatic project root detection
- Memory-efficient caching with reload capabilities

The configuration system supports:
- Dataset creation parameters (sources, sizes, quality thresholds)
- Model interface settings (API keys, timeouts, reasoning modes)
- Evaluation parameters (experimental types, metrics, prompt configurations)
- Generation settings (grid densities, algorithms, cultural scoring)
- Performance tuning (batch sizes, memory limits, caching strategies)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

__all__ = ["ConfigLoader", "get_config", "reload_config"]


class ConfigLoader:
    """
    Loads and manages configuration parameters for dataset creation.

    Provides type-safe access to configuration values with defaults.
    """

    def __init__(self, config_file: str = "crossword_config.txt"):
        """
        Initialize configuration loader.

        Args:
            config_file: Path to configuration file relative to project root
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        # Find project root (look for crossword_config.txt)
        current_path = Path(__file__).parent
        config_path = None

        # Search up the directory tree
        for _ in range(5):  # Limit search depth
            potential_path = current_path / self.config_file
            if potential_path.exists():
                config_path = potential_path
                break
            current_path = current_path.parent

        if config_path is None:
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            self._load_defaults()
            return

        logger.info(f"Loading configuration from {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Parse key=value pairs
                    if "=" not in line:
                        logger.warning(f"Invalid config line {line_num}: {line}")
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert value to appropriate type
                    self.config[key] = self._parse_value(value)

            logger.info(f"Loaded {len(self.config)} configuration parameters")

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._load_defaults()

    def _parse_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse string value to appropriate Python type."""
        # Handle boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Handle numeric values
        if value.replace(".", "").replace("-", "").isdigit():
            if "." in value:
                return float(value)
            else:
                return int(value)

        # Return as string
        return value

    def _load_defaults(self):
        """Load default configuration values for dataset creation only."""
        self.config = {
            # Dataset creation
            "BATCH_SIZE_PROCESSING": 15,
            "TARGET_DATASET_SIZE_PER_LANGUAGE": 1000,
            "DEFAULT_DATASET_SOURCES": "MILU,IndicWikiBio,IndoWordNet,Bhasha-Wiki",
            "MIN_QUALITY_SCORE": 0.3,
            "HF_DATASET_REPO": "selim-b-kh/Buddhi_pragati",
            # Word filtering
            "MIN_WORD_LENGTH": 2,
            "MAX_WORD_LENGTH": 12,
            "MIN_CLUE_LENGTH": 10,
            "MAX_CLUE_LENGTH": 500,
            # Source-specific parameters
            "MILU_FILTER_CONTEXTUAL_QUESTIONS": True,
            "BHASHA_WIKI_MIN_TEXT_LENGTH": 50,
            "INDIC_WIKIBIO_MIN_SUMMARY_LENGTH": 30,
            "INDOWORDNET_MIN_DEFINITION_LENGTH": 5,
            "NER_MODEL_NAME": "ai4bharat/IndicNER",
            "MAX_ENTITIES_PER_TEXT": 5,
            "REQUIRE_SINGLE_WORD_ENTITIES": True,
            # CLI defaults
            "DEFAULT_HF_TOKEN": "",
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration parameter name
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value for {key}: {value}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value for {key}: {value}")
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return default

    def get_string(self, key: str, default: str = "") -> str:
        """Get string configuration value."""
        value = self.get(key, default)
        return str(value) if value is not None else default

    def get_list_of_ints(self, key: str, default: str = "") -> list:
        """Get list of integers from comma-separated string configuration value."""
        value = self.get_string(key, default)
        if not value:
            return []
        try:
            return [int(item.strip()) for item in value.split(",")]
        except (ValueError, TypeError):
            logger.warning(f"Invalid list of integers for {key}: {value}")
            # Return default as list
            if default:
                return [int(item.strip()) for item in default.split(",")]
            return []

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset-related configuration."""
        return {
            "BATCH_SIZE_PROCESSING": self.get_int("BATCH_SIZE_PROCESSING", 15),
            "TARGET_DATASET_SIZE_PER_LANGUAGE": self.get_int(
                "TARGET_DATASET_SIZE_PER_LANGUAGE", 1000
            ),
            "DEFAULT_DATASET_SOURCES": self.get_string(
                "DEFAULT_DATASET_SOURCES", "MILU,IndicWikiBio,IndoWordNet,Bhasha-Wiki"
            ),
            "MIN_QUALITY_SCORE": self.get_float("MIN_QUALITY_SCORE", 0.3),
            "HF_DATASET_REPO": self.get_string(
                "HF_DATASET_REPO", "selim-b-kh/Buddhi_pragati"
            ),
            "MIN_WORD_LENGTH": self.get_int("MIN_WORD_LENGTH", 2),
            "MAX_WORD_LENGTH": self.get_int("MAX_WORD_LENGTH", 12),
            "MIN_CLUE_LENGTH": self.get_int("MIN_CLUE_LENGTH", 10),
            "MAX_CLUE_LENGTH": self.get_int("MAX_CLUE_LENGTH", 500),
            "MILU_FILTER_CONTEXTUAL_QUESTIONS": self.get_bool(
                "MILU_FILTER_CONTEXTUAL_QUESTIONS", True
            ),
            "BHASHA_WIKI_MIN_TEXT_LENGTH": self.get_int(
                "BHASHA_WIKI_MIN_TEXT_LENGTH", 50
            ),
            "INDIC_WIKIBIO_MIN_SUMMARY_LENGTH": self.get_int(
                "INDIC_WIKIBIO_MIN_SUMMARY_LENGTH", 30
            ),
            "INDOWORDNET_MIN_DEFINITION_LENGTH": self.get_int(
                "INDOWORDNET_MIN_DEFINITION_LENGTH", 5
            ),
            "NER_MODEL_NAME": self.get_string("NER_MODEL_NAME", "ai4bharat/IndicNER"),
            "MAX_ENTITIES_PER_TEXT": self.get_int("MAX_ENTITIES_PER_TEXT", 5),
            "REQUIRE_SINGLE_WORD_ENTITIES": self.get_bool(
                "REQUIRE_SINGLE_WORD_ENTITIES", True
            ),
        }

    def get_generation_config(self) -> Dict[str, Any]:
        """Get crossword generation configuration parameters."""
        return {
            "max_generation_attempts": self.get_int("MAX_GENERATION_ATTEMPTS", 50),
            "min_words_per_puzzle": self.get_int("MIN_WORDS_PER_PUZZLE", 8),
            "target_grid_density": self.get_float("TARGET_GRID_DENSITY", 0.75),
            "min_acceptable_density": self.get_float("MIN_ACCEPTABLE_DENSITY", 0.65),
            "indian_context_threshold": self.get_float("INDIAN_CONTEXT_THRESHOLD", 0.4),
            "generation_timeout_seconds": self.get_int(
                "GENERATION_TIMEOUT_SECONDS", 30
            ),
            "prefer_indian_entries": self.get_bool("PREFER_INDIAN_ENTRIES", True),
            "density_weight": self.get_float("DENSITY_WEIGHT", 0.6),
            "intersection_weight": self.get_float("INTERSECTION_WEIGHT", 0.25),
            "cultural_coherence_weight": self.get_float(
                "CULTURAL_COHERENCE_WEIGHT", 0.15
            ),
            "min_grid_size": self.get_int("MIN_GRID_SIZE", 3),
            "max_grid_size": self.get_int("MAX_GRID_SIZE", 30),
            "crossword_batch_size": self.get_int("CROSSWORD_BATCH_SIZE", 10),
            "max_concurrent_attempts": self.get_int("MAX_CONCURRENT_ATTEMPTS", 5),
            "retry_with_smaller_corpus": self.get_bool(
                "RETRY_WITH_SMALLER_CORPUS", True
            ),
        }

    def get_cli_defaults(self) -> Dict[str, Any]:
        """Get CLI default values."""
        return {
            "language": "English",
            "model": "gpt-4o",
            "model_source": "openai",
            "hf_token": self.get_string("DEFAULT_HF_TOKEN", ""),
            "openai_api_key": "",
            # Dataset creation defaults
            "target_size": self.get_int("TARGET_DATASET_SIZE_PER_LANGUAGE", 1000),
            "sources": self.get_string("DEFAULT_DATASET_SOURCES", "MILU").split(","),
            "batch_size": self.get_int("BATCH_SIZE_PROCESSING", 100),
            # Dataset management defaults
            "repo_id": self.get_string("HF_DATASET_REPO", "selim-b-kh/Buddhi_pragati"),
            # Generation defaults
            "grid_size": self.get_int("DEFAULT_GRID_SIZE", 15),
            "count": self.get_int("DEFAULT_PUZZLE_COUNT", 10),
            # Context scoring mode default
            "context_scoring_mode": self.get_string(
                "DEFAULT_CONTEXT_SCORING_MODE", "complete"
            ),
        }


# Global configuration instance
_config = None


def get_config() -> ConfigLoader:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config():
    """Reload configuration from file."""
    global _config
    _config = ConfigLoader()
    return _config
