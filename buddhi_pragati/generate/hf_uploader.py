"""
HuggingFace Uploader for Generated Crossword Puzzles

This module handles uploading generated crossword puzzles to HuggingFace Hub
in the CrosswordPuzzleEntry format for dataset storage with proper configuration
management, YAML header generation, and README updates.

Key Features:
- Language-specific configurations (subsets) for proper dataset organization
- Automatic YAML header generation with real statistics
- Comprehensive README updates after each upload
- Cache clearing for uploaded configurations
- Robust error handling and validation
"""

import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset, load_dataset, get_dataset_config_names
from huggingface_hub import HfApi, create_repo, DatasetCard
from datetime import datetime

from .puzzle_entry import CrosswordPuzzleEntry
from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class PuzzleHFUploader:
    """
    Handles uploading generated crossword puzzles to HuggingFace Hub with proper
    configuration management, YAML headers, and README updates.

    Creates and manages language-specific configurations (subsets) in the dataset
    repository, ensuring proper organization and metadata management.
    """

    def __init__(self, hf_token: str = None, generated_repo: str = None):
        """
        Initialize HuggingFace uploader for puzzle datasets.

        Args:
            hf_token: HuggingFace token for dataset operations
            generated_repo: Repository for generated puzzles (default from config)
        """
        self.config = get_config()
        self.hf_token = hf_token or self.config.get_string("DEFAULT_HF_TOKEN", "")

        # Use separate repository for generated puzzles
        self.generated_repo = generated_repo or self.config.get_string(
            "HF_GENERATED_PUZZLES_REPO", "selim-b-kh/buddhi-pragati-puzzles"
        )

        if self.hf_token:
            from huggingface_hub import login

            login(token=self.hf_token)
            self.hf_api = HfApi()
        else:
            self.hf_api = None
            logger.warning(
                "No HuggingFace token provided - upload functionality disabled"
            )

        logger.info(
            f"Initialized PuzzleHFUploader for repository: {self.generated_repo}"
        )

    def _clear_dataset_cache(self, repo_id: str, config_name: str = None):
        """
        Clear HuggingFace dataset cache for a specific repository.

        Args:
            repo_id: Repository ID (e.g., 'selim-b-kh/buddhi-pragati-puzzles')
            config_name: Optional language configuration to clear specifically.
                        If None, clears entire repository cache.
        """
        try:
            repo_name = repo_id.replace("/", "___")

            if config_name:
                logger.debug(
                    f"Clearing HuggingFace cache for {repo_id}, config: {config_name}"
                )
            else:
                logger.debug(
                    f"Clearing all HuggingFace cache for repository: {repo_id}"
                )

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
                    logger.debug(f"Cleared {cleared_count} dataset cache directories")

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
                    logger.debug(f"Cleared {cleared_count} hub cache directories")

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
                            logger.debug(
                                f"Cleared {cleared_count} internal cache directories"
                            )
            except Exception:
                pass

        except Exception as e:
            logger.warning(
                f"Failed to clear cache for {repo_id}{f' (config: {config_name})' if config_name else ''}: {e}"
            )

    def upload_puzzle_batch(
        self, puzzles: List[CrosswordPuzzleEntry], language: str, grid_size: int
    ) -> bool:
        """
        Upload a batch of generated puzzles to HuggingFace with proper configuration management.

        Args:
            puzzles: List of CrosswordPuzzleEntry instances to upload
            language: Language of the puzzles
            grid_size: Grid size of the puzzles

        Returns:
            True if upload successful, False otherwise
        """
        if not self.hf_api or not self.hf_token:
            logger.error("HuggingFace API not available - cannot upload")
            return False

        if not puzzles:
            logger.warning("No puzzles to upload")
            return False

        try:
            logger.info(
                f"Uploading {len(puzzles)} puzzles for {language} ({grid_size}x{grid_size})"
            )

            # Ensure repository exists
            self._ensure_repository_exists()

            # Use language-specific configuration (not grid-size specific)
            config_name = language.lower()

            # PERFORMANCE FIX: Clear cache only for current language, not entire repository
            self._clear_dataset_cache(self.generated_repo, config_name=config_name)
            logger.debug(
                f"Cleared cache for {config_name} before loading existing dataset"
            )

            # Convert puzzles to dataset format
            puzzle_dicts = [puzzle.to_dict() for puzzle in puzzles]
            new_df = pd.DataFrame(puzzle_dicts)

            # Try to load existing dataset and merge with forced redownload
            try:
                logger.debug(
                    f"Loading existing dataset for {config_name} with force_redownload"
                )
                existing_dataset = load_dataset(
                    self.generated_repo,
                    config_name,
                    split="train",
                    token=self.hf_token,
                    download_mode="force_redownload",  # Force fresh download
                    verification_mode="no_checks",  # Skip verification for speed
                )
                existing_df = existing_dataset.to_pandas()
                logger.debug(f"Found existing dataset with {len(existing_df)} entries")

                # Merge with existing data, avoiding duplicates by ID
                existing_ids = set(existing_df["id"].tolist())
                new_entries_to_add = [
                    entry for entry in puzzle_dicts if entry["id"] not in existing_ids
                ]

                if new_entries_to_add:
                    logger.info(
                        f"Adding {len(new_entries_to_add)} new entries (filtered {len(puzzle_dicts) - len(new_entries_to_add)} duplicates)"
                    )
                    new_entries_df = pd.DataFrame(new_entries_to_add)
                    combined_df = pd.concat(
                        [existing_df, new_entries_df], ignore_index=True
                    )
                else:
                    logger.info("No new entries to add (all were duplicates)")
                    combined_df = existing_df

            except Exception as e:
                logger.info(
                    f"No existing dataset found or error loading: {e}. Creating new dataset."
                )
                combined_df = new_df

            # Create HuggingFace Dataset from combined data
            final_dataset = Dataset.from_pandas(combined_df)
            logger.info(f"Uploading dataset with {len(combined_df)} total entries")

            # Upload to hub with language-specific configuration
            final_dataset.push_to_hub(
                self.generated_repo,
                config_name=config_name,
                split="train",
                token=self.hf_token,
            )

            # PERFORMANCE FIX: Update dataset card with intelligent caching for current language
            self._update_dataset_card_comprehensive(active_language=language)

            logger.info(
                f"✅ Successfully uploaded {len(puzzles)} puzzles to {self.generated_repo}/{config_name}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to upload puzzles: {e}")
            return False

    def upload_multi_language_batch(
        self, language_puzzles: Dict[str, List[CrosswordPuzzleEntry]]
    ) -> Dict[str, bool]:
        """
        Upload puzzles for multiple languages.

        Args:
            language_puzzles: Dictionary mapping language -> List[CrosswordPuzzleEntry]

        Returns:
            Dictionary mapping language -> success_status
        """
        results = {}

        for language, puzzles in language_puzzles.items():
            if not puzzles:
                results[language] = False
                continue

            # Group puzzles by grid size
            grid_size_groups = {}
            for puzzle in puzzles:
                grid_size = puzzle.grid_size  # Now stored as integer
                if grid_size not in grid_size_groups:
                    grid_size_groups[grid_size] = []
                grid_size_groups[grid_size].append(puzzle)

            # Upload each grid size separately
            language_success = True
            for grid_size, grid_puzzles in grid_size_groups.items():
                success = self.upload_puzzle_batch(grid_puzzles, language, grid_size)
                if not success:
                    language_success = False

            results[language] = language_success

        return results

    def _ensure_repository_exists(self):
        """Ensure the generated puzzles repository exists on HuggingFace."""
        try:
            create_repo(
                self.generated_repo,
                repo_type="dataset",
                token=self.hf_token,
                exist_ok=True,
                private=False,
            )
            logger.debug(f"Ensured repository {self.generated_repo} exists")

        except Exception as e:
            logger.error(f"Failed to ensure repository exists: {e}")
            raise

    def _scan_repository_configurations(self, repo_id: str) -> List[str]:
        """Scan HuggingFace repository to find all existing language configurations."""
        try:
            # First try the proper HuggingFace API for getting dataset configurations
            logger.debug(f"Getting dataset configurations using HF API for {repo_id}")
            configs = get_dataset_config_names(repo_id, token=self.hf_token)

            # Filter out any invalid configurations that might cause issues
            valid_configs = []
            for config in configs:
                # Skip common false-positive configuration names
                if config not in ["data", "default", "train", "test", "validation"]:
                    valid_configs.append(config)
                else:
                    logger.debug(f"Skipping invalid configuration name: {config}")

            config_list = sorted(valid_configs)
            logger.debug(
                f"Found {len(config_list)} valid configurations in {repo_id}: {config_list}"
            )
            return config_list

        except Exception as e:
            logger.warning(f"Could not get configurations using HF API: {e}")
            return []

    def _update_dataset_card_comprehensive(self, active_language: str = None):
        """
        Update dataset card with comprehensive statistics for all configurations.

        Args:
            active_language: Language currently being updated (gets fresh stats, others use cache)
        """
        try:
            if active_language:
                logger.debug(
                    f"Updating dataset card for {self.generated_repo} (intelligent caching for {active_language})"
                )
            else:
                logger.debug(
                    f"Updating dataset card for {self.generated_repo} (full refresh)"
                )

            # Scan all configurations in repository
            all_configs = self._scan_repository_configurations(self.generated_repo)

            if not all_configs:
                logger.info(
                    "No configurations found in repository, updating README to show clean state"
                )
                # Generate clean repository README
                yaml_header = self._generate_clean_yaml_header()
                readme_content = self._generate_clean_readme_content()
                full_readme = yaml_header + "\n" + readme_content
            else:
                # Generate dynamic YAML header with intelligent caching
                yaml_header = self._generate_dynamic_yaml_header(
                    self.generated_repo, all_configs, active_language=active_language
                )

                # Generate unified README content with intelligent caching
                readme_content = self._generate_dynamic_readme_content(
                    self.generated_repo, all_configs, active_language=active_language
                )

                # Combine YAML header and content
                full_readme = yaml_header + "\n" + readme_content

            # Update dataset card
            card = DatasetCard(full_readme)
            card.push_to_hub(self.generated_repo, token=self.hf_token)

            if all_configs:
                logger.info(
                    f"✅ Updated dataset card for {self.generated_repo} with {len(all_configs)} language configurations"
                )
            else:
                logger.info(
                    f"✅ Updated dataset card for {self.generated_repo} to show clean repository state"
                )

        except Exception as e:
            logger.error(f"Failed to update dataset card: {e}")
            # Don't raise - this shouldn't block the upload process

    def _get_configuration_stats(
        self, repo_id: str, config_name: str, force_refresh: bool = False
    ) -> dict:
        """
        Get statistics for a specific language configuration.

        Args:
            repo_id: Repository ID (e.g., 'selim-b-kh/buddhi-pragati-puzzles')
            config_name: Language configuration name (e.g., 'english', 'hindi')
            force_refresh: If True, force download fresh data. If False, reuse cache when possible.
        """
        try:
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

            stats = {
                "total_entries": len(df),
                "avg_density": float(df["density"].mean())
                if "density" in df.columns
                else 0.0,
                "avg_word_count": float(df["word_count"].mean())
                if "word_count" in df.columns
                else 0.0,
                "avg_quality_score": float(df["quality_score"].mean())
                if "quality_score" in df.columns
                else 0.0,
                "avg_context_score": float(df["context_score"].mean())
                if "context_score" in df.columns
                else 0.0,
                "grid_sizes": list(df["grid_size"].unique())
                if "grid_size" in df.columns
                else [],
                "num_bytes": dataset.info.dataset_size if dataset.info else 0,
                "download_size": dataset.info.download_size if dataset.info else 0,
            }

            logger.debug(
                f"Stats for {config_name}: {stats['total_entries']} entries, avg density: {stats['avg_density']:.1%}"
            )
            return stats

        except Exception as e:
            logger.warning(f"Could not get stats for {config_name}: {e}")
            return {
                "total_entries": 0,
                "avg_density": 0.0,
                "avg_word_count": 0.0,
                "avg_quality_score": 0.0,
                "avg_context_score": 0.0,
                "grid_sizes": [],
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

            # Dataset info section - simplified schema
            dataset_info_lines.extend(
                [
                    f"- config_name: {config}",
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
            "# Buddhi-Pragati Generated Crossword Puzzles Dataset",
            "",
            "This dataset contains crossword puzzles generated using memetic algorithms from the Buddhi-Pragati benchmark system.",
            "",
            "## Available Languages",
            "",
        ]

        total_entries = 0
        all_grid_sizes = set()

        for config in sorted(all_configs):
            # Use intelligent caching - force refresh only for active language
            force_refresh = active_language and config == active_language.lower()
            stats = self._get_configuration_stats(
                repo_id, config, force_refresh=force_refresh
            )
            total_entries += stats["total_entries"]
            all_grid_sizes.update(stats["grid_sizes"])

            content_lines.extend(
                [
                    f"### {config.title()}",
                    f"- **Total Puzzles**: {stats['total_entries']}",
                    f"- **Grid Sizes**: {', '.join(map(str, sorted(stats['grid_sizes'])))}",
                    f"- **Average Density**: {stats['avg_density']:.1%}",
                    f"- **Average Quality Score**: {stats['avg_quality_score']:.3f}",
                    f"- **Average Context Score**: {stats['avg_context_score']:.3f}",
                    "",
                ]
            )

        content_lines.extend(
            [
                "## Dataset Structure",
                "",
                "Each puzzle entry contains:",
                "- `id`: Unique puzzle identifier",
                "- `clues`: List of clue data with positions and directions",
                "- `empty_grid`: Grid with numbers and blocked cells for solving",
                "- `solved_grid`: Complete solution grid with all letters",
                "- `context_score`: Mean Indian cultural context score (0.0-1.0)",
                "- `quality_score`: Overall puzzle quality metric (0.0-1.0)",
                "- `source_mix`: Distribution of word sources used",
                "- `grid_size`: Grid size as integer for square grids (e.g., 10 for 10x10)",
                "- `density`: Actual grid fill density achieved (0.0-1.0)",
                "- `word_count`: Number of words placed in puzzle",
                "- `generation_metadata`: Algorithm and generation details",
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
                "",
                "# Access puzzle data",
                "puzzle = dataset['train'][0]",
                "print(f\"Puzzle ID: {puzzle['id']}\")",
                "print(f\"Grid size: {puzzle['grid_size']}x{puzzle['grid_size']}\")",
                "print(f\"Density: {puzzle['density']:.1%}\")",
                "print(f\"Word count: {puzzle['word_count']}\")",
                "```",
                "",
                "## Summary Statistics",
                "",
                f"- **Total Languages**: {len(all_configs)}",
                f"- **Total Puzzles**: {total_entries:,}",
                f"- **Grid Sizes Available**: {', '.join(map(str, sorted(all_grid_sizes)))}",
                "",
                "## Generation Method",
                "",
                "Puzzles are generated using memetic algorithms that combine:",
                "1. **Population-based search**: Multiple candidate grids compete",
                "2. **Cultural prioritization**: Indian context entries preferred",
                "3. **Genetic operators**: Crossover and mutation for optimization",
                "4. **Local search**: Hill-climbing for density improvement",
                "5. **Fitness evaluation**: Multi-objective scoring (density + intersections + cultural coherence)",
                "",
                "## Source Corpus",
                "",
                "Generated from clue-answer pairs in the Buddhi-Pragati corpus dataset:",
                "- **MILU**: Multi-choice questions from Indian examinations",
                "- **IndicWikiBio**: Biographical information from Wikipedia",
                "- **IndoWordNet**: Dictionary definitions and word relationships",
                "- **Bhasha-Wiki**: Named entities from multilingual Wikipedia",
                "",
            ]
        )

        return "\n".join(content_lines)

    def _generate_clean_readme_content(self) -> str:
        """Generate README content for clean repository (no configurations)."""
        return f"""# Buddhi-Pragati Generated Crossword Puzzles Dataset

This repository will contain crossword puzzles generated using memetic algorithms from the Buddhi-Pragati benchmark system.

## Status
Repository is currently clean and ready for new puzzle uploads.

## Planned Features
- **High Density Generation**: Target 75% grid fill using memetic algorithms
- **Cultural Prioritization**: Indian cultural entries preferred during generation
- **Wide Grid Size Support**: Supports 3x3 to 30x30 grid sizes
- **Multi-language**: Generated for 19 languages including Indic scripts
- **Quality Scoring**: Each puzzle includes quality and cultural context scores

## Dataset Structure

Each puzzle entry will contain:
- `id`: Unique puzzle identifier
- `clues`: List of clue data with positions and directions
- `empty_grid`: Grid with numbers and blocked cells for solving
- `solved_grid`: Complete solution grid with all letters
- `context_score`: Mean Indian cultural context score (0.0-1.0)
- `quality_score`: Overall puzzle quality metric (0.0-1.0)
- `source_mix`: Distribution of word sources used
- `grid_size`: Grid size as integer for square grids (e.g., 10 for 10x10)
- `density`: Actual grid fill density achieved (0.0-1.0)
- `word_count`: Number of words placed in puzzle
- `generation_metadata`: Algorithm and generation details

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

Automatically generated crossword puzzle dataset."""

    def get_upload_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about uploaded puzzles."""
        if not self.hf_api or not self.hf_token:
            return {"error": "HuggingFace API not available"}

        try:
            configs = self._scan_repository_configurations(self.generated_repo)

            stats = {
                "repository": self.generated_repo,
                "total_configurations": len(configs),
                "configurations": [],
                "total_puzzles": 0,
            }

            for config in configs:
                try:
                    config_stats = self._get_configuration_stats(
                        self.generated_repo, config
                    )
                    config_stats["name"] = config
                    stats["configurations"].append(config_stats)
                    stats["total_puzzles"] += config_stats["total_entries"]

                except Exception as e:
                    logger.error(f"Error getting stats for {config}: {e}")

            return stats

        except Exception as e:
            logger.error(f"Error getting upload statistics: {e}")
            return {"error": str(e)}

    def validate_upload(self, config_name: str, expected_count: int) -> bool:
        """
        Validate that upload was successful and entries went to correct configuration.

        Args:
            config_name: Configuration name to validate
            expected_count: Minimum expected number of entries

        Returns:
            True if validation passes, False otherwise
        """
        try:
            logger.info(f"Validating upload for configuration: {config_name}")

            # Clear cache and reload to get fresh data
            self._clear_dataset_cache(self.generated_repo)

            # Check if configuration exists
            configs = self._scan_repository_configurations(self.generated_repo)
            if config_name not in configs:
                logger.error(f"Configuration {config_name} not found after upload")
                return False

            # Get configuration statistics
            stats = self._get_configuration_stats(self.generated_repo, config_name)
            actual_count = stats["total_entries"]

            if actual_count >= expected_count:
                logger.info(
                    f"✅ Upload validation passed: {actual_count} entries in {config_name}"
                )
                return True
            else:
                logger.error(
                    f"❌ Upload validation failed: expected >= {expected_count}, got {actual_count}"
                )
                return False

        except Exception as e:
            logger.error(f"Upload validation failed: {e}")
            return False
