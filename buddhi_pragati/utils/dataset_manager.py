"""
Dataset Management Utility for Buddhi-Pragati HuggingFace Operations

This module provides utilities for managing HuggingFace datasets including deletion
of specific language configurations (subsets), dataset inspection, and maintenance
operations.

Key Features:
- Delete specific language configurations from multi-config datasets
- List and inspect dataset configurations and file structure
- Update dataset cards and metadata
- Backup and restore functionality for dataset configurations

Architecture:
- DatasetManager: Main orchestrator for dataset operations
- Configuration-aware operations using crossword_config.txt
- Safe deletion with backup options
- Comprehensive logging and error handling

Note: Since HuggingFace Hub doesn't directly support subset deletion via API,
this utility implements workarounds by manually managing dataset files and
configurations.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset, get_dataset_config_names
from huggingface_hub import HfApi, login, list_repo_files

from .config_loader import get_config

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Comprehensive dataset management utility for HuggingFace operations.

    Handles deletion, inspection, and maintenance of multi-configuration
    datasets with language-specific subsets.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize dataset manager with HuggingFace credentials.

        Args:
            hf_token: HuggingFace token for authentication
        """
        self.config = get_config()
        self.dataset_config = self.config.get_dataset_config()
        self.hf_token = (
            hf_token or os.getenv("HF_TOKEN") or self.config.get("DEFAULT_HF_TOKEN")
        )

        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN env var or pass hf_token parameter."
            )

        # Initialize HF API
        login(token=self.hf_token)
        self.hf_api = HfApi()

        # Default repository from config
        self.default_repo = self.dataset_config.get(
            "HF_DATASET_REPO", "selim-b-kh/Buddhi_pragati"
        )

        logger.info(f"DatasetManager initialized for repository: {self.default_repo}")

    def list_dataset_configurations(self, repo_id: Optional[str] = None) -> List[str]:
        """
        List all available configurations (language subsets) in the dataset.

        Args:
            repo_id: Repository ID (default from config)

        Returns:
            List of configuration names
        """
        repo_id = repo_id or self.default_repo

        try:
            configs = get_dataset_config_names(repo_id, token=self.hf_token)
            logger.info(f"Found {len(configs)} configurations in {repo_id}: {configs}")
            return configs
        except Exception as e:
            logger.error(f"Failed to get configurations for {repo_id}: {e}")
            return []

    def inspect_dataset_structure(self, repo_id: Optional[str] = None) -> Dict:
        """
        Inspect complete dataset structure including files and configurations.

        Args:
            repo_id: Repository ID (default from config)

        Returns:
            Dictionary with dataset structure information
        """
        repo_id = repo_id or self.default_repo

        try:
            # Get all files in repository
            repo_files = list_repo_files(
                repo_id, repo_type="dataset", token=self.hf_token
            )

            # Get configurations
            configs = self.list_dataset_configurations(repo_id)

            # Organize by file type and configuration
            structure = {
                "repository": repo_id,
                "configurations": configs,
                "files": {
                    "data_files": [],
                    "config_files": [],
                    "metadata_files": [],
                    "other_files": [],
                },
                "config_file_mapping": {},
            }

            for file_path in repo_files:
                if file_path.endswith((".parquet", ".json", ".jsonl", ".csv")):
                    structure["files"]["data_files"].append(file_path)

                    # Try to map to configuration
                    for config in configs:
                        if config in file_path:
                            if config not in structure["config_file_mapping"]:
                                structure["config_file_mapping"][config] = []
                            structure["config_file_mapping"][config].append(file_path)

                elif file_path in ["README.md", "dataset_infos.json"]:
                    structure["files"]["metadata_files"].append(file_path)
                elif file_path.endswith(".yaml") or "config" in file_path:
                    structure["files"]["config_files"].append(file_path)
                else:
                    structure["files"]["other_files"].append(file_path)

            logger.info(f"Dataset structure inspection complete for {repo_id}")
            logger.info(
                f"Configurations: {len(configs)}, Total files: {len(repo_files)}"
            )

            return structure

        except Exception as e:
            logger.error(f"Failed to inspect dataset structure for {repo_id}: {e}")
            return {"error": str(e)}

    def backup_configuration(
        self,
        config_name: str,
        repo_id: Optional[str] = None,
        backup_dir: Optional[str] = None,
    ) -> str:
        """
        Create a backup of a specific configuration before deletion.

        Args:
            config_name: Configuration name to backup
            repo_id: Repository ID (default from config)
            backup_dir: Local directory for backup (default: ./backups)

        Returns:
            Path to backup directory
        """
        repo_id = repo_id or self.default_repo
        backup_dir = backup_dir or "./dataset_backups"

        backup_path = (
            Path(backup_dir) / f"{repo_id.replace('/', '_')}_{config_name}_backup"
        )
        backup_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Creating backup for {config_name} configuration...")

            # Load and save the dataset
            dataset = load_dataset(
                repo_id, config_name, split="train", token=self.hf_token
            )
            dataset_path = backup_path / "dataset.json"

            # Convert to pandas and save as JSON for easy restoration
            df = dataset.to_pandas()
            df.to_json(dataset_path, orient="records", indent=2)

            # Save metadata
            metadata = {
                "repo_id": repo_id,
                "config_name": config_name,
                "num_rows": len(df),
                "columns": list(df.columns),
                "backup_timestamp": pd.Timestamp.now().isoformat(),
            }

            metadata_path = backup_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Backup created successfully at {backup_path}")
            logger.info(f"Backed up {len(df)} entries for {config_name}")

            return str(backup_path)

        except Exception as e:
            logger.error(f"Failed to create backup for {config_name}: {e}")
            raise

    def delete_configuration(
        self,
        config_name: str,
        repo_id: Optional[str] = None,
        create_backup: bool = True,
    ) -> bool:
        """
        Delete a specific configuration (language subset) from the dataset.

        Since HuggingFace doesn't support direct subset deletion, this method:
        1. Creates a backup if requested
        2. Identifies all files associated with the configuration
        3. Deletes those files from the repository
        4. Updates dataset metadata

        Args:
            config_name: Configuration name to delete (e.g., 'hindi')
            repo_id: Repository ID (default from config)
            create_backup: Whether to create backup before deletion

        Returns:
            True if deletion successful, False otherwise
        """
        repo_id = repo_id or self.default_repo

        logger.info(
            f"Starting deletion process for '{config_name}' configuration from {repo_id}"
        )

        # Safety check - confirm configuration exists
        existing_configs = self.list_dataset_configurations(repo_id)
        if config_name not in existing_configs:
            logger.error(f"Configuration '{config_name}' not found in {repo_id}")
            logger.info(f"Available configurations: {existing_configs}")
            return False

        # Create backup if requested
        backup_path = None
        if create_backup:
            try:
                backup_path = self.backup_configuration(config_name, repo_id)
                logger.info(f"✅ Backup created at: {backup_path}")
            except Exception as e:
                logger.error(f"Backup failed: {e}")
                return False

        try:
            # Inspect dataset structure to identify files to delete
            structure = self.inspect_dataset_structure(repo_id)

            if "error" in structure:
                logger.error(
                    f"Failed to inspect dataset structure: {structure['error']}"
                )
                return False

            # Identify files associated with this configuration
            files_to_delete = structure["config_file_mapping"].get(config_name, [])

            if not files_to_delete:
                logger.warning(f"No files found for configuration '{config_name}'")
                logger.info(
                    "Attempting to delete by rebuilding dataset without this configuration..."
                )
                return self._rebuild_dataset_without_config(
                    config_name, repo_id, existing_configs
                )

            logger.info(
                f"Found {len(files_to_delete)} files to delete for '{config_name}':"
            )
            for file_path in files_to_delete:
                logger.info(f"  - {file_path}")

            # Delete files one by one
            deleted_files = []
            failed_files = []

            for file_path in files_to_delete:
                try:
                    self.hf_api.delete_file(
                        path_in_repo=file_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=self.hf_token,
                    )
                    deleted_files.append(file_path)
                    logger.info(f"✅ Deleted: {file_path}")

                except Exception as e:
                    logger.error(f"❌ Failed to delete {file_path}: {e}")
                    failed_files.append(file_path)

            # Update dataset card to remove references to deleted configuration
            self._update_dataset_card_after_deletion(config_name, repo_id)

            # Report results
            logger.info(f"DELETION SUMMARY for '{config_name}':")
            logger.info(f"  ✅ Successfully deleted: {len(deleted_files)} files")
            if failed_files:
                logger.warning(f"  ❌ Failed to delete: {len(failed_files)} files")
                for file_path in failed_files:
                    logger.warning(f"    - {file_path}")

            if backup_path:
                logger.info(f"  💾 Backup available at: {backup_path}")

            success = len(failed_files) == 0
            if success:
                logger.info(
                    f"🎉 Configuration '{config_name}' successfully deleted from {repo_id}"
                )
            else:
                logger.error(
                    f"⚠️  Partial deletion completed with {len(failed_files)} failures"
                )

            return success

        except Exception as e:
            logger.error(f"Deletion process failed: {e}")
            import traceback

            logger.error(f"Full error: {traceback.format_exc()}")
            return False

    def _rebuild_dataset_without_config(
        self, config_to_exclude: str, repo_id: str, existing_configs: List[str]
    ) -> bool:
        """
        Rebuild entire dataset excluding a specific configuration.

        This is a fallback method when file-by-file deletion doesn't work.
        """
        logger.info(
            f"Rebuilding dataset {repo_id} without '{config_to_exclude}' configuration..."
        )

        try:
            # Load all configurations except the one to exclude
            remaining_configs = [c for c in existing_configs if c != config_to_exclude]

            if not remaining_configs:
                logger.error("Cannot delete the only configuration in the dataset")
                return False

            logger.info(f"Preserving configurations: {remaining_configs}")

            # For each remaining configuration, load and re-upload
            for config_name in remaining_configs:
                logger.info(f"Preserving configuration: {config_name}")

                # Load existing data
                dataset = load_dataset(
                    repo_id, config_name, split="train", token=self.hf_token
                )

                # Re-upload to preserve it
                dataset.push_to_hub(
                    repo_id, config_name=config_name, split="train", token=self.hf_token
                )

                logger.info(
                    f"✅ Preserved configuration '{config_name}' with {len(dataset)} entries"
                )

            logger.info(
                f"Dataset rebuild completed. '{config_to_exclude}' has been excluded."
            )
            return True

        except Exception as e:
            logger.error(f"Dataset rebuild failed: {e}")
            return False

    def _update_dataset_card_after_deletion(self, deleted_config: str, repo_id: str):
        """Update dataset card with dynamic README system after configuration deletion."""
        try:
            # Import the dynamic README system from dataset_builder
            from buddhi_pragati.data.dataset_builder import DatasetBuilder

            # Create a temporary DatasetBuilder instance to access dynamic README methods
            temp_builder = DatasetBuilder(hf_token=self.hf_token)

            # Use the dynamic README update system
            temp_builder._update_dataset_card_incremental("dummy", repo_id)

            logger.info(
                f"✅ Updated dataset card using dynamic system after deleting '{deleted_config}'"
            )

        except Exception as e:
            logger.warning(f"Failed to update dataset card with dynamic system: {e}")
            # Fallback to simple cleanup for empty repository
            self._create_empty_repository_readme(repo_id)

    def _create_empty_repository_readme(self, repo_id: str):
        """Create a clean README for empty repository as fallback."""
        try:
            from huggingface_hub import DatasetCard

            empty_readme = """---
license: apache-2.0
---
# Buddhi-Pragati Crossword Dataset

This dataset contains crossword clue-answer pairs extracted from multiple Indian language sources.

## Available Languages

*No language configurations currently available. Upload data to populate this dataset.*

## Dataset Structure

Each entry contains:
- `id`: Unique identifier
- `clue`: Crossword clue text
- `answer`: Single-word answer (uppercase)
- `source`: Original data source (MILU, IndicWikiBio, IndoWordNet, Bhasha-Wiki)
- `source_id`: Original ID in source dataset
- `context_score`: Indian cultural context score (0.0-1.0)
- `quality_score`: Crossword suitability score (0.0-1.0)

Automatically generated crossword benchmark dataset."""

            card = DatasetCard(empty_readme)
            card.push_to_hub(repo_id, token=self.hf_token)
            logger.info("✅ Created empty repository README as fallback")

        except Exception as e:
            logger.error(f"Failed to create empty repository README: {e}")

    def restore_configuration(
        self, backup_path: str, repo_id: Optional[str] = None
    ) -> bool:
        """
        Restore a configuration from backup.

        Args:
            backup_path: Path to backup directory
            repo_id: Target repository ID

        Returns:
            True if restoration successful
        """
        repo_id = repo_id or self.default_repo
        backup_path = Path(backup_path)

        if not backup_path.exists():
            logger.error(f"Backup path does not exist: {backup_path}")
            return False

        try:
            # Load backup metadata
            metadata_path = backup_path / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            config_name = metadata["config_name"]
            logger.info(f"Restoring configuration '{config_name}' to {repo_id}")

            # Load backup data
            dataset_path = backup_path / "dataset.json"
            df = pd.read_json(dataset_path, orient="records")

            # Convert to HF dataset and upload
            dataset = Dataset.from_pandas(df)
            dataset.push_to_hub(
                repo_id, config_name=config_name, split="train", token=self.hf_token
            )

            logger.info(
                f"✅ Successfully restored '{config_name}' with {len(df)} entries"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to restore configuration: {e}")
            return False

    def get_configuration_stats(
        self, config_name: str, repo_id: Optional[str] = None
    ) -> Dict:
        """
        Get detailed statistics for a specific configuration.

        Args:
            config_name: Configuration name
            repo_id: Repository ID

        Returns:
            Dictionary with configuration statistics
        """
        repo_id = repo_id or self.default_repo

        try:
            dataset = load_dataset(
                repo_id, config_name, split="train", token=self.hf_token
            )
            df = dataset.to_pandas()

            stats = {
                "config_name": config_name,
                "repo_id": repo_id,
                "total_entries": len(df),
                "columns": list(df.columns),
                "sources": df["source"].value_counts().to_dict()
                if "source" in df.columns
                else {},
                "quality_stats": {
                    "mean_quality": df["quality_score"].mean()
                    if "quality_score" in df.columns
                    else None,
                    "min_quality": df["quality_score"].min()
                    if "quality_score" in df.columns
                    else None,
                    "max_quality": df["quality_score"].max()
                    if "quality_score" in df.columns
                    else None,
                },
                "answer_length_stats": {
                    "mean_length": df["answer"].str.len().mean()
                    if "answer" in df.columns
                    else None,
                    "min_length": df["answer"].str.len().min()
                    if "answer" in df.columns
                    else None,
                    "max_length": df["answer"].str.len().max()
                    if "answer" in df.columns
                    else None,
                },
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats for {config_name}: {e}")
            return {"error": str(e)}

    def print_repository_overview(self, repo_id: Optional[str] = None):
        """Print a comprehensive overview of the repository."""
        repo_id = repo_id or self.default_repo

        print(f"\n{'=' * 60}")
        print(f"DATASET REPOSITORY OVERVIEW: {repo_id}")
        print(f"{'=' * 60}")

        # Get configurations
        configs = self.list_dataset_configurations(repo_id)
        print(f"\n📊 CONFIGURATIONS ({len(configs)} total):")

        for config in configs:
            stats = self.get_configuration_stats(config, repo_id)
            if "error" not in stats:
                print(f"  • {config}: {stats['total_entries']} entries")
                if stats["sources"]:
                    sources = ", ".join(stats["sources"].keys())
                    print(f"    Sources: {sources}")
            else:
                print(f"  • {config}: Error loading stats")

        # Get structure
        structure = self.inspect_dataset_structure(repo_id)
        if "error" not in structure:
            print("\n REPOSITORY FILES:")
            print(f"  • Data files: {len(structure['files']['data_files'])}")
            print(f"  • Metadata files: {len(structure['files']['metadata_files'])}")
            print(f"  • Config files: {len(structure['files']['config_files'])}")
            print(f"  • Other files: {len(structure['files']['other_files'])}")

        print("\n✅ Repository overview complete")
