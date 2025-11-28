"""
Model Versioning System

Manages ML model versions, metadata, and deployment tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ModelVersion:
    """Represents a versioned ML model."""

    def __init__(
        self,
        model_name: str,
        version: str,
        model_path: Path,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize model version.

        Args:
            model_name: Model identifier
            version: Version string (e.g., "1.0.0")
            model_path: Path to model file
            metadata: Additional metadata
        """
        self.model_name = model_name
        self.version = version
        self.model_path = model_path
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "model_path": str(self.model_path),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class ModelRegistry:
    """Registry for managing model versions."""

    def __init__(self, registry_path: Path):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = registry_path / "registry.json"
        self._versions: dict[str, list[ModelVersion]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)
                    for model_name, versions in data.items():
                        self._versions[model_name] = [
                            ModelVersion(
                                model_name=v['model_name'],
                                version=v['version'],
                                model_path=Path(v['model_path']),
                                metadata=v.get('metadata', {}),
                            )
                            for v in versions
                        ]
            except Exception as e:
                logger.warning("Failed to load registry", error=str(e))
                self._versions = {}
        else:
            self._versions = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {
            model_name: [v.to_dict() for v in versions]
            for model_name, versions in self._versions.items()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_version(
        self,
        model_name: str,
        version: str,
        model_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Model identifier
            version: Version string
            model_path: Path to model file
            metadata: Additional metadata

        Returns:
            ModelVersion instance
        """
        version_obj = ModelVersion(model_name, version, model_path, metadata)

        if model_name not in self._versions:
            self._versions[model_name] = []

        self._versions[model_name].append(version_obj)
        self._save_registry()

        logger.info(
            "Model version registered",
            model_name=model_name,
            version=version,
            path=str(model_path),
        )

        return version_obj

    def get_latest_version(self, model_name: str) -> ModelVersion | None:
        """Get latest version of a model."""
        if model_name not in self._versions or not self._versions[model_name]:
            return None

        versions = self._versions[model_name]
        # Sort by created_at, return most recent
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions[0]

    def get_all_versions(self, model_name: str) -> list[ModelVersion]:
        """Get all versions of a model."""
        return self._versions.get(model_name, [])

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._versions.keys())


# Global registry instance
_model_registry: ModelRegistry | None = None


def get_model_registry(registry_path: Path | None = None) -> ModelRegistry:
    """
    Get or create global model registry.

    Args:
        registry_path: Optional custom registry path

    Returns:
        ModelRegistry instance
    """
    global _model_registry

    if _model_registry is None:
        if registry_path is None:
            from shared.config import settings
            registry_path = Path(settings.ml_model_path) / "registry"
        _model_registry = ModelRegistry(registry_path)

    return _model_registry

