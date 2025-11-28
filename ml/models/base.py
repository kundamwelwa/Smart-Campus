"""
Base ML Model Abstract Class

Defines common interface for all ML models with train(), predict(), and explain() methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog
import torch

from ml.models.versioning import ModelRegistry, get_model_registry

logger = structlog.get_logger(__name__)


class BaseMLModel(ABC):
    """
    Abstract base class for all ML models.

    Provides common interface for training, prediction, and explainability.
    All ML models in Argos inherit from this class.
    """

    def __init__(self, model_name: str, version: str = "1.0.0"):
        """
        Initialize ML model.

        Args:
            model_name: Model identifier
            version: Model version
        """
        self.model_name = model_name
        self.version = version
        self.model: Any | None = None
        self.is_trained = False
        self.registry: ModelRegistry | None = None
        self._seed: int | None = None

    @abstractmethod
    async def train(self, training_data: Any, validation_data: Any | None = None, **kwargs) -> dict[str, Any]:
        """
        Train the model on provided data.

        Args:
            training_data: Training dataset
            validation_data: Validation dataset (optional)
            **kwargs: Additional training parameters

        Returns:
            dict: Training metrics and results
        """

    @abstractmethod
    async def predict(self, input_data: Any) -> Any:
        """
        Make predictions on input data.

        Args:
            input_data: Input features

        Returns:
            Model predictions
        """

    @abstractmethod
    async def explain(self, input_data: Any, prediction: Any) -> dict[str, Any]:
        """
        Explain a prediction (explainable AI).

        Provides feature importance, attention weights, or other
        interpretability information for the prediction.

        Args:
            input_data: Input that was predicted
            prediction: Model's prediction

        Returns:
            dict: Explanation data (feature importance, visualizations, etc.)
        """

    async def save(self, path: Path, register_version: bool = True) -> None:
        """
        Save model to disk and optionally register version.

        Args:
            path: Path to save model
            register_version: Whether to register this version in the model registry
        """
        if self.model is None:
            raise ValueError("No model to save")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        if isinstance(self.model, torch.nn.Module):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'version': self.version,
                'is_trained': self.is_trained,
                'seed': self._seed,
            }, path)

            logger.info("Model saved", model_name=self.model_name, path=str(path))

        # Register version in registry
        if register_version:
            if self.registry is None:
                self.registry = get_model_registry()

            metadata = {
                'is_trained': self.is_trained,
                'seed': self._seed,
            }

            self.registry.register_version(
                model_name=self.model_name,
                version=self.version,
                model_path=path,
                metadata=metadata,
            )

    async def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to saved model
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load PyTorch model
        checkpoint = torch.load(path)

        if self.model is not None and isinstance(self.model, torch.nn.Module):
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = checkpoint.get('is_trained', True)

            logger.info("Model loaded", model_name=self.model_name, path=str(path))

    def set_deterministic(self, seed: int = 42) -> None:
        """
        Set deterministic behavior for reproducibility.

        Seeds all random number generators (PyTorch, NumPy, Python).
        Critical for unit tests and reproducible research.

        Args:
            seed: Random seed value
        """
        import random

        import numpy as np

        # Set Python random seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._seed = seed
        logger.info("Deterministic mode enabled", seed=seed)
