"""
Machine Learning Model Abstraction

Provides abstract MLModel wrapper with training, prediction, explainability,
and versioning support. Enables pluggable ML frameworks and model management.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ModelStatus(str, Enum):
    """ML model lifecycle status."""

    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelFramework(str, Enum):
    """Supported ML frameworks."""

    SCIKIT_LEARN = "scikit_learn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    CUSTOM = "custom"


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    auc_roc: float | None = Field(default=None, ge=0.0, le=1.0)
    mse: float | None = Field(default=None, ge=0.0)
    mae: float | None = Field(default=None, ge=0.0)
    r2_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    custom_metrics: dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            **self.model_dump(exclude={"custom_metrics"}),
            **self.custom_metrics,
        }


class ModelVersion(BaseModel):
    """Model version information."""

    id: UUID = Field(default_factory=uuid4)
    version: str = Field(..., description="Semantic version (e.g., 1.0.0)")
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    trained_by: UUID | None = Field(default=None, description="User who trained model")
    training_dataset_id: UUID | None = Field(default=None)
    training_samples: int = Field(default=0, ge=0)
    metrics: ModelMetrics = Field(default_factory=ModelMetrics)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    artifacts_path: str | None = Field(default=None, description="Path to model files")
    notes: str | None = Field(default=None, max_length=2000)


class Explanation(BaseModel):
    """Model prediction explanation."""

    prediction: Any = Field(..., description="Model prediction")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_importance: dict[str, float] = Field(
        default_factory=dict, description="Feature importance scores"
    )
    explanation_method: str = Field(..., description="Explainability method used")
    explanation_data: dict[str, Any] = Field(
        default_factory=dict, description="Method-specific explanation data"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MLModel(ABC):
    """
    Abstract base class for machine learning models.

    Provides standardized interface for training, prediction, explainability,
    and versioning across different ML frameworks.
    """

    def __init__(
        self,
        model_id: UUID,
        name: str,
        framework: ModelFramework,
        model_type: str,
    ):
        """
        Initialize ML model.

        Args:
            model_id: Unique model identifier
            name: Model name
            framework: ML framework used
            model_type: Model type (classifier, regressor, etc.)
        """
        self.model_id = model_id
        self.name = name
        self.framework = framework
        self.model_type = model_type
        self.status = ModelStatus.UNTRAINED
        self.current_version: ModelVersion | None = None
        self.version_history: list[ModelVersion] = []
        self.error: str | None = None

    @abstractmethod
    async def train(
        self,
        X: Any,
        y: Any,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelVersion:
        """
        Train the model on provided data.

        Args:
            X: Training features
            y: Training labels
            hyperparameters: Model hyperparameters
            **kwargs: Additional training arguments

        Returns:
            ModelVersion: Version info for trained model

        Raises:
            ValueError: If training fails
        """

    @abstractmethod
    async def predict(self, X: Any) -> Any:
        """
        Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predictions (format depends on model type)

        Raises:
            ValueError: If model not trained
        """

    @abstractmethod
    async def predict_proba(self, X: Any) -> Any:
        """
        Predict class probabilities (for classifiers).

        Args:
            X: Features to predict on

        Returns:
            Class probabilities

        Raises:
            ValueError: If model not trained or not a classifier
        """

    @abstractmethod
    async def explain(
        self, X: Any, method: str = "shap", **kwargs: Any
    ) -> list[Explanation]:
        """
        Generate explanations for predictions.

        Args:
            X: Features to explain
            method: Explainability method (shap, lime, feature_importance)
            **kwargs: Method-specific arguments

        Returns:
            List of explanations for each sample
        """

    @abstractmethod
    async def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model artifacts
        """

    @abstractmethod
    async def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory path containing model artifacts
        """

    async def evaluate(self, X: Any, y_true: Any) -> ModelMetrics:
        """
        Evaluate model performance.

        Args:
            X: Test features
            y_true: True labels

        Returns:
            ModelMetrics: Performance metrics
        """
        # Default implementation - subclasses can override
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )

        predictions = await self.predict(X)

        metrics = ModelMetrics()

        # Try classification metrics
        try:
            metrics.accuracy = float(accuracy_score(y_true, predictions))
            metrics.precision = float(
                precision_score(y_true, predictions, average="weighted")
            )
            metrics.recall = float(
                recall_score(y_true, predictions, average="weighted")
            )
            metrics.f1_score = float(
                f1_score(y_true, predictions, average="weighted")
            )
        except Exception:
            # Try regression metrics
            try:
                metrics.mse = float(mean_squared_error(y_true, predictions))
                metrics.mae = float(mean_absolute_error(y_true, predictions))
                metrics.r2_score = float(r2_score(y_true, predictions))
            except Exception as e:
                logger.warning("Could not compute metrics", error=str(e))

        return metrics

    def create_version(
        self,
        version: str,
        trained_by: UUID | None,
        training_samples: int,
        metrics: ModelMetrics,
        hyperparameters: dict[str, Any],
    ) -> ModelVersion:
        """
        Create a new model version.

        Args:
            version: Semantic version string
            trained_by: User ID who trained model
            training_samples: Number of training samples
            metrics: Performance metrics
            hyperparameters: Model hyperparameters

        Returns:
            ModelVersion: New version object
        """
        model_version = ModelVersion(
            version=version,
            trained_by=trained_by,
            training_samples=training_samples,
            metrics=metrics,
            hyperparameters=hyperparameters,
        )

        self.version_history.append(model_version)
        self.current_version = model_version
        self.status = ModelStatus.TRAINED

        logger.info(
            "Model version created",
            model=self.name,
            version=version,
            metrics=metrics.to_dict(),
        )

        return model_version

    def get_info(self) -> dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model info
        """
        return {
            "model_id": str(self.model_id),
            "name": self.name,
            "framework": self.framework.value,
            "model_type": self.model_type,
            "status": self.status.value,
            "current_version": self.current_version.version
            if self.current_version
            else None,
            "total_versions": len(self.version_history),
            "error": self.error,
        }

    def get_latest_metrics(self) -> ModelMetrics | None:
        """Get metrics from latest version."""
        if self.current_version:
            return self.current_version.metrics
        return None


class ModelRegistry:
    """
    Registry for managing multiple ML models.

    Provides centralized model management, versioning, and deployment.
    """

    def __init__(self):
        """Initialize model registry."""
        self.models: dict[UUID, MLModel] = {}
        self.models_by_name: dict[str, UUID] = {}

    def register_model(self, model: MLModel) -> None:
        """
        Register a model in the registry.

        Args:
            model: ML model to register
        """
        self.models[model.model_id] = model
        self.models_by_name[model.name] = model.model_id
        logger.info("Model registered", model=model.name, id=str(model.model_id))

    def unregister_model(self, model_id: UUID) -> bool:
        """
        Unregister a model.

        Args:
            model_id: Model ID to unregister

        Returns:
            bool: True if unregistered successfully
        """
        if model_id not in self.models:
            return False

        model = self.models[model_id]
        del self.models[model_id]
        del self.models_by_name[model.name]
        logger.info("Model unregistered", model=model.name, id=str(model_id))
        return True

    def get_model(self, model_id: UUID) -> MLModel | None:
        """Get model by ID."""
        return self.models.get(model_id)

    def get_model_by_name(self, name: str) -> MLModel | None:
        """Get model by name."""
        model_id = self.models_by_name.get(name)
        if model_id:
            return self.models.get(model_id)
        return None

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model info dictionaries
        """
        return [model.get_info() for model in self.models.values()]

    def get_deployed_models(self) -> list[MLModel]:
        """Get all deployed models."""
        return [m for m in self.models.values() if m.status == ModelStatus.DEPLOYED]

