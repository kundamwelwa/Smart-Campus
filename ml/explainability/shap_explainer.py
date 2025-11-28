"""
SHAP Explainability Wrapper

Provides model-agnostic explanations using SHAP (SHapley Additive exPlanations).
"""

from typing import Any

import numpy as np
import shap
import structlog
import torch

logger = structlog.get_logger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for PyTorch models.

    Provides:
    - Global feature importance
    - Local instance explanations
    - Visualizations (summary plots, force plots)
    """

    def __init__(self, model: torch.nn.Module, background_data: np.ndarray | None = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained PyTorch model
            background_data: Background dataset for SHAP (optional)
        """
        self.model = model
        self.model.eval()

        # Create wrapper for PyTorch model
        self.predict_fn = self._create_predict_function()

        # Initialize SHAP explainer
        if background_data is not None:
            self.explainer = shap.KernelExplainer(self.predict_fn, background_data)
        else:
            # Use a small sample as background
            self.explainer = None

        logger.info("SHAP explainer initialized")

    def _create_predict_function(self):
        """Create prediction function for SHAP."""
        def predict(x: np.ndarray) -> np.ndarray:
            """Predict function that SHAP can call."""
            self.model.eval()

            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                predictions = self.model(x_tensor)

                return predictions.numpy()

        return predict

    def explain_instance(
        self,
        instance: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            instance: Input instance (features)
            feature_names: List of feature names

        Returns:
            dict: Explanation with SHAP values and feature importance
        """
        if self.explainer is None:
            # Create explainer on-the-fly
            background = instance.reshape(1, -1)
            self.explainer = shap.KernelExplainer(self.predict_fn, background)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))

        # Extract feature importance
        feature_importance = []
        for i, name in enumerate(feature_names):
            feature_importance.append({
                'feature': name,
                'shap_value': float(shap_values[0][i]),
                'feature_value': float(instance[i]),
                'impact': 'positive' if shap_values[0][i] > 0 else 'negative',
            })

        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        return {
            'shap_values': shap_values[0].tolist(),
            'feature_importance': feature_importance,
            'top_features': feature_importance[:5],
            'explanation_method': 'SHAP (Shapley Additive Explanations)',
        }

    def global_importance(
        self,
        data: np.ndarray,
        feature_names: list[str],
        max_samples: int = 100,
    ) -> dict[str, Any]:
        """
        Calculate global feature importance across dataset.

        Args:
            data: Dataset to analyze
            feature_names: Feature names
            max_samples: Maximum samples to use (for performance)

        Returns:
            dict: Global feature importance
        """
        logger.info("Calculating global SHAP importance", samples=min(len(data), max_samples))

        # Sample data if too large
        if len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data

        # Create explainer if needed
        if self.explainer is None:
            background = data[:min(100, len(data))]
            self.explainer = shap.KernelExplainer(self.predict_fn, background)

        # Calculate SHAP values for sample
        shap_values = self.explainer.shap_values(sample_data)

        # Average absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance ranking
        importance_ranking = []
        for i, name in enumerate(feature_names):
            importance_ranking.append({
                'feature': name,
                'mean_abs_shap_value': float(mean_abs_shap[i]),
                'rank': i + 1,
            })

        # Sort by importance
        importance_ranking.sort(key=lambda x: x['mean_abs_shap_value'], reverse=True)

        # Re-rank
        for i, item in enumerate(importance_ranking):
            item['rank'] = i + 1

        return {
            'global_feature_importance': importance_ranking,
            'top_5_features': importance_ranking[:5],
            'samples_analyzed': len(sample_data),
            'method': 'SHAP global importance',
        }

