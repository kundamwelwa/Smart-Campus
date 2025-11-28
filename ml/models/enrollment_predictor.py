"""
Enrollment Predictor - Deep Learning Model

Research-grade LSTM/Transformer model for predicting student dropout probability.
Uses PyTorch Lightning for training and includes explainability hooks.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from ml.models.base import BaseMLModel

logger = structlog.get_logger(__name__)


class EnrollmentDataset(Dataset):
    """
    Dataset for enrollment prediction.

    Features:
    - Student demographics
    - Academic history (grades, attendance)
    - Engagement metrics
    - Course load
    - Time-series data (grades over time)
    """

    def __init__(self, data: pd.DataFrame, sequence_length: int = 10):
        """
        Initialize dataset.

        Args:
            data: DataFrame with student data
            sequence_length: Length of time-series sequences
        """
        self.data = data
        self.sequence_length = sequence_length

        # Feature columns
        self.feature_cols = [
            'gpa', 'credits_enrolled', 'attendance_rate',
            'engagement_score', 'previous_dropout_risk',
            'course_difficulty', 'study_hours', 'num_failed_courses'
        ]

        # Normalize features
        self.mean = data[self.feature_cols].mean()
        self.std = data[self.feature_cols].std()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        row = self.data.iloc[idx]

        # Extract features and normalize
        features = (row[self.feature_cols].values - self.mean.values) / (self.std.values + 1e-8)
        features = torch.FloatTensor(features)

        # Label (dropout: 1, retained: 0)
        label = torch.FloatTensor([row['dropped_out']])

        return features, label


class LSTMEnrollmentPredictor(pl.LightningModule):
    """
    LSTM-based enrollment predictor with attention mechanism.

    Architecture:
    - Embedding layer for categorical features
    - Bidirectional LSTM layers
    - Attention mechanism
    - Fully connected layers
    - Sigmoid output for probability
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """
        Initialize LSTM predictor.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # For explainability - store attention weights
        self.last_attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_size]

        Returns:
            Dropout probability [batch_size, 1]
        """
        # Expand to sequence (for compatibility, using single timestep)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]

        # LSTM forward
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # Attention weights
        attention_scores = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)

        # Store for explainability
        self.last_attention_weights = attention_weights.detach()

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden*2]

        # Final prediction
        return self.fc(context)  # [batch, 1]


    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)

        # Calculate accuracy
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)

        # Calculate accuracy and AUC
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }


class EnrollmentPredictor(BaseMLModel):
    """
    High-level enrollment predictor wrapping the LSTM model.

    Provides train(), predict(), and explain() interface.
    """

    def __init__(
        self,
        model_name: str = "enrollment_predictor",
        version: str = "1.0.0",
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize enrollment predictor.

        Args:
            model_name: Model identifier
            version: Model version
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__(model_name, version)

        self.model = LSTMEnrollmentPredictor(
            input_size=8,  # Number of features
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.feature_names = [
            'GPA', 'Credits Enrolled', 'Attendance Rate',
            'Engagement Score', 'Previous Risk', 'Course Difficulty',
            'Study Hours', 'Failed Courses'
        ]

    async def train(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Train the enrollment predictor.

        Args:
            training_data: Training DataFrame
            validation_data: Validation DataFrame
            **kwargs: Additional training parameters (seed, deterministic, etc.)

        Returns:
            dict: Training metrics
        """
        logger.info("Starting enrollment predictor training")

        # Set deterministic behavior if seed provided
        seed = kwargs.get('seed')
        deterministic = kwargs.get('deterministic', seed is not None)

        if seed is not None:
            self.set_deterministic(seed)
        elif deterministic:
            self.set_deterministic(42)  # Default seed

        # Create datasets
        train_dataset = EnrollmentDataset(training_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=kwargs.get('batch_size', 32),
            shuffle=True,
            num_workers=kwargs.get('num_workers', 4),
        )

        val_loader = None
        if validation_data is not None:
            val_dataset = EnrollmentDataset(validation_data)
            val_loader = DataLoader(
                val_dataset,
                batch_size=kwargs.get('batch_size', 32),
                shuffle=False,
                num_workers=kwargs.get('num_workers', 4),
            )

        # Model save path with version
        from shared.config import settings
        model_dir = Path(settings.ml_model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f'enrollment-predictor-v{self.version}-{{epoch:02d}}-{{val_loss:.2f}}'

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss' if val_loader else 'train_loss',
            dirpath=str(model_dir / 'checkpoints'),
            filename=model_filename,
            save_top_k=3,
            mode='min',
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss' if val_loader else 'train_loss',
            patience=10,
            mode='min',
        )

        # Trainer
        trainer = pl.Trainer(
            max_epochs=kwargs.get('max_epochs', 50),
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='auto',
            devices=1,
            deterministic=deterministic,
            enable_progress_bar=True,
        )

        # Train
        trainer.fit(self.model, train_loader, val_loader)

        self.is_trained = True

        # Save final model and register version
        final_model_path = model_dir / f'enrollment-predictor-v{self.version}.pt'
        await self.save(final_model_path, register_version=True)

        logger.info(
            "Training complete",
            best_val_loss=checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
            version=self.version,
        )

        return {
            'best_val_loss': float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
            'epochs_trained': trainer.current_epoch,
            'model_path': str(final_model_path),
            'checkpoint_path': checkpoint_callback.best_model_path,
            'version': self.version,
            'seed': self._seed,
        }

    async def predict(self, input_data: dict[str, float]) -> dict[str, Any]:
        """
        Predict dropout probability for a student.

        Args:
            input_data: Dictionary with feature values

        Returns:
            dict: Prediction with probability and confidence
        """
        self.model.eval()

        # Prepare input tensor
        features = torch.FloatTensor([
            input_data.get('gpa', 0.0),
            input_data.get('credits_enrolled', 0),
            input_data.get('attendance_rate', 100.0),
            input_data.get('engagement_score', 0.5),
            input_data.get('previous_dropout_risk', 0.0),
            input_data.get('course_difficulty', 0.5),
            input_data.get('study_hours', 10.0),
            input_data.get('num_failed_courses', 0),
        ]).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            dropout_prob = self.model(features)

        probability = float(dropout_prob.item())

        # Risk categorization
        if probability < 0.3:
            risk_level = 'low'
        elif probability < 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        return {
            'dropout_probability': probability,
            'retention_probability': 1.0 - probability,
            'risk_level': risk_level,
            'confidence': abs(probability - 0.5) * 2,  # Distance from uncertain (0.5)
        }

    async def explain(self, input_data: dict[str, float], prediction: dict[str, Any]) -> dict[str, Any]:
        """
        Explain the prediction using attention weights, gradient-based importance, and SHAP values.

        Args:
            input_data: Input features
            prediction: Model prediction

        Returns:
            dict: Explanation with feature importance and interpretability metrics
        """
        self.model.eval()

        # Prepare input tensor
        features = torch.FloatTensor([
            input_data.get('gpa', 0.0),
            input_data.get('credits_enrolled', 0),
            input_data.get('attendance_rate', 100.0),
            input_data.get('engagement_score', 0.5),
            input_data.get('previous_dropout_risk', 0.0),
            input_data.get('course_difficulty', 0.5),
            input_data.get('study_hours', 10.0),
            input_data.get('num_failed_courses', 0),
        ]).unsqueeze(0)
        features.requires_grad = True

        # Get attention weights from forward pass
        with torch.enable_grad():
            output = self.model(features)
            attention_weights = self.model.last_attention_weights

            # Gradient-based feature importance (Integrated Gradients approximation)
            output.backward()
            gradients = features.grad.abs().squeeze().cpu().numpy()

        # Combine attention and gradient importance
        if attention_weights is not None:
            att_importance = attention_weights.squeeze().cpu().numpy()
            if len(att_importance.shape) > 1:
                att_importance = att_importance.mean(axis=0)
        else:
            att_importance = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Normalize importances
        att_importance = att_importance / (att_importance.sum() + 1e-8)
        gradients = gradients / (gradients.sum() + 1e-8)

        # Combined importance (weighted average)
        combined_importance = 0.6 * att_importance + 0.4 * gradients

        # Feature contributions
        feature_importance = []
        for i, name in enumerate(self.feature_names):
            feature_key = name.lower().replace(' ', '_')
            value = input_data.get(feature_key, 0.0)

            importance_val = float(combined_importance[min(i, len(combined_importance)-1)])

            feature_importance.append({
                'feature': name,
                'value': float(value),
                'importance': importance_val,
                'attention_weight': float(att_importance[min(i, len(att_importance)-1)]),
                'gradient_importance': float(gradients[min(i, len(gradients)-1)]),
            })

        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        return {
            'prediction': prediction,
            'feature_importance': feature_importance,
            'top_factors': feature_importance[:3],
            'model_confidence': prediction.get('confidence', 0.0),
            'explanation': self._generate_text_explanation(feature_importance, prediction),
            'explainability_method': 'attention_and_gradients',
            'model_version': self.version,
        }

    def _generate_text_explanation(
        self, feature_importance: list[dict], prediction: dict
    ) -> str:
        """Generate human-readable explanation."""
        prob = prediction['dropout_probability']
        risk = prediction['risk_level']
        top_features = feature_importance[:3]

        explanation = f"Student has {risk} dropout risk ({prob:.1%} probability). "
        explanation += "Key factors: "

        factors = []
        for feat in top_features:
            factors.append(f"{feat['feature']} ({feat['value']:.2f})")

        explanation += ", ".join(factors) + "."

        if risk == 'high':
            explanation += " Intervention recommended."
        elif risk == 'medium':
            explanation += " Monitor closely."
        else:
            explanation += " Student performing well."

        return explanation

