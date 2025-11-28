"""
Training Pipeline for Enrollment Predictor

Trains the LSTM enrollment predictor on synthetic data with proper validation.
"""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from ml.datasets.synthetic_generator import SyntheticDataGenerator
from ml.models.enrollment_predictor import EnrollmentPredictor

logger = structlog.get_logger(__name__)


async def train_enrollment_predictor(
    data_path: str | None = None,
    output_model_path: str = 'ml/models/saved/enrollment_predictor.pt',
    test_size: float = 0.2,
    val_size: float = 0.1,
    max_epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Complete training pipeline for enrollment predictor.

    Args:
        data_path: Path to training data CSV (if None, generates synthetic)
        output_model_path: Where to save trained model
        test_size: Test set proportion
        val_size: Validation set proportion
        max_epochs: Maximum training epochs
        batch_size: Batch size for training

    Returns:
        dict: Training results and metrics
    """
    logger.info("Starting enrollment predictor training pipeline")

    # Load or generate data
    if data_path and Path(data_path).exists():
        logger.info("Loading data from file", path=data_path)
        df = pd.read_csv(data_path)
    else:
        logger.info("Generating synthetic data")
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_student_data(num_students=10000)

    logger.info(
        "Data loaded",
        total_samples=len(df),
        dropout_rate=f"{df['dropped_out'].mean():.1%}",
    )

    # Split data
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['dropped_out']
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size / (1 - test_size), random_state=42, stratify=train_val_df['dropped_out']
    )

    logger.info(
        "Data split",
        train=len(train_df),
        val=len(val_df),
        test=len(test_df),
    )

    # Create model
    predictor = EnrollmentPredictor(
        model_name="enrollment_predictor_v1",
        version="1.0.0",
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )

    # Set deterministic for reproducibility
    predictor.set_deterministic(seed=42)

    # Train model
    training_results = await predictor.train(
        training_data=train_df,
        validation_data=val_df,
        max_epochs=max_epochs,
        batch_size=batch_size,
        deterministic=True,
    )

    # Save model
    output_path = Path(output_model_path)
    await predictor.save(output_path)

    logger.info("Model saved", path=str(output_path))

    # Evaluate on test set
    logger.info("Evaluating on test set")

    test_predictions = []
    test_labels = []

    for _idx, row in test_df.iterrows():
        input_data = {
            'gpa': row['current_gpa'],
            'credits_enrolled': row['credits_enrolled'],
            'attendance_rate': row['attendance_rate'],
            'engagement_score': row['engagement_score'],
            'previous_dropout_risk': row['previous_dropout_risk'],
            'course_difficulty': row['course_difficulty_avg'],
            'study_hours': row['study_hours_per_week'],
            'num_failed_courses': row['num_failed_courses'],
        }

        pred = await predictor.predict(input_data)
        test_predictions.append(pred['dropout_probability'])
        test_labels.append(row['dropped_out'])

    # Calculate test metrics
    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)

    # Accuracy
    test_acc = ((test_predictions > 0.5) == test_labels).mean()

    # AUC-ROC (simplified calculation)
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(test_labels, test_predictions)

    logger.info(
        "Test evaluation complete",
        test_accuracy=f"{test_acc:.3f}",
        test_auc=f"{test_auc:.3f}",
    )

    results = {
        'training': training_results,
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'model_path': str(output_path),
        'dataset_size': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
        },
    }

    print("\n" + "="*60)
    print("🎓 ENROLLMENT PREDICTOR TRAINING COMPLETE")
    print("="*60)
    print(f"✅ Test Accuracy: {test_acc:.1%}")
    print(f"✅ Test AUC-ROC: {test_auc:.3f}")
    print(f"✅ Model saved: {output_path}")
    print("="*60 + "\n")

    return results


if __name__ == '__main__':
    # Run training
    results = asyncio.run(train_enrollment_predictor())
    print("Training results:", results)

