from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import AppConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


def _build_model(config: AppConfig) -> Pipeline:
    """Build classifier pipeline with scaler + LDA or SVM."""
    clf_algorithm = config.classifier.algorithm.lower()
    
    if clf_algorithm == "lda":
        clf = LinearDiscriminantAnalysis()
    elif clf_algorithm == "svm":
        clf = SVC(kernel="linear")
    else:
        raise ValueError(f"Unknown classifier: {clf_algorithm}")
    
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    config: AppConfig,
    save_model: bool = True,
) -> Tuple[Pipeline, float]:
    """Train classifier and evaluate on test set.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        config: Application configuration
        save_model: Whether to save trained model to disk
    
    Returns:
        (trained_pipeline, test_accuracy)
    """
    test_size = config.classifier.test_size
    random_state = 42
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)}")
    
    model = _build_model(config)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    
    logger.info(f"Test accuracy: {acc:.4f}")
    logger.debug(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
    
    if save_model:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / "model_latest.joblib"
        joblib.dump(model, out_path)
        logger.info(f"Model saved to: {out_path}")
    
    return model, acc
