from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except Exception as e:  # pragma: no cover - environment-dependent
    # Provide a clearer, actionable error when binary dependencies fail to load
    msg = (
        "Nepodařilo se importovat scikit-learn nebo některou z jeho nativních závislostí\n"
        f"Původní chyba: {e}\n\n"
        "Možné opravy (Windows):\n"
        "1) Aktivujte své virtuální prostředí (pokud používáte) nebo otevřete PowerShell s Pythonem.\n"
        "2) Aktualizujte pip a nainstalujte/aktualizujte potřebné balíčky:\n"
        "   python -m pip install --upgrade pip\n"
        "   python -m pip install --upgrade numpy scipy scikit-learn pyarrow\n"
        "3) Pokud chyba přetrvává, nainstalujte Microsoft Visual C++ Redistributable (2015-2022),\n"
        "   restartujte systém a znovu spusťte aplikaci. Stažení: https://learn.microsoft.com/windows/win32/msi/download-the-latest-supported-visual-c-downloads\n\n"
        "Pokud chcete, mohu zkontrolovat verze nainstalovaných balíčků ve vašem prostředí nebo připravit přesné příkazy."
    )
    raise ImportError(msg) from e

from .config import AppConfig, PROJECT_ROOT
from .features import BandpowerFeatureExtractor, EpochSignalPreprocessor, FilterBankCSP

logger = logging.getLogger(__name__)


def _build_model(config: AppConfig) -> Pipeline:
    """Build a raw-epoch pipeline with preprocessing, feature extraction and classifier."""
    clf_algorithm = config.classifier.algorithm.lower()
    feature_method = config.features.method.lower()

    sfreq = float(config.preprocessing.sfreq)
    l_freq = float(config.preprocessing.l_freq)
    h_freq = float(config.preprocessing.h_freq)
    notch_freq = float(config.preprocessing.notch_freq)
    use_car = bool(getattr(config.preprocessing, "car", False))
    window_length_sec = float(config.training.window_length_sec)
    expected_n_times = int(round(window_length_sec * sfreq))
    bands = [tuple(map(float, band)) for band in config.features.bands]
    
    if clf_algorithm == "lda":
        clf = LinearDiscriminantAnalysis()
    elif clf_algorithm == "svm":
        clf = SVC(kernel="linear")
    else:
        raise ValueError(f"Unknown classifier: {clf_algorithm}")

    preprocess = EpochSignalPreprocessor(
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=notch_freq,
        use_car=use_car,
        expected_n_times=expected_n_times,
    )

    if feature_method in {"csp", "fbcsp"}:
        fbands = bands[:1] if feature_method == "csp" else bands
        feature_step = FilterBankCSP(
            sfreq=sfreq,
            bands=fbands,
            n_components=int(config.features.csp_components),
            top_k_features=int(config.features.fbcsp_top_k) if feature_method == "fbcsp" else 0,
            expected_n_times=expected_n_times,
        )
    elif feature_method == "psd":
        feature_step = BandpowerFeatureExtractor(
            sfreq=sfreq,
            bands=bands,
            expected_n_times=expected_n_times,
        )
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("features", feature_step),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    config: AppConfig,
    save_model: bool = True,
) -> Tuple[Pipeline, float]:
    """Train classifier and evaluate on test set.
    
    Args:
        X: Epoch array (n_epochs, n_channels, n_times)
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
    logger.info(
        f"Feature method={config.features.method.lower()}, classifier={config.classifier.algorithm.lower()}"
    )
    logger.info(f"Model window length: {config.training.window_length_sec}s")
    
    model = _build_model(config)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    preprocessed_sample = model.named_steps["preprocess"].transform(X_test[:1])
    features_shape = model.named_steps["features"].transform(preprocessed_sample).shape
    
    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info(f"Transformed feature shape: {features_shape}")
    logger.debug(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
    
    if save_model:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / "model_latest.joblib"
        joblib.dump(model, out_path)
        logger.info(f"Model saved to: {out_path}")
    
    return model, acc
