from __future__ import annotations

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


def _build_model(config: AppConfig) -> Pipeline:
  clf_type = config.classifier.get("type", "lda").lower()

  if clf_type == "lda":
    clf = LinearDiscriminantAnalysis()
  elif clf_type == "svm":
    clf = SVC(kernel="linear")
  else:
    raise ValueError(f"Neznámý typ klasifikátoru: {clf_type}")

  return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_and_evaluate(
  X: np.ndarray,
  y: np.ndarray,
  config: AppConfig,
  save_model: bool = True,
) -> Tuple[Pipeline, float]:
  """Rozdělí data na trénink/test, natrénuje model a vypíše metriky.

  Vrací (model, test_accuracy).
  """

  test_size = float(config.classifier.get("test_size", 0.2))
  random_state = int(config.classifier.get("random_state", 42))

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
  )

  model = _build_model(config)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  acc = float(accuracy_score(y_test, y_pred))

  print("Test accuracy:", acc)
  print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
  print("Classification report:\n", classification_report(y_test, y_pred))

  if save_model:
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "model_latest.joblib"
    joblib.dump(model, out_path)
    print(f"Model uložen do: {out_path}")

  return model, acc
