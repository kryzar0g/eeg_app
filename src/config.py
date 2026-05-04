from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
  """Konfigurace aplikace načtená z YAML souboru."""

  data: Dict[str, Any]

  @property
  def experiment(self) -> Dict[str, Any]:
    return self.data.get("experiment", {})

  @property
  def lsl(self) -> Dict[str, Any]:
    return self.data.get("lsl", {})

  @property
  def paradigm(self) -> Dict[str, Any]:
    return self.data.get("paradigm", {})

  @property
  def preprocessing(self) -> Dict[str, Any]:
    return self.data.get("preprocessing", {})

  @property
  def features(self) -> Dict[str, Any]:
    return self.data.get("features", {})

  @property
  def classifier(self) -> Dict[str, Any]:
    return self.data.get("classifier", {})

  @property
  def events(self) -> Dict[str, Any]:
    return self.data.get("events", {})


def load_config(path: Path | None = None) -> AppConfig:
  """Načte YAML konfiguraci a vrátí ji jako AppConfig.

  Pokud není cesta zadána, použije se `config/config.yaml` v kořeni projektu.
  """

  if path is None:
    path = PROJECT_ROOT / "config" / "config.yaml"

  if not path.is_file():
    raise FileNotFoundError(f"Konfigurační soubor nebyl nalezen: {path}")

  with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

  return AppConfig(data=data)
