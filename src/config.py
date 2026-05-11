from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfig(BaseModel):
    trials_per_class: int = Field(40, gt=0)
    baseline_duration: float = Field(2.0, gt=0)
    cue_duration: float = Field(1.0, gt=0)
    imagery_duration: float = Field(4.0, gt=0)
    iti_duration: float = Field(2.0, gt=0)


class LSLConfig(BaseModel):
    eeg_stream_name: str = Field("EEG", min_length=1)
    eeg_stream_type: str = Field("EEG", min_length=1)
    marker_stream_name: str = Field("Markers", min_length=1)
    resolution_timeout: float = Field(10.0, gt=0)


class PreprocessingConfig(BaseModel):
    sfreq: float = Field(250.0, gt=0, description="Sampling frequency in Hz")
    l_freq: float = Field(8.0, ge=0)
    h_freq: float = Field(30.0, ge=0)
    notch_freq: float = Field(50.0, gt=0)
    
    @validator("h_freq")
    def h_freq_greater_than_l_freq(cls, v, values):
        if "l_freq" in values and v <= values["l_freq"]:
            raise ValueError("h_freq must be greater than l_freq")
        return v


class FeaturesConfig(BaseModel):
    bands: list[tuple[float, float]] = Field(
        [(8, 12), (12, 30)],
        description="Frequency bands for feature extraction"
    )


class ClassifierConfig(BaseModel):
    algorithm: str = Field("lda", pattern="^(lda|svm)$")
    test_size: float = Field(0.2, ge=0.01, le=0.99)


class AppConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    lsl: LSLConfig = Field(default_factory=LSLConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    
    # Additional optional sections from YAML
    paradigm: Dict[str, Any] = Field(default_factory=dict)
    events: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow additional fields


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate YAML configuration.
    
    Args:
        path: Path to config.yaml. If None, uses config/config.yaml relative to project root.
        Can be overridden by EEG_CONFIG_PATH environment variable.
    
    Returns:
        Validated AppConfig instance.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
    """
    # Check environment variable first
    env_path = os.getenv("EEG_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        logger.debug(f"Using config from environment variable: {path}")
    
    if path is None:
        path = PROJECT_ROOT / "config" / "config.yaml"
    
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    
    try:
        config = AppConfig(**raw_data)
        logger.debug(f"Configuration loaded from: {path}")
        return config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
