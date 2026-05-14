from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import joblib
import numpy as np

from .config import AppConfig, PROJECT_ROOT, load_config
from .features import BandpowerFeatureExtractor
from .lsl_acquisition import pull_chunk, resolve_eeg_stream
from .preprocessing import preprocess_epoch_data

logger = logging.getLogger(__name__)

MAX_BUFFER_SECONDS = 30


def _get_class_names(config: AppConfig) -> Dict[int, str]:
    class_map = config.paradigm.get("classes", {})
    return {int(code): str(label) for label, code in class_map.items()}


def _get_model_path() -> str:
    env_path = os.getenv("EEG_MODEL_PATH")
    if env_path:
        return env_path
    return str(PROJECT_ROOT / "models" / "model_latest.joblib")


def _is_raw_epoch_pipeline(model: object) -> bool:
    named_steps = getattr(model, "named_steps", None)
    return isinstance(named_steps, dict) and {"preprocess", "features"}.issubset(named_steps)


def _predict_with_legacy_model(model: object, epoch_arr: np.ndarray, config: AppConfig) -> int:
    """Predict with older models that expect a 2D feature matrix."""
    processed = preprocess_epoch_data(
        epoch_arr,
        sfreq=float(config.preprocessing.sfreq),
        l_freq=float(config.preprocessing.l_freq),
        h_freq=float(config.preprocessing.h_freq),
        notch_freq=float(config.preprocessing.notch_freq),
        use_car=bool(getattr(config.preprocessing, "car", False)),
    )

    extractor = BandpowerFeatureExtractor(
        sfreq=float(config.preprocessing.sfreq),
        bands=config.features.bands,
        expected_n_times=int(round(float(config.training.window_length_sec) * float(config.preprocessing.sfreq))),
    )
    extractor.fit(processed)
    features = extractor.transform(processed)
    return int(model.predict(features)[0])


def run_online_bci(config: Optional[AppConfig] = None) -> None:
    """Run online BCI inference with the serialized pipeline trained offline."""
    if config is None:
        config = load_config()

    model_path = _get_model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run offline analysis first (mode='offline') to train a model.\n"
            "Or set EEG_MODEL_PATH environment variable to custom location."
        )

    model = joblib.load(model_path)
    logger.info(f"Model loaded from: {model_path}")

    raw_epoch_pipeline = _is_raw_epoch_pipeline(model)
    if not raw_epoch_pipeline:
        logger.warning(
            "Loaded a legacy model without embedded raw-epoch preprocessing. "
            "Falling back to bandpower features for online inference; retrain the model to use the new pipeline."
        )

    eeg_inlet = resolve_eeg_stream(config)
    class_names = _get_class_names(config)

    prep_cfg = config.preprocessing
    sfreq = float(prep_cfg.sfreq)
    epoch_length_sec = float(config.training.window_length_sec)
    n_samples_epoch = int(round(epoch_length_sec * sfreq))
    n_samples_slide = n_samples_epoch // 2

    buf_chunks: list[np.ndarray] = []
    buf_total = 0
    max_buffer_samples = int(MAX_BUFFER_SECONDS * sfreq)

    logger.info(
        f"Online BCI started. Epoch: {epoch_length_sec}s, overlap: {epoch_length_sec / 2}s"
    )

    try:
        while True:
            chunk, _ = pull_chunk(eeg_inlet)

            if chunk.size == 0:
                time.sleep(0.005)
                continue

            buf_chunks.append(chunk)
            buf_total += chunk.shape[0]

            if buf_total > max_buffer_samples:
                all_data = np.vstack(buf_chunks)
                all_data = all_data[-max_buffer_samples:]
                buf_chunks = [all_data]
                buf_total = all_data.shape[0]

            if buf_total < n_samples_epoch:
                continue

            all_data = np.vstack(buf_chunks)
            epoch_data = all_data[-n_samples_epoch:]
            epoch_arr = epoch_data[np.newaxis, :, :]

            try:
                if raw_epoch_pipeline:
                    pred = int(model.predict(epoch_arr)[0])
                else:
                    pred = _predict_with_legacy_model(model, epoch_arr, config)
                label = class_names.get(pred, str(pred))
                logger.info(f"Prediction: {label} (code={pred})")
            except Exception as e:
                logger.error(f"Classification error: {e}", exc_info=True)
                continue

            overlap_data = all_data[-n_samples_slide:]
            buf_chunks = [overlap_data]
            buf_total = overlap_data.shape[0]

    except KeyboardInterrupt:
        logger.info("Online BCI stopped by user")
    except Exception as e:
        logger.error(f"Online BCI fatal error: {e}", exc_info=True)
        raise
