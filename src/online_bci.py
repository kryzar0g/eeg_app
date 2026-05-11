from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import joblib
import mne
import numpy as np

from .config import AppConfig, PROJECT_ROOT, load_config
from .features import compute_bandpower_features
from .lsl_acquisition import pull_chunk, resolve_eeg_stream
from .preprocessing import bandpass_filter

logger = logging.getLogger(__name__)

MAX_BUFFER_SECONDS = 30  # Prevent unbounded buffer growth


def _get_class_names(config: AppConfig) -> Dict[int, str]:
    """Get class name mapping from configuration."""
    class_map = config.paradigm.get("classes", {})
    # Convert {label: code} to {code: label}
    return {int(code): str(label) for label, code in class_map.items()}


def _get_model_path() -> str:
    """Get model path from environment or default location."""
    env_path = os.getenv("EEG_MODEL_PATH")
    if env_path:
        return env_path
    return str(PROJECT_ROOT / "models" / "model_latest.joblib")


def _get_expected_feature_count(model) -> Optional[int]:
    """Get number of features expected by the model."""
    expected = getattr(model, "n_features_in_", None)
    if expected is not None:
        return int(expected)
    
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        scaler = named_steps.get("scaler")
        expected = getattr(scaler, "n_features_in_", None)
        if expected is not None:
            return int(expected)
    
    return None


def _validate_feature_matrix(model, X: np.ndarray, bands: list[list[float] | tuple[float, float]]) -> None:
    """Validate feature matrix dimensions match model expectations."""
    expected_features = _get_expected_feature_count(model)
    actual_features = int(X.shape[1])
    
    if expected_features is None or actual_features == expected_features:
        return
    
    n_bands = len(bands)
    channel_info = ""
    if n_bands > 0:
        expected_ch = expected_features // n_bands if expected_features % n_bands == 0 else None
        actual_ch = actual_features // n_bands if actual_features % n_bands == 0 else None
        if expected_ch and actual_ch:
            channel_info = f" ({actual_ch} online channels vs {expected_ch} training channels)"
    
    raise RuntimeError(
        f"Feature mismatch: model expects {expected_features} features, "
        f"got {actual_features}{channel_info}. "
        "Check that LSL stream and band configuration match training setup."
    )


def _build_info(n_channels: int, sfreq: float) -> mne.Info:
    """Create MNE Info object for given channel count and sampling rate."""
    ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]
    return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")


def run_online_bci(config: Optional[AppConfig] = None) -> None:
    """Run online BCI loop: load model and classify EEG in real-time.
    
    Algorithm:
        1. Connect to EEG LSL stream
        2. Buffer samples for epoch_length seconds
        3. Apply band-pass filter and extract features
        4. Classify and log prediction
        5. Shift buffer by 50% overlap
    
    Press Ctrl+C to stop.
    """
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
    
    logger.info("Connecting to EEG LSL stream...")
    eeg_inlet = resolve_eeg_stream(config)
    
    prep_cfg = config.preprocessing
    sfreq = float(prep_cfg.sfreq)
    l_freq = float(prep_cfg.l_freq)
    h_freq = float(prep_cfg.h_freq)
    
    # Get class names from config
    class_names = _get_class_names(config)
    
    # Get epoch length from config
    epoch_length_sec = float(config.events.get("epoch_length", 4.0))
    
    n_samples_epoch = int(round(epoch_length_sec * sfreq))
    n_samples_slide = n_samples_epoch // 2
    
    # Get feature bands
    bands = config.features.bands
    bands = [(float(b[0]), float(b[1])) for b in bands]
    
    buf_chunks: list[np.ndarray] = []
    buf_total: int = 0
    max_buffer_samples = int(MAX_BUFFER_SECONDS * sfreq)
    
    logger.info(
        f"Online BCI started. Epoch: {epoch_length_sec}s, "
        f"overlap: {epoch_length_sec/2}s, bands: {bands}"
    )
    
    try:
        while True:
            chunk, _ = pull_chunk(eeg_inlet)
            
            if chunk.size == 0:
                time.sleep(0.005)
                continue
            
            buf_chunks.append(chunk)
            buf_total += chunk.shape[0]
            
            # Prevent unbounded buffer growth
            if buf_total > max_buffer_samples:
                all_data = np.vstack(buf_chunks)
                excess = all_data.shape[0] - max_buffer_samples
                all_data = all_data[excess:]
                buf_chunks = [all_data]
                buf_total = all_data.shape[0]
            
            if buf_total < n_samples_epoch:
                continue
            
            all_data = np.vstack(buf_chunks)
            epoch_data = all_data[-n_samples_epoch:]
            
            n_channels = epoch_data.shape[1]
            info = _build_info(n_channels, sfreq)
            
            epoch_arr = epoch_data.T[np.newaxis, :, :]
            epochs = mne.EpochsArray(epoch_arr, info=info, verbose=False)
            
            try:
                bandpass_filter(epochs, l_freq, h_freq)
                X, _, _ = compute_bandpower_features(epochs, bands)
                _validate_feature_matrix(model, X, bands)
                
                pred = int(model.predict(X)[0])
                label = class_names.get(pred, str(pred))
                logger.info(f"Prediction: {label} (code={pred})")
            except Exception as e:
                logger.error(f"Classification error: {e}", exc_info=True)
                continue
            
            # Shift buffer
            overlap_data = all_data[-n_samples_slide:]
            buf_chunks = [overlap_data]
            buf_total = overlap_data.shape[0]
    
    except KeyboardInterrupt:
        logger.info("Online BCI stopped by user")
    except Exception as e:
        logger.error(f"Online BCI fatal error: {e}", exc_info=True)
        raise


def run_online_bci(config: Optional[AppConfig] = None) -> None:
  """Spustí online BCI smyčku: načte model a klasifikuje EEG v reálném čase.

  Algoritmus:
    1. Připojí se k EEG LSL streamu.
    2. Sbírá vzorky do bufferu délky ``epoch_length`` sekund.
    3. Aplikuje band-pass filtr a extrahuje log-bandpower příznaky.
    4. Klasifikuje epochu a tiskne predikci.
    5. Posune buffer o polovinu délky epochy (překrývání 50 %).

  Stiskněte Ctrl+C pro ukončení.
  """

  if config is None:
    config = load_config()

  model_path = PROJECT_ROOT / "models" / "model_latest.joblib"
  if not model_path.is_file():
    raise FileNotFoundError(
      f"Model nenalezen: {model_path}\n"
      "Nejprve proveďte offline analýzu (režim 'offline'), která model natrénuje."
    )

  model = joblib.load(model_path)
  print(f"Model načten z: {model_path}")

  eeg_inlet = resolve_eeg_stream(config)

  prep_cfg = config.preprocessing
  sfreq = float(prep_cfg.get("sfreq", 250.0))
  l_freq = float(prep_cfg.get("l_freq", 8.0))
  h_freq = float(prep_cfg.get("h_freq", 30.0))

  epoch_length_sec = float(config.events.get("epoch_length", 4.0))
  n_samples_epoch = int(round(epoch_length_sec * sfreq))
  n_samples_slide = n_samples_epoch // 2

  bands = config.features.get("bands", [[8.0, 12.0], [12.0, 30.0]])

  # Buffer jako python list fragmentů (n_samples_frag, n_channels)
  buf_chunks: list[np.ndarray] = []
  buf_total: int = 0

  print(
    f"Online BCI spuštěna. Délka epochy: {epoch_length_sec} s "
    f"(posun: {epoch_length_sec / 2} s). Stiskněte Ctrl+C pro ukončení."
  )

  try:
    while True:
      chunk, _ = pull_chunk(eeg_inlet)

      if chunk.size == 0:
        time.sleep(0.005)
        continue

      buf_chunks.append(chunk)  # chunk: (n_samples, n_channels)
      buf_total += chunk.shape[0]

      if buf_total < n_samples_epoch:
        continue

      # Sestavíme celý buffer a vezmeme poslední epochu
      all_data = np.vstack(buf_chunks)  # (buf_total, n_channels)
      epoch_data = all_data[-n_samples_epoch:]  # (n_samples_epoch, n_channels)

      n_channels = epoch_data.shape[1]
      info = _build_info(n_channels, sfreq)

      # MNE EpochsArray očekává tvar (n_epochs, n_channels, n_times)
      epoch_arr = epoch_data.T[np.newaxis, :, :]
      epochs = mne.EpochsArray(epoch_arr, info=info, verbose=False)

      bandpass_filter(epochs, l_freq, h_freq)

      X, _, _ = compute_bandpower_features(epochs, bands)

      _validate_feature_matrix(model, X, bands)

      pred = int(model.predict(X)[0])
      label = _CLASS_NAMES.get(pred, str(pred))
      print(f"[{time.strftime('%H:%M:%S')}] Predikce: {label} (kód={pred})")

      # Posun bufferu o n_samples_slide vzorků
      overlap_data = all_data[-n_samples_slide:]
      buf_chunks = [overlap_data]
      buf_total = overlap_data.shape[0]

  except KeyboardInterrupt:
    print("Online BCI ukončena.")
