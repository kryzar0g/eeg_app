from __future__ import annotations

import time
from typing import Optional

import joblib
import mne
import numpy as np

from .config import AppConfig, PROJECT_ROOT, load_config
from .features import compute_bandpower_features
from .lsl_acquisition import pull_chunk, resolve_eeg_stream
from .preprocessing import bandpass_filter


_CLASS_NAMES: dict[int, str] = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}


def _build_info(n_channels: int, sfreq: float) -> mne.Info:
  """Vytvoří MNE Info objekt pro zadaný počet kanálů a vzorkovací frekvenci."""

  ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]
  return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")


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

      pred = int(model.predict(X)[0])
      label = _CLASS_NAMES.get(pred, str(pred))
      print(f"[{time.strftime('%H:%M:%S')}] Predikce: {label} (kód={pred})")

      # Posun bufferu o n_samples_slide vzorků
      overlap_data = all_data[-n_samples_slide:]
      buf_chunks = [overlap_data]
      buf_total = overlap_data.shape[0]

  except KeyboardInterrupt:
    print("Online BCI ukončena.")
