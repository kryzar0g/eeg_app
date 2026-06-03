from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np

from .classifier import train_and_evaluate
from .config import AppConfig, load_config

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".edf", ".bdf", ".fif"}  # .fif = interní recorder výstup
MAX_FILE_SIZE_GB = 2


def _load_raw(file_path: Path) -> mne.io.BaseRaw:
  suffix = file_path.suffix.lower()

  if suffix not in SUPPORTED_FORMATS:
    raise ValueError(f"Unsupported format: {suffix}. Supported: {SUPPORTED_FORMATS}")

  file_size_gb = file_path.stat().st_size / (1024**3)
  preload = file_size_gb <= MAX_FILE_SIZE_GB
  if not preload:
    logger.warning(
      f"Large file ({file_size_gb:.1f} GB). Loading without preload to save memory."
    )

  try:
    if suffix == ".edf":
      raw = mne.io.read_raw_edf(file_path, preload=preload, verbose="ERROR")
    elif suffix == ".bdf":
      raw = mne.io.read_raw_bdf(file_path, preload=preload, verbose="ERROR")
    else:  # .fif – výstup interního EegRecorderu
      raw = mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")

    n_ch = len(raw.ch_names)  # raw.n_channels neexistuje ve starších verzích MNE
    n_times = raw.n_times if hasattr(raw, "n_times") else raw.last_samp - raw.first_samp + 1
    logger.info(
      f"Loaded {file_path.name}: {n_ch} channels, {n_times} samples @ {raw.info['sfreq']} Hz"
    )
    return raw
  except Exception as e:
    raise RuntimeError(f"Failed to load {file_path}: {e}") from e


def _prepare_epochs_from_csv(raw: mne.io.BaseRaw, config: AppConfig) -> mne.Epochs:
  events_cfg = config.events
  csv_path_str = events_cfg.get("timestamp_csv_path")
  epoch_len_sec = float(events_cfg.get("epoch_length", 0.0))

  if not csv_path_str:
    raise RuntimeError("CSV mode requires events.timestamp_csv_path in config")
  if epoch_len_sec <= 0:
    raise RuntimeError("CSV mode requires positive events.epoch_length in config")

  csv_path = Path(csv_path_str).expanduser()
  if not csv_path.is_file():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

  intervals: List[Dict[str, str]] = []
  with csv_path.open("r", encoding="utf-8") as f:
    next(f, None)
    for line_num, line in enumerate(f, start=2):
      line = line.strip()
      if not line:
        continue
      parts = [p.strip() for p in line.split(";")]
      if len(parts) < 3:
        logger.warning(f"Skipping line {line_num}: insufficient columns")
        continue
      intervals.append({"start": parts[0], "end": parts[1], "label": parts[2]})

  if not intervals:
    raise RuntimeError("CSV is empty or invalid")

  sfreq = float(raw.info["sfreq"])
  epoch_len_samples = int(round(epoch_len_sec * sfreq))
  raw_eeg = raw.copy().pick("eeg")

  data_list: List[np.ndarray] = []
  events_list: List[List[int]] = []
  label_to_code: Dict[str, int] = {}

  for interval in intervals:
    try:
      start_sec = float(interval["start"])
      end_sec = float(interval["end"])
    except ValueError:
      logger.warning(f"Skipping interval with invalid timestamps: {interval}")
      continue

    if end_sec <= start_sec:
      continue

    label = interval["label"].strip()
    if label not in label_to_code:
      label_to_code[label] = len(label_to_code) + 1

    code = label_to_code[label]
    start_sample = int(round(start_sec * sfreq))
    end_sample = int(round(end_sec * sfreq))
    total_samples = end_sample - start_sample

    if total_samples < epoch_len_samples:
      continue

    n_epochs_here = total_samples // epoch_len_samples
    for i in range(n_epochs_here):
      s = start_sample + i * epoch_len_samples
      e = s + epoch_len_samples
      if e > raw_eeg.n_times:
        break
      data_list.append(raw_eeg.get_data(start=s, stop=e))
      events_list.append([s, 0, code])

  if not data_list:
    raise RuntimeError("No valid epochs extracted from CSV.")

  data = np.stack(data_list, axis=0)
  events = np.asarray(events_list, dtype=int)
  event_id = {label: code for label, code in label_to_code.items()}

  epochs = mne.EpochsArray(
    data,
    info=raw_eeg.info.copy(),
    events=events,
    event_id=event_id,
    verbose="ERROR",
  )

  logger.info(f"Created {len(epochs)} epochs from CSV for classes: {list(event_id.keys())}")
  return epochs


def _prepare_epochs_from_stim(raw: mne.io.BaseRaw, config: AppConfig) -> mne.Epochs:
  events_cfg = config.events
  stim_channel = events_cfg.get("stim_channel", "STI 014")
  tmin = float(events_cfg.get("tmin", 0.0))
  tmax = float(events_cfg.get("tmax", 4.0))

  if stim_channel not in raw.info["ch_names"]:
    available = ", ".join(raw.info["ch_names"])
    raise RuntimeError(
      f"Stimulus channel '{stim_channel}' not found in data. Available channels: {available}"
    )

  logger.info(f"Detecting events in channel: {stim_channel}")
  events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1)
  if events.size == 0:
    raise RuntimeError("No events found in the EEG file")

  class_map = config.paradigm.get("classes", {})
  if not class_map:
    raise RuntimeError("Configuration missing: paradigm.classes")

  valid_codes = set(int(v) for v in class_map.values())
  events_sel = events[np.isin(events[:, 2], list(valid_codes))]
  if events_sel.size == 0:
    raise RuntimeError("No events matching class codes from paradigm.classes")

  event_id = {str(label): int(code) for label, code in class_map.items()}
  epochs = mne.Epochs(
    raw,
    events_sel,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
    picks="eeg",
    verbose="ERROR",
  )

  logger.info(f"Created {len(epochs)} epochs")
  return epochs


def _prepare_epochs_from_annotations(raw: mne.io.BaseRaw, config: AppConfig) -> mne.Epochs:
  """Epochy z MNE anotací – výstup interního EegRecorderu (FIF soubory).

  Anotace mají popis = kód třídy jako řetězec ("1", "2", …),
  duration = imagery_duration. Epochy začínají na onset markeru.
  """
  events_cfg = config.events
  tmin = float(events_cfg.get("tmin", 0.0))
  # tmax: explicitní z config, nebo imagery_duration jako záloha
  tmax_cfg = events_cfg.get("tmax", None)
  tmax = float(tmax_cfg) if tmax_cfg is not None else float(config.experiment.imagery_duration)

  events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
  if events.size == 0:
    raise RuntimeError(
      "FIF soubor neobsahuje žádné anotace/markery. "
      "Zkontrolujte, zda byl záznam pořízený interním EegRecorderem, "
      "nebo přepněte na mode: stim / csv v config.yaml."
    )

  logger.info(f"Anotace nalezeny: {event_id}")

  epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
    picks="eeg",
    verbose="ERROR",
  )
  logger.info(f"Vytvořeno {len(epochs)} epoch z anotací")
  return epochs


def _prepare_epochs(raw: mne.io.BaseRaw, config: AppConfig) -> mne.Epochs:
  mode = str(config.events.get("mode", "stim")).lower()

  # ── Auto-detekce výstupu interního EegRecorderu ────────────────────
  # Recorder vždy produkuje BDF se Status kanálem nebo EDF/FIF s MNE
  # anotacemi. Tato detekce má přednost před nastaveným mode, protože
  # je jednoznačná (žádný jiný zdroj neprodukuje "Status" kanál bez STI).
  filenames    = getattr(raw, "filenames", [])
  is_our_file  = bool(filenames) and any(
    str(f).lower().endswith((".edf", ".bdf", ".fif")) for f in filenames
  )
  ch_names        = raw.info.get("ch_names", [])
  has_status_ch   = "Status" in ch_names
  has_annotations = hasattr(raw, "annotations") and len(raw.annotations) > 0
  stim_channel    = config.events.get("stim_channel", "STI 014")
  has_stim_ch     = stim_channel in ch_names

  if is_our_file and has_status_ch and not has_stim_ch:
    # BDF z EegRecorderu: markery v Status kanálu
    logger.info("BDF z interního EegRecorderu – čtu markery ze Status kanálu")
    return _prepare_epochs_from_status(raw, config)

  if is_our_file and has_annotations and not has_stim_ch:
    # EDF+ nebo FIF z EegRecorderu: markery jako MNE anotace
    logger.info("EDF/FIF z interního EegRecorderu – čtu markery z MNE anotací")
    return _prepare_epochs_from_annotations(raw, config)

  # ── Explicitní mode z config.yaml ───────────────────────────────────
  if mode == "annotations":
    return _prepare_epochs_from_annotations(raw, config)
  if mode == "csv":
    return _prepare_epochs_from_csv(raw, config)
  return _prepare_epochs_from_stim(raw, config)


def _prepare_epochs_from_status(raw: mne.io.BaseRaw, config: AppConfig) -> mne.Epochs:
  """Epochování z BDF Status kanálu (výstup interního EegRecorderu).

  Status kanál obsahuje integer kódy tříd vložené na začátku každého stimulu.
  Funkce najde nenulové hodnoty, interpretuje je jako event kódy a epochuje.
  """
  events_cfg = config.events
  tmin = float(events_cfg.get("tmin", 0.0))
  tmax_cfg = events_cfg.get("tmax", None)
  tmax = float(tmax_cfg) if tmax_cfg is not None else float(config.experiment.imagery_duration)

  try:
    events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="ERROR")
  except Exception as e:
    raise RuntimeError(f"Nepodařilo se načíst markery ze Status kanálu: {e}") from e

  if events.size == 0:
    raise RuntimeError(
      "BDF Status kanál neobsahuje žádné markery. "
      "Zkontrolujte, zda byl záznam pořízen interním EegRecorderem."
    )

  # Sestavit event_id: kódy z paradigmatu nebo ze všech nalezených událostí
  class_map = config.paradigm.get("classes", {})
  if class_map:
    event_id = {str(label): int(code) for label, code in class_map.items()}
    # Filtrovat události na platné kódy
    valid_codes = set(event_id.values())
    events = events[np.isin(events[:, 2], list(valid_codes))]
    if events.size == 0:
      raise RuntimeError("Status kanál neobsahuje události odpovídající třídám v config.yaml")
  else:
    unique_codes = np.unique(events[:, 2])
    event_id = {str(c): int(c) for c in unique_codes}

  logger.info(f"Status kanál: nalezeno {len(events)} událostí, třídy: {event_id}")

  epochs = mne.Epochs(
    raw.copy().pick("eeg"),
    events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
    verbose="ERROR",
  )
  logger.info(f"Vytvořeno {len(epochs)} epoch z BDF Status kanálu")
  return epochs


def _average_epochs(
  X: np.ndarray,
  y: np.ndarray,
  n_averages: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """Průměruje skupiny n_averages epoch stejné třídy (ERP averaging).

  Pro každou třídu rozdělí dostupné epochy do skupin po n_averages
  a každou skupinu zprůměruje. Výsledkem je méně epoch s vyšším SNR.

  Příklad: 40 epoch × 4 třídy, n_averages=5 → 8 průměrovaných epoch × 4 třídy.
  """
  if n_averages <= 1:
    return X, y

  unique_labels = np.unique(y)
  X_out: List[np.ndarray] = []
  y_out: List[int] = []

  for label in unique_labels:
    idx = np.where(y == label)[0]
    n_groups = len(idx) // n_averages
    if n_groups == 0:
      logger.warning(
        f"Třída {label}: pouze {len(idx)} epoch, "
        f"n_averages={n_averages} – přeskakuji průměrování pro tuto třídu"
      )
      X_out.append(X[idx])
      y_out.extend([int(label)] * len(idx))
      continue

    for i in range(n_groups):
      group_idx = idx[i * n_averages : (i + 1) * n_averages]
      averaged = X[group_idx].mean(axis=0)  # (n_channels, n_times)
      X_out.append(averaged[np.newaxis])
      y_out.append(int(label))

    remainder = len(idx) % n_averages
    if remainder:
      logger.debug(
        f"Třída {label}: {remainder} zbývajících epoch vynecháno při průměrování"
      )

  logger.info(
    f"ERP averaging: {len(y)} epoch → {len(y_out)} průměrovaných epoch "
    f"(n_averages={n_averages})"
  )
  return np.vstack(X_out), np.array(y_out, dtype=int)


def _crop_epoch_array(
  X: np.ndarray,
  y: np.ndarray,
  crop_samples: int,
  step_samples: int,
  augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
  if crop_samples <= 0:
    raise ValueError("crop_samples must be positive")

  n_epochs, n_channels, n_times = X.shape
  if crop_samples > n_times:
    raise ValueError(
      f"Model window ({crop_samples} samples) is longer than the available epoch ({n_times} samples)"
    )

  if crop_samples == n_times:
    return X, y

  windows: List[np.ndarray] = []
  labels: List[int] = []

  for epoch, label in zip(X, y):
    if augment:
      starts = range(0, n_times - crop_samples + 1, step_samples)
    else:
      starts = [max((n_times - crop_samples) // 2, 0)]

    for start in starts:
      stop = start + crop_samples
      if stop > n_times:
        continue
      windows.append(epoch[:, start:stop])
      labels.append(int(label))

  if not windows:
    raise RuntimeError("Cropping produced no training windows")

  return np.stack(windows, axis=0), np.asarray(labels, dtype=int)


def run_offline_from_file(file_path_str: str) -> Tuple[float, int]:
  """Load EEG file, create epochs, and train the configured model pipeline."""
  config = load_config()
  file_path = Path(file_path_str).expanduser().resolve()

  if not file_path.is_file():
    raise FileNotFoundError(f"File not found: {file_path}")

  logger.info(f"Loading EEG from {file_path}")
  raw = _load_raw(file_path)

  logger.info("Creating epochs...")
  epochs = _prepare_epochs(raw, config)
  X = epochs.get_data()
  y = epochs.events[:, 2].astype(int)

  sfreq = float(config.preprocessing.sfreq)
  crop_samples = int(round(config.training.window_length_sec * sfreq))
  step_samples = int(round(config.training.crop_step_sec * sfreq))

  logger.info(
    f"Offline crop settings: window={config.training.window_length_sec}s, step={config.training.crop_step_sec}s, augment={config.training.crop_enabled}"
  )
  X, y = _crop_epoch_array(
    X,
    y,
    crop_samples=crop_samples,
    step_samples=step_samples,
    augment=bool(config.training.crop_enabled),
  )

  # ERP averaging: průměrovat skupiny epoch pro zvýšení SNR
  n_averages = int(getattr(config.experiment, "n_averages", 1))
  if n_averages > 1:
    X, y = _average_epochs(X, y, n_averages=n_averages)

  logger.info(f"Epoch array shape: {X.shape}, classes: {len(np.unique(y))}")
  logger.info("Training classifier pipeline...")
  model, acc = train_and_evaluate(X, y, config=config, save_model=True)

  return acc, len(epochs)
