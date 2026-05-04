from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np

from .config import AppConfig, PROJECT_ROOT, load_config
from .classifier import train_and_evaluate
from .features import compute_bandpower_features


def _load_raw(file_path: Path) -> mne.io.BaseRaw:
  suffix = file_path.suffix.lower()

  if suffix == ".edf":
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
  elif suffix == ".bdf":
    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose="ERROR")
  else:
    raise ValueError(f"Nepodporovaný formát souboru: {suffix} (očekávám .edf nebo .bdf)")

  return raw


def _prepare_epochs(raw: mne.io.BaseRaw, config: AppConfig) -> mne.Epochs:
  events_cfg = config.events
  mode = str(events_cfg.get("mode", "stim")).lower()

  if mode == "csv":
    return _prepare_epochs_from_csv(raw, config)

  stim_channel = events_cfg.get("stim_channel", "STI 014")
  tmin = float(events_cfg.get("tmin", 0.0))
  tmax = float(events_cfg.get("tmax", 4.0))

  print(f"Hledám události ve stimulačním kanálu: {stim_channel}")

  if stim_channel not in raw.info["ch_names"]:
    raise RuntimeError(
      "Stimulační kanál z konfigurace ('"
      + stim_channel
      + "') se v tomto souboru nenachází.\n"
      + "Dostupné kanály jsou: "
      + ", ".join(raw.info["ch_names"])
    )

  events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1)

  if events.size == 0:
    raise RuntimeError("V zadaném EEG souboru nebyly nalezeny žádné události.")

  # Mapování label -> kód z konfigurace paradigmatu
  class_map = config.paradigm.get("classes", {})
  if not class_map:
    raise RuntimeError("V konfiguraci není definováno 'paradigm.classes'.")

  valid_codes = set(int(v) for v in class_map.values())
  mask = np.isin(events[:, 2], list(valid_codes))
  events_sel = events[mask]

  if events_sel.size == 0:
    raise RuntimeError(
      "V EEG souboru nebyly nalezeny události odpovídající kódům tříd z konfigurace."
    )

  event_id = {str(label): int(code) for label, code in class_map.items()}

  print(f"Počet rozpoznaných událostí: {len(events_sel)}")

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

  print(f"Vytvořeno epoch: {len(epochs)}")
  return epochs


def _prepare_epochs_from_csv(raw: mne.io.BaseRaw, config: AppConfig) -> mne.Epochs:
  """Vytvoří epochy z časových intervalů v CSV souboru.

  CSV je typu: start_seconds; end_seconds; label
  Vytváří se fixed-length epochy (events.epoch_length) uvnitř každého intervalu.
  Každý unikátní textový label dostane vlastní celočíselný kód třídy.
  """

  events_cfg = config.events
  csv_path_str = events_cfg.get("timestamp_csv_path")
  epoch_len_sec = float(events_cfg.get("epoch_length", 0.0))

  if not csv_path_str:
    raise RuntimeError(
      "Pro mode='csv' musíte v konfiguraci nastavit 'events.timestamp_csv_path'."
    )
  if epoch_len_sec <= 0:
    raise RuntimeError(
      "Pro mode='csv' musíte v konfiguraci nastavit kladnou hodnotu 'events.epoch_length'."
    )

  csv_path = Path(csv_path_str).expanduser()
  if not csv_path.is_file():
    raise FileNotFoundError(f"CSV s časovými značkami neexistuje: {csv_path}")

  # Načtení CSV ručně (oddělovač ';')
  intervals: List[Dict[str, str]] = []
  with csv_path.open("r", encoding="utf-8") as f:
    # prvni radek je hlavicka
    header = f.readline()
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = [p.strip() for p in line.split(";")]
      if len(parts) < 3:
        continue
      start_s, end_s, label = parts[0], parts[1], parts[2]
      intervals.append({"start": start_s, "end": end_s, "label": label})

  if not intervals:
    raise RuntimeError("CSV s časovými značkami je prázdné nebo ve špatném formátu.")

  sfreq = float(raw.info["sfreq"])
  epoch_len_samples = int(round(epoch_len_sec * sfreq))
  if epoch_len_samples <= 0:
    raise RuntimeError("Z epoch_length a sfreq vyšla nulová délka epochy v samplech.")

  raw_eeg = raw.copy().pick("eeg")

  data_list: List[np.ndarray] = []
  events_list: List[List[int]] = []
  label_to_code: Dict[str, int] = {}

  for interval in intervals:
    try:
      start_sec = float(interval["start"])
      end_sec = float(interval["end"])
    except ValueError:
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
      # interval je kratší než jedna epocha – přeskočíme
      continue

    n_epochs_here = total_samples // epoch_len_samples
    for i in range(n_epochs_here):
      s = start_sample + i * epoch_len_samples
      e = s + epoch_len_samples
      if e > raw_eeg.n_times:
        break

      data_seg = raw_eeg.get_data(start=s, stop=e)  # (n_channels, n_times)
      data_list.append(data_seg)
      events_list.append([s, 0, code])

  if not data_list:
    raise RuntimeError(
      "Z dodaných časových intervalů se nepodařilo vytvořit žádné epochy. "
      "Zkontrolujte 'events.epoch_length' a rozsahy intervalů."
    )

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

  print(
    f"Vytvořeno epoch: {len(epochs)} (z CSV), třídy: "
    + ", ".join(f"{k} -> {v}" for k, v in event_id.items())
  )

  return epochs


def run_offline_from_file(file_path_str: str) -> Tuple[float, int]:
  """Načte EDF/BDF soubor, vytvoří epochy, spočítá příznaky a natrénuje klasifikátor.

  Vrací (test_accuracy, n_epochs).
  """

  config = load_config()
  file_path = Path(file_path_str).expanduser().resolve()

  if not file_path.is_file():
    raise FileNotFoundError(f"Soubor neexistuje: {file_path}")

  print(f"Načítám EEG soubor: {file_path}")
  raw = _load_raw(file_path)

  epochs = _prepare_epochs(raw, config)

  bands_cfg = config.features.get("bands", [])
  if not bands_cfg:
    raise RuntimeError("V konfiguraci nejsou definována žádná frekvenční pásma (features.bands).")

  X, y, feature_names = compute_bandpower_features(epochs, bands_cfg)
  print(f"Matice příznaků: {X.shape}, počet tříd: {len(np.unique(y))}")

  model, acc = train_and_evaluate(X, y, config=config, save_model=True)

  return acc, len(epochs)
