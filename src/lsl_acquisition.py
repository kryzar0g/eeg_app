from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop

from .config import AppConfig


@dataclass
class LSLStreams:
  """Drží objekty pro EEG inlet a marker outlet."""

  eeg_inlet: StreamInlet
  marker_outlet: StreamOutlet


def resolve_eeg_stream(config: AppConfig, timeout: float = 10.0) -> StreamInlet:
  """Najde EEG LSL stream podle jména/typu a vytvoří inlet.

  Timeout je v sekundách.
  """

  name = config.lsl.get("eeg_stream_name", "EEG")
  stype = config.lsl.get("eeg_stream_type", "EEG")

  streams = resolve_byprop("type", stype, timeout=timeout)
  if not streams:
    raise RuntimeError(f"Nebyl nalezen žádný LSL stream typu '{stype}'.")

  if name:
    named = [s for s in streams if s.name() == name]
    if not named:
      print(
        f"Varování: nenašel se stream se jménem '{name}', používám první dostupný."
      )
      info = streams[0]
    else:
      info = named[0]
  else:
    info = streams[0]

  inlet = StreamInlet(info, max_chunklen=0)
  return inlet


def create_marker_outlet(config: AppConfig) -> StreamOutlet:
  """Vytvoří LSL marker outlet pro experimentální události."""

  name = config.lsl.get("marker_stream_name", "Markers")
  info = StreamInfo(
    name=name,
    type="Markers",
    channel_count=1,
    nominal_srate=0.0,
    channel_format="string",
    source_id="eeg_app_markers",
  )
  outlet = StreamOutlet(info)
  return outlet


def create_streams(config: AppConfig) -> LSLStreams:
  """Připraví EEG inlet a marker outlet podle konfigurace."""

  eeg_inlet = resolve_eeg_stream(config)
  marker_outlet = create_marker_outlet(config)
  return LSLStreams(eeg_inlet=eeg_inlet, marker_outlet=marker_outlet)


def pull_chunk(eeg_inlet: StreamInlet, max_samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
  """Přečte chunk vzorků z EEG inletu.

  Vrací dvojici (data, timestamps), kde data je pole tvaru (n_samples, n_channels).
  """

  chunk, timestamps = eeg_inlet.pull_chunk(max_samples=max_samples)
  if not timestamps:
    return np.empty((0, 0)), np.empty((0,))

  data = np.asarray(chunk, dtype=float)
  ts = np.asarray(timestamps, dtype=float)
  return data, ts


def push_marker(marker_outlet: StreamOutlet, code: str) -> None:
  """Odešle marker (událost) přes LSL outlet."""

  marker_outlet.push_sample([code])
