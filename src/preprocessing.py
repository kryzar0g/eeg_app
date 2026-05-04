from __future__ import annotations

import mne

from .config import AppConfig


def bandpass_filter(
  raw: mne.io.BaseRaw,
  l_freq: float,
  h_freq: float,
) -> mne.io.BaseRaw:
  """Aplikuje band-pass FIR filtr na raw EEG data.

  Modifikuje objekt in-place a zároveň ho vrací.
  """

  raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose="ERROR")
  return raw


def notch_filter(raw: mne.io.BaseRaw, freq: float) -> mne.io.BaseRaw:
  """Aplikuje notch filtr na raw EEG data (typicky 50 nebo 60 Hz).

  Modifikuje objekt in-place a zároveň ho vrací.
  """

  raw.notch_filter(freqs=[freq], verbose="ERROR")
  return raw


def preprocess_raw(raw: mne.io.BaseRaw, config: AppConfig) -> mne.io.BaseRaw:
  """Předzpracuje raw EEG data: notch + band-pass filtr.

  Parametry jsou načteny z konfigurace (preprocessing sekce).
  Vrací předzpracovaný raw objekt.
  """

  prep = config.preprocessing
  l_freq = float(prep.get("l_freq", 8.0))
  h_freq = float(prep.get("h_freq", 30.0))
  notch_freq = float(prep.get("notch_freq", 50.0))

  notch_filter(raw, notch_freq)
  bandpass_filter(raw, l_freq, h_freq)

  return raw
