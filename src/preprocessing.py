from __future__ import annotations

import mne

from .config import AppConfig


def bandpass_filter(
    raw: mne.io.BaseRaw,
    l_freq: float,
    h_freq: float,
) -> mne.io.BaseRaw:
    """Apply band-pass FIR filter to EEG data (in-place)."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose="ERROR")
    return raw


def notch_filter(raw: mne.io.BaseRaw, freq: float) -> mne.io.BaseRaw:
    """Apply notch filter to remove power line frequency (in-place)."""
    raw.notch_filter(freqs=[freq], verbose="ERROR")
    return raw


def preprocess_raw(raw: mne.io.BaseRaw, config: AppConfig) -> mne.io.BaseRaw:
    """Apply standard preprocessing: notch filter + band-pass filter.
    
    Parameters loaded from config.preprocessing section.
    """
    prep = config.preprocessing
    l_freq = float(prep.l_freq)
    h_freq = float(prep.h_freq)
    notch_freq = float(prep.notch_freq)
    
    notch_filter(raw, notch_freq)
    bandpass_filter(raw, l_freq, h_freq)
    
    return raw
