from __future__ import annotations

import numpy as np
from scipy import signal

import mne

from .config import AppConfig


def _validate_epoch_array(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            f"Expected epoch data with shape (n_epochs, n_channels, n_times), got {array.shape}"
        )
    return array


def apply_notch_filter(data: np.ndarray, sfreq: float, freq: float, quality_factor: float = 30.0) -> np.ndarray:
    """Apply a notch filter to each epoch/channel time series."""
    array = _validate_epoch_array(data)
    if freq <= 0 or freq >= sfreq / 2:
        return array

    flat = array.reshape(-1, array.shape[-1])
    b, a = signal.iirnotch(w0=freq, Q=quality_factor, fs=sfreq)
    filtered = signal.filtfilt(b, a, flat, axis=-1)
    return filtered.reshape(array.shape)


def apply_bandpass_filter(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase band-pass filter to each epoch/channel time series."""
    array = _validate_epoch_array(data)
    nyquist = sfreq / 2.0

    if l_freq <= 0 and h_freq >= nyquist:
        return array

    if l_freq <= 0:
        cutoff = h_freq
        btype = "lowpass"
    elif h_freq >= nyquist:
        cutoff = l_freq
        btype = "highpass"
    else:
        cutoff = [l_freq, h_freq]
        btype = "bandpass"

    flat = array.reshape(-1, array.shape[-1])
    sos = signal.butter(order, cutoff, btype=btype, fs=sfreq, output="sos")

    try:
        filtered = signal.sosfiltfilt(sos, flat, axis=-1)
    except ValueError as exc:
        # Try to parse padlen from error message and pad reflectively to satisfy filtfilt
        msg = str(exc)
        import re

        m = re.search(r"padlen, which is (\d+)", msg)
        if m:
            padlen = int(m.group(1))
            n_signals, n_samples = flat.shape
            if n_samples <= padlen:
                need = padlen + 1 - n_samples
                left = need // 2 + (need % 2)
                right = need // 2
                padded = np.pad(flat, ((0, 0), (left, right)), mode="reflect")
                try:
                    filt_padded = signal.sosfiltfilt(sos, padded, axis=-1)
                    # remove padding
                    filtered = filt_padded[:, left : left + n_samples]
                except Exception:
                    # final fallback to forward/backward sosfilt (no padding)
                    filtered = signal.sosfilt(sos, flat, axis=-1)
                    filtered = np.flip(filtered, axis=-1)
                    filtered = signal.sosfilt(sos, filtered, axis=-1)
                    filtered = np.flip(filtered, axis=-1)
            else:
                # Unexpected: re-raise to be handled by outer code
                raise
        else:
            # Unknown ValueError message — fallback to non-pad filtfilt approx
            filtered = signal.sosfilt(sos, flat, axis=-1)
            filtered = np.flip(filtered, axis=-1)
            filtered = signal.sosfilt(sos, filtered, axis=-1)
            filtered = np.flip(filtered, axis=-1)

    return filtered.reshape(array.shape)


def apply_common_average_reference(data: np.ndarray) -> np.ndarray:
    """Apply common average reference across channels for each epoch/sample."""
    array = _validate_epoch_array(data)
    return array - array.mean(axis=1, keepdims=True)


def preprocess_epoch_data(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    notch_freq: float | None = None,
    use_car: bool = False,
) -> np.ndarray:
    """Apply notch, band-pass and optional CAR preprocessing to epoched data."""
    processed = _validate_epoch_array(data).copy()

    if notch_freq is not None:
        processed = apply_notch_filter(processed, sfreq=sfreq, freq=notch_freq)

    processed = apply_bandpass_filter(processed, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq)

    if use_car:
        processed = apply_common_average_reference(processed)

    return processed


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
