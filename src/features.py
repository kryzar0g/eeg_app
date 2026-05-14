from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy import linalg, signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted

from .preprocessing import preprocess_epoch_data, apply_bandpass_filter


def _ensure_epoch_array(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            f"Expected epoch data with shape (n_epochs, n_channels, n_times), got {array.shape}"
        )
    return array


def _compute_covariance(epoch: np.ndarray) -> np.ndarray:
    centered = epoch - epoch.mean(axis=-1, keepdims=True)
    cov = centered @ centered.T
    trace = float(np.trace(cov))
    if trace <= 0:
        n_channels = epoch.shape[0]
        return np.eye(n_channels, dtype=float) / max(n_channels, 1)
    return cov / trace


def _compute_bandpower_array(data: np.ndarray, sfreq: float, bands: Sequence[Sequence[float]]) -> np.ndarray:
    array = _ensure_epoch_array(data)
    n_epochs, n_channels, n_times = array.shape
    nperseg = min(256, n_times)

    flat = array.reshape(-1, n_times)
    freqs, psd = signal.welch(flat, fs=sfreq, axis=-1, nperseg=nperseg)
    psd = psd.reshape(n_epochs, n_channels, -1)

    features: List[np.ndarray] = []
    for fmin, fmax in bands:
        band_mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
        if not np.any(band_mask):
            band_power = np.zeros((n_epochs, n_channels), dtype=float)
        else:
            band_power = np.log10(psd[:, :, band_mask].mean(axis=-1) + 1e-12)
        features.append(band_power)

    return np.concatenate(features, axis=1)


class EpochSignalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sfreq: float,
        l_freq: float,
        h_freq: float,
        notch_freq: float | None = 50.0,
        use_car: bool = False,
        expected_n_times: int | None = None,
    ) -> None:
        self.sfreq = float(sfreq)
        self.l_freq = float(l_freq)
        self.h_freq = float(h_freq)
        self.notch_freq = None if notch_freq is None else float(notch_freq)
        self.use_car = bool(use_car)
        self.expected_n_times = None if expected_n_times is None else int(expected_n_times)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        array = _ensure_epoch_array(X)
        self.n_channels_ = int(array.shape[1])
        self.n_times_ = int(array.shape[2])
        if self.expected_n_times is not None and self.n_times_ != self.expected_n_times:
            raise ValueError(
                f"Expected epochs with {self.expected_n_times} samples, got {self.n_times_}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["n_channels_", "n_times_"])
        array = _ensure_epoch_array(X)
        if array.shape[1] != self.n_channels_:
            raise ValueError(f"Expected {self.n_channels_} channels, got {array.shape[1]}")
        if array.shape[2] != self.n_times_:
            raise ValueError(f"Expected {self.n_times_} samples per epoch, got {array.shape[2]}")
        return preprocess_epoch_data(
            array,
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            notch_freq=self.notch_freq,
            use_car=self.use_car,
        )


class BandpowerFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq: float, bands: Sequence[Sequence[float]], expected_n_times: int | None = None) -> None:
        self.sfreq = float(sfreq)
        self.bands = [tuple(map(float, band)) for band in bands]
        self.expected_n_times = None if expected_n_times is None else int(expected_n_times)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        array = _ensure_epoch_array(X)
        self.n_channels_ = int(array.shape[1])
        self.n_times_ = int(array.shape[2])
        if self.expected_n_times is not None and self.n_times_ != self.expected_n_times:
            raise ValueError(
                f"Expected epochs with {self.expected_n_times} samples, got {self.n_times_}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["n_channels_", "n_times_"])
        array = _ensure_epoch_array(X)
        if array.shape[1] != self.n_channels_:
            raise ValueError(f"Expected {self.n_channels_} channels, got {array.shape[1]}")
        if array.shape[2] != self.n_times_:
            raise ValueError(f"Expected {self.n_times_} samples per epoch, got {array.shape[2]}")
        return _compute_bandpower_array(array, sfreq=self.sfreq, bands=self.bands)


class BinaryCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 2, reg: float = 1e-6) -> None:
        self.n_components = int(n_components)
        self.reg = float(reg)

    def fit(self, X: np.ndarray, y: np.ndarray):
        array = _ensure_epoch_array(X)
        labels = np.asarray(y).astype(int)
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("BinaryCSP requires exactly two classes")

        cov_a = np.mean([_compute_covariance(epoch) for epoch in array[labels == classes[0]]], axis=0)
        cov_b = np.mean([_compute_covariance(epoch) for epoch in array[labels == classes[1]]], axis=0)
        cov_total = cov_a + cov_b

        n_channels = cov_total.shape[0]
        cov_total = cov_total + np.eye(n_channels, dtype=float) * self.reg

        evals, evecs = linalg.eigh(cov_a, cov_total)
        order = np.argsort(evals)[::-1]
        filters = evecs[:, order]

        n_components = min(self.n_components, filters.shape[1])
        n_top = int(np.ceil(n_components / 2.0))
        n_bottom = n_components - n_top
        selected_indices = list(range(n_top)) + list(range(filters.shape[1] - n_bottom, filters.shape[1]))
        self.filters_ = filters[:, selected_indices]
        self.n_channels_ = int(array.shape[1])
        self.n_times_ = int(array.shape[2])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["filters_", "n_channels_", "n_times_"])
        array = _ensure_epoch_array(X)
        if array.shape[1] != self.n_channels_:
            raise ValueError(f"Expected {self.n_channels_} channels, got {array.shape[1]}")
        if array.shape[2] != self.n_times_:
            raise ValueError(f"Expected {self.n_times_} samples per epoch, got {array.shape[2]}")

        features = []
        for epoch in array:
            projected = self.filters_.T @ epoch
            variance = np.var(projected, axis=-1)
            normalized = variance / (variance.sum() + 1e-12)
            features.append(np.log10(normalized + 1e-12))
        return np.asarray(features, dtype=float)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq: float, bands: Sequence[Sequence[float]], n_components: int = 2, reg: float = 1e-6, top_k_features: int = 0, expected_n_times: int | None = None) -> None:
        self.sfreq = float(sfreq)
        self.bands = [tuple(map(float, band)) for band in bands]
        self.n_components = int(n_components)
        self.reg = float(reg)
        self.top_k_features = int(top_k_features)
        self.expected_n_times = None if expected_n_times is None else int(expected_n_times)

    def fit(self, X: np.ndarray, y: np.ndarray):
        array = _ensure_epoch_array(X)
        labels = np.asarray(y).astype(int)
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError("FilterBankCSP requires at least two classes")

        self.classes_ = classes.tolist()
        self.n_channels_ = int(array.shape[1])
        self.n_times_ = int(array.shape[2])
        if self.expected_n_times is not None and self.n_times_ != self.expected_n_times:
            raise ValueError(
                f"Expected epochs with {self.expected_n_times} samples, got {self.n_times_}"
            )

        self.band_models_ = []
        for fmin, fmax in self.bands:
            band_data = apply_bandpass_filter(array, sfreq=self.sfreq, l_freq=fmin, h_freq=fmax)
            class_models = []
            for cls in self.classes_:
                binary_y = (labels == cls).astype(int)
                model = BinaryCSP(n_components=self.n_components, reg=self.reg)
                model.fit(band_data, binary_y)
                class_models.append(model)
            self.band_models_.append(class_models)

        raw_features = self._transform_features(array)
        if self.top_k_features > 0 and raw_features.shape[1] > self.top_k_features:
            scores = mutual_info_classif(raw_features, labels, random_state=42)
            self.selected_feature_indices_ = np.argsort(scores)[::-1][: self.top_k_features]
            self.n_features_in_ = int(self.top_k_features)
        else:
            self.selected_feature_indices_ = None
            self.n_features_in_ = int(raw_features.shape[1])
        return self

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        features = []
        for band_index, (fmin, fmax) in enumerate(self.bands):
            band_data = apply_bandpass_filter(X, sfreq=self.sfreq, l_freq=fmin, h_freq=fmax)
            for class_model in self.band_models_[band_index]:
                features.append(class_model.transform(band_data))
        return np.concatenate(features, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["band_models_", "classes_", "n_channels_", "n_times_"])
        array = _ensure_epoch_array(X)
        if array.shape[1] != self.n_channels_:
            raise ValueError(f"Expected {self.n_channels_} channels, got {array.shape[1]}")
        if array.shape[2] != self.n_times_:
            raise ValueError(f"Expected {self.n_times_} samples per epoch, got {array.shape[2]}")

        features = self._transform_features(array)
        if self.selected_feature_indices_ is not None:
            return features[:, self.selected_feature_indices_]
        return features
