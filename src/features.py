from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import mne
import numpy as np


def compute_bandpower_features(
  epochs: mne.Epochs,
  bands: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
  """Vypočítá log-bandpower příznaky pro zadaná pásma.

  Vrací (X, y, feature_names), kde:
  - X má tvar (n_epochs, n_features),
  - y je vektor tříd (kódů událostí),
  - feature_names jsou názvy příznaků.
  """

  # PSD přes všechny epochy / kanály
  psd_obj = epochs.compute_psd(method="welch", fmin=0.0, fmax=max(b[1] for b in bands))
  psd = psd_obj.get_data()  # shape: (n_epochs, n_channels, n_freqs)
  freqs = psd_obj.freqs

  n_epochs, n_channels, _ = psd.shape
  bands = [tuple(map(float, b)) for b in bands]

  features: List[np.ndarray] = []
  feature_names: List[str] = []

  for band_idx, (fmin, fmax) in enumerate(bands):
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx.size == 0:
      # pokud pásmo neobsahuje žádné frekvence, vytvoříme nuly
      band_power = np.zeros((n_epochs, n_channels), dtype=float)
    else:
      band_psd = psd[:, :, idx].mean(axis=-1)  # (n_epochs, n_channels)
      # log-transformace pro stabilizaci rozdělení
      band_power = np.log10(band_psd + 1e-12)

    features.append(band_power)

    for ch_name in epochs.ch_names:
      feature_names.append(f"{ch_name}_band{band_idx}")

  # spojíme pásma do jedné matice (n_epochs, n_channels * n_bands)
  X = np.concatenate(features, axis=1)

  # třídy odpovídají ID událostí v epochách
  y = epochs.events[:, 2].astype(int)

  return X, y, feature_names
