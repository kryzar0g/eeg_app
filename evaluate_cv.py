"""Podrobna cross-validovana evaluace FBCSP+LDA na namerených datech."""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = sys.stdout
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              confusion_matrix, balanced_accuracy_score)
from src.config import load_config
from src.offline_analysis import _load_raw, _prepare_epochs, _crop_epoch_array
from src.classifier import _build_model

config = load_config()
target_sfreq = float(config.preprocessing.sfreq)
CLASS_NAMES = {1: 'LEVA RUKA', 2: 'PRAVA RUKA', 3: 'OBE NOHY', 4: 'JAZYK'}
SEP = "=" * 60


def evaluate_file(fname, label):
    raw = _load_raw(Path(fname))
    epochs = _prepare_epochs(raw, config)
    if abs(float(epochs.info['sfreq']) - target_sfreq) > 1.0:
        print(f"  Resampling {epochs.info['sfreq']:.0f} -> {target_sfreq:.0f} Hz")
        epochs = epochs.resample(target_sfreq, verbose=False)

    X = epochs.get_data()
    y = epochs.events[:, 2].astype(int)

    crop_s = int(round(config.training.window_length_sec * target_sfreq))
    step_s = int(round(config.training.crop_step_sec * target_sfreq))
    # augment=False: jedno okno na epochu (stred) pro cistou CV
    X, y = _crop_epoch_array(X, y, crop_s, step_s, augment=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        m = _build_model(config)
        m.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])

    acc  = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=[1, 2, 3, 4])

    print(SEP)
    print(f"  {label}")
    print(SEP)
    print(f"  Epoch: {len(epochs)}   |  Okno: {config.training.window_length_sec}s  |  5-fold CV")
    print(f"  Presnost (accuracy):      {acc*100:.1f}%")
    print(f"  Balanced accuracy:        {bacc*100:.1f}%")
    print(f"  Cohen kappa:              {kappa:.3f}  (nahodne=0, perfektni=1)")
    print(f"  Nahodna volba (4 tridy):  25.0%")
    print()
    print("  Confusion matrix [skutecnost x predikce]:")
    header = f"  {'':12}" + "".join(f"{CLASS_NAMES[c]:>12}" for c in [1,2,3,4])
    print(header)
    for i, cls in enumerate([1, 2, 3, 4]):
        row = cm[i]
        total = row.sum()
        pct = row / total * 100 if total > 0 else np.zeros(4)
        cells = "".join(f"{v:>9}({p:2.0f}%)" for v, p in zip(row, pct))
        print(f"  {CLASS_NAMES[cls]:12}{cells}")
    print()
    per_class = cm.diagonal() / cm.sum(axis=1)
    for i, cls in enumerate([1, 2, 3, 4]):
        print(f"  {CLASS_NAMES[cls]:12} presnost: {per_class[i]*100:.1f}%")
    return kappa, acc, bacc


def load_epochs_xy(fname):
    raw = _load_raw(Path(fname))
    epochs = _prepare_epochs(raw, config)
    if abs(float(epochs.info['sfreq']) - target_sfreq) > 1.0:
        epochs = epochs.resample(target_sfreq, verbose=False)
    X = epochs.get_data()
    y = epochs.events[:, 2].astype(int)
    crop_s = int(round(config.training.window_length_sec * target_sfreq))
    step_s = int(round(config.training.crop_step_sec * target_sfreq))
    X, y = _crop_epoch_array(X, y, crop_s, step_s, augment=False)
    return X, y


def evaluate_combined(fnames, label):
    all_X, all_y = [], []
    total = 0
    for fname in fnames:
        raw = _load_raw(Path(fname))
        epochs = _prepare_epochs(raw, config)
        if abs(float(epochs.info['sfreq']) - target_sfreq) > 1.0:
            epochs = epochs.resample(target_sfreq, verbose=False)
        X = epochs.get_data()
        y = epochs.events[:, 2].astype(int)
        crop_s = int(round(config.training.window_length_sec * target_sfreq))
        step_s = int(round(config.training.crop_step_sec * target_sfreq))
        X, y = _crop_epoch_array(X, y, crop_s, step_s, augment=False)
        all_X.append(X)
        all_y.append(y)
        total += len(epochs)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        m = _build_model(config)
        m.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])

    acc  = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=[1, 2, 3, 4])

    print(SEP)
    print(f"  {label}")
    print(SEP)
    print(f"  Epoch celkem: {total}   |  Okno: {config.training.window_length_sec}s  |  5-fold CV")
    print(f"  Presnost (accuracy):      {acc*100:.1f}%")
    print(f"  Balanced accuracy:        {bacc*100:.1f}%")
    print(f"  Cohen kappa:              {kappa:.3f}")
    print()
    print("  Confusion matrix:")
    header = f"  {'':12}" + "".join(f"{CLASS_NAMES[c]:>12}" for c in [1,2,3,4])
    print(header)
    for i, cls in enumerate([1, 2, 3, 4]):
        row = cm[i]
        total_row = row.sum()
        pct = row / total_row * 100 if total_row > 0 else np.zeros(4)
        cells = "".join(f"{v:>9}({p:2.0f}%)" for v, p in zip(row, pct))
        print(f"  {CLASS_NAMES[cls]:12}{cells}")
    return kappa, acc


print()
k1, a1, b1 = evaluate_file('data/recordings/mereni1.bdf', 'MERENI 1  (80 trialu, 20x kazda trida)')
print()
k2, a2, b2 = evaluate_file('data/recordings/mereni2.bdf', 'MERENI 2  (36 trialu, neuplne)')

# Cross-session: trénink na mereni1, test na mereni2
print()
print(SEP)
print("  CROSS-SESSION: train=mereni1, test=mereni2")
print(SEP)
X_tr, y_tr = load_epochs_xy('data/recordings/mereni1.bdf')
X_te, y_te = load_epochs_xy('data/recordings/mereni2.bdf')
m = _build_model(config)
m.fit(X_tr, y_tr)
y_p = m.predict(X_te)
acc_cs  = accuracy_score(y_te, y_p)
kappa_cs = cohen_kappa_score(y_te, y_p)
print(f"  Train: {len(y_tr)} epoch   |   Test: {len(y_te)} epoch")
print(f"  Presnost:    {acc_cs*100:.1f}%   |   Kappa: {kappa_cs:.3f}")

print()
k_comb, a_comb = evaluate_combined(
    ['data/recordings/mereni1.bdf', 'data/recordings/mereni2.bdf'],
    'KOMBINOVANE  (mereni1 + mereni2)',
)

print()
print(SEP)
print("  SOUHRN")
print(SEP)
print(f"  mereni1 (5-fold CV)   acc={a1*100:.1f}%  kappa={k1:.3f}")
print(f"  mereni2 (5-fold CV)   acc={a2*100:.1f}%  kappa={k2:.3f}")
print(f"  kombinovane CV        acc={a_comb*100:.1f}%  kappa={k_comb:.3f}")
print(f"  cross-session         acc={acc_cs*100:.1f}%  kappa={kappa_cs:.3f}")
print(f"  Nahodna volba:        25.0%  (kappa=0.0)")
print()
print("  => Potreba vice dat (72+ trialu na tridu, vice subjektu)")
