"""Test ERD analyzy na syntetickem signalu s kontralateralnim utlumem."""
import sys, io, os, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mne
from pathlib import Path
from src.config import load_config
from src.analysis_erd import run_erd_analysis

cfg = load_config()
sfreq = 250.0
rng = np.random.default_rng(0)

# 8 kanalu vcetne C3, C4, Cz
ch_names = ['Fp1', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1']
n_ch = len(ch_names)

# Casovani: marker = zacatek imaginace. Pred markerem baseline.
# Trial: 2s baseline + 4s imagery = 6s. Marker na 2s.
imagery_dur = 4.0
baseline_dur = 2.0
trial_dur = baseline_dur + imagery_dur
n_trials_per_class = 8
classes = {'LEFT_HAND': 1, 'RIGHT_HAND': 2, 'BOTH_FEET': 3, 'TONGUE': 4}

# Sestavit kontinualni signal
trials = []
for label, code in classes.items():
    for _ in range(n_trials_per_class):
        trials.append((label, code))
rng.shuffle(trials)

total_samples = int(len(trials) * trial_dur * sfreq)
data = rng.standard_normal((n_ch, total_samples)) * 10e-6  # sum
t = np.arange(total_samples) / sfreq

# mu rytmus 10 Hz pridame vsude jako baseline
mu = np.sin(2 * np.pi * 10 * t) * 20e-6
for ch in range(n_ch):
    data[ch] += mu

events_list = []
sample = 0
for label, code in trials:
    marker_sample = sample + int(baseline_dur * sfreq)
    imag_start = marker_sample
    imag_end = marker_sample + int(imagery_dur * sfreq)

    # Simulovat ERD: utlumit mu v kontralateralni hemisfere behem imaginace
    idx_c3 = ch_names.index('C3')
    idx_c4 = ch_names.index('C4')
    idx_cz = ch_names.index('Cz')

    if code == 2:  # RIGHT_HAND -> utlum C3 (leva hemisfera)
        data[idx_c3, imag_start:imag_end] -= mu[imag_start:imag_end] * 0.8
    elif code == 1:  # LEFT_HAND -> utlum C4 (prava hemisfera)
        data[idx_c4, imag_start:imag_end] -= mu[imag_start:imag_end] * 0.8
    elif code == 3:  # BOTH_FEET -> utlum Cz
        data[idx_cz, imag_start:imag_end] -= mu[imag_start:imag_end] * 0.8

    events_list.append((marker_sample, code))
    sample += int(trial_dur * sfreq)

info = mne.create_info(ch_names, sfreq, 'eeg')
raw = mne.io.RawArray(data, info, verbose='ERROR')

# Pridat anotace (markery na zacatku imaginace)
onsets = [s / sfreq for s, _ in events_list]
descs = [str(c) for _, c in events_list]
raw.set_annotations(mne.Annotations(onset=onsets, duration=[imagery_dur]*len(onsets), description=descs))

with tempfile.TemporaryDirectory() as tmp:
    fif = Path(tmp) / 'test_erd_raw.fif'
    raw.save(str(fif), overwrite=True, verbose='ERROR')

    result = run_erd_analysis(str(fif), config=cfg)
    print(result['report_text'])
    print()
    print('Graf:', result['figure_path'])

    # Overit ze lateralizace je spravna
    r = result['classes']
    print()
    print('=== KONTROLA LATERALIZACE ===')
    ok_right = r['RIGHT_HAND']['C3']['mu_erd'] < r['RIGHT_HAND']['C4']['mu_erd']
    ok_left  = r['LEFT_HAND']['C4']['mu_erd'] < r['LEFT_HAND']['C3']['mu_erd']
    print(f"RIGHT_HAND: C3 utlum silnejsi nez C4? {ok_right}  (C3={r['RIGHT_HAND']['C3']['mu_erd']:.1f}, C4={r['RIGHT_HAND']['C4']['mu_erd']:.1f})")
    print(f"LEFT_HAND:  C4 utlum silnejsi nez C3? {ok_left}  (C3={r['LEFT_HAND']['C3']['mu_erd']:.1f}, C4={r['LEFT_HAND']['C4']['mu_erd']:.1f})")
    if ok_right and ok_left:
        print()
        print('PASS - ERD spravne detekuje kontralateralni utlum')
        sys.exit(0)
    else:
        print('FAIL - lateralizace nesedi')
        sys.exit(1)
