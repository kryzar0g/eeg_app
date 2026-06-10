"""Test spravneho meritka BDF zapisu pro oba typy vstupu (uV a V)."""
import sys, io, numpy as np, tempfile, os, mne
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['EEG_LSL_TIMEOUT'] = '0.1'

from src.config import load_config
from src.eeg_recorder import EegRecorder
from pathlib import Path

cfg = load_config()
errors = []

cases = [
    ('LSL uV vstup (typicky z EEG zarizeni)',  50.0,   'uV'),
    ('MNE V vstup  (SI jednotky)',             50e-6,  'V' ),
]

with tempfile.TemporaryDirectory() as tmp:
    for label, data_scale, unit in cases:
        rec = EegRecorder(cfg, output_dir=Path(tmp))
        rec._sfreq = 250.0
        rec._ch_names = ['CH'+str(i+1) for i in range(4)]
        rec._first_timestamp = 0.0
        rng = np.random.default_rng(42)
        rec._data_chunks = [rng.standard_normal((2500, 4)) * data_scale]
        rec._marker_events = [(int(s*250.0), c) for s, c in [(1.0,'1'),(3.0,'2'),(5.0,'3'),(7.0,'4')]]
        path = rec._save_edf('test')

        if path.suffix == '.bdf':
            raw = mne.io.read_raw_bdf(str(path), preload=True, verbose='ERROR')
        else:
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose='ERROR')

        eeg_picks = mne.pick_types(raw.info, eeg=True)
        data_back = raw.get_data(picks=eeg_picks)

        # MNE vraci data v V (SI), prevedt na uV pro kontrolu
        rms_uv = float(np.sqrt(np.mean(data_back**2))) * 1e6
        ok = 20.0 < rms_uv < 200.0

        status = 'OK  ' if ok else 'FAIL'
        print(f'[{status}] {label}')
        print(f'       format={path.suffix}, RMS={rms_uv:.1f} uV (ocekavano ~50 uV)')
        if not ok:
            errors.append(label)

        # Overit markery
        if path.suffix == '.bdf':
            try:
                events = mne.find_events(raw, stim_channel='Status',
                                         shortest_event=1, verbose='ERROR')
                print(f'       markery: {len(events)} eventi nalezeno (ocekavano 4)')
            except Exception as e:
                print(f'       markery: chyba - {e}')

print()
if errors:
    print('CELKEM: FAIL')
    for e in errors:
        print(' -', e)
    sys.exit(1)
else:
    print('CELKEM: PASS - meritko BDF je spravne')
