"""Test priameho BDF zapisu a nacteni zpet pres MNE."""
import sys, io, numpy as np, tempfile, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['EEG_LSL_TIMEOUT'] = '0.1'

from src.config import load_config
from src.eeg_recorder import EegRecorder
from pathlib import Path
import mne

cfg = load_config()

PASS = '[OK  ]'
FAIL = '[FAIL]'
errors = []

def check(label, cond, detail=''):
    if cond:
        print(f'{PASS} {label}' + (f'  -> {detail}' if detail else ''))
    else:
        print(f'{FAIL} {label}' + (f'  -> {detail}' if detail else ''))
        errors.append(label)

with tempfile.TemporaryDirectory() as tmp:
    sfreq = 250.0
    n_ch = 8
    n_samples = int(10 * sfreq)

    rec = EegRecorder(cfg, output_dir=Path(tmp))
    rec._sfreq = sfreq
    rec._ch_names = [f'CH{i+1}' for i in range(n_ch)]
    rec._first_timestamp = 0.0

    # Synteticka EEG data v mikrovoltach (realny rozsah ~20 uV RMS)
    rng = np.random.default_rng(42)
    chunk = rng.standard_normal((n_samples, n_ch)) * 20e-6  # 20 uV v [V]
    rec._data_chunks = [chunk]

    # 4 tridy markeru
    rec._marker_events = [
        (1.0, '1'), (2.5, '2'), (4.0, '3'), (5.5, '4'),
        (6.0, '1'), (7.0, '2'), (8.0, '3'), (8.5, '4'),
    ]

    # --- Ulozit ---
    path = rec._save_edf(patient_name='TestPacient')

    check('Soubor byl ulozen', path is not None and path.exists())
    if path:
        check('Format je BDF nebo EDF (ne FIF)',
              path.suffix in ('.bdf', '.edf'),
              path.suffix)
        check('Soubor neni prazdny', path.stat().st_size > 1000,
              f'{path.stat().st_size} B')

        # --- Nacist zpet ---
        try:
            if path.suffix == '.bdf':
                raw = mne.io.read_raw_bdf(str(path), preload=True, verbose='ERROR')
            else:
                raw = mne.io.read_raw_edf(str(path), preload=True, verbose='ERROR')

            check('MNE precte soubor zpet', True)
            n_loaded = len(raw.ch_names)
            check('Spravny pocet EEG kanalu', n_loaded >= n_ch,
                  f'{n_loaded} kanalu')
            dur = float(raw.times[-1])
            check('Delka zaznamu ~10 s', 9.0 <= dur <= 11.0,
                  f'{dur:.1f} s')
            # Markery: BDF pouziva Status kanal, EDF/FIF pouziva anotace
            has_status = 'Status' in raw.ch_names
            has_annot  = len(raw.annotations) > 0
            check('Markery zachovany (Status kanal nebo anotace)',
                  has_status or has_annot,
                  f'Status={has_status}, ann={len(raw.annotations)}')

            # Overit data nejsou vse nuly
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            data = raw.get_data(picks=eeg_picks)
            check('EEG data nejsou nula', data.std() > 1e-9,
                  f'std={data.std():.2e}')

            # Zkusit epochovat pres _prepare_epochs (simulace treninku)
            from src.offline_analysis import _prepare_epochs
            test_cfg = cfg.model_copy(deep=True)
            test_cfg.events['tmin'] = 0.0
            test_cfg.events['tmax'] = 1.0
            try:
                epochs = _prepare_epochs(raw, test_cfg)
                check('Epochovani pres _prepare_epochs funguje',
                      len(epochs) > 0, f'{len(epochs)} epoch')
            except Exception as e:
                check('Epochovani pres _prepare_epochs funguje', False, str(e))

        except Exception as e:
            check('MNE precte soubor zpet', False, str(e))

    # --- CSV markery ---
    if path:
        csv = path.parent / (path.stem + '.markers.csv')
        check('CSV markery ulozeny', csv.exists())
        if csv.exists():
            lines = csv.read_text().splitlines()
            check('CSV ma spravny format', lines[0] == 'start;end;label')
            check('CSV ma vsechny markery', len(lines) - 1 == len(rec._marker_events),
                  f'{len(lines)-1} radku')

print()
print('=' * 50)
if errors:
    print(f'FAIL — {len(errors)} problemu:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('PASS — BDF zapis a cteni funguje')
print('=' * 50)
