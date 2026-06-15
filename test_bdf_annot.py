"""Test ze EegRecorder uklada BDF+ s pravymi anotacemi ctitelnymi MNE."""
import sys, io, os, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['EEG_LSL_TIMEOUT'] = '0.1'

import numpy as np, mne
from pathlib import Path
from src.config import load_config
from src.eeg_recorder import EegRecorder

cfg = load_config()
sfreq = 250.0
errors = []
def check(l, c, d=''):
    print(f"[{'OK  ' if c else 'FAIL'}] {l}" + (f"  -> {d}" if d else ""))
    if not c: errors.append(l)

with tempfile.TemporaryDirectory() as tmp:
    rec = EegRecorder(cfg, output_dir=Path(tmp))
    rec._sfreq = sfreq
    rec._ch_names = ['Fp1','C3','Cz','C4','P3','Pz','P4','O1']
    n = int(40 * sfreq)
    rec._data_chunks = [np.random.default_rng(0).standard_normal((n, 8)) * 30e-6]  # 30uV ve V
    # 8 markeru trid (1-4) + 4 klidove markery (kod 0)
    rec._marker_events = [(int(s*sfreq), c) for s, c in
                          [(2,'1'),(5,'0'),(8,'2'),(11,'0'),(14,'3'),(17,'0'),
                           (20,'4'),(23,'0'),(26,'1'),(29,'2'),(32,'3'),(35,'4')]]

    path = rec._save_edf('TestAnot')
    check("1. BDF ulozen", path is not None and path.exists(), str(path.suffix))
    check("2. Pripona .bdf", path.suffix == '.bdf')

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose='ERROR')
    check("3. MNE precetl vsechny anotace", len(raw.annotations) == 12,
          f"{len(raw.annotations)} anotaci")
    check("4. Zadny Status kanal", 'Status' not in raw.ch_names)

    descs = [a['description'] for a in sorted(raw.annotations, key=lambda a: a['onset'])]
    check("5. Citelne popisky (PRAVA RUKA atd.)",
          'PRAVA RUKA' in descs and 'LEVA RUKA' in descs and 'JAZYK' in descs and 'OBE NOHY' in descs,
          f"{set(descs)}")
    check("6. Klidove anotace 'KLID'", descs.count('KLID') == 4, f"KLID x{descs.count('KLID')}")

    d = raw.get_data(picks=mne.pick_types(raw.info, eeg=True))
    rms = float(np.sqrt(np.mean(d**2)))*1e6
    check("7. Meritko ~30uV", 20 < rms < 50, f"{rms:.1f} uV")

    # Offline epochovani: klid se vynecha, zustanou jen 4 tridy (8 epoch)
    from src.offline_analysis import _prepare_epochs
    tc = cfg.model_copy(deep=True)
    tc.events['tmin']=0.0; tc.events['tmax']=2.0
    eps = _prepare_epochs(raw, tc)
    check("8. Trenink vynecha KLID, 8 epoch trid", len(eps) == 8, f"{len(eps)} epoch")
    check("9. Tridy spravne namapovany", set(eps.event_id.keys()) ==
          {'LEFT_HAND','RIGHT_HAND','BOTH_FEET','TONGUE'}, f"{set(eps.event_id.keys())}")

print()
if errors: print("FAIL:", errors); sys.exit(1)
print("PASS - BDF+ anotace funguji, MNE je cte do raw.annotations")
