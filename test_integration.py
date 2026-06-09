"""Integracni test: 4-tridni BDF -> trenink modelu -> ERD analyza."""
import sys, io, os, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['EEG_LSL_TIMEOUT'] = '0.1'

import numpy as np
from pathlib import Path
from src.config import load_config
from src.eeg_recorder import EegRecorder

PASS, FAIL = '[OK  ]', '[FAIL]'
errors = []
def check(label, cond, detail=''):
    print(f"{PASS if cond else FAIL} {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        errors.append(label)

cfg = load_config()
sfreq = 250.0
ch_names = ['Fp1','C3','Cz','C4','P3','Pz','P4','O1']
classes = cfg.paradigm['classes']   # 4 tridy
imagery = cfg.experiment.imagery_duration
baseline = cfg.experiment.baseline_duration
trial = baseline + imagery
n_per = 10
rng = np.random.default_rng(7)

# Sestavit trialy
trials = [(l, c) for l, c in classes.items() for _ in range(n_per)]
rng.shuffle(trials)
total = int(len(trials) * trial * sfreq)
data = rng.standard_normal((len(ch_names), total)) * 8e-6
t = np.arange(total) / sfreq
mu = np.sin(2*np.pi*10*t) * 18e-6
for ch in range(len(ch_names)):
    data[ch] += mu

# Markery na zacatku imaginace + kontralateralni ERD
markers = []
s = 0
ci = {n: ch_names.index(n) for n in ('C3','C4','Cz')}
for label, code in trials:
    m = s + int(baseline*sfreq)
    e = m + int(imagery*sfreq)
    if code == 2: data[ci['C3'], m:e] -= mu[m:e]*0.8    # prava ruka
    elif code == 1: data[ci['C4'], m:e] -= mu[m:e]*0.8  # leva ruka
    elif code == 3: data[ci['Cz'], m:e] -= mu[m:e]*0.8  # nohy
    markers.append((m/sfreq, str(code)))
    s += int(trial*sfreq)

with tempfile.TemporaryDirectory() as tmp:
    # Zapsat BDF pres EegRecorder (realny format)
    rec = EegRecorder(cfg, output_dir=Path(tmp))
    rec._sfreq = sfreq
    rec._ch_names = ch_names
    rec._first_timestamp = 0.0
    rec._data_chunks = [data.T]   # (n_samples, n_channels)
    rec._marker_events = markers
    bdf_path = rec._save_edf('IntegrationTest')

    check("1. BDF zaznam vytvoren", bdf_path is not None and bdf_path.exists(), bdf_path.suffix)

    # 2. Trenink modelu na 4-tridnim BDF
    try:
        from src.offline_analysis import run_offline_from_file
        acc, n_epochs = run_offline_from_file(str(bdf_path))
        check("2. Trenink modelu probehl", n_epochs > 0, f"{n_epochs} epoch, acc={acc:.3f}")
        # 4 tridy -> chance ~25%, se simulovanym signalem by melo byt vyssi
        check("2b. Presnost nad nahodou (>0.3)", acc > 0.3, f"acc={acc:.3f}")
    except Exception as ex:
        check("2. Trenink modelu probehl", False, str(ex)[:120])

    # 3. Model byl ulozen
    from src.config import PROJECT_ROOT
    model_path = PROJECT_ROOT / "models" / "model_latest.joblib"
    check("3. Model ulozen", model_path.exists(), str(model_path.name))

    # 4. ERD analyza
    try:
        from src.analysis_erd import run_erd_analysis
        res = run_erd_analysis(str(bdf_path), config=cfg)
        r = res['classes']
        check("4. ERD analyza probehla", len(r) > 0, f"{len(r)} trid")
        ok_r = r['RIGHT_HAND']['C3']['mu_erd'] < r['RIGHT_HAND']['C4']['mu_erd']
        ok_l = r['LEFT_HAND']['C4']['mu_erd'] < r['LEFT_HAND']['C3']['mu_erd']
        check("4b. Kontralateralni lateralizace OK", ok_r and ok_l,
              f"R:C3={r['RIGHT_HAND']['C3']['mu_erd']:.0f} L:C4={r['LEFT_HAND']['C4']['mu_erd']:.0f}")
        check("4c. ERD graf vytvoren", res.get('figure_path') and Path(res['figure_path']).exists())
    except Exception as ex:
        import traceback; traceback.print_exc()
        check("4. ERD analyza probehla", False, str(ex)[:120])

print()
print("=" * 52)
if errors:
    print(f"FAIL - {len(errors)} problemu: {errors}")
    sys.exit(1)
print("PASS - cely flow (BDF -> trenink -> ERD) funguje")
print("=" * 52)
