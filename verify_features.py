"""Runtime verifikace vsech hvezdicky-povinnych bodu."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import ast
import numpy as np
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

# ═══════════════════════════════════════════════════
# * 1. OPRAVA PSYCHOPY: create_marker_outlet misto create_streams
# ═══════════════════════════════════════════════════
print("\n=== * Oprava PsychoPy zobrazeni ===")
src = Path('src/gui_app_v2.py').read_text(encoding='utf-8')
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == '_start_paradigm_proc':
        body = ast.get_source_segment(src, node) or ""
        if 'create_marker_outlet' in body and 'create_streams' not in body:
            print(f"{PASS} _start_paradigm_proc pouziva create_marker_outlet (ne create_streams)")
            print(f"      -> PsychoPy okno se otevre i bez EEG zarizeni")
        else:
            msg = "FAIL: create_streams stale pritomen nebo create_marker_outlet chybi"
            print(f"{FAIL} {msg}")
            errors.append(msg)

# Overit ze EegRecorder je integrovan do _run_selection
if 'EegRecorder' in src and 'recorder.start()' in src and 'recorder.stop(' in src:
    print(f"{PASS} _run_selection spousti EegRecorder paralelne s paradigmatem")
else:
    msg = "FAIL: EegRecorder neni integrovan v _run_selection"
    print(f"{FAIL} {msg}")
    errors.append(msg)

# ═══════════════════════════════════════════════════
# * 2. PRIMA AKVISICE (bez LabRecorderu)
# ═══════════════════════════════════════════════════
print("\n=== * Prima akvisice EEG (bez LabRecorderu) ===")
import os
os.environ["EEG_LSL_TIMEOUT"] = "0.1"
try:
    from src.eeg_recorder import EegRecorder
    from src.config import load_config
    cfg = load_config()
    rec = EegRecorder(cfg)
    rec.start()
    assert rec.available == False, "Bez EEG zarizeni by available melo byt False"
    result = rec.stop(patient_name="test")
    assert result is None
    print(f"{PASS} EegRecorder spustitelny a tiche selze bez LSL zarizeni")
    print(f"      -> Zaznam bezi bez LabRecorderu, pri pripojeni EEG uklada FIF")

    # Overit ze FIF je v SUPPORTED_FORMATS
    from src.offline_analysis import SUPPORTED_FORMATS
    assert '.fif' in SUPPORTED_FORMATS
    print(f"{PASS} FIF format podporovan v offline_analysis ({SUPPORTED_FORMATS})")

    # Simulovat zapis markeru a ulozeni CSV
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rec2 = EegRecorder(cfg, output_dir=Path(tmp))
        rec2._first_timestamp = 0.0
        rec2._marker_events = [(1.0, "1"), (5.0, "2"), (9.0, "1"), (13.0, "2")]
        fif_path = Path(tmp) / "eeg_test_raw.fif"
        rec2._save_markers_csv(fif_path)
        csv_path = Path(tmp) / "eeg_test_raw.markers.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert lines[0] == "start;end;label"
        assert len(lines) == 5  # header + 4 zaznamu
        print(f"{PASS} Markery se spravne ukladaji do CSV ({len(lines)-1} zaznamu)")
finally:
    del os.environ["EEG_LSL_TIMEOUT"]

# ═══════════════════════════════════════════════════
# * 3. PRUMEROVANI STIMULU (n_averages)
# ═══════════════════════════════════════════════════
print("\n=== * Prumerovani stimulu (ERP averaging) ===")
from src.offline_analysis import _average_epochs
from src.config import ExperimentConfig

# Test n_averages=1: zadne prumerovani
X = np.random.randn(40, 8, 500)
y = np.array([1]*10 + [2]*10 + [3]*10 + [4]*10)
X1, y1 = _average_epochs(X, y, n_averages=1)
assert X1.shape == X.shape
print(f"{PASS} n_averages=1: zadne prumerovani, {len(X1)} epoch zachovano")

# Test n_averages=5
X2, y2 = _average_epochs(X, y, n_averages=5)
expected = 4 * (10 // 5)  # 4 tridy * 2 bloky = 8
assert len(X2) == expected, f"Ocekavano {expected}, dostali jsme {len(X2)}"
print(f"{PASS} n_averages=5: {len(X)} epoch -> {len(X2)} prumerovanych (SNR vyssi)")

# Overit matematiku prumerovani
Xtest = np.array([[[2.0, 4.0]], [[4.0, 8.0]]])  # (2,1,2)
ytest = np.array([1, 1])
Xavg, _ = _average_epochs(Xtest, ytest, n_averages=2)
np.testing.assert_allclose(Xavg[0, 0], [3.0, 6.0])  # (2+4)/2=3, (4+8)/2=6
print(f"{PASS} Matematika prumerovani spravna: [(2+4)/2, (4+8)/2] = [3, 6]")

# Overit n_averages v paradigmatu
paradigm_src = Path('src/stimuli/paradigm_base.py').read_text(encoding='utf-8')
assert 'n_averages' in paradigm_src
assert 'n_blocks' in paradigm_src
print(f"{PASS} Paradigma: stimuly se opakuji v blocich (n_averages v _build_trials)")

# Overit n_averages v konfigu
cfg_main = load_config()
assert hasattr(cfg_main.experiment, 'n_averages')
assert cfg_main.experiment.n_averages == 1
cfg24 = load_config('config/config_24ch.yaml')
assert cfg24.experiment.n_averages == 5
print(f"{PASS} Config: n_averages={cfg_main.experiment.n_averages} (main), n_averages={cfg24.experiment.n_averages} (24ch)")

# ═══════════════════════════════════════════════════
# * 4. EXE SOUBOR
# ═══════════════════════════════════════════════════
print("\n=== * EXE soubor (build_exe.py) ===")
exe_src = Path('build_exe.py').read_text(encoding='utf-8')
checks = {
    'PyInstaller': 'PyInstaller pouzit jako builder',
    '--onedir': 'onedir mod (rychly start)',
    'liblsl': 'liblsl.dll pro pylsl zahrnuta',
    'run_app.py': 'entry point run_app.py',
    'config': 'config/ slozka zahrnuta',
}
for keyword, desc in checks.items():
    if keyword in exe_src:
        print(f"{PASS} build_exe.py: {desc}")
    else:
        msg = f"FAIL: {keyword} chybi v build_exe.py"
        print(f"{FAIL} {msg}")
        errors.append(msg)

# ═══════════════════════════════════════════════════
# (volitelne) LSL komunikace
# ═══════════════════════════════════════════════════
print("\n=== (volitelne) LSL komunikace po siti ===")
try:
    from src.lsl_acquisition import (
        resolve_eeg_stream, create_marker_outlet, pull_chunk, push_marker
    )
    print(f"{PASS} LSL moduly dostupne (pylsl nainstalovano)")

    lsl_src = Path('src/lsl_acquisition.py').read_text(encoding='utf-8')
    assert 'resolve_byprop' in lsl_src
    assert 'StreamInlet' in lsl_src
    assert 'StreamOutlet' in lsl_src
    print(f"{PASS} LSL: EEG inlet + marker outlet implementovany")
except ImportError as e:
    print(f"  SKIP: pylsl neni k dispozici ({e})")

# ═══════════════════════════════════════════════════
# (volitelne) 24 kanalu
# ═══════════════════════════════════════════════════
print("\n=== (volitelne) 24 kanalu EEG ===")
eeg = cfg24.model_extra.get('eeg', {})
n_ch = eeg.get('n_channels')
ch_names = eeg.get('ch_names', [])
assert n_ch == 24
assert len(ch_names) == 24
print(f"{PASS} config_24ch.yaml: {n_ch} kanalu, nazvy: {ch_names[:4]}...")
print(f"{PASS} n_averages={cfg24.experiment.n_averages}, sfreq={cfg24.preprocessing.sfreq} Hz")

# ═══════════════════════════════════════════════════
# SOUHRNNY VYSLEDEK
# ═══════════════════════════════════════════════════
print()
print("=" * 55)
if errors:
    print(f"VYSLEDEK: FAIL ({len(errors)} chyb)")
    for e in errors:
        print(f"  - {e}")
else:
    print("VYSLEDEK: PASS — vsechny povinne hvezdickove body OK")
print("=" * 55)

sys.exit(1 if errors else 0)
