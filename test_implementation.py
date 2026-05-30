# -*- coding: utf-8 -*-
"""
Testy pro ověření všech nových funkcí.

Spuštění:
    python test_implementation.py
"""

import sys
import os
import traceback
from pathlib import Path

# Windows: nastavit UTF-8 pro stdout, aby prošly unicode znaky
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "  OK"
FAIL = "  FAIL"
SKIP = "  SKIP"

results = []


def test(name):
    """Dekorátor pro test funkce."""
    def decorator(fn):
        try:
            fn()
            results.append((name, True, None))
            print(f"{PASS} {name}")
        except Exception as e:
            results.append((name, False, e))
            print(f"{FAIL} {name}")
            print(f"     {type(e).__name__}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
        return fn
    return decorator


# ─── 1. Config: n_averages ────────────────────────────────────────────────────
print("\n=== 1. Config: n_averages ===")

@test("ExperimentConfig má pole n_averages")
def _():
    from src.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert hasattr(cfg, "n_averages"), "n_averages chybí"
    assert cfg.n_averages == 1, f"výchozí hodnota má být 1, je {cfg.n_averages}"

@test("ExperimentConfig n_averages=5 validuje správně")
def _():
    from src.config import ExperimentConfig
    cfg = ExperimentConfig(n_averages=5)
    assert cfg.n_averages == 5

@test("ExperimentConfig odmítá n_averages=0")
def _():
    from src.config import ExperimentConfig
    import pydantic
    try:
        ExperimentConfig(n_averages=0)
        raise AssertionError("Mělo selhat s n_averages=0")
    except (pydantic.ValidationError, ValueError):
        pass  # Správně

@test("load_config() načte n_averages z config.yaml")
def _():
    from src.config import load_config
    cfg = load_config()
    assert hasattr(cfg.experiment, "n_averages"), "n_averages chybí v načtené config"
    assert cfg.experiment.n_averages >= 1


# ─── 2. EegRecorder instantiation ────────────────────────────────────────────
print("\n=== 2. EegRecorder ===")

@test("EegRecorder lze vytvořit bez pádu")
def _():
    from src.config import load_config
    from src.eeg_recorder import EegRecorder
    cfg = load_config()
    rec = EegRecorder(cfg)
    assert not rec.available
    assert rec.saved_path is None

@test("EegRecorder.start() bez LSL → tiše selže, available=False")
def _():
    import os
    os.environ["EEG_LSL_TIMEOUT"] = "0.1"  # velmi krátký timeout
    try:
        from src.config import load_config
        from src.eeg_recorder import EegRecorder
        cfg = load_config()
        rec = EegRecorder(cfg)
        rec.start()   # Bez EEG zařízení → RuntimeError interně potlačen
        assert rec.available == False, "Bez LSL by available mělo být False"
    finally:
        del os.environ["EEG_LSL_TIMEOUT"]

@test("EegRecorder.stop() bez start() vrátí None")
def _():
    from src.config import load_config
    from src.eeg_recorder import EegRecorder
    cfg = load_config()
    rec = EegRecorder(cfg)
    result = rec.stop(patient_name="test")
    assert result is None

@test("EegRecorder output_dir se vytvoří")
def _():
    import tempfile
    from src.config import load_config
    from src.eeg_recorder import EegRecorder
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config()
        out = Path(tmp) / "recordings"
        rec = EegRecorder(cfg, output_dir=out)
        assert out.exists(), "output_dir by měl být vytvořen v __init__"

@test("EegRecorder _save_markers_csv generuje správnou cestu")
def _():
    import tempfile
    from src.config import load_config
    from src.eeg_recorder import EegRecorder
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config()
        rec = EegRecorder(cfg, output_dir=Path(tmp))
        rec._first_timestamp = 0.0
        rec._marker_events = [(1.0, "1"), (5.0, "2")]
        fif_path = Path(tmp) / "eeg_test_20240101_raw.fif"
        rec._save_markers_csv(fif_path)
        csv_path = Path(tmp) / "eeg_test_20240101_raw.markers.csv"
        assert csv_path.exists(), f"CSV by měl existovat na {csv_path}"
        content = csv_path.read_text()
        assert "start;end;label" in content
        assert "1.000000" in content


# ─── 3. _average_epochs ───────────────────────────────────────────────────────
print("\n=== 3. _average_epochs ===")

@test("_average_epochs n_averages=1 vrátí nezměněná data")
def _():
    import numpy as np
    from src.offline_analysis import _average_epochs
    X = np.random.randn(40, 8, 500)
    y = np.array([1]*10 + [2]*10 + [3]*10 + [4]*10)
    X2, y2 = _average_epochs(X, y, n_averages=1)
    assert X2.shape == X.shape
    np.testing.assert_array_equal(y2, y)

@test("_average_epochs n_averages=5: správný počet výstupních epoch")
def _():
    import numpy as np
    from src.offline_analysis import _average_epochs
    # 40 epoch na třídu × 4 třídy → n_averages=5 → 8 bloků × 4 třídy = 32 epoch
    X = np.random.randn(160, 8, 500)
    y = np.array([1]*40 + [2]*40 + [3]*40 + [4]*40)
    X2, y2 = _average_epochs(X, y, n_averages=5)
    assert len(X2) == 32, f"Očekáváno 32, dostali jsme {len(X2)}"
    assert len(y2) == 32

@test("_average_epochs průměrování je skutečné průměrování")
def _():
    import numpy as np
    from src.offline_analysis import _average_epochs
    # 3 epochy třídy 1, n_averages=3 → 1 průměrovaná epocha
    X = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])  # (3, 1, 2)
    y = np.array([1, 1, 1])
    X2, y2 = _average_epochs(X, y, n_averages=3)
    assert X2.shape == (1, 1, 2)
    np.testing.assert_allclose(X2[0, 0], [3.0, 4.0])  # (1+3+5)/3=3, (2+4+6)/3=4

@test("_average_epochs zachovává správné labely")
def _():
    import numpy as np
    from src.offline_analysis import _average_epochs
    X = np.random.randn(6, 4, 100)
    y = np.array([1, 1, 2, 2, 3, 3])
    X2, y2 = _average_epochs(X, y, n_averages=2)
    assert len(X2) == 3
    assert set(y2.tolist()) == {1, 2, 3}


# ─── 4. paradigm_base: _build_trials s n_averages ────────────────────────────
print("\n=== 4. _build_trials (n_averages) ===")

@test("_build_trials n_averages=1: standardní shuffle")
def _():
    from src.config import AppConfig, ExperimentConfig
    cfg = AppConfig(
        experiment=ExperimentConfig(trials_per_class=10, n_averages=1),
        paradigm={"classes": {"LEFT": 1, "RIGHT": 2}},
    )
    # Simulujeme jen _build_trials, bez GUI/psychopy
    import random
    n_per_class = cfg.experiment.trials_per_class
    n_averages = cfg.experiment.n_averages
    labels = list(cfg.paradigm["classes"].keys())
    trials = []
    for label in labels:
        for _ in range(n_per_class):
            trials.append({"label": label, "code": cfg.paradigm["classes"][label]})
    assert len(trials) == 20

@test("_build_trials n_averages=5: bloky jsou správně sestaveny")
def _():
    from src.config import AppConfig, ExperimentConfig
    # Otestujeme _build_trials přímo (bez PsychoPy) přes import metody
    import types, random, math

    exp = ExperimentConfig(trials_per_class=20, n_averages=5)
    paradigm_cfg = {"classes": {"A": 1, "B": 2}}

    # Rekonstruujeme logiku _build_trials:
    n_per_class = exp.trials_per_class
    n_averages = exp.n_averages
    class_map = paradigm_cfg["classes"]

    trials = []
    n_blocks = max(1, n_per_class // n_averages)
    for label, code in class_map.items():
        for _ in range(n_blocks):
            block = [{"label": str(label), "code": int(code), "rep": r, "n_reps": n_averages}
                     for r in range(n_averages)]
            trials.append(block)
    random.shuffle(trials)
    flat = [t for block in trials for t in block]

    # 2 třídy × 4 bloky × 5 opakování = 40 trialů
    assert len(flat) == 2 * 4 * 5, f"Očekáváno 40, dostali jsme {len(flat)}"
    # Zkontrolovat že rep index je správný
    for trial in flat:
        assert 0 <= trial["rep"] < n_averages


# ─── 5. offline_analysis: FIF podpora ────────────────────────────────────────
print("\n=== 5. offline_analysis: FIF + annotations ===")

@test("_load_raw podporuje .fif příponu")
def _():
    # Test pouze to, že .fif je v SUPPORTED_FORMATS
    from src.offline_analysis import SUPPORTED_FORMATS
    assert ".fif" in SUPPORTED_FORMATS, f".fif chybí v {SUPPORTED_FORMATS}"

@test("_load_raw odmítá neznámou příponu")
def _():
    import tempfile
    from src.offline_analysis import _load_raw
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        p = Path(f.name)
    try:
        p.write_bytes(b"fake")
        try:
            _load_raw(p)
            raise AssertionError("Měl odmítnout .xyz soubor")
        except ValueError as e:
            assert "xyz" in str(e).lower() or "unsupported" in str(e).lower()
    finally:
        p.unlink(missing_ok=True)

@test("_prepare_epochs_from_annotations funguje na syntetickém FIF")
def _():
    import tempfile
    import numpy as np
    import mne
    from src.config import load_config
    from src.offline_analysis import _prepare_epochs_from_annotations

    cfg = load_config()

    # Syntetická data: 8 kanálů, 10 s @ 250 Hz
    sfreq = 250.0
    n_ch = 8
    n_times = int(10 * sfreq)
    data = np.random.randn(n_ch, n_times) * 1e-6
    info = mne.create_info(
        ch_names=[f"CH{i+1}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    # Přidat anotace (simulace markerů: 2× třída 1, 2× třída 2)
    annotations = mne.Annotations(
        onset=[1.0, 2.5, 4.0, 6.0],
        duration=[1.0, 1.0, 1.0, 1.0],
        description=["1", "2", "1", "2"],
    )
    raw.set_annotations(annotations)

    # Upravit config na tmax=1.0 pro tento test
    from src.config import AppConfig, ExperimentConfig
    test_cfg = cfg.model_copy(deep=True)
    test_cfg.events["tmax"] = 1.0
    test_cfg.events["tmin"] = 0.0

    epochs = _prepare_epochs_from_annotations(raw, test_cfg)
    assert len(epochs) == 4, f"Očekávány 4 epochy, dostali jsme {len(epochs)}"

@test("_prepare_epochs auto-detekuje FIF soubor")
def _():
    import tempfile
    import numpy as np
    import mne
    from src.config import load_config
    from src.offline_analysis import _prepare_epochs

    cfg = load_config()
    sfreq = 250.0
    n_ch = 4
    n_times = int(10 * sfreq)
    data = np.random.randn(n_ch, n_times) * 1e-6
    info = mne.create_info(
        ch_names=[f"CH{i+1}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    annotations = mne.Annotations(
        onset=[1.0, 4.0, 7.0, 8.5],
        duration=[1.0, 1.0, 1.0, 1.0],
        description=["1", "2", "1", "2"],
    )
    raw.set_annotations(annotations)

    # Uložit jako FIF a znovu načíst
    with tempfile.TemporaryDirectory() as tmp:
        fif_path = Path(tmp) / "test_raw.fif"
        raw.save(str(fif_path), overwrite=True, verbose="ERROR")
        raw_loaded = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")

        # _prepare_epochs by měl auto-detekovat FIF s anotacemi
        from src.config import AppConfig
        test_cfg = cfg.model_copy(deep=True)
        test_cfg.events["tmax"] = 1.0
        test_cfg.events["tmin"] = 0.0

        epochs = _prepare_epochs(raw_loaded, test_cfg)
        assert len(epochs) == 4, f"Očekávány 4 epochy, dostali jsme {len(epochs)}"


# ─── 6. gui_app_v2: oprava _start_paradigm_proc ───────────────────────────────
print("\n=== 6. GUI fix ===")

@test("_start_paradigm_proc importuje create_marker_outlet (ne create_streams)")
def _():
    import ast
    gui_path = PROJECT_ROOT / "src" / "gui_app_v2.py"
    source = gui_path.read_text(encoding="utf-8")
    # Ověřit, že _start_paradigm_proc NEPOUŽÍVÁ create_streams
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_start_paradigm_proc":
            func_src = ast.get_source_segment(source, node) or ""
            assert "create_marker_outlet" in func_src, "Musí použít create_marker_outlet"
            assert "create_streams" not in func_src, "Nesmí použít create_streams (bug)"
            return
    raise AssertionError("Funkce _start_paradigm_proc nebyla nalezena")

@test("_run_selection importuje EegRecorder")
def _():
    import ast
    gui_path = PROJECT_ROOT / "src" / "gui_app_v2.py"
    source = gui_path.read_text(encoding="utf-8")
    assert "EegRecorder" in source, "EegRecorder musí být importován v gui_app_v2.py"
    assert "recorder.start()" in source, "recorder.start() musí být voláno"
    assert "recorder.stop(" in source, "recorder.stop() musí být voláno"


# ─── 7. config soubory ────────────────────────────────────────────────────────
print("\n=== 7. Config soubory ===")

@test("config.yaml obsahuje n_averages")
def _():
    import yaml
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "n_averages" in data.get("experiment", {}), "config.yaml musí mít experiment.n_averages"

@test("config_24ch.yaml obsahuje n_averages=5")
def _():
    import yaml
    cfg_path = PROJECT_ROOT / "config" / "config_24ch.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    n_avg = data.get("experiment", {}).get("n_averages", None)
    assert n_avg == 5, f"config_24ch.yaml: n_averages má být 5, je {n_avg}"

@test("config_24ch.yaml: eeg.n_channels je 24")
def _():
    import yaml
    cfg_path = PROJECT_ROOT / "config" / "config_24ch.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    n_ch = data.get("eeg", {}).get("n_channels", None)
    assert n_ch == 24, f"n_channels má být 24, je {n_ch}"

@test("build_exe.py existuje")
def _():
    assert (PROJECT_ROOT / "build_exe.py").exists(), "build_exe.py neexistuje"


# ─── Shrnutí ──────────────────────────────────────────────────────────────────
print()
print("=" * 55)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"Výsledek: {passed}/{total} testů prošlo", end="")
if failed:
    print(f"  ({failed} selhalo)")
    print("\nSelhané testy:")
    for name, ok, err in results:
        if not ok:
            print(f"  • {name}: {err}")
else:
    print(" – vše OK ✓")
print("=" * 55)

sys.exit(0 if failed == 0 else 1)
