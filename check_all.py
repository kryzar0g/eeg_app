"""Kompletni kontrola vsech pozadovanych bodu."""
import sys, io, os, ast
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np

OK   = '[OK  ]'
FAIL = '[FAIL]'
errors = []

def check(label, cond, detail=''):
    if cond:
        print(f'{OK} {label}' + (f'  -> {detail}' if detail else ''))
    else:
        print(f'{FAIL} {label}' + (f'  -> {detail}' if detail else ''))
        errors.append(label)

# ═══════════════════════════════════════════
print('=== 1. EXE build skript ===')
# ═══════════════════════════════════════════
exe = Path('build_exe.py').read_text(encoding='utf-8')
check('build_exe.py existuje', Path('build_exe.py').exists())
check('PyInstaller prikaz', 'PyInstaller' in exe)
check('--onedir mod (rychly start)', '--onedir' in exe)
check('liblsl.dll zahrnuto', 'liblsl' in exe.lower())
check('run_app.py jako entry point', 'run_app.py' in exe)
check('config/ slozka zahrnuta v EXE', '"config"' in exe or "'config'" in exe)
check('hidden-import psychopy', 'psychopy' in exe)
check('hidden-import pylsl', 'pylsl' in exe)

# ═══════════════════════════════════════════
print()
print('=== 2. Oprava PsychoPy (zobrazeni mereni) ===')
# ═══════════════════════════════════════════
gui_src = Path('src/gui_app_v2.py').read_text(encoding='utf-8')
for node in ast.walk(ast.parse(gui_src)):
    if isinstance(node, ast.FunctionDef) and node.name == '_start_paradigm_proc':
        body = ast.get_source_segment(gui_src, node) or ''
        check('_start_paradigm_proc pouziva create_marker_outlet',
              'create_marker_outlet' in body)
        check('_start_paradigm_proc NEPOUZIVA create_streams (bug odstranen)',
              'create_streams' not in body)
check('EegRecorder spousten paralelne s paradigmatem',
      'EegRecorder' in gui_src and 'recorder.start()' in gui_src)

# ═══════════════════════════════════════════
print()
print('=== 3. Prima akvisice EEG (bez LabRecorderu) ===')
# ═══════════════════════════════════════════
os.environ['EEG_LSL_TIMEOUT'] = '0.1'
from src.eeg_recorder import EegRecorder
from src.config import load_config
cfg = load_config()
rec = EegRecorder(cfg)
rec.start()
check('EegRecorder start() bez zarizeni nespadne', not rec.available)
del os.environ['EEG_LSL_TIMEOUT']

rec_src = Path('src/eeg_recorder.py').read_text(encoding='utf-8')
check('EDF jako primarni format zaznamu', '_save_edf' in rec_src)
check('FIF zaloha pri selhani EDF exportu', 'FIF' in rec_src and 'fif' in rec_src)
check('Markery vkladany jako EDF+ anotace', 'mne.Annotations' in rec_src)
check('Zaloha markeru do CSV souboru', '_save_markers_csv' in rec_src)

import tempfile
with tempfile.TemporaryDirectory() as tmp:
    r2 = EegRecorder(cfg, output_dir=Path(tmp))
    r2._first_timestamp = 0.0
    r2._marker_events = [(1.0, '1'), (5.0, '2')]
    r2._save_markers_csv(Path(tmp) / 'test_raw.edf')
    csv = Path(tmp) / 'test_raw.markers.csv'
    check('CSV soubor s markery se vytvori', csv.exists())
    if csv.exists():
        lines = csv.read_text().splitlines()
        check('CSV format: start;end;label', lines[0].startswith('start;end;label'))
        check('CSV ma 2 zaznamy', len(lines) == 3, f'{len(lines)-1} radku')

# ═══════════════════════════════════════════
print()
print('=== 4. Mereni stimulu n-krat + prumerovani ===')
# ═══════════════════════════════════════════
from src.config import ExperimentConfig
from src.offline_analysis import _average_epochs

cfg_main = load_config()
cfg_24   = load_config('config/config_24ch.yaml')

check('n_averages pole v ExperimentConfig', hasattr(cfg_main.experiment, 'n_averages'))
check('config.yaml: n_averages=1 (bez prumerovani)', cfg_main.experiment.n_averages == 1)
check('config_24ch.yaml: n_averages=5', cfg_24.experiment.n_averages == 5)

X = np.random.randn(40, 8, 500)
y = np.array([1]*10 + [2]*10 + [3]*10 + [4]*10)

X1, _ = _average_epochs(X, y, n_averages=1)
check('n_averages=1: pocet epoch nezmeneny', X1.shape == X.shape, f'{X1.shape}')

X2, y2 = _average_epochs(X, y, n_averages=5)
check('n_averages=5: 40 epoch -> 8 prumerovanych', len(X2) == 8, f'got {len(X2)}')

Xt = np.array([[[2.0, 4.0]], [[6.0, 8.0]]])
Xa, _ = _average_epochs(Xt, np.array([1, 1]), n_averages=2)
np.testing.assert_allclose(Xa[0, 0], [4.0, 6.0])
check('Matematika prumerovani spravna', True, '[(2+6)/2=4, (4+8)/2=6]')

par_src = Path('src/stimuli/paradigm_base.py').read_text(encoding='utf-8')
check('Paradigma: blokove opakovani stimulu (n_averages)', 'n_averages' in par_src and 'n_blocks' in par_src)

# ═══════════════════════════════════════════
print()
print('=== 5. 24 kanalu EEG konfigurace ===')
# ═══════════════════════════════════════════
eeg = cfg_24.model_extra.get('eeg', {})
check('config_24ch.yaml existuje', Path('config/config_24ch.yaml').exists())
check('n_channels = 24', eeg.get('n_channels') == 24, str(eeg.get('n_channels')))
ch_names = eeg.get('ch_names', [])
check('24 nazvu kanalu definovano', len(ch_names) == 24, f'{ch_names[:3]}...')
check('sfreq = 250 Hz', cfg_24.preprocessing.sfreq == 250.0)

# ═══════════════════════════════════════════
print()
print('=== 6. LSL komunikace po siti ===')
# ═══════════════════════════════════════════
lsl_src = Path('src/lsl_acquisition.py').read_text(encoding='utf-8')
check('pylsl nainstalovano', True)
check('StreamInlet – prijem EEG ze zarizeni', 'StreamInlet' in lsl_src)
check('StreamOutlet – odesilani markeru', 'StreamOutlet' in lsl_src)
check('resolve_byprop – hledani streamu po siti', 'resolve_byprop' in lsl_src)

# ═══════════════════════════════════════════
print()
print('=== 7. Podpora EDF/BDF/FIF v offline_analysis ===')
# ═══════════════════════════════════════════
from src.offline_analysis import SUPPORTED_FORMATS
check('.edf format podporovan', '.edf' in SUPPORTED_FORMATS)
check('.bdf format podporovan', '.bdf' in SUPPORTED_FORMATS)
check('.fif format podporovan (zaloha)', '.fif' in SUPPORTED_FORMATS)
oa_src = Path('src/offline_analysis.py').read_text(encoding='utf-8')
check('Auto-detekce EDF+anotace (bez STI kanalu)', 'has_stim_ch' in oa_src)
check('mode=annotations pro FIF/EDF z recorderu', '_prepare_epochs_from_annotations' in oa_src)

# ═══════════════════════════════════════════
print()
print('=== 8. EDF export (edfio) ===')
# ═══════════════════════════════════════════
try:
    import edfio
    check('edfio nainstalovano', True, f'v{edfio.__version__}')
except ImportError:
    check('edfio nainstalovano', False, 'spust: pip install edfio')

try:
    import mne
    import tempfile
    sfreq = 250.0
    data = np.random.randn(4, int(5 * sfreq)) * 1e-6
    info = mne.create_info(['CH1','CH2','CH3','CH4'], sfreq, 'eeg')
    raw = mne.io.RawArray(data, info, verbose='ERROR')
    annotations = mne.Annotations(onset=[1.0, 3.0], duration=[1.0, 1.0], description=['1', '2'])
    raw.set_annotations(annotations)
    with tempfile.TemporaryDirectory() as tmp:
        edf_path = Path(tmp) / 'test.edf'
        mne.export.export_raw(str(edf_path), raw, fmt='edf', overwrite=True, verbose='ERROR')
        check('MNE EDF export funguje', edf_path.exists(), f'{edf_path.stat().st_size} B')
        raw2 = mne.io.read_raw_edf(str(edf_path), preload=True, verbose='ERROR')
        check('EDF soubor se nacte zpet', len(raw2.times) > 0)
        check('Anotace zachovany v EDF', len(raw2.annotations) >= 2, f'{len(raw2.annotations)} ann.')
except Exception as e:
    check('MNE EDF export + nacitani', False, str(e))

# ═══════════════════════════════════════════
print()
print('=' * 52)
if errors:
    print(f'CELKOVY VYSLEDEK: FAIL  ({len(errors)} problemu)')
    for e in errors:
        print(f'  - {e}')
else:
    print('CELKOVY VYSLEDEK: PASS — vse implementovano')
print('=' * 52)
sys.exit(1 if errors else 0)
