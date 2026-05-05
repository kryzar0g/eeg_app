import importlib
import sys
mods = ['yaml','numpy','scipy','sklearn','mne','joblib','pylsl','psychopy']
for name in mods:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name}: FAIL -> {type(exc).__name__}: {exc}")

sys.exit(0)
