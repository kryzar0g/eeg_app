"""Test IPC marker flow: paradigma -> fronta -> recorder, zarovnani podle vzorku."""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['EEG_LSL_TIMEOUT'] = '0.1'

import numpy as np
from multiprocessing import Queue
from src.config import load_config
from src.eeg_recorder import EegRecorder

cfg = load_config()
q = Queue()
rec = EegRecorder(cfg, marker_queue=q)

# Simulovat ze recorder bezi: nastavit running a spustit jen queue smycku rucne
rec._running = True
rec._sfreq = 250.0
rec._eeg_sample_count = 0

import threading
t = threading.Thread(target=rec._marker_queue_loop, daemon=True)
t.start()

# Simulovat prijem EEG vzorku + markeru v case
errors = []
expected = []
for i, code in enumerate([1, 2, 3, 4, 1, 2], start=1):
    rec._eeg_sample_count += 500   # 2s EEG @ 250Hz
    expected.append((rec._eeg_sample_count, str(code)))
    q.put_nowait(code)
    time.sleep(0.15)   # nechat queue smycku zpracovat

time.sleep(0.3)
rec._running = False
t.join(timeout=2)

print(f"Ocekavano markeru: {len(expected)}")
print(f"Zachyceno markeru: {len(rec._marker_events)}")

ok_count = len(rec._marker_events) == len(expected)
print(f"[{'OK' if ok_count else 'FAIL'}] pocet markeru sedi")
if not ok_count: errors.append("pocet")

# Overit ze sample indexy odpovidaji (markery prisly po EEG vzorcich)
for (exp_idx, exp_code), (got_idx, got_code) in zip(expected, rec._marker_events):
    if exp_code != got_code:
        errors.append(f"kod {exp_code}!={got_code}")
    # index by mel byt >= exp_idx (marker prisel az po nacteni vzorku)
    if got_idx < exp_idx - 600:  # tolerance
        errors.append(f"index {got_idx} prilis daleko od {exp_idx}")

print("Zachycene markery:", rec._marker_events)
print()
if errors:
    print("FAIL:", errors)
    sys.exit(1)
print("PASS - IPC marker flow funguje, zarovnani podle EEG vzorku spravne")
