"""Test IPC marker synchronizace: (code, local_clock_ts) -> presny EEG vzorek."""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['EEG_LSL_TIMEOUT'] = '0.1'

import numpy as np
from multiprocessing import Queue
from src.config import load_config
from src.eeg_recorder import EegRecorder

cfg = load_config()
sfreq = 250.0
errors = []

# === Test 1: _marker_time_to_sample - presny prevod casu na vzorek ===
rec = EegRecorder(cfg)
rec._sfreq = sfreq
rec._first_eeg_local_time = 1000.0   # prvni EEG vzorek v lokalnim case = t=1000s

# Marker v case 1002.0s -> mel by byt na vzorku (1002-1000)*250 = 500
idx = rec._marker_time_to_sample(1002.0)
ok = idx == 500
print(f"[{'OK' if ok else 'FAIL'}] cas 1002.0s -> vzorek {idx} (ocekavano 500)")
if not ok: errors.append("time_to_sample")

# Marker pred zacatkem -> clamp na 0
idx0 = rec._marker_time_to_sample(999.0)
ok0 = idx0 == 0
print(f"[{'OK' if ok0 else 'FAIL'}] cas pred zacatkem -> vzorek {idx0} (clamp 0)")
if not ok0: errors.append("clamp")

# Bez referencniho casu -> fallback na pocet vzorku
rec._first_eeg_local_time = None
rec._eeg_sample_count = 777
idxf = rec._marker_time_to_sample(1234.0)
okf = idxf == 777
print(f"[{'OK' if okf else 'FAIL'}] bez ref. casu -> fallback pocet vzorku {idxf} (777)")
if not okf: errors.append("fallback")

# === Test 2: IPC fronta s (code, ts) formatem ===
rec2 = EegRecorder(cfg, marker_queue=Queue())
rec2._sfreq = sfreq
rec2._first_eeg_local_time = 5000.0
rec2._running = True

import threading
t = threading.Thread(target=rec2._marker_queue_loop, daemon=True)
t.start()

# Poslat markery s timestampy (kazdy o 2s pozdeji)
expected_samples = []
for i, code in enumerate([1, 2, 3, 4]):
    marker_time = 5000.0 + (i + 1) * 2.0   # 5002, 5004, 5006, 5008
    expected_samples.append(int((marker_time - 5000.0) * sfreq))  # 500,1000,1500,2000
    rec2._marker_queue.put_nowait((code, marker_time))
    time.sleep(0.12)

time.sleep(0.3)
rec2._running = False
t.join(timeout=2)

got_samples = [s for s, _ in rec2._marker_events]
print(f"\nOcekavane vzorky: {expected_samples}")
print(f"Zachycene vzorky: {got_samples}")
ok2 = got_samples == expected_samples
print(f"[{'OK' if ok2 else 'FAIL'}] IPC marker synchronizace presna")
if not ok2: errors.append("ipc_sync")

print()
if errors:
    print("FAIL:", errors); sys.exit(1)
print("PASS - marker synchronizace (time_correction + local_clock) funguje")
