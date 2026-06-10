"""Test REALNEHO cross-process IPC (Windows spawn): paradigma proces -> recorder.

Overuje ze multiprocessing.Queue spravne predava markery z child procesu
(jako paradigma) do hlavniho procesu (recorder). Bez PsychoPy.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['EEG_LSL_TIMEOUT'] = '0.1'

import multiprocessing


def _fake_paradigm(marker_queue, n):
    """Simuluje paradigma: posila markery (code, local_clock) do fronty."""
    try:
        from pylsl import local_clock
    except Exception:
        local_clock = time.time
    for code in range(1, n + 1):
        marker_queue.put_nowait((code, float(local_clock())))
        time.sleep(0.05)


def main():
    from src.config import load_config
    from src.eeg_recorder import EegRecorder
    import threading

    cfg = load_config()
    q = multiprocessing.Queue()

    # Recorder bez realneho EEG - rucne spustime queue smycku
    rec = EegRecorder(cfg, marker_queue=q)
    rec._running = True
    rec._sfreq = 250.0
    rec._first_eeg_local_time = None   # fallback na pocet vzorku
    rec._eeg_sample_count = 1000

    t = threading.Thread(target=rec._marker_queue_loop, daemon=True)
    t.start()

    # Spustit REALNY child proces (jako paradigma)
    n = 6
    proc = multiprocessing.Process(target=_fake_paradigm, args=(q, n))
    proc.start()
    proc.join(timeout=10)

    time.sleep(0.5)
    rec._running = False
    t.join(timeout=2)

    print(f"Posláno markeru: {n}")
    print(f"Zachyceno: {len(rec._marker_events)}")
    print(f"Markery: {rec._marker_events}")

    if len(rec._marker_events) == n:
        codes = [c for _, c in rec._marker_events]
        if codes == [str(i) for i in range(1, n + 1)]:
            print("\nPASS - cross-process IPC funguje, vsechny markery dorucены spravne")
            return 0
    print("\nFAIL - markery se nepredaly spravne pres procesy")
    return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
