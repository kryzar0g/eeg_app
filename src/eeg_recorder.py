"""Interní EEG recorder: přijímá LSL stream a ukládá do FIF/EDF bez potřeby LabRecorderu."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import AppConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


class EegRecorder:
    """Nahrává EEG data z LSL streamu a ukládá jako MNE FIF soubor.

    Spouští se ve vlákně paralelně s paradigmatem. Zároveň poslouchá marker
    LSL stream a vkládá značky jako anotace → výsledný FIF lze přímo použít
    pro trénování (mode: annotations v config).

    Použití::

        recorder = EegRecorder(config)
        recorder.start()          # připojí se k LSL a začne nahrávat
        # ... spustit paradigma ...
        saved = recorder.stop()   # uloží FIF a vrátí Path
    """

    def __init__(self, config: AppConfig, output_dir: Optional[Path] = None) -> None:
        self.config = config
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "recordings"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._data_chunks: List[np.ndarray] = []
        self._marker_events: List[Tuple[float, str]] = []  # (lsl_timestamp, marker_code)
        self._first_timestamp: Optional[float] = None
        self._sfreq: float = float(config.preprocessing.sfreq)
        self._ch_names: List[str] = []

        self._running = False
        self._eeg_thread: Optional[threading.Thread] = None
        self._marker_thread: Optional[threading.Thread] = None
        self._eeg_inlet = None
        self._marker_inlet = None

        self.saved_path: Optional[Path] = None
        self.available: bool = False  # True po úspěšném start()

    # ─── Veřejné metody ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Připojí se k LSL a spustí nahrávání na pozadí.

        Pokud LSL stream není dostupný, zaznamená varování a nahrávání přeskočí
        (aplikace dál funguje, jen bez interního záznamu).
        """
        from .lsl_acquisition import resolve_eeg_stream

        logger.info("EegRecorder: hledám LSL stream...")
        try:
            self._eeg_inlet = resolve_eeg_stream(self.config)
        except RuntimeError as exc:
            logger.warning(
                f"EegRecorder: LSL stream nedostupný ({exc}). "
                "Interní nahrávání přeskočeno – použijte LabRecorder."
            )
            return

        # Zjistit informace o streamu
        info = self._eeg_inlet.info()
        n_ch = info.channel_count()
        nominal_srate = info.nominal_srate()
        if nominal_srate > 0:
            self._sfreq = float(nominal_srate)

        # Načíst názvy kanálů z info (pokud jsou)
        ch_node = info.desc().child("channels").child("channel")
        names: List[str] = []
        for i in range(n_ch):
            label = ch_node.child_value("label")
            names.append(label if label else f"CH{i + 1}")
            ch_node = ch_node.next_sibling()
        self._ch_names = names

        # Volitelně: připojit se k marker streamu
        try:
            from pylsl import resolve_byprop, StreamInlet
            marker_streams = resolve_byprop("type", "Markers", timeout=2.0)
            if marker_streams:
                self._marker_inlet = StreamInlet(marker_streams[0])
                logger.info("EegRecorder: připojen k marker streamu")
        except Exception as exc:
            logger.debug(f"EegRecorder: marker stream nedostupný ({exc})")

        self._running = True
        self.available = True

        self._eeg_thread = threading.Thread(target=self._eeg_loop, daemon=True, name="eeg-recorder-eeg")
        self._eeg_thread.start()

        if self._marker_inlet is not None:
            self._marker_thread = threading.Thread(
                target=self._marker_loop, daemon=True, name="eeg-recorder-markers"
            )
            self._marker_thread.start()

        logger.info(
            f"EegRecorder: nahrávání spuštěno ({n_ch} kanálů @ {self._sfreq:.0f} Hz, "
            f"výstup: {self.output_dir})"
        )

    def stop(self, patient_name: str = "pacient") -> Optional[Path]:
        """Zastaví nahrávání a uloží FIF soubor.

        Args:
            patient_name: Jméno pacienta, použije se v názvu souboru.

        Returns:
            Cesta k uloženému FIF souboru, nebo None pokud nebyla dostupná data.
        """
        if not self.available:
            return None

        self._running = False
        for t in (self._eeg_thread, self._marker_thread):
            if t is not None:
                t.join(timeout=5.0)

        if not self._data_chunks:
            logger.warning("EegRecorder: žádná data k uložení")
            return None

        return self._save_edf(patient_name)

    # ─── Vlákna ───────────────────────────────────────────────────────────────

    def _eeg_loop(self) -> None:
        """Smyčka příjmu EEG dat (běží v separátním vlákně)."""
        from .lsl_acquisition import pull_chunk

        while self._running:
            chunk, timestamps = pull_chunk(self._eeg_inlet)
            if chunk.size == 0:
                time.sleep(0.002)
                continue
            if self._first_timestamp is None and timestamps.size > 0:
                self._first_timestamp = float(timestamps[0])
            self._data_chunks.append(chunk)  # (n_samples, n_channels)

        logger.debug("EegRecorder: EEG smyčka ukončena")

    def _marker_loop(self) -> None:
        """Smyčka příjmu markerů (běží v separátním vlákně)."""
        while self._running:
            try:
                sample, timestamp = self._marker_inlet.pull_sample(timeout=0.1)
                if sample and timestamp is not None:
                    marker_code = str(sample[0])
                    self._marker_events.append((float(timestamp), marker_code))
                    logger.debug(f"EegRecorder: marker {marker_code!r} @ t={timestamp:.3f}")
            except Exception as exc:
                logger.debug(f"EegRecorder: chyba čtení markeru: {exc}")

        logger.debug("EegRecorder: marker smyčka ukončena")

    # ─── Ukládání ─────────────────────────────────────────────────────────────

    def _build_raw(self) -> "mne.io.RawArray":
        """Sestaví MNE RawArray z nahraných dat včetně anotací markerů."""
        import mne

        all_data = np.vstack(self._data_chunks)  # (n_samples, n_channels)
        data_t = all_data.T  # MNE: (n_channels, n_samples)

        info = mne.create_info(
            ch_names=self._ch_names,
            sfreq=self._sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data_t, info, verbose="ERROR")

        # Vložit markery jako EDF+ anotace
        if self._marker_events and self._first_timestamp is not None:
            onsets = [
                max(0.0, ts - self._first_timestamp)
                for ts, _ in self._marker_events
            ]
            durations = [float(self.config.experiment.imagery_duration)] * len(self._marker_events)
            descriptions = [code for _, code in self._marker_events]
            raw.set_annotations(mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
            ))
            logger.info(
                f"EegRecorder: vloženo {len(self._marker_events)} markerů jako anotace"
            )
        return raw

    def _save_edf(self, patient_name: str) -> Optional[Path]:
        """Uloží záznam jako EDF soubor s EDF+ anotacemi.

        EDF je primární formát aplikace (načítá offline_analysis.py).
        Pokud export selže (chybí edfio), uloží jako záložní FIF.
        """
        import mne

        raw = self._build_raw()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in patient_name
        )

        # ── Primární: EDF ────────────────────────────────────────────
        fname_edf = self.output_dir / f"eeg_{safe_name}_{timestamp_str}_raw.edf"
        try:
            mne.export.export_raw(
                str(fname_edf),
                raw,
                fmt="edf",
                overwrite=True,
                verbose="ERROR",
            )
            self.saved_path = fname_edf
            logger.info(f"EegRecorder: uloženo jako EDF → {fname_edf}")
            self._save_markers_csv(fname_edf)
            return fname_edf

        except Exception as edf_exc:
            logger.warning(
                f"EegRecorder: EDF export selhal ({edf_exc}). "
                "Zálohuji do FIF. Nainstalujte 'edfio' pro EDF podporu: "
                "pip install edfio"
            )

        # ── Záloha: FIF ──────────────────────────────────────────────
        fname_fif = self.output_dir / f"eeg_{safe_name}_{timestamp_str}_raw.fif"
        try:
            raw.save(str(fname_fif), overwrite=True, verbose="ERROR")
            self.saved_path = fname_fif
            logger.info(f"EegRecorder: uloženo jako FIF záloha → {fname_fif}")
            self._save_markers_csv(fname_fif)
            return fname_fif
        except Exception as fif_exc:
            logger.error(f"EegRecorder: i FIF záloha selhala: {fif_exc}", exc_info=True)
            return None

    # Zpětná kompatibilita — old name
    _save_fif = _save_edf

    def _save_markers_csv(self, eeg_path: Path) -> None:
        """Uloží markery jako CSV kompatibilní s mode='csv' v offline_analysis."""
        if not self._marker_events or self._first_timestamp is None:
            return
        try:
            # Pozor: Path.with_suffix neumožňuje více teček → musíme použít string
            csv_path = eeg_path.parent / (eeg_path.stem + ".markers.csv")
            imagery_dur = float(self.config.experiment.imagery_duration)
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("start;end;label\n")
                for ts, code in self._marker_events:
                    start_sec = max(0.0, ts - self._first_timestamp)
                    end_sec = start_sec + imagery_dur
                    f.write(f"{start_sec:.6f};{end_sec:.6f};{code}\n")
            logger.info(f"EegRecorder: markery CSV uloženy → {csv_path}")
        except Exception as exc:
            logger.warning(f"EegRecorder: nelze uložit markers CSV: {exc}")
