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

    def __init__(self, config: AppConfig, output_dir: Optional[Path] = None,
                 marker_queue=None) -> None:
        self.config = config
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "recordings"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Volitelna IPC fronta - paradigma do ni posila markery primo
        # (zaloha pro pripad ze LSL marker discovery selze v sitovem setupu).
        self._marker_queue = marker_queue

        self._data_chunks: List[np.ndarray] = []
        # Markery ukladame jako (sample_index, code) = poradi EEG vzorku v dobe
        # prijeti markeru. Tim se vyhneme problemu s ruznymi LSL hodinami,
        # kdyz EEG prichazi ze vzdaleneho PC a markery z lokalniho paradigmatu.
        self._marker_events: List[Tuple[int, str]] = []  # (eeg_sample_index, code)
        self._eeg_sample_count: int = 0  # pocet dosud prijatych EEG vzorku
        self._first_timestamp: Optional[float] = None
        # Lokalni cas (LSL local_clock) prvniho EEG vzorku po time_correction.
        # Slouzi k presnemu zarovnani markeru s EEG napric pocitaci.
        self._first_eeg_local_time: Optional[float] = None
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

        self._running = True
        self.available = True

        # EEG akvisice – spustit hned
        self._eeg_thread = threading.Thread(
            target=self._eeg_loop, daemon=True, name="eeg-recorder-eeg"
        )
        self._eeg_thread.start()

        # Markery: pokud je k dispozici IPC fronta (interni zaznam s nasim
        # paradigmatem), pouzij JI jako primarni - je spolehlivejsi nez LSL
        # discovery v sitovem setupu. Jinak pouzij LSL marker stream.
        if self._marker_queue is not None:
            self._marker_thread = threading.Thread(
                target=self._marker_queue_loop, daemon=True, name="eeg-recorder-queue",
            )
            logger.info("EegRecorder: markery pres IPC frontu (primo z paradigmatu)")
        else:
            self._marker_thread = threading.Thread(
                target=self._marker_connect_and_loop, daemon=True, name="eeg-recorder-markers",
            )
            logger.info("EegRecorder: markery pres LSL stream")
        self._marker_thread.start()

        logger.info(
            "EegRecorder: nahravani spusteno (%d kanalu @ %.0f Hz, vystup: %s)",
            n_ch, self._sfreq, self.output_dir,
        )

    def _marker_queue_loop(self) -> None:
        """Drenuje IPC frontu s markery z paradigma procesu.

        Paradigma posila (code, local_clock_timestamp). Protoze obe procesy
        bezi na STEJNEM pocitaci, local_clock je stejny hodinovy domen jako
        time_correction-zarovnany EEG -> sample-accurate synchronizace.
        Zpetna kompatibilita: pokud prijde holy kod, pouzije se citac vzorku.
        """
        import queue as _queue
        seen = 0
        while self._running:
            try:
                item = self._marker_queue.get(timeout=0.1)
            except (_queue.Empty, Exception):
                continue
            if item is None:
                continue
            # Format (code, local_clock_ts) nebo holy code
            if isinstance(item, (tuple, list)) and len(item) == 2:
                code, ts = item
                sample_idx = self._marker_time_to_sample(float(ts))
            else:
                code = item
                sample_idx = int(self._eeg_sample_count)
            self._marker_events.append((sample_idx, str(code)))
            seen += 1
            logger.info("EegRecorder: marker(IPC) %r @ EEG vzorek %d", code, sample_idx)
        logger.info("EegRecorder: IPC marker smycka ukoncena, prijato %d", seen)

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
                # Prepocet na LOKALNI hodiny: remote_timestamp + time_correction.
                # Tim ziskame cas prvniho EEG vzorku ve stejnem hodinovem
                # domenu jako paradigma (local_clock) -> presne zarovnani.
                try:
                    tc = float(self._eeg_inlet.time_correction())
                except Exception:
                    tc = 0.0
                self._first_eeg_local_time = float(timestamps[0]) + tc
                logger.info("EegRecorder: time_correction=%.4f s, prvni EEG vzorek @ local=%.4f",
                            tc, self._first_eeg_local_time)
            self._data_chunks.append(chunk)  # (n_samples, n_channels)
            self._eeg_sample_count += int(chunk.shape[0])

        logger.debug("EegRecorder: EEG smyčka ukončena")

    def _marker_time_to_sample(self, marker_local_time: Optional[float]) -> int:
        """Prevede lokalni cas markeru na index EEG vzorku.

        Pouziva time_correction-zarovnany cas prvniho EEG vzorku a vzorkovaci
        frekvenci. Tim je marker umisten na presny EEG vzorek i kdyz EEG
        prichazi ze vzdaleneho PC s jinymi hodinami. Fallback (kdyz cas neni
        k dispozici): aktualni pocet prijatych vzorku.
        """
        if marker_local_time is None or self._first_eeg_local_time is None:
            return int(self._eeg_sample_count)
        delta = marker_local_time - self._first_eeg_local_time
        return max(0, int(round(delta * self._sfreq)))

    def _marker_connect_and_loop(self) -> None:
        """Hledá marker LSL stream a po nalezení přijímá markery.

        Retry smyčka: paradigma (marker outlet) se spouští až po start(),
        proto zkouší každou sekundu až 60 sekund.
        """
        from pylsl import resolve_byprop, StreamInlet

        deadline = time.time() + 60.0
        inlet = None

        while self._running and time.time() < deadline:
            try:
                streams = resolve_byprop("type", "Markers", timeout=1.0)
                if streams:
                    inlet = StreamInlet(streams[0])
                    logger.info("EegRecorder: marker stream nalezen a pripojen")
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if inlet is None:
            logger.warning("EegRecorder: marker stream nenalezen do 60s - BDF nebude mit markery")
            return

        self._marker_inlet = inlet
        self._marker_loop()

    def _marker_loop(self) -> None:
        """Přijímá markery z LSL a zarovnava je presne podle time_correction.

        Marker timestamp (clock odesilatele) + time_correction = lokalni cas,
        ktery se prevede na index EEG vzorku. Sample-accurate synchronizace.
        """
        try:
            marker_tc = float(self._marker_inlet.time_correction())
        except Exception:
            marker_tc = 0.0

        n_received = 0
        while self._running:
            try:
                sample, timestamp = self._marker_inlet.pull_sample(timeout=0.1)
                if sample and timestamp is not None:
                    marker_code = str(sample[0])
                    marker_local = float(timestamp) + marker_tc
                    sample_idx = self._marker_time_to_sample(marker_local)
                    self._marker_events.append((sample_idx, marker_code))
                    n_received += 1
                    logger.info("EegRecorder: marker(LSL) %r @ EEG vzorek %d (celkem %d)",
                                marker_code, sample_idx, n_received)
            except Exception as exc:
                logger.debug("EegRecorder: chyba cteni markeru: %s", exc)

        logger.info("EegRecorder: marker smycka ukoncena, prijato %d markeru", n_received)

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

        # Vložit markery jako EDF+ anotace.
        # marker_events obsahuje (sample_index, code) -> onset = index / sfreq
        if self._marker_events:
            onsets = [
                max(0.0, float(sample_idx) / self._sfreq)
                for sample_idx, _ in self._marker_events
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
        """Uloží záznam jako BDF soubor (primárně) nebo EDF.

        Pořadí pokusů:
          1. BDF přímý zápis (čistý Python, bez edfio) — vždy funguje
          2. EDF přes mne.export (vyžaduje edfio) — záloha
          3. FIF přes mne.save — poslední záloha
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in patient_name
        )

        # ── 1. Primární: BDF (přímý zápis, bez závislostí) ──────────
        fname_bdf = self.output_dir / f"eeg_{safe_name}_{timestamp_str}_raw.bdf"
        try:
            self._write_bdf(fname_bdf, patient_name=patient_name)
            self.saved_path = fname_bdf
            logger.info(f"EegRecorder: uloženo jako BDF → {fname_bdf}")
            self._save_markers_csv(fname_bdf)
            return fname_bdf
        except Exception as bdf_exc:
            logger.warning(f"EegRecorder: BDF zápis selhal ({bdf_exc}), zkouším EDF...")

        # ── 2. Záloha: EDF přes MNE (vyžaduje edfio) ────────────────
        fname_edf = self.output_dir / f"eeg_{safe_name}_{timestamp_str}_raw.edf"
        try:
            import mne
            raw = self._build_raw()
            mne.export.export_raw(str(fname_edf), raw, fmt="edf", overwrite=True, verbose="ERROR")
            self.saved_path = fname_edf
            logger.info(f"EegRecorder: uloženo jako EDF → {fname_edf}")
            self._save_markers_csv(fname_edf)
            return fname_edf
        except Exception as edf_exc:
            logger.warning(f"EegRecorder: EDF export selhal ({edf_exc}), ukládám FIF...")

        # ── 3. Poslední záloha: FIF ──────────────────────────────────
        fname_fif = self.output_dir / f"eeg_{safe_name}_{timestamp_str}_raw.fif"
        try:
            import mne
            raw = self._build_raw()
            raw.save(str(fname_fif), overwrite=True, verbose="ERROR")
            self.saved_path = fname_fif
            logger.info(f"EegRecorder: uloženo jako FIF záloha → {fname_fif}")
            self._save_markers_csv(fname_fif)
            return fname_fif
        except Exception as fif_exc:
            logger.error(f"EegRecorder: všechny formáty selhaly: {fif_exc}", exc_info=True)
            return None

    # Zpětná kompatibilita
    _save_fif = _save_edf

    def _write_bdf(self, path: Path, patient_name: str = "X") -> None:
        """Zapíše data přímo do BDF souboru bez externích závislostí.

        BDF (BioSemi Data Format) = EDF s 24-bitovými vzorky a Status kanálem.
        Markery se uloží do Status kanálu jako integer kódy.
        Čistá Python + numpy implementace — funguje v každém prostředí.
        """
        import struct
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in patient_name)

        all_data = np.vstack(self._data_chunks)   # (n_samples, n_eeg_ch)
        n_samples_total, n_eeg = all_data.shape
        sfreq = self._sfreq

        # Sanitizovat názvy kanálů pro BDF (max 16 ASCII znaků)
        def _edf_label(name: str) -> str:
            safe = "".join(c for c in name if c.isascii() and c.isprintable())
            return safe[:16]

        ch_labels = [_edf_label(n) for n in self._ch_names] if self._ch_names \
                    else [f"CH{i+1}" for i in range(n_eeg)]

        # Přidat Status kanál pro markery
        status_labels = ch_labels + ["Status"]
        n_ch = n_eeg + 1

        # Délka jednoho záznamu = 1 sekunda
        record_dur_sec = 1
        samples_per_record = int(round(sfreq * record_dur_sec))
        n_records = int(np.ceil(n_samples_total / samples_per_record))

        # Doplnit data na celý počet záznamů
        pad = n_records * samples_per_record - n_samples_total
        if pad > 0:
            all_data = np.vstack([all_data, np.zeros((pad, n_eeg))])

        # Sestavit Status kanál: 0 všude, markery na správných vzorcích.
        # marker_events = (sample_index, code) -> primo index do status kanalu.
        status_ch = np.zeros(n_records * samples_per_record, dtype=np.int32)
        if self._marker_events:
            # Marker drzime po dobu nekolika vzorku, aby ho find_events spolehlive
            # zachytil i pri filtraci (sirka pulsu ~ 10 vzorku).
            pulse_width = max(2, int(round(0.02 * sfreq)))  # ~20 ms
            for sample_idx, code in self._marker_events:
                idx = int(sample_idx)
                if 0 <= idx < len(status_ch):
                    try:
                        status_ch[idx:idx + pulse_width] = int(code)
                    except (ValueError, OverflowError):
                        pass

        # ── Detekce jednotek LSL streamu ─────────────────────────────
        # LSL EEG zařízení posílají data typicky v µV (hodnoty ~1–200).
        # Pokud jsou hodnoty velmi malé (<0.01), jsou pravděpodobně ve V
        # a je třeba je převést na µV. Tato auto-detekce zabraňuje
        # saturaci signálu (clipping) při zápisu do BDF.
        data_abs_max = float(np.abs(all_data).max()) if all_data.size > 0 else 1.0

        if data_abs_max < 0.1:
            # Data jsou ve Voltech (SI) → převést na µV
            eeg_uv_data = all_data * 1e6
            logger.info("EegRecorder BDF: data detekována v V → převod na µV (×1e6)")
        else:
            # Data jsou již v µV (typické pro LSL EEG streamy)
            eeg_uv_data = all_data.copy()
            logger.info(f"EegRecorder BDF: data v µV, max={data_abs_max:.1f} µV")

        # Fyzikální rozsah: nastav dynamicky podle skutečných dat + 20% rezerva
        p_abs = max(float(np.abs(eeg_uv_data).max()) * 1.2, 500.0)
        phys_min = -p_abs
        phys_max =  p_abs
        digi_min = -8388608
        digi_max =  8388607
        gain = (phys_max - phys_min) / (digi_max - digi_min)  # µV/digit

        def _edf_field(value: str, width: int) -> bytes:
            s = str(value)[:width]
            return s.ljust(width).encode("ascii")

        now = datetime.now()

        # ── BDF hlavička (EDF spec: přesně 256 bytů) ─────────────────
        # Pole: version(8) + patient(80) + recording(80) + date(8) +
        #       time(8) + hdr_bytes(8) + reserved(44) + records(8) +
        #       duration(8) + n_signals(4) = 256 bytů
        hdr = bytearray()

        # Version: 8 bytů — BDF = 0xFF + "BIOSEMI" (EDF = "0       ")
        hdr += b'\xff' + b'BIOSEMI'                     # 8 bytů
        # Local patient identification — 80 bytů
        hdr += _edf_field(f"X X X {safe_name}", 80)    # 80 bytů
        # Local recording identification — 80 bytů
        hdr += _edf_field(
            f"Startdate {now.strftime('%d-%b-%Y').upper()} X X EegRecorder", 80
        )                                                # 80 bytů
        # Start date & time
        hdr += _edf_field(now.strftime("%d.%m.%y"), 8)  # 8 bytů
        hdr += _edf_field(now.strftime("%H.%M.%S"), 8)  # 8 bytů
        # Počet bytů v hlavičce
        n_header_bytes = (n_ch + 1) * 256
        hdr += _edf_field(str(n_header_bytes), 8)       # 8 bytů
        # Reserved (44 bytů) — BDF identifikátor "24BIT"
        hdr += b'24BIT' + b' ' * 39                     # 44 bytů
        # Počet datových záznamů
        hdr += _edf_field(str(n_records), 8)            # 8 bytů
        # Délka záznamu v sekundách
        hdr += _edf_field(str(record_dur_sec), 8)       # 8 bytů
        # Počet signálů
        hdr += _edf_field(str(n_ch), 4)

        assert len(hdr) == 256, f"Hlavicka ma {len(hdr)} bytu misto 256"

        # Signálové hlavičky (po 256 bytech na kanál, ale pole jsou rozložena)
        # Label (16 × n_ch)
        for lbl in status_labels:
            hdr += _edf_field(lbl, 16)
        # Transducer type (80 × n_ch)
        for i in range(n_ch):
            hdr += _edf_field("AgAgCl" if i < n_eeg else "Trigger", 80)
        # Physical dimension (8 × n_ch)
        for i in range(n_ch):
            hdr += _edf_field("uV" if i < n_eeg else "Boolean", 8)
        # Physical minimum (8 × n_ch)
        for i in range(n_ch):
            hdr += _edf_field(str(phys_min if i < n_eeg else 0), 8)
        # Physical maximum (8 × n_ch)
        for i in range(n_ch):
            hdr += _edf_field(str(phys_max if i < n_eeg else 1), 8)
        # Digital minimum (8 × n_ch)
        for i in range(n_ch):
            hdr += _edf_field(str(digi_min if i < n_eeg else 0), 8)
        # Digital maximum (8 × n_ch)
        for i in range(n_ch):
            hdr += _edf_field(str(digi_max if i < n_eeg else 1), 8)
        # Prefiltering (80 × n_ch)
        for _ in range(n_ch):
            hdr += _edf_field("", 80)
        # Samples per record (8 × n_ch)
        for _ in range(n_ch):
            hdr += _edf_field(str(samples_per_record), 8)
        # Reserved (32 × n_ch)
        for _ in range(n_ch):
            hdr += _edf_field("", 32)

        assert len(hdr) == n_header_bytes, \
            f"Hlavicka: ocekavano {n_header_bytes} B, got {len(hdr)} B"

        # ── Datové záznamy ───────────────────────────────────────────
        def _float_to_int24(arr_uv: np.ndarray) -> np.ndarray:
            """µV float → int24 (ořezáno na ±8 388 607)."""
            digital = np.round(arr_uv / gain).astype(np.int64)
            return np.clip(digital, digi_min, digi_max).astype(np.int32)

        eeg_digital = _float_to_int24(eeg_uv_data)  # (total_samples, n_eeg)

        with path.open("wb") as f:
            f.write(bytes(hdr))
            for rec in range(n_records):
                start = rec * samples_per_record
                end   = start + samples_per_record
                # EEG kanály
                for ch in range(n_eeg):
                    chunk = eeg_digital[start:end, ch]
                    for val in chunk:
                        # int24 little-endian = 3 byty
                        b = int(val) & 0xFFFFFF
                        f.write(struct.pack('<I', b)[:3])
                # Status kanál (int24)
                for val in status_ch[start:end]:
                    b = int(val) & 0xFFFFFF
                    f.write(struct.pack('<I', b)[:3])

        logger.debug(
            f"EegRecorder: BDF zápis dokončen — {n_records} záznamů, "
            f"{n_ch} kanálů, {samples_per_record} vzorků/záznam"
        )

    def _save_markers_csv(self, eeg_path: Path) -> None:
        """Uloží markery jako CSV kompatibilní s mode='csv' v offline_analysis."""
        if not self._marker_events:
            return
        try:
            # Pozor: Path.with_suffix neumožňuje více teček → musíme použít string
            csv_path = eeg_path.parent / (eeg_path.stem + ".markers.csv")
            imagery_dur = float(self.config.experiment.imagery_duration)
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("start;end;label\n")
                for sample_idx, code in self._marker_events:
                    start_sec = max(0.0, float(sample_idx) / self._sfreq)
                    end_sec = start_sec + imagery_dur
                    f.write(f"{start_sec:.6f};{end_sec:.6f};{code}\n")
            logger.info(f"EegRecorder: markery CSV uloženy → {csv_path}")
        except Exception as exc:
            logger.warning(f"EegRecorder: nelze uložit markers CSV: {exc}")
