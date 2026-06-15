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
        """Uloží záznam jako BDF+ s pravymi anotacemi.

        Pořadí pokusů:
          1. BDF+ anotace cistym Pythonem (TAL) - PRIMARNI, bez zavislosti
          2. BDF+ pres edfio - zaloha pokud by pure-python selhal
          3. FIF pres mne.save - posledni zaloha
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in patient_name
        )

        # ── 1. Primární: BDF+ anotace cistym Pythonem (bez zavislosti) ──
        fname_bdf = self.output_dir / f"eeg_{safe_name}_{timestamp_str}_raw.bdf"
        try:
            self._write_bdf(fname_bdf, patient_name=patient_name)
            self.saved_path = fname_bdf
            logger.info(f"EegRecorder: uloženo jako BDF+ s anotacemi → {fname_bdf}")
            self._save_markers_csv(fname_bdf)
            return fname_bdf
        except Exception as bdf_exc:
            logger.warning(
                "EegRecorder: pure-python BDF+ selhal (%s), zkousim edfio...", bdf_exc
            )

        # ── 2. Záloha: BDF+ pres edfio ──────────────────────────────
        try:
            self._write_bdf_annotations(fname_bdf)
            self.saved_path = fname_bdf
            logger.info(f"EegRecorder: uloženo jako BDF+ (edfio) → {fname_bdf}")
            self._save_markers_csv(fname_bdf)
            return fname_bdf
        except Exception as exc:
            logger.warning(f"EegRecorder: edfio BDF+ selhal ({exc}), ukladam FIF...")

        # ── 2. Poslední záloha: FIF (s MNE anotacemi) ───────────────
        fname_fif = self.output_dir / f"eeg_{safe_name}_{timestamp_str}_raw.fif"
        try:
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

    def _code_to_description(self) -> dict:
        """Mapa kod tridy -> citelny popisek pro anotace.

        Pouzije cesky popisek z paradigm.cues (napr. 'PRAVA RUKA').
        Kod 0 = klidovy interval -> 'KLID'.
        """
        classes = self.config.paradigm.get("classes", {})
        cues = self.config.paradigm.get("cues", {})
        mapping = {}
        for key, code in classes.items():
            label = str(cues.get(key, {}).get("label", key)).strip()
            mapping[int(code)] = label or str(key)
        mapping[0] = "KLID"
        return mapping

    def _write_bdf_annotations(self, path: Path) -> None:
        """Zapíše BDF+ s pravymi anotacemi (markery) pres edfio.

        Markery jsou ulozeny jako BDF+ anotace -> MNE je nacte do
        raw.annotations a EDFbrowser je zobrazi jako anotace.
        Vyzaduje balik edfio.
        """
        from edfio import Bdf, BdfSignal, EdfAnnotation

        all_data = np.vstack(self._data_chunks)   # (n_samples, n_eeg)
        n_samples, n_eeg = all_data.shape

        # Detekce jednotek: data ve V -> prevest na uV (viz _write_bdf)
        data_abs_max = float(np.abs(all_data).max()) if all_data.size else 1.0
        if data_abs_max < 0.1:
            eeg_uv = all_data * 1e6
        else:
            eeg_uv = all_data
        p_abs = max(float(np.abs(eeg_uv).max()) * 1.2, 100.0)

        ch_names = self._ch_names if self._ch_names else [f"CH{i+1}" for i in range(n_eeg)]

        def _lbl(name: str) -> str:
            safe = "".join(c for c in name if c.isascii() and c.isprintable())
            return (safe[:16] or "CH")

        signals = [
            BdfSignal(
                eeg_uv[:, i].astype(np.float64),
                sampling_frequency=self._sfreq,
                label=_lbl(ch_names[i]),
                physical_dimension="uV",
                physical_range=(-p_abs, p_abs),
            )
            for i in range(n_eeg)
        ]

        # Markery -> BDF+ anotace s citelnym popiskem (PRAVA RUKA, KLID...)
        imagery_dur = float(self.config.experiment.imagery_duration)
        iti_dur = float(getattr(self.config.experiment, "iti_duration", 1.5))
        code_to_desc = self._code_to_description()
        annotations = [
            EdfAnnotation(
                onset=max(0.0, float(sample_idx) / self._sfreq),
                duration=(iti_dur if int(code) == 0 else imagery_dur),
                text=code_to_desc.get(int(code), str(code)),
            )
            for sample_idx, code in self._marker_events
        ]

        bdf = Bdf(signals, annotations=annotations, data_record_duration=1.0)
        bdf.write(str(path))
        logger.info(
            "EegRecorder: BDF+ zapsano (%d kanalu, %d anotaci)",
            n_eeg, len(annotations),
        )

    def _write_bdf(self, path: Path, patient_name: str = "X") -> None:
        """Zapíše BDF+ s pravymi anotacemi v cistem Pythonu (bez edfio).

        Markery se ulozi jako BDF+ anotace (TAL) v kanalu "BDF Annotations".
        MNE je nacte do raw.annotations, EDFbrowser zobrazi jako anotace.
        24-bit vzorky (BioSemi), funguje v kazdem prostredi.
        """
        import struct
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in patient_name)

        all_data = np.vstack(self._data_chunks)   # (n_samples, n_eeg_ch)
        n_samples_total, n_eeg = all_data.shape
        sfreq = self._sfreq

        def _edf_label(name: str) -> str:
            safe = "".join(c for c in name if c.isascii() and c.isprintable())
            return safe[:16]

        ch_labels = [_edf_label(n) for n in self._ch_names] if self._ch_names \
                    else [f"CH{i+1}" for i in range(n_eeg)]

        # Posledni kanal = "BDF Annotations" (TAL), ne Status
        n_ch = n_eeg + 1
        record_dur_sec = 1
        eeg_spr = int(round(sfreq * record_dur_sec))           # vzorku/zaznam (EEG)
        n_records = int(np.ceil(n_samples_total / eeg_spr))

        pad = n_records * eeg_spr - n_samples_total
        if pad > 0:
            all_data = np.vstack([all_data, np.zeros((pad, n_eeg))])

        # ── Sestavit TAL anotace pro kazdy zaznam ────────────────────
        imagery_dur = float(self.config.experiment.imagery_duration)
        iti_dur = float(getattr(self.config.experiment, "iti_duration", 1.5))
        code_to_desc = self._code_to_description()

        # markery rozdelit do zaznamu podle onsetu; popis = citelny nazev,
        # delka = imagery_dur (klid 'KLID' kod 0 ma delku iti)
        markers_by_rec: dict = {}
        for sample_idx, code in self._marker_events:
            onset = float(sample_idx) / sfreq
            rec_idx = int(onset // record_dur_sec)
            desc = code_to_desc.get(int(code), str(code))
            dur = iti_dur if int(code) == 0 else imagery_dur
            markers_by_rec.setdefault(rec_idx, []).append((onset, dur, desc))

        def _num(x: float) -> bytes:
            s = ("%f" % x).rstrip("0").rstrip(".")
            return (s if s else "0").encode("ascii")

        def _tal_for_record(rec_idx: int) -> bytes:
            # timekeeping TAL (povinny prvni v kazdem zaznamu)
            tal = b"+" + _num(rec_idx * record_dur_sec) + b"\x14\x14\x00"
            for onset, dur, desc in markers_by_rec.get(rec_idx, []):
                tal += (b"+" + _num(onset) + b"\x15" + _num(dur)
                        + b"\x14" + str(desc).encode("ascii", "replace") + b"\x14\x00")
            return tal

        # Velikost annotation kanalu: nejdelsi TAL blok + rezerva
        max_tal = max((len(_tal_for_record(r)) for r in range(n_records)), default=16)
        annot_bytes = max(max_tal, 16)
        annot_spr = int(np.ceil(annot_bytes / 3.0))    # BDF: 3 byty/vzorek
        annot_bytes = annot_spr * 3

        # ── Detekce jednotek (V vs µV) ───────────────────────────────
        data_abs_max = float(np.abs(all_data).max()) if all_data.size > 0 else 1.0
        if data_abs_max < 0.1:
            eeg_uv_data = all_data * 1e6
            logger.info("EegRecorder BDF: data ve V → prevod na µV")
        else:
            eeg_uv_data = all_data.copy()
            logger.info("EegRecorder BDF: data v µV, max=%.1f µV", data_abs_max)

        p_abs = max(float(np.abs(eeg_uv_data).max()) * 1.2, 500.0)
        phys_min, phys_max = -p_abs, p_abs
        digi_min, digi_max = -8388608, 8388607
        gain = (phys_max - phys_min) / (digi_max - digi_min)

        def _edf_field(value: str, width: int) -> bytes:
            return str(value)[:width].ljust(width).encode("ascii")

        now = datetime.now()
        spr_list = [eeg_spr] * n_eeg + [annot_spr]  # samples/record per signal

        # ── Hlavni hlavička (256 B) ──────────────────────────────────
        hdr = bytearray()
        hdr += b'\xff' + b'BIOSEMI'
        hdr += _edf_field(f"X X X {safe_name}", 80)
        hdr += _edf_field(f"Startdate {now.strftime('%d-%b-%Y').upper()} X X EegRecorder", 80)
        hdr += _edf_field(now.strftime("%d.%m.%y"), 8)
        hdr += _edf_field(now.strftime("%H.%M.%S"), 8)
        n_header_bytes = (n_ch + 1) * 256
        hdr += _edf_field(str(n_header_bytes), 8)
        # Reserved: musi zacinat "BDF+C" pro BDF+ s anotacemi
        hdr += _edf_field("BDF+C", 44)
        hdr += _edf_field(str(n_records), 8)
        hdr += _edf_field(str(record_dur_sec), 8)
        hdr += _edf_field(str(n_ch), 4)
        assert len(hdr) == 256

        def _is_annot(i: int) -> bool:
            return i == n_eeg  # posledni kanal

        # Labels
        for i in range(n_ch):
            hdr += _edf_field("BDF Annotations" if _is_annot(i) else ch_labels[i], 16)
        # Transducer
        for i in range(n_ch):
            hdr += _edf_field("" if _is_annot(i) else "AgAgCl", 80)
        # Physical dimension
        for i in range(n_ch):
            hdr += _edf_field("" if _is_annot(i) else "uV", 8)
        # Physical min
        for i in range(n_ch):
            hdr += _edf_field("-1" if _is_annot(i) else str(phys_min), 8)
        # Physical max
        for i in range(n_ch):
            hdr += _edf_field("1" if _is_annot(i) else str(phys_max), 8)
        # Digital min
        for i in range(n_ch):
            hdr += _edf_field(str(digi_min), 8)
        # Digital max
        for i in range(n_ch):
            hdr += _edf_field(str(digi_max), 8)
        # Prefiltering
        for i in range(n_ch):
            hdr += _edf_field("", 80)
        # Samples per record
        for i in range(n_ch):
            hdr += _edf_field(str(spr_list[i]), 8)
        # Reserved
        for i in range(n_ch):
            hdr += _edf_field("", 32)

        assert len(hdr) == n_header_bytes, f"hdr {len(hdr)} != {n_header_bytes}"

        # ── Data ─────────────────────────────────────────────────────
        digital = np.clip(np.round(eeg_uv_data / gain), digi_min, digi_max).astype(np.int64)

        def _int24le(val: int) -> bytes:
            # little-endian 24-bit signed (low 3 byty z 32-bit two's complement)
            return struct.pack('<i', int(val))[:3]

        with path.open("wb") as f:
            f.write(bytes(hdr))
            for rec in range(n_records):
                s0 = rec * eeg_spr
                s1 = s0 + eeg_spr
                for ch in range(n_eeg):
                    for val in digital[s0:s1, ch]:
                        f.write(_int24le(val))
                # Annotation kanal: raw TAL byty, doplnene \x00
                tal = _tal_for_record(rec)
                f.write(tal.ljust(annot_bytes, b"\x00"))

        logger.info(
            "EegRecorder: BDF+ zapsano (%d EEG kanalu + anotace, %d zaznamu, %d markeru)",
            n_eeg, n_records, len(self._marker_events),
        )

    def _save_markers_csv(self, eeg_path: Path) -> None:
        """Uloží markery jako CSV s citelnym popiskem (start;end;label;code)."""
        if not self._marker_events:
            return
        try:
            csv_path = eeg_path.parent / (eeg_path.stem + ".markers.csv")
            imagery_dur = float(self.config.experiment.imagery_duration)
            iti_dur = float(getattr(self.config.experiment, "iti_duration", 1.5))
            code_to_desc = self._code_to_description()
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("start;end;label;code\n")
                for sample_idx, code in self._marker_events:
                    start_sec = max(0.0, float(sample_idx) / self._sfreq)
                    dur = iti_dur if int(code) == 0 else imagery_dur
                    desc = code_to_desc.get(int(code), str(code))
                    f.write(f"{start_sec:.6f};{start_sec + dur:.6f};{desc};{code}\n")
            logger.info(f"EegRecorder: markery CSV uloženy → {csv_path}")
        except Exception as exc:
            logger.warning(f"EegRecorder: nelze uložit markers CSV: {exc}")
