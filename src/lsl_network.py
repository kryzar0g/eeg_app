"""LSL network utilities – discovery, configuration and connection helpers.

Pouziti:
    from src.lsl_network import scan_streams, configure_network, test_connection

EEG zarizeni streamuje pres LSL po siti (TCP/UDP multicast).
Tento modul zajistuje:
  1. Skenovani dostupnych streamu na lokalni siti
  2. Konfiguraci lsl_api.cfg pro prime IP pripojeni (cross-subnet)
  3. Test pripojeni k EEG zarizeni
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Umisteni lsl_api.cfg – vedle EXE nebo v projektu
_CFG_CANDIDATES = [
    Path.cwd() / "lsl_api.cfg",
    Path(__file__).resolve().parents[1] / "lsl_api.cfg",
]


# ─── Data třídy ──────────────────────────────────────────────────────────────

class StreamDesc:
    """Popis nalezeného LSL streamu."""

    def __init__(self, info: Any) -> None:
        self.name: str       = info.name()
        self.stream_type: str = info.type()
        self.channels: int   = info.channel_count()
        self.sfreq: float    = float(info.nominal_srate())
        self.hostname: str   = info.hostname()
        self.source_id: str  = info.source_id()
        self._info           = info

    def __str__(self) -> str:
        return (
            f"{self.name} [{self.stream_type}] "
            f"{self.channels}ch @ {self.sfreq:.0f}Hz  "
            f"host={self.hostname}"
        )

    def make_inlet(self) -> Any:
        """Vytvoří StreamInlet pro tento stream."""
        from pylsl import StreamInlet
        return StreamInlet(self._info, max_chunklen=0)


# ─── Skenování sítě ──────────────────────────────────────────────────────────

def scan_streams(timeout: float = 3.0, stream_type: str = "") -> List[StreamDesc]:
    """Najde všechny LSL streamy na lokální síti.

    Args:
        timeout:     Jak dlouho čekat na odpovědi (sekundy).
        stream_type: Filtrovat podle typu (např. "EEG"). Prázdné = vše.

    Returns:
        Seznam StreamDesc objektů seřazených: EEG první, pak ostatní.
    """
    try:
        from pylsl import resolve_streams, resolve_byprop
    except ImportError:
        logger.error("pylsl neni nainstalovano")
        return []

    logger.info("LSL scan: hledam streamy (timeout=%.1fs) ...", timeout)
    t0 = time.time()
    try:
        if stream_type:
            raw = resolve_byprop("type", stream_type, timeout=timeout)
        else:
            raw = resolve_streams(wait_time=timeout)
    except Exception as exc:
        logger.warning("LSL scan selhal: %s", exc)
        return []

    elapsed = time.time() - t0
    descs = [StreamDesc(s) for s in raw]

    # Seradit: EEG napred
    descs.sort(key=lambda d: (0 if d.stream_type.upper() == "EEG" else 1, d.name))

    logger.info("LSL scan: nalezeno %d streamu za %.1fs", len(descs), elapsed)
    for d in descs:
        logger.info("  %s", d)

    return descs


def test_connection(host_or_stream_name: str = "",
                    stream_type: str = "EEG",
                    timeout: float = 5.0) -> Optional[StreamDesc]:
    """Otestuje připojení k EEG streamu.

    Args:
        host_or_stream_name: IP / hostname nebo název streamu. Prázdné = první nalezený.
        stream_type:         Typ streamu (default "EEG").
        timeout:             Timeout v sekundách.

    Returns:
        StreamDesc pokud je stream dostupný, jinak None.
    """
    streams = scan_streams(timeout=timeout, stream_type=stream_type)
    if not streams:
        return None

    if host_or_stream_name:
        target = host_or_stream_name.strip().lower()
        # Hledat podle hostname nebo jmena streamu
        for s in streams:
            if s.hostname.lower() == target or s.name.lower() == target:
                return s
        # Zadny presny match – zkusit castecny
        for s in streams:
            if target in s.hostname.lower() or target in s.name.lower():
                return s
        return None

    # Prvni nalezeny EEG stream
    return streams[0]


# ─── Konfigurace lsl_api.cfg ─────────────────────────────────────────────────

def configure_network(
    known_peers: Optional[List[str]] = None,
    multicast_port: int = 16571,
    base_port: int = 16572,
    ipv6: bool = True,
    output_path: Optional[Path] = None,
) -> Path:
    """Vygeneruje lsl_api.cfg pro síťové připojení k EEG zařízení.

    Použití pro přímé IP připojení (cross-subnet, bez multicastu):
        configure_network(known_peers=["192.168.1.100"])

    Args:
        known_peers:    Seznam IP adres EEG zařízení.
        multicast_port: Port pro LSL multicast (default 16571).
        base_port:      Základní port pro datové přenosy (default 16572).
        ipv6:           Povolit IPv6.
        output_path:    Kde uložit cfg soubor (default: vedle projektu).

    Returns:
        Cesta k uloženému cfg souboru.
    """
    if output_path is None:
        output_path = _CFG_CANDIDATES[0]

    peers_str = ""
    if known_peers:
        sanitized = [p.strip() for p in known_peers if p.strip()]
        if sanitized:
            peers_str = "{" + ", ".join(sanitized) + "}"

    cfg_content = f"""; lsl_api.cfg – automaticky vygenerováno aplikací EEG BCI
; Editujte pro nastavení síťového streamování EEG.

[multicast]
; Pridat IP adresy EEG zarizeni pro prime pripojeni (cross-subnet)
KnownPeers = {peers_str}
MulticastPort = {multicast_port}
IPv6 = {"allow" if ipv6 else "disable"}

[ports]
BasePort = {base_port}
PortRange = 32
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cfg_content, encoding="utf-8")
    logger.info("lsl_api.cfg zapsan: %s", output_path)
    return output_path


def get_lsl_cfg_path() -> Optional[Path]:
    """Vrátí cestu k existujícímu lsl_api.cfg nebo None."""
    for p in _CFG_CANDIDATES:
        if p.is_file():
            return p
    return None


def read_known_peers() -> List[str]:
    """Přečte seznam KnownPeers z existujícího lsl_api.cfg."""
    cfg_path = get_lsl_cfg_path()
    if cfg_path is None:
        return []
    try:
        text = cfg_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("knownpeers"):
                # KnownPeers = {192.168.1.1, 192.168.1.2}
                _, _, value = line.partition("=")
                value = value.strip().strip("{}")
                return [ip.strip() for ip in value.split(",") if ip.strip()]
    except Exception:
        pass
    return []
