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
import socket
import time
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Umisteni lsl_api.cfg – liblsl prohledava tato mista (v poradi priority):
# 1. %LOCALAPPDATA%\labstreaminglayer\liblsl\  (Windows – nejvyssi priorita)
# 2. ~/.config/labstreaminglayer/liblsl/        (Linux/Mac)
# 3. Adresar EXE / projekt
import os as _os, sys as _sys

def _lsl_cfg_paths() -> list:
    paths = []
    # Windows: %LOCALAPPDATA%\labstreaminglayer\liblsl\lsl_api.cfg
    local_app = _os.environ.get("LOCALAPPDATA", "")
    if local_app:
        paths.append(Path(local_app) / "labstreaminglayer" / "liblsl" / "lsl_api.cfg")
    # Linux/Mac: ~/.config/labstreaminglayer/liblsl/lsl_api.cfg
    paths.append(Path.home() / ".config" / "labstreaminglayer" / "liblsl" / "lsl_api.cfg")
    # Projekt / EXE adresar
    if getattr(_sys, "frozen", False):
        paths.append(Path(_sys.executable).parent / "lsl_api.cfg")
    else:
        paths.append(Path(__file__).resolve().parents[1] / "lsl_api.cfg")
    paths.append(Path.cwd() / "lsl_api.cfg")
    return paths

_CFG_CANDIDATES = _lsl_cfg_paths()


def _local_hostnames() -> set:
    """Vrátí množinu názvů tohoto počítače (hostname + varianty)."""
    names = set()
    try:
        h = socket.gethostname()
        names.add(h.lower())
        names.add(h.split(".")[0].lower())          # bez domény
        names.add(socket.getfqdn(h).lower())        # plné jméno
        # Přidat lokální IP adresy
        for info in socket.getaddrinfo(h, None):
            names.add(info[4][0].lower())
    except Exception:
        pass
    names.update({"localhost", "127.0.0.1", "::1"})
    return names


# Singleton – vypočteme jednou
_LOCAL_NAMES: set = set()


def _is_local(hostname: str) -> bool:
    """Vrátí True pokud stream pochází z tohoto počítače."""
    global _LOCAL_NAMES
    if not _LOCAL_NAMES:
        _LOCAL_NAMES = _local_hostnames()
    return hostname.lower() in _LOCAL_NAMES


# ─── Data třída ──────────────────────────────────────────────────────────────

class StreamDesc:
    """Popis nalezeného LSL streamu."""

    def __init__(self, info: Any) -> None:
        self.name: str        = info.name()
        self.stream_type: str = info.type()
        self.channels: int    = info.channel_count()
        self.sfreq: float     = float(info.nominal_srate())
        self.hostname: str    = info.hostname()
        self.source_id: str   = info.source_id()
        self.is_local: bool   = _is_local(info.hostname())
        self._info            = info

    @property
    def source_label(self) -> str:
        """'LOCAL' nebo 'NETWORK' – odkud stream pochází."""
        return "LOCAL" if self.is_local else "NETWORK"

    def __str__(self) -> str:
        return (
            f"[{self.source_label}] {self.name} [{self.stream_type}] "
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


# ─── Ping / dosažitelnost ────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class HostStatus:
    """Výsledek testu dosažitelnosti hostitele."""
    host: str
    reachable: bool          # reaguje na ping / TCP spojení
    ping_ms: float           # průměrná latence v ms (-1 pokud nedostupné)
    lsl_port_open: bool      # LSL port 16571 otevřen
    error: str               # popis chyby (prázdné = OK)

    def __str__(self) -> str:
        if not self.reachable:
            return f"{self.host}: NEDOSTUPNY ({self.error})"
        lsl = "LSL-PORT-OK" if self.lsl_port_open else "LSL-PORT-ZAVRENY"
        return f"{self.host}: OK  ping={self.ping_ms:.0f}ms  {lsl}"


def ping_host(host: str, count: int = 3, timeout_sec: float = 2.0) -> HostStatus:
    """Otestuje dosažitelnost hostitele přes ICMP ping a TCP LSL port.

    Funguje na Windows i Linux. Firewall může blokovat ICMP – proto
    fallback přes TCP spojení na port 16571 (LSL).

    Args:
        host:        IP adresa nebo hostname (např. "192.168.1.100").
        count:       Počet ping pokusů.
        timeout_sec: Timeout na jeden pokus v sekundách.

    Returns:
        HostStatus s výsledkem testu.
    """
    import subprocess
    import platform

    host = host.strip()
    if not host:
        return HostStatus(host, False, -1, False, "Prazdna adresa")

    # ── 1. ICMP ping ─────────────────────────────────────────────────
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", str(count), "-w", str(int(timeout_sec * 1000)), host]
    else:
        cmd = ["ping", "-c", str(count), "-W", str(int(timeout_sec)), host]

    ping_ms = -1.0
    reachable = False
    error = ""

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec * count + 2,
        )
        output = result.stdout + result.stderr

        if result.returncode == 0:
            reachable = True
            # Parsovat průměrnou latenci z výstupu
            import re
            # Windows: "Minimum = Xms, Maximum = Xms, Average = Xms"
            m = re.search(r"[Aa]verage[^=]*=\s*(\d+)", output)
            if not m:
                # Linux: "rtt min/avg/max/mdev = X/X/X/X ms"
                m = re.search(r"=\s*[\d.]+/([\d.]+)/", output)
            if m:
                ping_ms = float(m.group(1))
        else:
            error = "Ping selhal (ICMP mozna blokovan firewallom)"
    except subprocess.TimeoutExpired:
        error = "Ping timeout"
    except FileNotFoundError:
        error = "Prikaz ping nenalezen"
    except Exception as exc:
        error = str(exc)

    # ── 2. TCP fallback – LSL port 16571 ─────────────────────────────
    lsl_port_open = False
    try:
        import socket as _socket
        with _socket.create_connection((host, 16571), timeout=timeout_sec):
            lsl_port_open = True
            if not reachable:
                reachable = True
                error = ""
    except OSError:
        pass  # Port zavreny nebo host nedostupny

    # ── 3. TCP fallback – port 16572 (alternativni LSL port) ─────────
    if not lsl_port_open:
        try:
            import socket as _socket
            with _socket.create_connection((host, 16572), timeout=timeout_sec):
                lsl_port_open = True
                if not reachable:
                    reachable = True
                    error = ""
        except OSError:
            pass

    return HostStatus(
        host=host,
        reachable=reachable,
        ping_ms=ping_ms,
        lsl_port_open=lsl_port_open,
        error=error,
    )


def scan_subnet(
    subnet: str = "",
    timeout_sec: float = 0.5,
    max_workers: int = 50,
) -> List[HostStatus]:
    """Prohledá celou podsíť a najde aktivní hostitele.

    Pokud subnet není zadán, odvodí ho z lokální IP adresy.

    Args:
        subnet:      Prefix podsítě, např. "192.168.1" (bez poslední oktety).
        timeout_sec: Timeout pro každý ping.
        max_workers: Počet paralelních vláken.

    Returns:
        Seznam HostStatus pro dosažitelné hostitele (seřazeno podle IP).
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Odvodit podsíť z lokální IP pokud není zadána
    if not subnet:
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            parts = local_ip.split(".")
            if len(parts) == 4:
                subnet = ".".join(parts[:3])
        except Exception:
            pass
    if not subnet:
        return []

    logger.info("Subnet scan: %s.1-%s.254 (timeout=%.1fs)", subnet, subnet, timeout_sec)

    hosts = [f"{subnet}.{i}" for i in range(1, 255)]

    results: List[HostStatus] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(ping_host, h, 1, timeout_sec): h for h in hosts}
        for future in as_completed(futures):
            try:
                status = future.result()
                if status.reachable:
                    results.append(status)
            except Exception:
                pass

    results.sort(key=lambda s: tuple(int(x) for x in s.host.split(".")))
    logger.info("Subnet scan: nalezeno %d aktivnich hostu", len(results))
    return results


# ─── Konfigurace lsl_api.cfg ─────────────────────────────────────────────────

def configure_network(
    known_peers: Optional[List[str]] = None,
    multicast_port: int = 16571,
    base_port: int = 16572,
    ipv6: bool = True,
    output_path: Optional[Path] = None,
) -> Path:
    """Vygeneruje lsl_api.cfg pro síťové připojení k EEG zařízení.

    Zapíše soubor do VŠECH míst kde liblsl hledá konfiguraci:
      - %LOCALAPPDATA%\\labstreaminglayer\\liblsl\\lsl_api.cfg  (Windows – nejvyšší priorita)
      - ~/.config/labstreaminglayer/liblsl/lsl_api.cfg           (Linux/Mac)
      - projektový adresář

    Použití pro přímé IP připojení (cross-subnet, bez multicastu):
        configure_network(known_peers=["192.168.1.100"])
    """
    peers_str = ""
    if known_peers:
        sanitized = [p.strip() for p in known_peers if p.strip()]
        if sanitized:
            peers_str = "{" + ", ".join(sanitized) + "}"

    cfg_content = (
        "; lsl_api.cfg – automaticky vygenerovano aplikaci EEG BCI\n"
        "; Editujte pro nastaveni sitoveho streamovani EEG.\n"
        "\n"
        "[multicast]\n"
        "; IP adresy EEG zarizeni pro prime pripojeni (cross-subnet)\n"
        f"KnownPeers = {peers_str}\n"
        f"MulticastPort = {multicast_port}\n"
        f"IPv6 = {'allow' if ipv6 else 'disable'}\n"
        "\n"
        "[ports]\n"
        f"BasePort = {base_port}\n"
        "PortRange = 32\n"
    )

    # Zapsat do vsech umisteni kde liblsl hleda konfiguraci
    written: List[Path] = []
    targets = list(_lsl_cfg_paths())
    if output_path:
        targets.insert(0, Path(output_path))

    for p in targets:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(cfg_content, encoding="utf-8")
            written.append(p)
            logger.info("lsl_api.cfg zapsan: %s", p)
        except Exception as exc:
            logger.debug("lsl_api.cfg nelze zapsat do %s: %s", p, exc)

    if not written:
        raise RuntimeError("Nepodarilo se zapsat lsl_api.cfg do zadneho umisteni")

    return written[0]


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
