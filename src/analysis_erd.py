"""ERD / koherence analyza motoricke imaginace.

Vedecky princip
---------------
Pri predstave pohybu KONCETINY se v MOTORICKEM KORTEXU utlumi
senzomotoricky rytmus (mu 8-12 Hz a beta 13-30 Hz). Tento utlum je
KONTRALATERALNI:
    * predstava PRAVE ruky  -> utlum v LEVE hemisfere  (elektroda C3)
    * predstava LEVE ruky   -> utlum v PRAVE hemisfere (elektroda C4)
    * predstava NOHOU       -> utlum centralne          (elektroda Cz)

Tomuto jevu se rika ERD (Event-Related Desynchronization) a meri se v %:

    ERD% = (P_imagery - P_baseline) / P_baseline * 100

    P_baseline = vykon v pasmu pred markerem (klid)
    P_imagery  = vykon v pasmu behem imaginace
    ZAPORNA hodnota = utlum (desynchronizace) = ocekavany jev.

Koherence
---------
Koherence mezi C3 a C4 vyjadruje funkcni propojeni hemisfer.
Pri unilateralni imaginaci ruky obvykle KLESA (hemisfery se "rozpoji").

Vystup
------
* slovnik s ERD% a koherenci pro kazdou tridu
* obrazek (PNG) ulozeny do reports/
* textovy souhrn s interpretaci lateralizace
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from .config import AppConfig, PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


# ─── Pomocne ──────────────────────────────────────────────────────────────────

def _find_channel(ch_names: List[str], candidates: List[str]) -> Optional[int]:
    """Najde index kanalu podle presneho jmena, pak podle podretezce."""
    upper = [c.upper() for c in ch_names]
    # 1. presna shoda
    for cand in candidates:
        cu = cand.upper()
        if cu in upper:
            return upper.index(cu)
    # 2. podretezec (napr. 'EEG C3' obsahuje 'C3')
    for cand in candidates:
        cu = cand.upper()
        for i, name in enumerate(upper):
            if cu in name:
                return i
    return None


def _resolve_motor_channels(
    ch_names: List[str], analysis_cfg: Dict
) -> Dict[str, Optional[int]]:
    """Urci indexy elektrod C3 (leva), C4 (prava), Cz (central).

    Pokud presne nazvy nejsou, zkusi podretezec; nakonec fallback
    na rozdeleni kanalu na levou/pravou polovinu.
    """
    chans = analysis_cfg.get("channels", {})
    left_cands = chans.get("left_hemisphere", ["C3"])
    right_cands = chans.get("right_hemisphere", ["C4"])
    central_cands = chans.get("central", ["Cz"])

    c3 = _find_channel(ch_names, left_cands)
    c4 = _find_channel(ch_names, right_cands)
    cz = _find_channel(ch_names, central_cands)

    # Fallback: rozdelit kanaly na poloviny
    n = len(ch_names)
    if c3 is None and n >= 2:
        c3 = n // 4              # nekde v leve polovine
        logger.warning("C3 nenalezen, fallback na kanal index %d (%s)", c3, ch_names[c3])
    if c4 is None and n >= 2:
        c4 = (3 * n) // 4        # nekde v prave polovine
        logger.warning("C4 nenalezen, fallback na kanal index %d (%s)", c4, ch_names[c4])
    if cz is None and n >= 1:
        cz = n // 2
        logger.warning("Cz nenalezen, fallback na kanal index %d (%s)", cz, ch_names[cz])

    return {"C3": c3, "C4": c4, "Cz": cz}


def _band_power(sig: np.ndarray, sfreq: float, band: Tuple[float, float]) -> float:
    """Prumerny vykon signalu v danem frekvencnim pasmu (Welch PSD)."""
    if sig.size < 16:
        return float("nan")
    nperseg = min(256, sig.size)
    freqs, psd = signal.welch(sig, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return float("nan")
    return float(np.mean(psd[mask]))


def _coherence(sig_a: np.ndarray, sig_b: np.ndarray, sfreq: float,
               band: Tuple[float, float]) -> float:
    """Prumerna magnitude-squared koherence mezi dvema signaly v pasmu."""
    n = min(sig_a.size, sig_b.size)
    if n < 32:
        return float("nan")
    nperseg = min(256, n)
    freqs, cxy = signal.coherence(sig_a[:n], sig_b[:n], fs=sfreq, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return float("nan")
    return float(np.mean(cxy[mask]))


# ─── Hlavni analyza ─────────────────────────────────────────────────────────

def run_erd_analysis(
    file_path_str: str,
    config: Optional[AppConfig] = None,
) -> Dict:
    """Spocita ERD a koherenci z nahraneho BDF/EDF/FIF souboru.

    Returns:
        slovnik s vysledky: {
            'classes': {label: {'C3': {'mu_erd': ..., 'beta_erd': ...}, ...,
                                'coherence_mu': ..., 'lateralization': ...}},
            'report_text': str,
            'figure_path': str | None,
            'channels': {'C3': name, 'C4': name, 'Cz': name},
        }
    """
    import mne
    from .offline_analysis import _load_raw

    if config is None:
        config = load_config()

    analysis_cfg = config.analysis or {}
    bands = analysis_cfg.get("bands", {"mu": [8.0, 12.0], "beta": [13.0, 30.0]})
    mu_band = tuple(bands.get("mu", [8.0, 12.0]))
    beta_band = tuple(bands.get("beta", [13.0, 30.0]))
    baseline_win = analysis_cfg.get("baseline_window", [-1.5, -0.5])
    imagery_win = analysis_cfg.get("imagery_window", [0.5, 3.5])

    file_path = Path(file_path_str).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Soubor nenalezen: {file_path}")

    logger.info("ERD analyza: nacitam %s", file_path.name)
    raw = _load_raw(file_path)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.info["ch_names"])

    # ── Udalosti (markery) -> seznam (onset_sample, label) ─────────────
    from .offline_analysis import _annotation_desc_to_code
    class_map = config.paradigm.get("classes", {})
    code_to_label = {int(v): str(k) for k, v in class_map.items()}

    trial_events: List[Tuple[int, str]] = []  # (onset_sample, label)

    if "Status" in ch_names:
        # Stary format: Status kanal s integer kody
        try:
            ev = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="ERROR")
            for e in ev:
                trial_events.append((int(e[0]), code_to_label.get(int(e[2]), f"code_{int(e[2])}")))
        except Exception:
            pass

    if not trial_events:
        # Novy format: BDF+ anotace s citelnymi popisky
        try:
            ev, ann_id = mne.events_from_annotations(raw, verbose="ERROR")
            int_to_label = {}
            for desc, mne_int in ann_id.items():
                code = _annotation_desc_to_code(desc, config)
                if code is not None:
                    int_to_label[mne_int] = code_to_label.get(code, str(desc))
                # klid/nezname vynechame (KLID se do ERD nepocita)
            for e in ev:
                if int(e[2]) in int_to_label:
                    trial_events.append((int(e[0]), int_to_label[int(e[2])]))
        except Exception:
            pass

    if not trial_events:
        raise RuntimeError("Zadne markery trid v souboru - nelze pocitat ERD.")

    # ── Motoricke kanaly ──────────────────────────────────────────────
    motor = _resolve_motor_channels(ch_names, analysis_cfg)
    c3_idx, c4_idx, cz_idx = motor["C3"], motor["C4"], motor["Cz"]
    motor_names = {
        "C3": ch_names[c3_idx] if c3_idx is not None else "?",
        "C4": ch_names[c4_idx] if c4_idx is not None else "?",
        "Cz": ch_names[cz_idx] if cz_idx is not None else "?",
    }
    logger.info("Motoricke kanaly: C3=%s C4=%s Cz=%s",
                motor_names["C3"], motor_names["C4"], motor_names["Cz"])

    data = raw.get_data()  # (n_channels, n_samples) ve V

    def seg(ch_idx: int, onset_sample: int, t0: float, t1: float) -> np.ndarray:
        s0 = onset_sample + int(round(t0 * sfreq))
        s1 = onset_sample + int(round(t1 * sfreq))
        s0 = max(0, s0)
        s1 = min(data.shape[1], s1)
        if s1 <= s0 or ch_idx is None:
            return np.array([])
        return data[ch_idx, s0:s1]

    # ── Akumulace ERD/koherence po tridach ────────────────────────────
    # Pro kazdou tridu sbirame ERD% per kanal/pasmo a koherenci
    per_class: Dict[str, Dict[str, List[float]]] = {}

    for onset, label in trial_events:
        acc = per_class.setdefault(label, {
            "C3_mu": [], "C4_mu": [], "Cz_mu": [],
            "C3_beta": [], "C4_beta": [], "Cz_beta": [],
            "coh_mu": [], "coh_beta": [],
        })

        for ch_key, ch_idx in (("C3", c3_idx), ("C4", c4_idx), ("Cz", cz_idx)):
            if ch_idx is None:
                continue
            base = seg(ch_idx, onset, baseline_win[0], baseline_win[1])
            imag = seg(ch_idx, onset, imagery_win[0], imagery_win[1])
            for band_key, band in (("mu", mu_band), ("beta", beta_band)):
                p_base = _band_power(base, sfreq, band)
                p_imag = _band_power(imag, sfreq, band)
                if p_base and p_base > 0 and not np.isnan(p_imag):
                    erd = (p_imag - p_base) / p_base * 100.0
                    acc[f"{ch_key}_{band_key}"].append(erd)

        # Koherence C3-C4 behem imaginace
        if c3_idx is not None and c4_idx is not None:
            a = seg(c3_idx, onset, imagery_win[0], imagery_win[1])
            b = seg(c4_idx, onset, imagery_win[0], imagery_win[1])
            acc["coh_mu"].append(_coherence(a, b, sfreq, mu_band))
            acc["coh_beta"].append(_coherence(a, b, sfreq, beta_band))

    # ── Agregace (prumer pres trialy) ─────────────────────────────────
    results: Dict[str, Dict] = {}
    for label, acc in per_class.items():
        def m(key: str) -> float:
            vals = [v for v in acc[key] if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        c3_mu, c4_mu, cz_mu = m("C3_mu"), m("C4_mu"), m("Cz_mu")
        c3_beta, c4_beta = m("C3_beta"), m("C4_beta")

        # Lateralizacni index (mu): rozdil ERD mezi hemisferami.
        # Pro pravou ruku ocekavame silnejsi ERD (zapornejsi) na C3.
        lateralization = c3_mu - c4_mu  # <0 => silnejsi utlum vlevo (prava ruka)

        results[label] = {
            "n_trials": len(acc["C3_mu"]) or len(acc["Cz_mu"]),
            "C3": {"mu_erd": c3_mu, "beta_erd": c3_beta},
            "C4": {"mu_erd": c4_mu, "beta_erd": c4_beta},
            "Cz": {"mu_erd": cz_mu, "beta_erd": m("Cz_beta")},
            "coherence_mu": m("coh_mu"),
            "coherence_beta": m("coh_beta"),
            "lateralization_mu": lateralization,
        }

    report_text = _build_report(results, motor_names, mu_band, beta_band)
    figure_path = _build_figure(results, motor_names, file_path)

    return {
        "classes": results,
        "report_text": report_text,
        "figure_path": str(figure_path) if figure_path else None,
        "channels": motor_names,
    }


def _build_report(results: Dict, motor_names: Dict,
                  mu_band, beta_band) -> str:
    """Sestavi citelny textovy souhrn s interpretaci."""
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("ERD / KOHERENCE - ANALYZA MOTORICKE IMAGINACE")
    lines.append("=" * 64)
    lines.append(f"Kanaly: C3={motor_names['C3']}  C4={motor_names['C4']}  Cz={motor_names['Cz']}")
    lines.append(f"Pasma: mu={mu_band[0]:.0f}-{mu_band[1]:.0f} Hz, "
                 f"beta={beta_band[0]:.0f}-{beta_band[1]:.0f} Hz")
    lines.append("ERD% < 0 = utlum (desynchronizace) = ocekavany jev pri imaginaci")
    lines.append("")

    # Ocekavana lateralizace podle tridy
    expected = {
        "RIGHT_HAND": "C3 (leva hemisfera) - kontralateralni",
        "LEFT_HAND": "C4 (prava hemisfera) - kontralateralni",
        "BOTH_FEET": "Cz (centralne)",
        "TONGUE": "Cz / lateralne",
    }

    for label, r in results.items():
        lines.append("-" * 64)
        lines.append(f"TRIDA: {label}   (n={r['n_trials']} trialu)")
        lines.append(f"  Ocekavany utlum: {expected.get(label, '?')}")
        lines.append(f"  mu ERD%   : C3={r['C3']['mu_erd']:+6.1f}   "
                     f"C4={r['C4']['mu_erd']:+6.1f}   Cz={r['Cz']['mu_erd']:+6.1f}")
        lines.append(f"  beta ERD% : C3={r['C3']['beta_erd']:+6.1f}   "
                     f"C4={r['C4']['beta_erd']:+6.1f}")
        lines.append(f"  Koherence C3-C4: mu={r['coherence_mu']:.3f}  "
                     f"beta={r['coherence_beta']:.3f}")

        # Interpretace
        lat = r["lateralization_mu"]
        if label == "RIGHT_HAND":
            ok = r["C3"]["mu_erd"] < r["C4"]["mu_erd"] and r["C3"]["mu_erd"] < 0
            lines.append(f"  -> {'OK' if ok else 'SLABE'}: "
                         f"{'silnejsi utlum vlevo (C3) = spravna lateralizace' if ok else 'ocekavan silnejsi utlum na C3'}")
        elif label == "LEFT_HAND":
            ok = r["C4"]["mu_erd"] < r["C3"]["mu_erd"] and r["C4"]["mu_erd"] < 0
            lines.append(f"  -> {'OK' if ok else 'SLABE'}: "
                         f"{'silnejsi utlum vpravo (C4) = spravna lateralizace' if ok else 'ocekavan silnejsi utlum na C4'}")
        elif label == "BOTH_FEET":
            ok = r["Cz"]["mu_erd"] < 0
            lines.append(f"  -> {'OK' if ok else 'SLABE'}: "
                         f"{'utlum centralne (Cz)' if ok else 'ocekavan utlum na Cz'}")

    lines.append("=" * 64)
    return "\n".join(lines)


def _build_figure(results: Dict, motor_names: Dict, file_path: Path) -> Optional[Path]:
    """Vykresli graf ERD% per trida/kanal a ulozi PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib nedostupny, graf se neulozi: %s", e)
        return None

    if not results:
        return None

    labels = list(results.keys())
    x = np.arange(len(labels))
    width = 0.25

    c3 = [results[l]["C3"]["mu_erd"] for l in labels]
    c4 = [results[l]["C4"]["mu_erd"] for l in labels]
    cz = [results[l]["Cz"]["mu_erd"] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Graf 1: mu ERD per kanal
    ax1.bar(x - width, c3, width, label=f"C3 ({motor_names['C3']})", color="#FF6B6B")
    ax1.bar(x, c4, width, label=f"C4 ({motor_names['C4']})", color="#4DA3FF")
    ax1.bar(x + width, cz, width, label=f"Cz ({motor_names['Cz']})", color="#9B9B9B")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("mu ERD %  (zaporne = utlum)")
    ax1.set_title("ERD v mu pasmu (8-12 Hz) per trida")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Graf 2: koherence C3-C4
    coh_mu = [results[l]["coherence_mu"] for l in labels]
    coh_beta = [results[l]["coherence_beta"] for l in labels]
    ax2.bar(x - width / 2, coh_mu, width, label="mu", color="#6BCB77")
    ax2.bar(x + width / 2, coh_beta, width, label="beta", color="#FFB86C")
    ax2.set_ylabel("Koherence C3-C4")
    ax2.set_title("Koherence mezi hemisferami")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"ERD / koherence - {file_path.name}", fontsize=13)
    fig.tight_layout()

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"erd_{file_path.stem}.png"
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    logger.info("ERD graf ulozen: %s", out_path)
    return out_path
