from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop

from .config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_LSL_TIMEOUT = 10.0
DEFAULT_MAX_CHUNK = 512


@dataclass
class LSLStreams:
    """Holds EEG inlet and marker outlet objects."""
    eeg_inlet: StreamInlet
    marker_outlet: StreamOutlet


def _get_lsl_timeout() -> float:
    """Get LSL resolution timeout from environment or default."""
    env_timeout = os.getenv("EEG_LSL_TIMEOUT")
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError:
            logger.warning(f"Invalid EEG_LSL_TIMEOUT: {env_timeout}, using default")
    return DEFAULT_LSL_TIMEOUT


def resolve_eeg_stream(config: AppConfig, timeout: Optional[float] = None) -> StreamInlet:
    """Find and connect to EEG LSL stream.
    
    Args:
        config: Application configuration
        timeout: Resolution timeout in seconds (default: 10s or EEG_LSL_TIMEOUT env var)
    
    Returns:
        StreamInlet connected to EEG stream
    
    Raises:
        RuntimeError: If no EEG stream found within timeout
    """
    if timeout is None:
        timeout = _get_lsl_timeout()
    
    eeg_name = config.lsl.eeg_stream_name
    eeg_type = config.lsl.eeg_stream_type

    logger.info(f"Searching for LSL stream (type={eeg_type}, name={eeg_name}, timeout={timeout}s)...")

    # Pokus 1: hledat podle typu (napr. "EEG")
    streams = []
    try:
        streams = resolve_byprop("type", eeg_type, timeout=timeout)
    except Exception as e:
        logger.warning(f"LSL resolve by type failed: {e}")

    # Pokus 2: pokud nenasel nic a mame konkretni jmeno, hledat podle jmena
    if not streams and eeg_name and eeg_name != eeg_type:
        logger.info(f"Type search empty, trying by name='{eeg_name}'...")
        try:
            streams = resolve_byprop("name", eeg_name, timeout=timeout)
        except Exception as e:
            logger.warning(f"LSL resolve by name failed: {e}")

    # Pokus 3: najit jakykoli stream (posledni moznost)
    if not streams:
        logger.info("Name search empty, trying any stream...")
        try:
            from pylsl import resolve_streams
            streams = resolve_streams(wait_time=min(timeout, 3.0))
        except Exception:
            pass

    if not streams:
        raise RuntimeError(
            f"No LSL stream found (type='{eeg_type}', name='{eeg_name}') within {timeout}s.\n"
            "Steps to fix:\n"
            "  1. Make sure LSL software is running on the EEG computer\n"
            "  2. Use Network EEG tab -> Scan -> select your stream -> 'Use selected'\n"
            "  3. Then start recording"
        )

    # Vybrat spravny stream: prednostne podle jmena, pak prvni nalezeny
    info = None
    if eeg_name:
        named = [s for s in streams if s.name() == eeg_name]
        if named:
            info = named[0]
            logger.info(f"Connected to stream by name: '{eeg_name}'")
    if info is None:
        # Prednostne EEG typ
        eeg_typed = [s for s in streams if s.type().upper() == "EEG"]
        info = eeg_typed[0] if eeg_typed else streams[0]
        logger.info(f"Connected to stream: '{info.name()}' (type={info.type()})")

    inlet = StreamInlet(info, max_chunklen=0)
    logger.info(f"EEG stream ready: {info.channel_count()} channels @ {info.nominal_srate()} Hz  host={info.hostname()}")
    return inlet


def create_marker_outlet(config: AppConfig) -> StreamOutlet:
    """Create LSL marker outlet for experimental events."""
    marker_name = config.lsl.marker_stream_name
    
    info = StreamInfo(
        name=marker_name,
        type="Markers",
        channel_count=1,
        nominal_srate=0.0,
        channel_format="string",
        source_id="eeg_app_markers",
    )
    outlet = StreamOutlet(info)
    logger.info(f"Marker outlet '{marker_name}' created")
    return outlet


def create_streams(config: AppConfig) -> LSLStreams:
    """Create EEG inlet and marker outlet based on configuration."""
    eeg_inlet = resolve_eeg_stream(config)
    marker_outlet = create_marker_outlet(config)
    return LSLStreams(eeg_inlet=eeg_inlet, marker_outlet=marker_outlet)


def pull_chunk(
    eeg_inlet: StreamInlet, max_samples: int = DEFAULT_MAX_CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """Read data chunk from EEG inlet.
    
    Returns:
        (data, timestamps) where data has shape (n_samples, n_channels)
    """
    try:
        chunk, timestamps = eeg_inlet.pull_chunk(max_samples=max_samples)
    except Exception as e:
        logger.error(f"Failed to pull chunk: {e}")
        return np.empty((0, 0)), np.empty((0,))
    
    if not timestamps:
        return np.empty((0, 0)), np.empty((0,))
    
    data = np.asarray(chunk, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    return data, ts


def push_marker(marker_outlet: StreamOutlet, code: str) -> None:
    """Send marker event via LSL outlet."""
    try:
        marker_outlet.push_sample([code])
        logger.debug(f"Marker sent: {code}")
    except Exception as e:
        logger.error(f"Failed to send marker: {e}")
