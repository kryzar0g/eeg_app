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
    
    logger.info(f"Searching for LSL stream (type={eeg_type}, timeout={timeout}s)...")
    
    try:
        streams = resolve_byprop("type", eeg_type, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"LSL stream resolution failed: {e}") from e
    
    if not streams:
        raise RuntimeError(
            f"No LSL stream found with type '{eeg_type}' within {timeout}s. "
            "Ensure LabRecorder or another LSL provider is running."
        )
    
    # Try to match by name
    if eeg_name:
        named = [s for s in streams if s.name() == eeg_name]
        if named:
            info = named[0]
            logger.info(f"Connected to '{eeg_name}' stream")
        else:
            info = streams[0]
            logger.warning(
                f"Stream '{eeg_name}' not found. Using first available: '{info.name()}'"
            )
    else:
        info = streams[0]
        logger.info(f"Connected to stream: '{info.name()}'")
    
    inlet = StreamInlet(info, max_chunklen=0)
    logger.debug(f"EEG stream: {info.channel_count()} channels @ {info.nominal_srate()} Hz")
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
