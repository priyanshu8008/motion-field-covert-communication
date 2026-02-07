"""
Utility functions for temporal signal encoding.
"""

import numpy as np
from typing import Tuple
from .exceptions import TemporalEncodingError


def validate_flow_input(flow_sequence: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Validate and normalize input flow sequence.
    
    Args:
        flow_sequence: Input flow array
        
    Returns:
        Tuple of (normalized_flow, num_frames)
        
    Raises:
        TemporalEncodingError: If input is invalid
    """
    if not isinstance(flow_sequence, np.ndarray):
        raise TemporalEncodingError(
            f"Input must be numpy.ndarray, got {type(flow_sequence).__name__}"
        )
    
    if flow_sequence.size == 0:
        raise TemporalEncodingError("Input flow sequence is empty")
    
    if flow_sequence.dtype != np.float32:
        raise TemporalEncodingError(
            f"Input dtype must be float32, got {flow_sequence.dtype}"
        )
    
    # Normalize shape to (T, H, W, 2)
    if flow_sequence.ndim == 3:
        # Single frame: (H, W, 2) -> (1, H, W, 2)
        if flow_sequence.shape[-1] != 2:
            raise TemporalEncodingError(
                f"Invalid shape {flow_sequence.shape}. Expected (H, W, 2) or (T, H, W, 2)"
            )
        flow_sequence = flow_sequence[np.newaxis, ...]
        num_frames = 1
    elif flow_sequence.ndim == 4:
        # Temporal sequence: (T, H, W, 2)
        if flow_sequence.shape[-1] != 2:
            raise TemporalEncodingError(
                f"Invalid shape {flow_sequence.shape}. Last dimension must be 2 for (dx, dy)"
            )
        num_frames = flow_sequence.shape[0]
    else:
        raise TemporalEncodingError(
            f"Invalid shape {flow_sequence.shape}. Expected (H, W, 2) or (T, H, W, 2)"
        )
    
    # Check for NaN or Inf in input
    if not np.isfinite(flow_sequence).all():
        raise TemporalEncodingError("Input contains NaN or Inf values")
    
    return flow_sequence, num_frames


def compute_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of optical flow vectors.
    
    Args:
        flow: Flow array of shape (..., 2) where last dim is (dx, dy)
        
    Returns:
        Magnitude array with shape (...,)
    """
    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    return magnitude


def compute_flow_direction(flow: np.ndarray) -> np.ndarray:
    """
    Compute direction (angle) of optical flow vectors.
    
    Args:
        flow: Flow array of shape (..., 2) where last dim is (dx, dy)
        
    Returns:
        Direction array in radians with shape (...,)
    """
    dx = flow[..., 0]
    dy = flow[..., 1]
    direction = np.arctan2(dy, dx)
    return direction


def check_signal_validity(signal: np.ndarray, name: str = "signal") -> None:
    """
    Check if a signal contains NaN or Inf values.
    
    Args:
        signal: Signal array to check
        name: Name of the signal for error message
        
    Raises:
        TemporalEncodingError: If signal contains NaN or Inf
    """
    if not np.isfinite(signal).all():
        raise TemporalEncodingError(
            f"Generated {name} contains NaN or Inf values"
        )


def compute_signal_statistics(signal: np.ndarray) -> dict:
    """
    Compute summary statistics for a signal.
    
    Args:
        signal: 1D signal array
        
    Returns:
        Dictionary of statistics
    """
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "median": float(np.median(signal)),
        "length": int(len(signal))
    }