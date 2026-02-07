"""
Temporal filtering functions for signal smoothing.
"""

import numpy as np
from .exceptions import TemporalEncodingError
from .encoding_utils import check_signal_validity


def apply_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Apply moving average filter to temporal signal.
    
    Args:
        signal: 1D temporal signal of shape (T,)
        window: Window size for moving average (must be odd and >= 1)
        
    Returns:
        Filtered signal of shape (T,)
        
    Raises:
        TemporalEncodingError: If window is invalid
    """
    if window < 1:
        raise TemporalEncodingError(
            f"Moving average window must be >= 1, got {window}"
        )
    
    if window == 1:
        return signal.copy()
    
    T = len(signal)
    filtered = np.zeros(T, dtype=np.float32)
    
    # Use reflection padding for boundary handling
    half_window = window // 2
    
    for i in range(T):
        # Determine the window bounds
        start = max(0, i - half_window)
        end = min(T, i + half_window + 1)
        
        # Compute mean over the window
        filtered[i] = np.mean(signal[start:end])
    
    check_signal_validity(filtered, "moving_average filtered signal")
    return filtered


def apply_exponential_smoothing(signal: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply exponential smoothing to temporal signal.
    
    Args:
        signal: 1D temporal signal of shape (T,)
        alpha: Smoothing factor in [0, 1]
               alpha=1 means no smoothing (original signal)
               alpha=0 means maximum smoothing (constant)
        
    Returns:
        Smoothed signal of shape (T,)
        
    Raises:
        TemporalEncodingError: If alpha is out of range
    """
    if not (0.0 <= alpha <= 1.0):
        raise TemporalEncodingError(
            f"Alpha must be in [0, 1], got {alpha}"
        )
    
    if alpha == 1.0:
        return signal.copy()
    
    T = len(signal)
    smoothed = np.zeros(T, dtype=np.float32)
    
    # Initialize with first value
    smoothed[0] = signal[0]
    
    # Apply exponential smoothing: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
    for t in range(1, T):
        smoothed[t] = alpha * signal[t] + (1.0 - alpha) * smoothed[t - 1]
    
    check_signal_validity(smoothed, "exponentially smoothed signal")
    return smoothed


def apply_filter(signal: np.ndarray, filter_type: str, filter_params: dict) -> np.ndarray:
    """
    Apply temporal filter to signal.
    
    Args:
        signal: 1D temporal signal
        filter_type: Type of filter ('moving_average' or 'exponential_smoothing')
        filter_params: Parameters for the filter
        
    Returns:
        Filtered signal
        
    Raises:
        TemporalEncodingError: If filter type is unsupported
    """
    if filter_type == "moving_average":
        window = filter_params.get("window", 5)
        return apply_moving_average(signal, window)
    
    elif filter_type == "exponential_smoothing":
        alpha = filter_params.get("alpha", 0.3)
        return apply_exponential_smoothing(signal, alpha)
    
    else:
        raise TemporalEncodingError(
            f"Unsupported filter type: {filter_type}. "
            f"Supported: 'moving_average', 'exponential_smoothing'"
        )