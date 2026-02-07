"""
Signal quantization functions.
"""

import numpy as np
from .exceptions import TemporalEncodingError
from .encoding_utils import check_signal_validity


def quantize_signal(signal: np.ndarray, levels: int) -> np.ndarray:
    """
    Quantize continuous signal to discrete levels using uniform quantization.
    
    Args:
        signal: 1D continuous signal of shape (T,)
        levels: Number of quantization levels (must be >= 2)
        
    Returns:
        Quantized signal with integer values in [0, levels-1]
        dtype: uint8 if levels <= 256, otherwise int32
        
    Raises:
        TemporalEncodingError: If levels is invalid
    """
    if levels < 2:
        raise TemporalEncodingError(
            f"Number of quantization levels must be >= 2, got {levels}"
        )
    
    # Find min and max of signal
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    
    # Handle edge case where signal is constant
    if signal_min == signal_max:
        # All values map to middle level
        middle_level = (levels - 1) // 2
        if levels <= 256:
            return np.full(len(signal), middle_level, dtype=np.uint8)
        else:
            return np.full(len(signal), middle_level, dtype=np.int32)
    
    # Normalize signal to [0, 1]
    normalized = (signal - signal_min) / (signal_max - signal_min)
    
    # Map to [0, levels-1]
    # Use floor to ensure deterministic behavior
    quantized_float = normalized * (levels - 1)
    quantized_int = np.floor(quantized_float).astype(np.int32)
    
    # Clip to valid range (handles potential floating point edge cases)
    quantized_int = np.clip(quantized_int, 0, levels - 1)
    
    # Use uint8 for 256 levels or fewer, int32 otherwise
    if levels <= 256:
        quantized_int = quantized_int.astype(np.uint8)
    
    return quantized_int


def dequantize_signal(quantized: np.ndarray, levels: int, 
                      original_min: float, original_max: float) -> np.ndarray:
    """
    Dequantize discrete signal back to continuous values.
    
    Args:
        quantized: Quantized signal with integer values in [0, levels-1]
        levels: Number of quantization levels used
        original_min: Minimum value of original signal
        original_max: Maximum value of original signal
        
    Returns:
        Dequantized continuous signal
    """
    if original_min == original_max:
        # Signal was constant, return constant array
        return np.full(len(quantized), original_min, dtype=np.float32)
    
    # Convert quantized levels back to [0, 1]
    normalized = quantized.astype(np.float32) / (levels - 1)
    
    # Scale back to original range
    dequantized = normalized * (original_max - original_min) + original_min
    
    check_signal_validity(dequantized, "dequantized signal")
    return dequantized