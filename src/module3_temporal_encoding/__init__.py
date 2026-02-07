"""
Module 3: Temporal Signal Encoding

This module converts optical flow sequences from Module 2 into 1D temporal signals.

Public Interface:
    - TemporalSignalEncoder: Main encoder class
    - TemporalEncodingError: Custom exception class

Example usage:
    >>> from module3_temporal_encoding import TemporalSignalEncoder
    >>> encoder = TemporalSignalEncoder()
    >>> result = encoder.encode(
    ...     flow_sequence,
    ...     method="magnitude_mean",
    ...     filter="moving_average",
    ...     quantize=True
    ... )
    >>> signal = result["signal"]
    >>> quantized = result["quantized_signal"]
    >>> metadata = result["metadata"]
"""

from .signal_encoder import TemporalSignalEncoder
from .exceptions import TemporalEncodingError

__all__ = [
    "TemporalSignalEncoder",
    "TemporalEncodingError"
]

__version__ = "1.0.0"