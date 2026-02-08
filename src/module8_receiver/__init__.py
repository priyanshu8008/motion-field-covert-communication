"""
Module 8: Receiver / Extraction Engine

This module implements the receiver pipeline for the motion-field covert
communication system. It extracts embedded data from stego videos by:
1. Recomputing optical flow from stego frames
2. Demodulating bits using QIM (Quantization Index Modulation)
3. Aggregating extracted bits into a raw bitstream

This module does NOT:
- Perform ECC decoding (handled by Module 4)
- Perform cryptographic decryption (handled by Module 3)
- Handle video I/O (handled by Module 1)

All operations are deterministic and follow exact inverse of the encoder.
"""

from .receiver import ReceiverEngine
from .flow_recompute import FlowRecomputeWrapper
from .capacity import CapacityEstimator
from .region_selection import RegionSelector
from .qim_demod import QIMDemodulator
from .bitstream import BitstreamAggregator

__all__ = [
    'ReceiverEngine',
    'FlowRecomputeWrapper',
    'CapacityEstimator',
    'RegionSelector',
    'QIMDemodulator',
    'BitstreamAggregator',
]

__version__ = '1.0.0'