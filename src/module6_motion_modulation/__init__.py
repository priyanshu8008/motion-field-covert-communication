"""
Motion-Field Modulation Engine (Architectural Module 5).

This module implements Quantization Index Modulation (QIM) for embedding
encrypted data into optical flow fields.

Public API:
    - MotionFieldModulator: Main interface class
    - EmbeddingMetadata: Result metadata structure
"""

from .modulator import MotionFieldModulator, EmbeddingMetadata

__all__ = [
    'MotionFieldModulator',
    'EmbeddingMetadata'
]

__version__ = '1.0.0'