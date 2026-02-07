# file: src/module5_ecc/__init__.py

"""
Module 5: Error Correction Coding (ECC)

Provides deterministic forward error correction for encrypted bitstreams.
Operates on opaque bytes between cryptographic pipeline (Module 4) and
motion-field modulation (Module 6).

Public API:
    - ecc_encode(data: bytes, config) -> bytes
    - ecc_decode(data: bytes, config) -> bytes
    - compute_ber(original: bytes, received: bytes) -> float
    - compute_ser(original: bytes, received: bytes, symbol_size: int) -> float
"""

from .encoder import ecc_encode
from .decoder import ecc_decode
from .metrics import compute_ber, compute_ser, compute_redundancy_overhead
from .errors import (
    ECCError,
    ECCEncodingError,
    ECCDecodingError,
    ECCCorrectionError,
    ECCConfigurationError,
)

__version__ = "1.0.0"

__all__ = [
    "ecc_encode",
    "ecc_decode",
    "compute_ber",
    "compute_ser",
    "compute_redundancy_overhead",
    "ECCError",
    "ECCEncodingError",
    "ECCDecodingError",
    "ECCCorrectionError",
    "ECCConfigurationError",
]