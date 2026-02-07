# file: module4_crypto/framing.py
"""
Frame assembly and parsing for encrypted bitstreams.
"""

import struct
from typing import Tuple
from .crypto_errors import MalformedFrameError, TruncatedFrameError, UnsupportedVersionError


VERSION = 0x01
HEADER_SIZE = 33
TAG_SIZE = 16
SALT_SIZE = 16
NONCE_SIZE = 12


def assemble_frame(
    salt: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes
) -> bytes:
    """
    Assemble encrypted frame from components.
    
    Frame structure (49 + N bytes):
        [version:1][salt:16][nonce:12][length:4][ciphertext:N][tag:16]
    
    Args:
        salt: 16-byte Argon2id salt
        nonce: 12-byte ChaCha20 nonce
        ciphertext: Encrypted payload (variable length)
        tag: 16-byte Poly1305 authentication tag
    
    Returns:
        Complete frame ready for ECC encoding
    """
    payload_length = len(ciphertext)
    
    frame = (
        bytes([VERSION]) +
        salt +
        nonce +
        struct.pack('>I', payload_length) +  # Big-endian uint32
        ciphertext +
        tag
    )
    
    return frame


def parse_frame(frame_bytes: bytes) -> Tuple[bytes, bytes, bytes, bytes]:
    """
    Parse encrypted frame into components.
    
    Args:
        frame_bytes: Complete encrypted frame
    
    Returns:
        Tuple of (salt, nonce, ciphertext, tag)
    
    Raises:
        TruncatedFrameError: If frame is too short
        UnsupportedVersionError: If version != 0x01
        MalformedFrameError: If frame structure is invalid
    """
    # Validate minimum length
    if len(frame_bytes) < HEADER_SIZE + TAG_SIZE:
        raise TruncatedFrameError(
            f"Frame too short: {len(frame_bytes)} bytes (minimum {HEADER_SIZE + TAG_SIZE})"
        )
    
    # Parse header
    version = frame_bytes[0]
    if version != VERSION:
        raise UnsupportedVersionError(f"Unsupported version: 0x{version:02x}")
    
    salt = frame_bytes[1:17]
    nonce = frame_bytes[17:29]
    payload_length = struct.unpack('>I', frame_bytes[29:33])[0]
    
    # Validate total length
    expected_length = HEADER_SIZE + payload_length + TAG_SIZE
    if len(frame_bytes) != expected_length:
        raise MalformedFrameError(
            f"Length mismatch: got {len(frame_bytes)} bytes, expected {expected_length}"
        )
    
    # Extract ciphertext and tag
    ciphertext_start = HEADER_SIZE
    ciphertext_end = HEADER_SIZE + payload_length
    
    ciphertext = frame_bytes[ciphertext_start:ciphertext_end]
    tag = frame_bytes[ciphertext_end:ciphertext_end + TAG_SIZE]
    
    return salt, nonce, ciphertext, tag