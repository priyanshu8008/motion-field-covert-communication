# file: module4_crypto/deterministic.py
"""
Deterministic utilities for salt and nonce derivation using HKDF.
"""

import hashlib
import hmac
from typing import Tuple


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """
    HKDF-Extract step using SHA-256.
    
    Args:
        salt: Salt value (key for HMAC)
        ikm: Input keying material (message for HMAC)
    
    Returns:
        Pseudorandom key (PRK) of length 32 bytes
    """
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    HKDF-Expand step using SHA-256.
    
    Args:
        prk: Pseudorandom key from HKDF-Extract
        info: Context and application specific information
        length: Length of output keying material in bytes
    
    Returns:
        Output keying material of specified length
    """
    hash_len = 32  # SHA-256 output length
    n = (length + hash_len - 1) // hash_len
    
    okm = b""
    previous = b""
    
    for i in range(1, n + 1):
        previous = hmac.new(
            prk,
            previous + info + bytes([i]),
            hashlib.sha256
        ).digest()
        okm += previous
    
    return okm[:length]


def derive_deterministic_params(random_seed: int) -> Tuple[bytes, bytes]:
    """
    Derive deterministic salt and nonce from system random seed.
    
    Args:
        random_seed: System random seed from config
    
    Returns:
        Tuple of (salt, nonce) where:
        - salt: 16 bytes for Argon2id
        - nonce: 12 bytes for ChaCha20
    """
    seed_bytes = random_seed.to_bytes(8, 'big')
    
    # Domain separation via fixed salt
    domain_salt = b"motion-covert-comm-v1"
    
    # Extract PRK from seed
    prk = hkdf_extract(salt=domain_salt, ikm=seed_bytes)
    
    # Expand into salt and nonce with different info strings
    salt = hkdf_expand(prk, info=b"argon2-salt", length=16)
    nonce = hkdf_expand(prk, info=b"chacha20-nonce", length=12)
    
    return salt, nonce