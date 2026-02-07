# file: module4_crypto/module4.py
# ============================================
# MODULE 4 — FROZEN
# Deterministic cryptographic pipeline
# DO NOT MODIFY WITHOUT FULL SYSTEM REVIEW
# ============================================

"""
Module 4: Cryptographic Pipeline

Main encryption and decryption functions for temporal signal protection.
"""

import struct
from cryptography.exceptions import InvalidTag
from .crypto_errors import AuthenticationFailureError
from .deterministic import derive_deterministic_params
from .kdf import derive_key
from .aead import encrypt_aead, decrypt_aead
from .framing import assemble_frame, parse_frame, VERSION


def encrypt(temporal_signal: bytes, password: str, config) -> bytes:
    """
    Encrypt temporal signal with password using deterministic AEAD.
    
    Args:
        temporal_signal: Output from Module 3 (opaque bitstream)
        password: User passphrase (UTF-8 string)
        config: Cryptographic configuration object
    
    Returns:
        Encrypted frame (49 + len(temporal_signal) bytes)
    
    Raises:
        ValueError: If password is too short or config is invalid
    """
    # Validate password length
    min_length = config.crypto.security.min_password_length
    if len(password) < min_length:
        raise ValueError(f"Password must be at least {min_length} characters")
    
    # Derive deterministic salt and nonce from config seed
    salt, nonce = derive_deterministic_params(config.system.random_seed)
    
    # Derive encryption key via Argon2id
    key = derive_key(password, salt, config)
    
    # Prepare AAD (bind version and payload length)
    payload_length = len(temporal_signal)
    aad = bytes([VERSION]) + struct.pack('>I', payload_length)
    
    # Encrypt with ChaCha20-Poly1305
    ciphertext, auth_tag = encrypt_aead(key, nonce, temporal_signal, aad)
    
    # Assemble frame
    encrypted_frame = assemble_frame(salt, nonce, ciphertext, auth_tag)
    
    return encrypted_frame


def decrypt(encrypted_frame: bytes, password: str, config) -> bytes:
    """
    Decrypt and authenticate encrypted frame.
    
    Args:
        encrypted_frame: Output from Module 5 ECC decode (or Module 4 encrypt)
        password: User passphrase (must match encryption password)
        config: Cryptographic configuration (must match encryption config)
    
    Returns:
        Plaintext temporal signal (original bitstream from Module 3)
    
    Raises:
        UnsupportedVersionError: If version != 0x01
        MalformedFrameError: If frame structure is invalid
        TruncatedFrameError: If frame is incomplete
        AuthenticationFailureError: If tag verification fails
    """
    # Parse frame (validates structure and version)
    salt, nonce, ciphertext, auth_tag = parse_frame(encrypted_frame)
    
    # Derive encryption key (same as encryption)
    key = derive_key(password, salt, config)
    
    # Prepare AAD (same as encryption)
    payload_length = len(ciphertext)
    aad = bytes([VERSION]) + struct.pack('>I', payload_length)
    
    # Decrypt and verify with ChaCha20-Poly1305
    try:
        plaintext = decrypt_aead(key, nonce, ciphertext, auth_tag, aad)
    except InvalidTag:
        # Authentication failed: could be corruption, wrong password, or tampering
        raise AuthenticationFailureError(
            "Authentication tag verification failed"
        )
    
    return plaintext