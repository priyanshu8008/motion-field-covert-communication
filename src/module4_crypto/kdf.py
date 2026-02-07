# file: module4_crypto/kdf.py
"""
Key derivation using Argon2id.
"""

from argon2 import PasswordHasher
from argon2.low_level import Type, hash_secret_raw


def derive_key(password: str, salt: bytes, config) -> bytes:
    """
    Derive 256-bit encryption key from password using Argon2id.
    
    Args:
        password: User-provided passphrase
        salt: 16-byte salt (deterministic or random)
        config: Configuration object with crypto.kdf parameters
    
    Returns:
        32-byte (256-bit) encryption key
    """
    password_bytes = password.encode('utf-8')
    
    key = hash_secret_raw(
        secret=password_bytes,
        salt=salt,
        time_cost=config.crypto.kdf.time_cost,
        memory_cost=config.crypto.kdf.memory_cost,
        parallelism=config.crypto.kdf.parallelism,
        hash_len=config.crypto.kdf.key_length,
        type=Type.ID  # Argon2id
    )
    
    return key