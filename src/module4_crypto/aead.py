# file: module4_crypto/aead.py
"""
Authenticated encryption using ChaCha20-Poly1305.
"""

from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


def encrypt_aead(
    key: bytes,
    nonce: bytes,
    plaintext: bytes,
    aad: bytes
) -> Tuple[bytes, bytes]:
    """
    Encrypt and authenticate data using ChaCha20-Poly1305.
    
    Args:
        key: 32-byte encryption key
        nonce: 12-byte nonce (must be unique per key)
        plaintext: Data to encrypt
        aad: Additional authenticated data (not encrypted)
    
    Returns:
        Tuple of (ciphertext, auth_tag) where:
        - ciphertext: Encrypted plaintext (same length as plaintext)
        - auth_tag: 16-byte Poly1305 authentication tag
    """
    cipher = ChaCha20Poly1305(key)
    
    # ChaCha20Poly1305.encrypt returns ciphertext || tag
    ciphertext_with_tag = cipher.encrypt(nonce, plaintext, aad)
    
    # Split into ciphertext and tag
    ciphertext = ciphertext_with_tag[:-16]
    auth_tag = ciphertext_with_tag[-16:]
    
    return ciphertext, auth_tag


def decrypt_aead(
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    aad: bytes
) -> bytes:
    """
    Decrypt and verify authenticated data using ChaCha20-Poly1305.
    
    Args:
        key: 32-byte encryption key
        nonce: 12-byte nonce (same as encryption)
        ciphertext: Encrypted data
        tag: 16-byte authentication tag
        aad: Additional authenticated data (same as encryption)
    
    Returns:
        Decrypted plaintext
    
    Raises:
        cryptography.exceptions.InvalidTag: If authentication fails
    """
    cipher = ChaCha20Poly1305(key)
    
    # ChaCha20Poly1305.decrypt expects ciphertext || tag
    ciphertext_with_tag = ciphertext + tag
    
    # Will raise InvalidTag if verification fails
    plaintext = cipher.decrypt(nonce, ciphertext_with_tag, aad)
    
    return plaintext