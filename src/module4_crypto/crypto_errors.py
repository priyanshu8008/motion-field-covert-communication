# file: module4_crypto/crypto_errors.py
"""
Cryptographic error types for Module 4.
"""


class CryptoError(Exception):
    """Base exception for Module 4 cryptographic operations."""
    pass


class UnsupportedVersionError(CryptoError):
    """Raised when frame version is not supported."""
    pass


class MalformedFrameError(CryptoError):
    """Raised when frame structure is invalid."""
    pass


class TruncatedFrameError(CryptoError):
    """Raised when frame data is incomplete."""
    pass


class AuthenticationFailureError(CryptoError):
    """Raised when authentication tag verification fails."""
    pass