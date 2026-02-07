# file: module4_crypto/__init__.py
"""
Module 4: Cryptographic Pipeline

Provides authenticated encryption for temporal signal bitstreams.
"""

from .module4 import encrypt, decrypt
from .crypto_errors import (
    CryptoError,
    UnsupportedVersionError,
    MalformedFrameError,
    TruncatedFrameError,
    AuthenticationFailureError
)


__all__ = [
    'encrypt',
    'decrypt',
    'CryptoError',
    'UnsupportedVersionError',
    'MalformedFrameError',
    'TruncatedFrameError',
    'AuthenticationFailureError',
]


__version__ = '1.0.0'