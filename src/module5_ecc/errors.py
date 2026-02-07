# file: src/module5_ecc/errors.py

"""
ECC-specific exception hierarchy.

All exceptions inherit from ECCError for unified handling.
"""


class ECCError(Exception):
    """Base exception for all ECC-related errors."""
    pass


class ECCEncodingError(ECCError):
    """Raised when encoding fails."""
    pass


class ECCDecodingError(ECCError):
    """Raised when decoding fails."""
    pass


class ECCCorrectionError(ECCDecodingError):
    """Raised when error correction capability is exceeded."""
    
    def __init__(self, message: str, num_errors: int = None, max_correctable: int = None):
        super().__init__(message)
        self.num_errors = num_errors
        self.max_correctable = max_correctable


class ECCConfigurationError(ECCError):
    """Raised when ECC configuration is invalid."""
    pass