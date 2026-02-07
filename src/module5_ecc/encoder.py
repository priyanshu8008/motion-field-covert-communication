# file: src/module5_ecc/encoder.py

"""
ECC encoding entry point.

Provides ecc_encode() function that operates on opaque encrypted bytes.
"""

from typing import Any, Dict
from .rs_codec import ReedSolomonCodec
from .errors import ECCEncodingError, ECCConfigurationError


def ecc_encode(data: bytes, config: Dict[str, Any]) -> bytes:
    """
    Encode encrypted data with forward error correction.
    
    This function treats input data as opaque bytes and adds redundancy
    to enable recovery from channel errors (compression, transmission noise).
    
    Args:
        data: Opaque encrypted byte stream from Module 4
        config: Configuration dictionary with 'ecc' section
    
    Returns:
        ECC-protected byte stream ready for Module 6
    
    Raises:
        ECCEncodingError: If encoding fails
        ECCConfigurationError: If configuration is invalid
    
    Configuration Schema:
        config['ecc']['type']: 'reed_solomon' (required)
        config['ecc']['reed_solomon']['n']: Total codeword length (default: 255)
        config['ecc']['reed_solomon']['k']: Message length (default: 223)
        config['ecc']['reed_solomon']['nsym']: Parity symbols (default: 32)
    
    Example:
        >>> config = {
        ...     'ecc': {
        ...         'type': 'reed_solomon',
        ...         'reed_solomon': {'n': 255, 'k': 223, 'nsym': 32}
        ...     }
        ... }
        >>> encoded = ecc_encode(encrypted_data, config)
    """
    if not isinstance(data, bytes):
        raise ECCEncodingError(f"Input must be bytes, got {type(data)}")
    
    # Extract ECC configuration
    try:
        ecc_config = config['ecc']
        ecc_type = ecc_config['type']
    except KeyError as e:
        raise ECCConfigurationError(f"Missing required config key: {e}") from e
    
    # Dispatch to appropriate codec
    if ecc_type == 'reed_solomon':
        return _encode_reed_solomon(data, config)
    elif ecc_type == 'ldpc':
        raise ECCConfigurationError("LDPC codec not yet implemented (extensible)")
    else:
        raise ECCConfigurationError(f"Unknown ECC type: {ecc_type}")


def _encode_reed_solomon(data: bytes, config: Dict[str, Any]) -> bytes:
    """
    Encode using Reed-Solomon codec.
    
    Args:
        data: Input bytes
        config: Configuration dictionary
    
    Returns:
        RS-encoded bytes
    """
    rs_config = config['ecc']['reed_solomon']
    
    # Extract parameters with defaults
    n = rs_config.get('n', 255)
    k = rs_config.get('k', 223)
    nsym = rs_config.get('nsym', 32)
    
    # Create codec and encode
    codec = ReedSolomonCodec(n=n, k=k, nsym=nsym)
    encoded_data = codec.encode(data)
    
    return encoded_data