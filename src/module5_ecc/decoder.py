# file: src/module5_ecc/decoder.py

"""
ECC decoding entry point.

Provides ecc_decode() function with explicit error correction and failure handling.
"""

from typing import Any, Dict
from .rs_codec import ReedSolomonCodec
from .errors import ECCDecodingError, ECCCorrectionError, ECCConfigurationError


def ecc_decode(data: bytes, config: Dict[str, Any]) -> bytes:
    """
    Decode ECC-protected data with error correction.
    
    This function attempts to correct errors introduced by the channel
    (compression, transmission noise) and recover the original encrypted payload.
    
    Args:
        data: ECC-protected byte stream from transmission
        config: Configuration dictionary with 'ecc' section
    
    Returns:
        Recovered encrypted byte stream for Module 4 decryption
    
    Raises:
        ECCDecodingError: If decoding fails due to format errors
        ECCCorrectionError: If errors exceed correction capability
        ECCConfigurationError: If configuration is invalid
    
    Configuration Schema:
        config['ecc']['type']: 'reed_solomon' (required)
        config['ecc']['reed_solomon']['n']: Total codeword length (default: 255)
        config['ecc']['reed_solomon']['k']: Message length (default: 223)
        config['ecc']['reed_solomon']['nsym']: Parity symbols (default: 32)
    
    Error Handling:
        - If errors <= nsym/2: Successful correction, returns original data
        - If errors > nsym/2: Raises ECCCorrectionError with diagnostic info
        - Never returns silently corrupted data
    
    Example:
        >>> try:
        ...     decrypted = ecc_decode(received_data, config)
        ... except ECCCorrectionError as e:
        ...     print(f"Too many errors: {e.num_errors} > {e.max_correctable}")
    """
    if not isinstance(data, bytes):
        raise ECCDecodingError(f"Input must be bytes, got {type(data)}")
    
    # Extract ECC configuration
    try:
        ecc_config = config['ecc']
        ecc_type = ecc_config['type']
    except KeyError as e:
        raise ECCConfigurationError(f"Missing required config key: {e}") from e
    
    # Dispatch to appropriate codec
    if ecc_type == 'reed_solomon':
        return _decode_reed_solomon(data, config)
    elif ecc_type == 'ldpc':
        raise ECCConfigurationError("LDPC codec not yet implemented (extensible)")
    else:
        raise ECCConfigurationError(f"Unknown ECC type: {ecc_type}")


def _decode_reed_solomon(data: bytes, config: Dict[str, Any]) -> bytes:
    """
    Decode using Reed-Solomon codec.
    
    Args:
        data: RS-encoded bytes
        config: Configuration dictionary
    
    Returns:
        Decoded original bytes
    
    Raises:
        ECCDecodingError: Format/structure errors
        ECCCorrectionError: Too many errors to correct
    """
    rs_config = config['ecc']['reed_solomon']
    
    # Extract parameters with defaults
    n = rs_config.get('n', 255)
    k = rs_config.get('k', 223)
    nsym = rs_config.get('nsym', 32)
    
    # Create codec and decode
    codec = ReedSolomonCodec(n=n, k=k, nsym=nsym)
    decoded_data = codec.decode(data)
    
    return decoded_data