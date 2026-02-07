# file: src/module5_ecc/metrics.py

"""
ECC performance metrics.

Provides utilities to compute Bit Error Rate (BER), Symbol Error Rate (SER),
and redundancy overhead for evaluating ECC performance.
"""

from typing import Optional


def compute_ber(original: bytes, received: bytes) -> float:
    """
    Compute Bit Error Rate (BER) between two byte sequences.
    
    BER = (number of bit errors) / (total number of bits)
    
    Args:
        original: Original transmitted data
        received: Received (possibly corrupted) data
    
    Returns:
        BER as a float in [0.0, 1.0]
    
    Raises:
        ValueError: If inputs have different lengths
    
    Example:
        >>> original = b'\x00\x00'
        >>> received = b'\x01\x00'  # 1 bit flipped
        >>> ber = compute_ber(original, received)
        >>> assert ber == 1.0 / 16  # 1 error in 16 bits
    """
    if len(original) != len(received):
        raise ValueError(
            f"Length mismatch: original={len(original)}, received={len(received)}"
        )
    
    if len(original) == 0:
        return 0.0
    
    # Count bit differences
    bit_errors = 0
    for b1, b2 in zip(original, received):
        xor = b1 ^ b2
        # Count set bits in XOR result
        bit_errors += bin(xor).count('1')
    
    total_bits = len(original) * 8
    return bit_errors / total_bits


def compute_ser(original: bytes, received: bytes, symbol_size: int = 8) -> float:
    """
    Compute Symbol Error Rate (SER) between two byte sequences.
    
    SER = (number of symbol errors) / (total number of symbols)
    
    A symbol is considered erroneous if any bit within it differs.
    
    Args:
        original: Original transmitted data
        received: Received (possibly corrupted) data
        symbol_size: Number of bits per symbol (default: 8 for byte-level)
    
    Returns:
        SER as a float in [0.0, 1.0]
    
    Raises:
        ValueError: If inputs have different lengths or invalid symbol_size
    
    Example:
        >>> original = b'\x00\x00\x00\x00'
        >>> received = b'\x01\x00\x02\x00'  # 2 symbols corrupted
        >>> ser = compute_ser(original, received, symbol_size=8)
        >>> assert ser == 2.0 / 4  # 2 errors in 4 symbols
    """
    if len(original) != len(received):
        raise ValueError(
            f"Length mismatch: original={len(original)}, received={len(received)}"
        )
    
    if symbol_size not in [1, 2, 4, 8, 16, 32]:
        raise ValueError(f"symbol_size must be power of 2 <= 32, got {symbol_size}")
    
    if len(original) == 0:
        return 0.0
    
    # Convert to bit arrays
    bits_per_byte = 8
    if symbol_size > bits_per_byte:
        raise ValueError(f"symbol_size {symbol_size} > 8 not supported for byte inputs")
    
    if symbol_size == 8:
        # Fast path: byte-level symbols
        symbol_errors = sum(1 for b1, b2 in zip(original, received) if b1 != b2)
        total_symbols = len(original)
    else:
        # General case: sub-byte symbols
        symbol_errors = 0
        total_bits = len(original) * bits_per_byte
        total_symbols = total_bits // symbol_size
        
        for byte_idx in range(len(original)):
            b1 = original[byte_idx]
            b2 = received[byte_idx]
            
            # Extract symbols from byte
            for shift in range(0, bits_per_byte, symbol_size):
                mask = (1 << symbol_size) - 1
                s1 = (b1 >> shift) & mask
                s2 = (b2 >> shift) & mask
                if s1 != s2:
                    symbol_errors += 1
    
    return symbol_errors / total_symbols


def compute_redundancy_overhead(
    original_length: int,
    encoded_length: int
) -> float:
    """
    Compute redundancy overhead as a percentage.
    
    Overhead = ((encoded_length - original_length) / original_length) * 100
    
    Args:
        original_length: Length of original data in bytes
        encoded_length: Length of ECC-protected data in bytes
    
    Returns:
        Overhead percentage
    
    Example:
        >>> overhead = compute_redundancy_overhead(1000, 1143)  # RS(255, 223)
        >>> assert abs(overhead - 14.3) < 0.1
    """
    if original_length <= 0:
        raise ValueError(f"original_length must be > 0, got {original_length}")
    
    if encoded_length < original_length:
        raise ValueError(
            f"encoded_length {encoded_length} < original_length {original_length}"
        )
    
    overhead = ((encoded_length - original_length) / original_length) * 100.0
    return overhead


def estimate_correction_capability(
    ber_before: float,
    ber_after: float,
    total_bits: int
) -> dict:
    """
    Estimate error correction effectiveness.
    
    Args:
        ber_before: BER before ECC decoding
        ber_after: BER after ECC decoding
        total_bits: Total number of bits
    
    Returns:
        Dictionary with correction statistics:
            - bits_corrected: Number of bits corrected
            - correction_rate: Fraction of errors corrected
            - residual_ber: Remaining BER after correction
    
    Example:
        >>> stats = estimate_correction_capability(0.01, 0.0001, 10000)
        >>> print(stats['bits_corrected'])  # ~99 bits
    """
    errors_before = int(ber_before * total_bits)
    errors_after = int(ber_after * total_bits)
    bits_corrected = errors_before - errors_after
    
    correction_rate = bits_corrected / errors_before if errors_before > 0 else 0.0
    
    return {
        'bits_corrected': bits_corrected,
        'errors_before': errors_before,
        'errors_after': errors_after,
        'correction_rate': correction_rate,
        'residual_ber': ber_after,
    }