# file: src/module5_ecc/testing_utils.py

"""
Testing utilities for ECC module.

Provides error injection for validation and robustness testing.
Used only in test/evaluation contexts.
"""

import random
from typing import Optional


def inject_bit_errors(
    data: bytes,
    error_rate: float,
    seed: Optional[int] = None
) -> bytes:
    """
    Inject random bit errors into data for testing.
    
    WARNING: This function introduces non-determinism and should ONLY
    be used in testing/evaluation contexts, never in production pipeline.
    
    Args:
        data: Original data
        error_rate: Probability of bit flip (0.0 to 1.0)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Data with injected errors
    
    Example:
        >>> original = b'\x00' * 100
        >>> corrupted = inject_bit_errors(original, error_rate=0.01, seed=42)
        >>> ber = compute_ber(original, corrupted)
        >>> assert 0.005 < ber < 0.015  # Approximately 1% BER
    """
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")
    
    if seed is not None:
        random.seed(seed)
    
    # Convert to mutable bytearray
    corrupted = bytearray(data)
    
    total_bits = len(data) * 8
    num_errors = int(total_bits * error_rate)
    
    # Randomly select bit positions to flip
    error_positions = random.sample(range(total_bits), num_errors)
    
    for pos in error_positions:
        byte_idx = pos // 8
        bit_idx = pos % 8
        corrupted[byte_idx] ^= (1 << bit_idx)
    
    return bytes(corrupted)


def inject_burst_errors(
    data: bytes,
    num_bursts: int,
    burst_length: int,
    seed: Optional[int] = None
) -> bytes:
    """
    Inject burst errors into data for testing.
    
    Burst errors simulate consecutive bit errors, common in channel noise.
    
    Args:
        data: Original data
        num_bursts: Number of error bursts
        burst_length: Length of each burst in bits
        seed: Random seed for reproducibility
    
    Returns:
        Data with burst errors
    
    Example:
        >>> original = b'\x00' * 100
        >>> corrupted = inject_burst_errors(original, num_bursts=5, burst_length=8, seed=42)
    """
    if seed is not None:
        random.seed(seed)
    
    corrupted = bytearray(data)
    total_bits = len(data) * 8
    
    if num_bursts * burst_length > total_bits:
        raise ValueError("Total burst bits exceed data length")
    
    for _ in range(num_bursts):
        # Random burst start position
        start_pos = random.randint(0, total_bits - burst_length)
        
        # Flip consecutive bits
        for offset in range(burst_length):
            pos = start_pos + offset
            byte_idx = pos // 8
            bit_idx = pos % 8
            corrupted[byte_idx] ^= (1 << bit_idx)
    
    return bytes(corrupted)