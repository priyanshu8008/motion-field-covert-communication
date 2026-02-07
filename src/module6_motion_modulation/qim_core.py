"""
Core Quantization Index Modulation (QIM) algorithms for motion-field embedding.

This module implements the fundamental QIM operations for embedding and extracting
bits from motion vectors while preserving direction.
"""

import numpy as np
from typing import Tuple


def qim_embed_vector(
    v: np.ndarray,
    bit: int,
    delta: float
) -> np.ndarray:
    """
    Embed one bit into a single motion vector using QIM.
    
    Algorithm (from ARCHITECTURE.md):
        1. m = ||v|| = sqrt(v[0]^2 + v[1]^2)
        2. q = round(m / delta)
        3. if bit == 0:
               m' = q * delta
           else:
               m' = (q + 0.5) * delta
        4. v' = v * (m' / m)
    
    Args:
        v: Motion vector (2,) [dx, dy]
        bit: Bit to embed (0 or 1)
        delta: Quantization step size (Δ)
        
    Returns:
        v': Modified motion vector with embedded bit
        
    Notes:
        - Direction is preserved: v' = v * scale_factor
        - Magnitude is quantized to encode the bit
        - If m ≈ 0, returns original vector (no embedding)
    """
    # Compute magnitude
    m = np.linalg.norm(v)
    
    # Skip zero or near-zero vectors
    if m < 1e-6:
        return v.copy()
    
    # Quantization index
    q = np.round(m / delta)
    
    # Compute modified magnitude based on bit
    if bit == 0:
        m_prime = q * delta
    else:
        m_prime = (q + 0.5) * delta
    
    # Scale vector to new magnitude (preserves direction)
    scale_factor = m_prime / m
    v_prime = v * scale_factor
    
    return v_prime


def qim_extract_bit(
    v: np.ndarray,
    delta: float,
    decision_boundary: float = 0.25
) -> int:
    """
    Extract bit from a motion vector using QIM.
    
    Algorithm (from ARCHITECTURE.md):
        1. m = ||v||
        2. q = round(m / delta)
        3. frac = (m / delta) - q
        4. bit = 0 if |frac| < decision_boundary else 1
    
    Args:
        v: Motion vector (2,) [dx, dy]
        delta: Quantization step size (Δ)
        decision_boundary: Threshold for bit decision (default: 0.25)
        
    Returns:
        bit: Extracted bit (0 or 1)
        
    Notes:
        - decision_boundary = 0.25 means:
          * bit = 0 if fractional part is in [-0.25, 0.25]
          * bit = 1 if fractional part is in (0.25, 0.75) or outside [-0.5, 0.5]
    """
    # Compute magnitude
    m = np.linalg.norm(v)
    
    # Handle zero vectors
    if m < 1e-6:
        return 0
    
    # Quantization index
    q = np.round(m / delta)
    
    # Fractional part
    frac = (m / delta) - q
    
    # Decision rule
    if np.abs(frac) < decision_boundary:
        return 0
    else:
        return 1


def qim_embed_flow(
    flow: np.ndarray,
    bits: np.ndarray,
    embedding_map: np.ndarray,
    delta: float
) -> np.ndarray:
    """
    Embed bit sequence into flow field using QIM.
    
    Args:
        flow: Flow field (H, W, 2) - ORIGINAL, NOT MODIFIED
        bits: Bit array (N,) - bits to embed
        embedding_map: Boolean array (H, W) - where to embed
        delta: Quantization step size
        
    Returns:
        modified_flow: Flow field with embedded bits (H, W, 2)
        
    Notes:
        - Only embeds in locations where embedding_map is True
        - Processes in raster-scan order (row-major)
        - Deterministic: same inputs → same outputs
    """
    H, W, _ = flow.shape
    modified_flow = flow.copy()
    
    # Get embedding locations in raster-scan order
    embed_indices = np.argwhere(embedding_map)  # (N, 2) array of [row, col]
    
    # Embed bits sequentially
    for bit_idx, (row, col) in enumerate(embed_indices):
        if bit_idx >= len(bits):
            break
        
        v = flow[row, col]
        bit = int(bits[bit_idx])
        v_prime = qim_embed_vector(v, bit, delta)
        modified_flow[row, col] = v_prime
    
    return modified_flow


def qim_extract_flow(
    flow: np.ndarray,
    embedding_map: np.ndarray,
    delta: float,
    decision_boundary: float,
    num_bits: int
) -> np.ndarray:
    """
    Extract bit sequence from flow field using QIM.
    
    Args:
        flow: Flow field (H, W, 2) - possibly corrupted
        embedding_map: Boolean array (H, W) - where bits are embedded
        delta: Quantization step size
        decision_boundary: Threshold for bit decisions
        num_bits: Number of bits to extract (REQUIRED)
        
    Returns:
        bits: Extracted bit array (num_bits,)
        
    Notes:
        - Extracts from locations where embedding_map is True
        - Processes in raster-scan order (same as embedding)
        - num_bits MUST be specified (no implicit inference)
    """
    # Get embedding locations in raster-scan order
    embed_indices = np.argwhere(embedding_map)  # (N, 2) array of [row, col]
    
    # Extract bits sequentially
    bits = np.zeros(num_bits, dtype=np.uint8)
    
    for bit_idx in range(num_bits):
        if bit_idx >= len(embed_indices):
            # Not enough locations - pad with zeros
            break
        
        row, col = embed_indices[bit_idx]
        v = flow[row, col]
        bit = qim_extract_bit(v, delta, decision_boundary)
        bits[bit_idx] = bit
    
    return bits


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert bit array to bytes.
    
    Args:
        bits: Bit array (N,) with values 0 or 1
        
    Returns:
        data: Byte string (padded to multiple of 8 bits)
    """
    # Pad to multiple of 8
    n_bits = len(bits)
    n_bytes = (n_bits + 7) // 8
    padded_bits = np.zeros(n_bytes * 8, dtype=np.uint8)
    padded_bits[:n_bits] = bits
    
    # Pack into bytes
    bytes_data = np.packbits(padded_bits)
    return bytes_data.tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes to bit array.
    
    Args:
        data: Byte string
        
    Returns:
        bits: Bit array (N*8,) with values 0 or 1
    """
    bytes_array = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(bytes_array)
    return bits