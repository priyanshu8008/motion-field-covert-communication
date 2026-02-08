"""
Bitstream Aggregator

Aggregates extracted bits from multiple frames into a raw bitstream.
Outputs bytes for downstream processing by ECC decoder (Module 4).
"""

import numpy as np
from typing import List, Dict, Optional


class BitstreamAggregator:
    """
    Aggregates extracted bits into output bitstream.
    
    Collects bits from all frames and converts to bytes.
    NO framing, NO padding, NO alignment - just raw bits → bytes.
    """
    
    def __init__(self, config: dict):
        """
        Initialize bitstream aggregator.
        
        Args:
            config: Configuration dictionary (currently unused, but kept for consistency)
        """
        self.config = config
    
    def aggregate(
        self,
        per_frame_bits: List[np.ndarray],
        collect_metadata: bool = False
    ) -> bytes:
        """
        Aggregate bits from all frames into a single bitstream.
        
        Args:
            per_frame_bits: List of bit arrays, one per frame
                           Each element is np.ndarray of shape (N,) with 0/1 integers
            collect_metadata: If True, also collect per-frame statistics
            
        Returns:
            bitstream: Raw bytes (NO framing, NO padding)
        """
        # Concatenate all bits
        if len(per_frame_bits) == 0:
            return b''
        
        # Filter out empty arrays
        non_empty_bits = [bits for bits in per_frame_bits if len(bits) > 0]
        
        if len(non_empty_bits) == 0:
            return b''
        
        # Concatenate all bits into single array
        all_bits = np.concatenate(non_empty_bits)
        
        # Convert bits to bytes
        bitstream = self._bits_to_bytes(all_bits)
        
        return bitstream
    
    def aggregate_with_metadata(
        self,
        per_frame_bits: List[np.ndarray]
    ) -> tuple[bytes, Dict[str, any]]:
        """
        Aggregate bits and collect metadata.
        
        Args:
            per_frame_bits: List of bit arrays, one per frame
            
        Returns:
            bitstream: Raw bytes
            metadata: Dictionary containing:
                - total_bits: Total number of bits extracted
                - num_frames: Number of frames processed
                - bits_per_frame: List of bit counts per frame
                - skipped_frames: Number of frames with zero capacity
        """
        # Collect statistics
        bits_per_frame = [len(bits) for bits in per_frame_bits]
        total_bits = sum(bits_per_frame)
        num_frames = len(per_frame_bits)
        skipped_frames = sum(1 for count in bits_per_frame if count == 0)
        
        metadata = {
            'total_bits': total_bits,
            'num_frames': num_frames,
            'bits_per_frame': bits_per_frame,
            'skipped_frames': skipped_frames,
            'avg_bits_per_frame': total_bits / max(num_frames - skipped_frames, 1),
        }
        
        # Aggregate bits
        bitstream = self.aggregate(per_frame_bits, collect_metadata=False)
        
        return bitstream, metadata
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """
        Convert bit array to bytes.
        
        Packs bits into bytes in big-endian order (MSB first).
        If number of bits is not multiple of 8, the last byte is padded with zeros.
        
        Args:
            bits: Bit array (N,) with 0/1 integers
            
        Returns:
            data: Bytes
        """
        if len(bits) == 0:
            return b''
        
        # Ensure bits are 0 or 1
        bits = bits.astype(int) & 1
        
        # Pad to multiple of 8
        remainder = len(bits) % 8
        if remainder != 0:
            padding = np.zeros(8 - remainder, dtype=int)
            bits = np.concatenate([bits, padding])
        
        # Reshape to (N/8, 8) and pack
        bits = bits.reshape(-1, 8)
        
        # Convert each row to a byte (big-endian: MSB first)
        bytes_array = np.packbits(bits, axis=1, bitorder='big').flatten()
        
        return bytes(bytes_array)
    
    def _bytes_to_bits(self, data: bytes) -> np.ndarray:
        """
        Convert bytes to bit array (for testing/verification).
        
        Args:
            data: Bytes
            
        Returns:
            bits: Bit array (N*8,) with 0/1 integers
        """
        if len(data) == 0:
            return np.array([], dtype=int)
        
        # Unpack bytes to bits
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='big')
        
        return bits.astype(int)