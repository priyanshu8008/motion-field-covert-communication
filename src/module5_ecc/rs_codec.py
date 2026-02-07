# file: src/module5_ecc/rs_codec.py

"""
Reed-Solomon codec implementation.

Uses the reedsolo library for Galois Field arithmetic and RS encoding/decoding.
Handles byte-to-symbol conversion and deterministic padding.
"""

import struct
from typing import Tuple
from reedsolo import RSCodec

from .errors import ECCEncodingError, ECCDecodingError, ECCCorrectionError, ECCConfigurationError


class ReedSolomonCodec:
    """
    Reed-Solomon codec wrapper with deterministic padding and error handling.
    
    Parameters:
        n (int): Total codeword length (symbols)
        k (int): Message length (data symbols)
        nsym (int): Number of parity symbols (n - k)
    
    Invariants:
        - n = k + nsym
        - n <= 255 (GF(256) constraint)
        - Corrects up to nsym // 2 symbol errors
    """
    
    def __init__(self, n: int = 255, k: int = 223, nsym: int = 32):
        # Validate parameters
        if n > 255:
            raise ECCConfigurationError(f"Reed-Solomon n={n} exceeds GF(256) limit of 255")
        if nsym != n - k:
            raise ECCConfigurationError(f"Inconsistent RS parameters: n={n}, k={k}, nsym={nsym}")
        if nsym < 2:
            raise ECCConfigurationError(f"nsym={nsym} must be >= 2")
        
        self.n = n
        self.k = k
        self.nsym = nsym
        self.max_correctable_errors = nsym // 2
        
        # Initialize reedsolo codec
        # reedsolo uses nsym as the number of ECC symbols
        self.codec = RSCodec(nsym)
    
    def encode(self, data: bytes) -> bytes:
        """
        Encode data with Reed-Solomon error correction.
        
        Args:
            data: Arbitrary-length byte stream
        
        Returns:
            ECC-protected byte stream with framing:
                [4 bytes: original_length][encoded chunks][padding if needed]
        
        Raises:
            ECCEncodingError: If encoding fails
        """
        if not isinstance(data, bytes):
            raise ECCEncodingError(f"Expected bytes, got {type(data)}")
        
        try:
            original_length = len(data)
            
            # Prepend length header (4 bytes, big-endian)
            length_header = struct.pack(">I", original_length)
            data_with_header = length_header + data
            
            # Split into k-byte chunks
            chunks = []
            offset = 0
            total_length = len(data_with_header)
            
            while offset < total_length:
                chunk = data_with_header[offset:offset + self.k]
                
                # Pad last chunk deterministically if needed
                if len(chunk) < self.k:
                    padding_length = self.k - len(chunk)
                    chunk = chunk + b'\x00' * padding_length
                
                # Encode chunk (returns k + nsym bytes)
                encoded_chunk = self.codec.encode(chunk)
                chunks.append(encoded_chunk)
                
                offset += self.k
            
            # Concatenate all encoded chunks
            encoded_data = b''.join(chunks)
            
            return encoded_data
            
        except Exception as e:
            raise ECCEncodingError(f"Reed-Solomon encoding failed: {e}") from e
    
    def decode(self, data: bytes) -> bytes:
        """
        Decode Reed-Solomon protected data with error correction.
        
        Args:
            data: ECC-protected byte stream from encode()
        
        Returns:
            Original data (without length header or padding)
        
        Raises:
            ECCDecodingError: If data format is invalid
            ECCCorrectionError: If errors exceed correction capability
        """
        if not isinstance(data, bytes):
            raise ECCDecodingError(f"Expected bytes, got {type(data)}")
        
        if len(data) == 0:
            raise ECCDecodingError("Cannot decode empty data")
        
        # Each encoded chunk is n bytes
        if len(data) % self.n != 0:
            raise ECCDecodingError(
                f"Data length {len(data)} is not a multiple of codeword length {self.n}"
            )
        
        try:
            # Decode all chunks
            decoded_chunks = []
            num_chunks = len(data) // self.n
            
            for i in range(num_chunks):
                chunk = data[i * self.n:(i + 1) * self.n]
                
                try:
                    # Decode and correct errors
                    # reedsolo.decode returns (message, ecc) tuple or raises ReedSolomonError
                    decoded = self.codec.decode(chunk)
                    decoded_chunk = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
                    decoded_chunks.append(decoded_chunk)

                    
                except Exception as decode_error:
                    # Error correction failed - too many errors
                    raise ECCCorrectionError(
                        f"Reed-Solomon correction failed on chunk {i}/{num_chunks}: {decode_error}",
                        max_correctable=self.max_correctable_errors
                    ) from decode_error
            
            # Concatenate decoded chunks
            decoded_data = b''.join(decoded_chunks)
            
            # Extract length header (first 4 bytes)
            if len(decoded_data) < 4:
                raise ECCDecodingError("Decoded data too short to contain length header")
            
            original_length = struct.unpack(">I", decoded_data[:4])[0]
            
            # Validate length
            if original_length > len(decoded_data) - 4:
                raise ECCDecodingError(
                    f"Length header {original_length} exceeds available data {len(decoded_data) - 4}"
                )
            
            # Extract original data (strip header and padding)
            original_data = decoded_data[4:4 + original_length]
            
            return original_data
            
        except ECCCorrectionError:
            # Re-raise correction errors as-is
            raise
        except Exception as e:
            raise ECCDecodingError(f"Reed-Solomon decoding failed: {e}") from e
    
    def get_redundancy_overhead(self) -> float:
        """
        Calculate redundancy overhead as a fraction.
        
        Returns:
            Overhead ratio: (nsym / k)
        """
        return self.nsym / self.k
    
    def get_code_rate(self) -> float:
        """
        Calculate code rate.
        
        Returns:
            Code rate: k / n
        """
        return self.k / self.n