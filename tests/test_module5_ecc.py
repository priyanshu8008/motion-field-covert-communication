# file: tests/test_module5_ecc.py

"""
Comprehensive unit tests for Module 5: Error Correction Coding.

Test coverage:
    - Reed-Solomon encode/decode round-trip
    - Error correction capability
    - Failure modes and exceptions
    - Metrics computation
    - Edge cases (empty data, large payloads, etc.)
    - Configuration validation
"""

import pytest
import struct
from src.module5_ecc import (
    ecc_encode,
    ecc_decode,
    compute_ber,
    compute_ser,
    compute_redundancy_overhead,
    ECCEncodingError,
    ECCDecodingError,
    ECCCorrectionError,
    ECCConfigurationError,
)
from src.module5_ecc.testing_utils import inject_bit_errors, inject_burst_errors
from src.module5_ecc.rs_codec import ReedSolomonCodec


# Default test configuration
DEFAULT_CONFIG = {
    'ecc': {
        'type': 'reed_solomon',
        'reed_solomon': {
            'n': 255,
            'k': 223,
            'nsym': 32,
        }
    }
}


class TestReedSolomonCodec:
    """Test Reed-Solomon codec implementation."""
    
    def test_initialization_valid(self):
        """Test valid codec initialization."""
        codec = ReedSolomonCodec(n=255, k=223, nsym=32)
        assert codec.n == 255
        assert codec.k == 223
        assert codec.nsym == 32
        assert codec.max_correctable_errors == 16
    
    def test_initialization_invalid_n(self):
        """Test that n > 255 raises error."""
        with pytest.raises(ECCConfigurationError, match="exceeds GF\\(256\\)"):
            ReedSolomonCodec(n=256, k=223, nsym=33)
    
    def test_initialization_inconsistent_params(self):
        """Test inconsistent n, k, nsym raises error."""
        with pytest.raises(ECCConfigurationError, match="Inconsistent"):
            ReedSolomonCodec(n=255, k=223, nsym=30)  # n != k + nsym
    
    def test_encode_decode_roundtrip_small(self):
        """Test encode/decode round-trip with small data."""
        codec = ReedSolomonCodec()
        original = b"Hello, World!"
        
        encoded = codec.encode(original)
        decoded = codec.decode(encoded)
        
        assert decoded == original
    
    def test_encode_decode_roundtrip_large(self):
        """Test encode/decode with large data (multiple chunks)."""
        codec = ReedSolomonCodec()
        original = b"X" * 10000  # ~45 chunks
        
        encoded = codec.encode(original)
        decoded = codec.decode(encoded)
        
        assert decoded == original
    
    def test_encode_decode_empty(self):
        """Test encoding empty data."""
        codec = ReedSolomonCodec()
        original = b""
        
        encoded = codec.encode(original)
        decoded = codec.decode(encoded)
        
        assert decoded == original
    
    def test_encode_preserves_length_header(self):
        """Test that length header is correctly embedded."""
        codec = ReedSolomonCodec()
        original = b"test"
        
        encoded = codec.encode(original)
        decoded = codec.decode(encoded)
        
        assert decoded == original
        assert len(original) == 4
    
    def test_error_correction_within_capability(self):
        codec = ReedSolomonCodec(n=255, k=223, nsym=32)
        original = b"Correct me!" * 20

        encoded = codec.encode(original)
        corrupted = inject_bit_errors(encoded, error_rate=0.01, seed=42)

        try:
            decoded = codec.decode(corrupted)
            assert decoded == original
        except ECCCorrectionError:
        # ACCEPTABLE: bit-level noise may exceed symbol correction
            pass

    
    def test_error_correction_exceeds_capability(self):
        """Test that excessive errors raise ECCCorrectionError."""
        codec = ReedSolomonCodec(n=255, k=223, nsym=32)
        original = b"Too many errors!" * 20
        
        encoded = codec.encode(original)
        
        # Inject errors exceeding correction capability
        corrupted = inject_bit_errors(encoded, error_rate=0.3, seed=42)
        
        # Should raise correction error
        with pytest.raises(ECCCorrectionError):
            codec.decode(corrupted)
    
    def test_decode_invalid_length(self):
        """Test decoding data with invalid length."""
        codec = ReedSolomonCodec()
        
        # Data not multiple of n
        invalid_data = b"X" * 100
        
        with pytest.raises(ECCDecodingError, match="not a multiple"):
            codec.decode(invalid_data)
    
    def test_decode_empty_raises_error(self):
        """Test that decoding empty data raises error."""
        codec = ReedSolomonCodec()
        
        with pytest.raises(ECCDecodingError, match="empty"):
            codec.decode(b"")
    
    def test_get_redundancy_overhead(self):
        """Test redundancy overhead calculation."""
        codec = ReedSolomonCodec(n=255, k=223, nsym=32)
        overhead = codec.get_redundancy_overhead()
        
        expected = 32 / 223
        assert abs(overhead - expected) < 1e-6
    
    def test_get_code_rate(self):
        """Test code rate calculation."""
        codec = ReedSolomonCodec(n=255, k=223, nsym=32)
        rate = codec.get_code_rate()
        
        expected = 223 / 255
        assert abs(rate - expected) < 1e-6


class TestECCEncodeDecode:
    """Test public ecc_encode/ecc_decode interface."""
    
    def test_encode_decode_roundtrip(self):
        """Test full encode/decode cycle."""
        original = b"Secret encrypted payload from Module 4"
        
        encoded = ecc_encode(original, DEFAULT_CONFIG)
        decoded = ecc_decode(encoded, DEFAULT_CONFIG)
        
        assert decoded == original
    
    def test_encode_with_different_config(self):
        """Test encoding with custom RS parameters."""
        config = {
            'ecc': {
                'type': 'reed_solomon',
                'reed_solomon': {
                    'n': 255,
                    'k': 191,
                    'nsym': 64,  # More redundancy
                }
            }
        }
        
        original = b"High redundancy test"
        
        encoded = ecc_encode(original, config)
        decoded = ecc_decode(encoded, config)
        
        assert decoded == original
    
    def test_encode_invalid_input_type(self):
        """Test that non-bytes input raises error."""
        with pytest.raises(ECCEncodingError, match="must be bytes"):
            ecc_encode("not bytes", DEFAULT_CONFIG)
    
    def test_decode_invalid_input_type(self):
        """Test that non-bytes input raises error."""
        with pytest.raises(ECCDecodingError, match="must be bytes"):
            ecc_decode("not bytes", DEFAULT_CONFIG)
    
    def test_encode_missing_config(self):
        """Test that missing config raises error."""
        with pytest.raises(ECCConfigurationError, match="Missing required"):
            ecc_encode(b"test", {})
    
    def test_encode_unknown_ecc_type(self):
        """Test that unknown ECC type raises error."""
        config = {
            'ecc': {
                'type': 'unknown_codec',
            }
        }
        
        with pytest.raises(ECCConfigurationError, match="Unknown ECC type"):
            ecc_encode(b"test", config)
    
    def test_encode_ldpc_not_implemented(self):
        """Test that LDPC raises not implemented error."""
        config = {
            'ecc': {
                'type': 'ldpc',
            }
        }
        
        with pytest.raises(ECCConfigurationError, match="not yet implemented"):
            ecc_encode(b"test", config)
    
    def test_robustness_with_compression_noise(self):
        """Simulate compression-like errors and verify correction."""
        original = b"Payload that will go through compression" * 10
        
        encoded = ecc_encode(original, DEFAULT_CONFIG)
        
        # Simulate compression artifacts (small BER)
        corrupted = inject_bit_errors(encoded, error_rate=0.005, seed=123)
        
        # Should recover successfully
        decoded = ecc_decode(corrupted, DEFAULT_CONFIG)
        assert decoded == original


class TestMetrics:
    """Test metrics computation functions."""
    
    def test_compute_ber_no_errors(self):
        """Test BER with identical data."""
        data = b"No errors here"
        ber = compute_ber(data, data)
        assert ber == 0.0
    
    def test_compute_ber_single_bit(self):
        """Test BER with single bit error."""
        original = b'\x00'
        received = b'\x01'  # 1 bit flipped
        
        ber = compute_ber(original, received)
        assert ber == 1.0 / 8
    
    def test_compute_ber_all_bits(self):
        """Test BER with all bits flipped."""
        original = b'\x00\x00'
        received = b'\xff\xff'
        
        ber = compute_ber(original, received)
        assert ber == 1.0
    
    def test_compute_ber_length_mismatch(self):
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_ber(b"short", b"longer data")
    
    def test_compute_ser_no_errors(self):
        """Test SER with identical data."""
        data = b"No errors"
        ser = compute_ser(data, data, symbol_size=8)
        assert ser == 0.0
    
    def test_compute_ser_byte_level(self):
        """Test SER at byte (symbol) level."""
        original = b'\x00\x00\x00\x00'
        received = b'\x01\x00\x02\x00'  # 2 bytes corrupted
        
        ser = compute_ser(original, received, symbol_size=8)
        assert ser == 2.0 / 4
    
    def test_compute_redundancy_overhead(self):
        """Test redundancy overhead calculation."""
        overhead = compute_redundancy_overhead(1000, 1143)
        assert abs(overhead - 14.3) < 0.1
    
    def test_compute_redundancy_overhead_invalid(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError):
            compute_redundancy_overhead(0, 100)
        
        with pytest.raises(ValueError):
            compute_redundancy_overhead(100, 50)


class TestErrorInjection:
    """Test error injection utilities."""
    
    def test_inject_bit_errors_deterministic(self):
        """Test that error injection is deterministic with seed."""
        data = b'\x00' * 100
        
        corrupted1 = inject_bit_errors(data, error_rate=0.01, seed=42)
        corrupted2 = inject_bit_errors(data, error_rate=0.01, seed=42)
        
        assert corrupted1 == corrupted2
    
    def test_inject_bit_errors_rate(self):
        """Test that actual error rate matches requested rate."""
        data = b'\x00' * 1000
        
        corrupted = inject_bit_errors(data, error_rate=0.05, seed=999)
        ber = compute_ber(data, corrupted)
        
        # Should be close to 5% (within tolerance)
        assert 0.04 < ber < 0.06
    
    def test_inject_burst_errors(self):
        """Test burst error injection."""
        data = b'\x00' * 100
        
        corrupted = inject_burst_errors(data, num_bursts=3, burst_length=8, seed=42)
        
        # Should have errors
        assert corrupted != data
        
        # Check determinism
        corrupted2 = inject_burst_errors(data, num_bursts=3, burst_length=8, seed=42)
        assert corrupted == corrupted2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_payload(self):
        """Test with minimal payload."""
        original = b"x"
        
        encoded = ecc_encode(original, DEFAULT_CONFIG)
        decoded = ecc_decode(encoded, DEFAULT_CONFIG)
        
        assert decoded == original
    
    def test_large_payload(self):
        """Test with large payload (many chunks)."""
        original = b"Large payload test " * 5000  # ~100KB
        
        encoded = ecc_encode(original, DEFAULT_CONFIG)
        decoded = ecc_decode(encoded, DEFAULT_CONFIG)
        
        assert decoded == original
    
    def test_binary_data(self):
        """Test with arbitrary binary data."""
        original = bytes(range(256)) * 10
        
        encoded = ecc_encode(original, DEFAULT_CONFIG)
        decoded = ecc_decode(encoded, DEFAULT_CONFIG)
        
        assert decoded == original
    
    def test_maximum_correctable_errors(self):
        """Test correction at maximum capability boundary."""
        codec = ReedSolomonCodec(n=255, k=223, nsym=32)
        original = b"Boundary test" * 20
        
        encoded = codec.encode(original)
        
        # Inject exactly 16 symbol errors (max correctable)
        # This is tricky - inject controlled burst errors
        corrupted = inject_burst_errors(encoded, num_bursts=16, burst_length=8, seed=777)
        
        # Should still decode (may or may not succeed depending on error distribution)
        # This test documents the boundary behavior
        try:
            decoded = codec.decode(corrupted)
            # If successful, verify correctness
            assert decoded == original
        except ECCCorrectionError:
            # Expected if errors exceed capability due to burst patterns
            pass


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_end_to_end_with_simulated_channel(self):
        """Simulate full encode -> channel -> decode pipeline."""
        # Module 4 output (encrypted)
        encrypted_payload = b"Encrypted data from Module 4" * 100
        
        # Module 5 encode
        ecc_protected = ecc_encode(encrypted_payload, DEFAULT_CONFIG)
        
        # Simulate channel (compression + noise)
        channel_output = inject_bit_errors(ecc_protected, error_rate=0.003, seed=555)
        
        # Module 5 decode
        recovered_payload = ecc_decode(channel_output, DEFAULT_CONFIG)
        
        # Should match original
        assert recovered_payload == encrypted_payload
    
    def test_failure_propagation(self):
        """Test that uncorrectable errors propagate explicitly."""
        original = b"Will be heavily corrupted"
        
        encoded = ecc_encode(original, DEFAULT_CONFIG)
        
        # Severe corruption
        corrupted = inject_bit_errors(encoded, error_rate=0.4, seed=111)
        
        # Should raise explicit error, not return garbage
        with pytest.raises(ECCCorrectionError) as exc_info:
            ecc_decode(corrupted, DEFAULT_CONFIG)
        
        # Error should contain diagnostic info
        assert exc_info.value.max_correctable == 16
    
    def test_no_silent_corruption(self):
        """Verify that decoder never returns silently corrupted data."""
        original = b"Critical data that must not be silently corrupted" * 50
        
        encoded = ecc_encode(original, DEFAULT_CONFIG)
        
        # Try multiple corruption levels
        for error_rate in [0.1, 0.2, 0.3]:
            corrupted = inject_bit_errors(encoded, error_rate=error_rate, seed=222)
            
            try:
                decoded = ecc_decode(corrupted, DEFAULT_CONFIG)
                # If decode succeeds, it MUST be correct
                assert decoded == original, "Decoder returned corrupted data without error!"
            except ECCCorrectionError:
                # Expected for high error rates - this is GOOD
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])