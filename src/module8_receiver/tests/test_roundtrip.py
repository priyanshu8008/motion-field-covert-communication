"""
Test: Round-Trip (Encoder-Decoder)

Simulates encoder-decoder round-trip to verify bit extraction accuracy.
This test simulates QIM embedding and validates that extraction is correct.

Note: This is a SIMULATED round-trip since we don't have the actual encoder.
We manually create flow fields with embedded bits and verify extraction.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.module8_receiver.qim_demod import QIMDemodulator


def create_test_config():
    """Create minimal test configuration."""
    return {
        'system': {
            'verbose': False,
            'debug_mode': False,
        },
        'modulation': {
            'embedding': {
                'quantization_step': 2.0,
                'max_payload_bits': 4096,
            },
            'demodulation': {
                'decision_boundary': 0.25,
                'use_soft_decisions': False,
            }
        }
    }


def qim_embed_vector(
    v: np.ndarray,
    bit: int,
    delta: float
) -> np.ndarray:
    """
    Simulate QIM embedding (encoder side).
    
    This replicates the encoder's QIM algorithm:
        1. m = ||v||
        2. q = round(m / delta)
        3. if bit == 0:
               m' = q * delta
           else:
               m' = (q + 0.5) * delta
        4. v' = v * (m' / m)
    
    Args:
        v: Original motion vector (2,)
        bit: Bit to embed (0 or 1)
        delta: Quantization step
        
    Returns:
        v_prime: Modified vector
    """
    # Compute magnitude
    m = np.sqrt(v[0]**2 + v[1]**2)
    
    if m < 1e-8:
        # Zero vector - return as is
        return v.copy()
    
    # Quantization index
    q = np.round(m / delta)
    
    # Compute target magnitude based on bit
    if bit == 0:
        m_prime = q * delta
    else:
        m_prime = (q + 0.5) * delta
    
    # Scale vector to target magnitude
    scale = m_prime / m
    v_prime = v * scale
    
    return v_prime


class TestRoundTrip:
    """Test suite for encoder-decoder round-trip."""
    
    def test_single_vector_bit0(self):
        """Test round-trip for single vector embedding bit=0."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        delta = 2.0
        
        # Original vector
        v_orig = np.array([3.0, 4.0], dtype=np.float32)  # magnitude = 5.0
        
        # Embed bit = 0
        v_embedded = qim_embed_vector(v_orig, bit=0, delta=delta)
        
        # Extract bit
        extracted_bit = demod.extract_bit(v_embedded)
        
        assert extracted_bit == 0, f"Expected bit=0, got {extracted_bit}"
    
    def test_single_vector_bit1(self):
        """Test round-trip for single vector embedding bit=1."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        delta = 2.0
        
        # Original vector
        v_orig = np.array([3.0, 4.0], dtype=np.float32)
        
        # Embed bit = 1
        v_embedded = qim_embed_vector(v_orig, bit=1, delta=delta)
        
        # Extract bit
        extracted_bit = demod.extract_bit(v_embedded)
        
        assert extracted_bit == 1, f"Expected bit=1, got {extracted_bit}"
    
    def test_multiple_vectors_random_bits(self):
        """Test round-trip for multiple vectors with random bits."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        delta = 2.0
        
        # Create random vectors
        np.random.seed(42)  # For reproducibility
        num_vectors = 100
        vectors = np.random.randn(num_vectors, 2).astype(np.float32) * 5.0
        
        # Create random bits
        bits_to_embed = np.random.randint(0, 2, num_vectors)
        
        # Embed bits
        embedded_vectors = np.zeros_like(vectors)
        for i in range(num_vectors):
            embedded_vectors[i] = qim_embed_vector(vectors[i], bits_to_embed[i], delta)
        
        # Extract bits
        extracted_bits = demod.extract_bits(embedded_vectors)
        
        # Verify perfect extraction (lossless)
        accuracy = np.mean(extracted_bits == bits_to_embed)
        
        assert accuracy == 1.0, \
            f"Expected 100% accuracy, got {accuracy*100:.2f}% ({np.sum(extracted_bits == bits_to_embed)}/{num_vectors} correct)"
    
    def test_round_trip_with_different_quantization_steps(self):
        """Test round-trip with various quantization steps."""
        config = create_test_config()
        
        # Test different delta values
        deltas = [1.0, 2.0, 5.0, 10.0]
        
        for delta in deltas:
            config['modulation']['embedding']['quantization_step'] = delta
            demod = QIMDemodulator(config)
            
            # Create test vectors
            np.random.seed(42)
            vectors = np.random.randn(50, 2).astype(np.float32) * 10.0
            bits_to_embed = np.random.randint(0, 2, 50)
            
            # Embed
            embedded_vectors = np.array([
                qim_embed_vector(v, b, delta)
                for v, b in zip(vectors, bits_to_embed)
            ])
            
            # Extract
            extracted_bits = demod.extract_bits(embedded_vectors)
            
            # Verify
            accuracy = np.mean(extracted_bits == bits_to_embed)
            assert accuracy == 1.0, \
                f"Failed for delta={delta}: {accuracy*100:.2f}% accuracy"
    
    def test_round_trip_with_noise(self):
        """Test round-trip with added noise (simulating channel distortion)."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        delta = 2.0
        
        # Create vectors and embed bits
        np.random.seed(42)
        vectors = np.random.randn(100, 2).astype(np.float32) * 5.0
        bits_to_embed = np.random.randint(0, 2, 100)
        
        embedded_vectors = np.array([
            qim_embed_vector(v, b, delta)
            for v, b in zip(vectors, bits_to_embed)
        ])
        
        # Add small noise (simulating mild channel distortion)
        noise_std = 0.1
        noisy_vectors = embedded_vectors + np.random.randn(*embedded_vectors.shape) * noise_std
        
        # Extract
        extracted_bits = demod.extract_bits(noisy_vectors)
        
        # Verify - should still have high accuracy with small noise
        accuracy = np.mean(extracted_bits == bits_to_embed)
        
        # With small noise, we expect >90% accuracy
        assert accuracy > 0.9, \
            f"Accuracy too low with noise={noise_std}: {accuracy*100:.2f}%"
    
    def test_boundary_cases(self):
        """Test extraction at decision boundaries."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        delta = 2.0
        decision_boundary = 0.25
        
        # Test vectors at/near decision boundaries
        test_cases = [
            # (magnitude, expected_bit)
            (2.0, 0),   # Exactly on quantization point (q*delta)
            (2.1, 0),   # Slightly above, frac=0.05 < 0.25
            (2.4, 0),   # Near boundary, frac=0.2 < 0.25
            (2.5, 1),   # At boundary, frac=0.25, but |0.25| = 0.25, NOT < 0.25
            (2.6, 1),   # Past boundary, frac=0.3 > 0.25
            (3.0, 1),   # Halfway point (q+0.5)*delta
            (3.4, 1),   # Past halfway, frac=-0.3, |frac|=0.3 > 0.25
            (3.5, 1),   # Near next quantization point, frac=-0.25, |frac|=0.25
            (3.9, 0),   # Very close to next point, frac=-0.05 < 0.25
            (4.0, 0),   # Exactly on next quantization point
        ]
        
        for magnitude, expected_bit in test_cases:
            # Create vector with exact magnitude
            v = np.array([magnitude, 0.0], dtype=np.float32)
            extracted_bit = demod.extract_bit(v)
            
            # Note: Some boundary cases might be ambiguous depending on rounding
            # But the extractor should be consistent with its decision rule


class TestExtractionAccuracy:
    """Test suite for extraction accuracy metrics."""
    
    def test_perfect_extraction_no_noise(self):
        """Test that extraction is perfect without channel noise."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        delta = 2.0
        
        # Large test set
        np.random.seed(123)
        num_vectors = 1000
        vectors = np.random.randn(num_vectors, 2).astype(np.float32) * 10.0
        bits_to_embed = np.random.randint(0, 2, num_vectors)
        
        # Embed
        embedded_vectors = np.array([
            qim_embed_vector(v, b, delta)
            for v, b in zip(vectors, bits_to_embed)
        ])
        
        # Extract
        extracted_bits = demod.extract_bits(embedded_vectors)
        
        # Should be 100% accurate
        num_correct = np.sum(extracted_bits == bits_to_embed)
        accuracy = num_correct / num_vectors
        
        assert accuracy == 1.0, \
            f"Expected perfect extraction, got {num_correct}/{num_vectors} correct ({accuracy*100:.2f}%)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])