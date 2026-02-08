"""
Test: Determinism

Verifies that the receiver produces identical outputs for identical inputs.
This is CRITICAL for proper synchronization with the encoder.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.module8_receiver.receiver import ReceiverEngine
from src.module8_receiver.qim_demod import QIMDemodulator
from src.module8_receiver.capacity import CapacityEstimator


def create_test_config():
    """Create minimal test configuration."""
    return {
        'system': {
            'verbose': False,
            'debug_mode': False,
        },
        'optical_flow': {
            'preprocessing': {
                'normalize': True,
                'max_flow_magnitude': 100.0,
            }
        },
        'modulation': {
            'embedding': {
                'quantization_step': 2.0,
                'max_payload_bits': 4096,
            },
            'selection': {
                'motion_threshold': 1.0,
                'spatial_distribution': 'uniform',
                'use_high_motion_regions': True,
            },
            'demodulation': {
                'decision_boundary': 0.25,
                'use_soft_decisions': False,
            }
        }
    }


def create_synthetic_frames(num_frames: int = 10, height: int = 64, width: int = 64):
    """
    Create synthetic video frames with motion.
    
    Returns:
        frames: List of RGB frames
    """
    frames = []
    for i in range(num_frames):
        # Create frame with gradient pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add horizontal gradient with shift
        offset = i * 5  # Shift pattern to simulate motion
        for x in range(width):
            frame[:, x, :] = ((x + offset) % 256)
        
        frames.append(frame)
    
    return frames


def create_synthetic_flow(height: int = 64, width: int = 64, magnitude: float = 5.0):
    """
    Create synthetic flow field with known properties.
    
    Args:
        height, width: Flow field dimensions
        magnitude: Average motion magnitude
        
    Returns:
        flow: FlowField (H, W, 2)
    """
    flow = np.zeros((height, width, 2), dtype=np.float32)
    
    # Create flow with varying magnitudes
    for y in range(height):
        for x in range(width):
            # Create smooth motion pattern
            flow[y, x, 0] = magnitude * np.sin(2 * np.pi * x / width)
            flow[y, x, 1] = magnitude * np.cos(2 * np.pi * y / height)
    
    return flow


class TestDeterminism:
    """Test suite for deterministic behavior."""
    
    def test_qim_extraction_deterministic(self):
        """Test that QIM extraction is deterministic."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        
        # Create test vectors
        vectors = np.array([
            [3.0, 4.0],   # magnitude = 5.0
            [1.0, 1.0],   # magnitude = sqrt(2)
            [0.0, 2.0],   # magnitude = 2.0
            [5.0, 0.0],   # magnitude = 5.0
        ], dtype=np.float32)
        
        # Extract bits multiple times
        bits1 = demod.extract_bits(vectors)
        bits2 = demod.extract_bits(vectors)
        bits3 = demod.extract_bits(vectors)
        
        # All extractions should be identical
        assert np.array_equal(bits1, bits2), "QIM extraction is not deterministic!"
        assert np.array_equal(bits2, bits3), "QIM extraction is not deterministic!"
    
    def test_capacity_computation_deterministic(self):
        """Test that capacity computation is deterministic."""
        config = create_test_config()
        estimator = CapacityEstimator(config)
        
        # Create test flow
        flow = create_synthetic_flow(64, 64, magnitude=5.0)
        
        # Compute capacity multiple times
        cap1 = estimator.compute_capacity(flow)
        cap2 = estimator.compute_capacity(flow)
        cap3 = estimator.compute_capacity(flow)
        
        # All should be identical
        assert cap1 == cap2 == cap3, "Capacity computation is not deterministic!"
    
    def test_embedding_map_deterministic(self):
        """Test that embedding map generation is deterministic."""
        config = create_test_config()
        estimator = CapacityEstimator(config)
        
        # Create test flow
        flow = create_synthetic_flow(64, 64, magnitude=5.0)
        capacity = estimator.compute_capacity(flow)
        
        # Generate embedding maps multiple times
        map1 = estimator.compute_embedding_map(flow, capacity)
        map2 = estimator.compute_embedding_map(flow, capacity)
        map3 = estimator.compute_embedding_map(flow, capacity)
        
        # All should be identical
        assert np.array_equal(map1, map2), "Embedding map is not deterministic!"
        assert np.array_equal(map2, map3), "Embedding map is not deterministic!"
    
    def test_full_receiver_deterministic(self):
        """Test that full receiver pipeline is deterministic."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        # Create test frames
        frames = create_synthetic_frames(num_frames=10)
        
        # Note: Since we're using stub flow extraction (zero flow), 
        # this will extract zero bits. But it should still be deterministic!
        
        # Extract multiple times
        bits1 = receiver.extract(frames)
        bits2 = receiver.extract(frames)
        bits3 = receiver.extract(frames)
        
        # All should be identical
        assert bits1 == bits2 == bits3, "Receiver pipeline is not deterministic!"


class TestQIMDemodulator:
    """Test suite for QIM demodulator."""
    
    def test_qim_decision_boundary(self):
        """Test QIM decision rule with known vectors."""
        config = create_test_config()
        config['modulation']['embedding']['quantization_step'] = 2.0
        config['modulation']['demodulation']['decision_boundary'] = 0.25
        
        demod = QIMDemodulator(config)
        
        # Test vectors with known expected outputs
        # For Δ=2.0, decision_boundary=0.25:
        
        # Vector with magnitude = 2.0 (exactly on quantization point)
        # m/Δ = 1.0, q = 1, frac = 0.0, |frac| < 0.25 → bit = 0
        v1 = np.array([2.0, 0.0], dtype=np.float32)
        assert demod.extract_bit(v1) == 0
        
        # Vector with magnitude = 3.0 (halfway between points)
        # m/Δ = 1.5, q = 2, frac = -0.5, |frac| = 0.5 > 0.25 → bit = 1
        v2 = np.array([3.0, 0.0], dtype=np.float32)
        assert demod.extract_bit(v2) == 1
        
        # Vector with magnitude = 2.4 (close to quantization point)
        # m/Δ = 1.2, q = 1, frac = 0.2, |frac| < 0.25 → bit = 0
        v3 = np.array([2.4, 0.0], dtype=np.float32)
        assert demod.extract_bit(v3) == 0
        
        # Vector with magnitude = 2.6 (past decision boundary)
        # m/Δ = 1.3, q = 1, frac = 0.3, |frac| > 0.25 → bit = 1
        v4 = np.array([2.6, 0.0], dtype=np.float32)
        assert demod.extract_bit(v4) == 1
    
    def test_qim_zero_vector(self):
        """Test QIM extraction from zero vector."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        
        # Zero vector should default to bit = 0
        v = np.array([0.0, 0.0], dtype=np.float32)
        assert demod.extract_bit(v) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])