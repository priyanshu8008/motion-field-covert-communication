"""
Test: Zero-Motion Frame Handling

Verifies that the receiver correctly handles frames with insufficient motion.
Such frames should be skipped without crashing or producing errors.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.module8_receiver.receiver import ReceiverEngine
from src.module8_receiver.capacity import CapacityEstimator
from src.module8_receiver.qim_demod import QIMDemodulator


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
                'motion_threshold': 1.0,  # Minimum motion required
                'spatial_distribution': 'uniform',
                'use_high_motion_regions': True,
            },
            'demodulation': {
                'decision_boundary': 0.25,
                'use_soft_decisions': False,
            }
        }
    }


def create_zero_motion_frames(num_frames: int = 10, height: int = 64, width: int = 64):
    """
    Create frames with NO motion (all identical).
    
    Returns:
        frames: List of identical frames
    """
    # Create one frame
    base_frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Repeat it
    frames = [base_frame.copy() for _ in range(num_frames)]
    
    return frames


def create_low_motion_flow(height: int = 64, width: int = 64, magnitude: float = 0.5):
    """
    Create flow field with motion below threshold.
    
    Args:
        magnitude: Motion magnitude (should be < motion_threshold for testing)
        
    Returns:
        flow: FlowField with low motion
    """
    flow = np.random.randn(height, width, 2).astype(np.float32) * magnitude
    return flow


class TestZeroMotionHandling:
    """Test suite for zero/low motion frame handling."""
    
    def test_zero_capacity_flow(self):
        """Test that zero-motion flow yields zero capacity."""
        config = create_test_config()
        estimator = CapacityEstimator(config)
        
        # Create flow with all magnitudes below threshold
        flow = create_low_motion_flow(64, 64, magnitude=0.5)
        # motion_threshold = 1.0, so all vectors should be rejected
        
        capacity = estimator.compute_capacity(flow)
        
        assert capacity == 0, f"Expected capacity=0 for low motion, got {capacity}"
    
    def test_zero_motion_embedding_map(self):
        """Test that zero-capacity flow produces empty embedding map."""
        config = create_test_config()
        estimator = CapacityEstimator(config)
        
        # Create zero-motion flow
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        
        capacity = estimator.compute_capacity(flow)
        assert capacity == 0
        
        # Embedding map should be all False
        embedding_map = estimator.compute_embedding_map(flow, capacity)
        assert not np.any(embedding_map), "Embedding map should be empty for zero motion"
    
    def test_extraction_from_zero_motion(self):
        """Test bit extraction from zero-motion flow."""
        config = create_test_config()
        demod = QIMDemodulator(config)
        estimator = CapacityEstimator(config)
        
        # Create zero-motion flow
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        
        # Get embedding map
        capacity = estimator.compute_capacity(flow)
        embedding_map = estimator.compute_embedding_map(flow, capacity)
        
        # Extract bits - should return empty array
        bits = demod.extract_from_flow(flow, embedding_map)
        
        assert len(bits) == 0, "Should extract zero bits from zero-motion flow"
    
    def test_receiver_with_all_zero_motion(self):
        """Test receiver with all frames having zero motion."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        # Create frames with zero motion
        frames = create_zero_motion_frames(num_frames=10)
        
        # Extract - should return empty bitstream without crashing
        bitstream = receiver.extract(frames)
        
        assert len(bitstream) == 0, "Should extract empty bitstream from zero-motion video"
    
    def test_receiver_with_mixed_motion(self):
        """Test receiver with mix of high and low motion frames."""
        config = create_test_config()
        estimator = CapacityEstimator(config)
        
        # Create flows with varying motion
        flows = []
        
        # Low motion (below threshold)
        low_flow = create_low_motion_flow(64, 64, magnitude=0.5)
        flows.append(low_flow)
        
        # High motion (above threshold)
        high_flow = np.random.randn(64, 64, 2).astype(np.float32) * 5.0
        flows.append(high_flow)
        
        # Zero motion
        zero_flow = np.zeros((64, 64, 2), dtype=np.float32)
        flows.append(zero_flow)
        
        # Check capacities
        capacities = [estimator.compute_capacity(flow) for flow in flows]
        
        # First should be zero (low motion)
        assert capacities[0] == 0, "Low motion flow should have zero capacity"
        
        # Second should be positive (high motion)
        assert capacities[1] > 0, "High motion flow should have positive capacity"
        
        # Third should be zero (no motion)
        assert capacities[2] == 0, "Zero motion flow should have zero capacity"
    
    def test_receiver_metadata_skipped_frames(self):
        """Test that receiver metadata correctly counts skipped frames."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        # Create zero-motion frames
        frames = create_zero_motion_frames(num_frames=10)
        
        # Extract with metadata
        bitstream, metadata = receiver.extract_with_metadata(frames)
        
        # All frames should be skipped (9 flows from 10 frames)
        assert metadata['num_frames_skipped'] == 9, \
            f"Expected 9 skipped frames, got {metadata['num_frames_skipped']}"
        
        # Total bits should be zero
        assert metadata['total_bits'] == 0, \
            f"Expected 0 total bits, got {metadata['total_bits']}"


class TestPartialExtraction:
    """Test suite for partial extraction scenarios."""
    
    def test_extraction_with_some_skipped_frames(self):
        """Test extraction when some frames are skipped."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        # Create frames (at least 2 needed)
        frames = create_zero_motion_frames(num_frames=5)
        
        # Extract with metadata
        bitstream, metadata = receiver.extract_with_metadata(frames)
        
        # Should not crash
        assert isinstance(bitstream, bytes)
        assert 'num_frames_skipped' in metadata
        assert 'total_bits' in metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])