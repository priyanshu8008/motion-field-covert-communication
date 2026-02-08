"""
Test: General Receiver Functionality

Comprehensive tests for ReceiverEngine and all subcomponents.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.module8_receiver.receiver import ReceiverEngine
from src.module8_receiver.flow_recompute import FlowRecomputeWrapper
from src.module8_receiver.capacity import CapacityEstimator
from src.module8_receiver.region_selection import RegionSelector
from src.module8_receiver.qim_demod import QIMDemodulator
from src.module8_receiver.bitstream import BitstreamAggregator


def create_test_config():
    """Create test configuration."""
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
                'epsilon': 0.5,
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
            },
            'constraints': {
                'enforce_smoothness': True,
                'smoothness_kernel_size': 5,
            }
        }
    }


class TestFlowRecomputeWrapper:
    """Test suite for FlowRecomputeWrapper."""
    
    def test_initialization(self):
        """Test flow wrapper initialization."""
        config = create_test_config()
        wrapper = FlowRecomputeWrapper(config)
        
        assert wrapper.normalize == True
        assert wrapper.max_flow_magnitude == 100.0
    
    def test_extract_flow_validation(self):
        """Test flow extraction input validation."""
        config = create_test_config()
        wrapper = FlowRecomputeWrapper(config)
        
        # Create mismatched frames
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="shape mismatch"):
            wrapper.extract_flow(frame1, frame2)
    
    def test_batch_extract_validation(self):
        """Test batch extraction validation."""
        config = create_test_config()
        wrapper = FlowRecomputeWrapper(config)
        
        # Empty frames list
        with pytest.raises(ValueError, match="at least 2 frames"):
            wrapper.batch_extract([])
        
        # Single frame
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="at least 2 frames"):
            wrapper.batch_extract([frame])


class TestCapacityEstimator:
    """Test suite for CapacityEstimator."""
    
    def test_initialization(self):
        """Test capacity estimator initialization."""
        config = create_test_config()
        estimator = CapacityEstimator(config)
        
        assert estimator.motion_threshold == 1.0
        assert estimator.spatial_distribution == 'uniform'
        assert estimator.max_payload_bits == 4096
    
    def test_invalid_spatial_distribution(self):
        """Test that invalid spatial distribution is rejected."""
        config = create_test_config()
        config['modulation']['selection']['spatial_distribution'] = 'random'
        
        with pytest.raises(ValueError, match="Invalid spatial_distribution"):
            estimator = CapacityEstimator(config)
    
    def test_capacity_with_high_motion(self):
        """Test capacity computation with high motion."""
        config = create_test_config()
        estimator = CapacityEstimator(config)
        
        # Create flow with high motion
        flow = np.random.randn(64, 64, 2).astype(np.float32) * 5.0
        
        capacity = estimator.compute_capacity(flow)
        
        # Should have positive capacity
        assert capacity > 0, "High motion flow should have positive capacity"
        assert capacity <= 4096, "Capacity should not exceed max_payload_bits"
    
    def test_capacity_capping(self):
        """Test that capacity is capped at max_payload_bits."""
        config = create_test_config()
        config['modulation']['embedding']['max_payload_bits'] = 100
        estimator = CapacityEstimator(config)
        
        # Create flow with very high motion everywhere
        flow = np.ones((64, 64, 2), dtype=np.float32) * 10.0
        
        capacity = estimator.compute_capacity(flow)
        
        # Should be capped at 100
        assert capacity <= 100, f"Capacity should be capped at 100, got {capacity}"


class TestRegionSelector:
    """Test suite for RegionSelector."""
    
    def test_extraction_vectors(self):
        """Test extraction vector retrieval."""
        config = create_test_config()
        selector = RegionSelector(config)
        
        # Create flow
        flow = np.random.randn(64, 64, 2).astype(np.float32)
        
        # Create embedding map
        embedding_map = np.zeros((64, 64), dtype=bool)
        embedding_map[10:20, 10:20] = True  # Select 10x10 region
        
        # Get vectors
        vectors, positions = selector.get_extraction_vectors(flow, embedding_map)
        
        # Should get 100 vectors (10x10)
        assert len(vectors) == 100, f"Expected 100 vectors, got {len(vectors)}"
        assert len(positions) == 100, f"Expected 100 positions, got {len(positions)}"
    
    def test_extraction_order(self):
        """Test extraction order is deterministic (raster scan)."""
        config = create_test_config()
        selector = RegionSelector(config)
        
        # Create flow
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        
        # Create positions in random order
        positions = np.array([
            [30, 20],
            [10, 10],
            [20, 15],
            [10, 11],
        ])
        
        # Get extraction order
        order = selector.get_extraction_order(positions, flow)
        
        # Should be sorted by row-major order
        # (10,10) < (10,11) < (20,15) < (30,20)
        expected_order = [1, 3, 2, 0]  # Indices in sorted order
        
        assert list(order) == expected_order, \
            f"Expected order {expected_order}, got {list(order)}"


class TestBitstreamAggregator:
    """Test suite for BitstreamAggregator."""
    
    def test_empty_aggregation(self):
        """Test aggregation of empty bit arrays."""
        config = create_test_config()
        aggregator = BitstreamAggregator(config)
        
        # Empty list
        bitstream = aggregator.aggregate([])
        assert bitstream == b'', "Empty input should yield empty output"
        
        # List of empty arrays
        bitstream = aggregator.aggregate([
            np.array([], dtype=int),
            np.array([], dtype=int),
        ])
        assert bitstream == b'', "Empty arrays should yield empty output"
    
    def test_bits_to_bytes_conversion(self):
        """Test bit-to-byte conversion."""
        config = create_test_config()
        aggregator = BitstreamAggregator(config)
        
        # 8 bits = 1 byte
        bits = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
        bitstream = aggregator.aggregate([bits])
        
        assert len(bitstream) == 1, "8 bits should yield 1 byte"
        
        # Check value: 10101010 (binary) = 0xAA (hex) = 170 (decimal)
        assert bitstream[0] == 0xAA, f"Expected 0xAA, got {hex(bitstream[0])}"
    
    def test_padding(self):
        """Test that partial bytes are padded."""
        config = create_test_config()
        aggregator = BitstreamAggregator(config)
        
        # 5 bits - should be padded to 8
        bits = np.array([1, 0, 1, 0, 1], dtype=int)
        bitstream = aggregator.aggregate([bits])
        
        assert len(bitstream) == 1, "5 bits should yield 1 byte (padded)"
    
    def test_aggregation_with_metadata(self):
        """Test aggregation with metadata collection."""
        config = create_test_config()
        aggregator = BitstreamAggregator(config)
        
        # Create per-frame bits
        per_frame_bits = [
            np.array([1, 0, 1, 0], dtype=int),  # 4 bits
            np.array([], dtype=int),             # 0 bits (skipped frame)
            np.array([1, 1, 0, 0, 1, 1], dtype=int),  # 6 bits
        ]
        
        bitstream, metadata = aggregator.aggregate_with_metadata(per_frame_bits)
        
        assert metadata['total_bits'] == 10, "Should count 10 total bits"
        assert metadata['num_frames'] == 3, "Should count 3 frames"
        assert metadata['skipped_frames'] == 1, "Should count 1 skipped frame"
        assert metadata['bits_per_frame'] == [4, 0, 6], "Should track per-frame counts"


class TestReceiverEngine:
    """Test suite for ReceiverEngine."""
    
    def test_initialization(self):
        """Test receiver engine initialization."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        assert receiver.config == config
        assert receiver.flow_recompute is not None
        assert receiver.capacity_estimator is not None
        assert receiver.qim_demod is not None
    
    def test_extract_validation(self):
        """Test extract method input validation."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        # Empty frames
        with pytest.raises(ValueError, match="at least 2 frames"):
            receiver.extract([])
        
        # Single frame
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="at least 2 frames"):
            receiver.extract([frame])
    
    def test_extract_basic(self):
        """Test basic extraction (with stub flow)."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        # Create test frames
        frames = [
            np.zeros((64, 64, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        # Extract (will use stub zero flow, so should get empty output)
        bitstream = receiver.extract(frames)
        
        # Should not crash and return bytes
        assert isinstance(bitstream, bytes)
    
    def test_extract_with_metadata(self):
        """Test extraction with metadata collection."""
        config = create_test_config()
        receiver = ReceiverEngine(config)
        
        # Create test frames
        frames = [
            np.zeros((64, 64, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        # Extract with metadata
        bitstream, metadata = receiver.extract_with_metadata(frames)
        
        # Check metadata structure
        assert 'total_bits' in metadata
        assert 'num_frames_processed' in metadata
        assert 'num_frames_skipped' in metadata
        assert 'extraction_time_seconds' in metadata
        assert 'per_frame_stats' in metadata
        
        # Should process 4 flows (5 frames)
        assert metadata['num_frames_processed'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])