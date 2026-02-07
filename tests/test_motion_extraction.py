"""
Unit Tests for Module 2: Motion Field Extraction

Tests for OpticalFlowExtractor class and related utilities.
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from module2_motion_extraction import (
    OpticalFlowExtractor,
    FlowExtractionError,
    visualize_flow,
    compute_flow_statistics,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def test_frames():
    """Create synthetic test frames."""
    # Create two simple test frames (small for fast testing)
    height, width = 64, 64
    
    # Frame 1: Gradient pattern
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        frame1[i, :, 0] = int(255 * i / height)  # R channel gradient
        frame1[:, i, 1] = int(255 * i / width)   # G channel gradient
    
    # Frame 2: Slightly shifted version
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        frame2[i, :, 0] = int(255 * i / height)
        frame2[:, i, 1] = int(255 * i / width)
    # Add some displacement
    frame2[2:, 2:, :] = frame1[:-2, :-2, :]
    
    return frame1, frame2


@pytest.fixture
def test_flow():
    """Create synthetic flow field for testing."""
    height, width = 64, 64
    flow = np.random.randn(height, width, 2).astype(np.float32) * 2.0
    return flow


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model weights file."""
    model_path = tmp_path / "raft-things.pth"
    
    # Create a minimal mock checkpoint
    import torch
    mock_state = {
        'model': {}  # Empty state dict for testing
    }
    torch.save(mock_state, model_path)
    
    return str(model_path)


# ============================================================================
# TEST OPTICAL FLOW EXTRACTOR INITIALIZATION
# ============================================================================

def test_extractor_initialization_cpu(mock_model_path):
    """Test extractor initialization on CPU."""
    extractor = OpticalFlowExtractor(
        model_path=mock_model_path,
        device="cpu"
    )
    assert extractor.device == "cpu"


def test_extractor_initialization_invalid_path():
    """Test that initialization fails with invalid model path."""
    with pytest.raises(FileNotFoundError):
        OpticalFlowExtractor(
            model_path="/nonexistent/path/model.pth",
            device="cpu"
        )


# ============================================================================
# TEST EXTRACT_FLOW
# ============================================================================

def test_extract_flow_output_shape(mock_model_path, test_frames):
    """Test that extract_flow returns correct shape."""
    frame1, frame2 = test_frames
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    flow = extractor.extract_flow(frame1, frame2)
    
    # Verify shape: (H, W, 2)
    assert flow.shape == (frame1.shape[0], frame1.shape[1], 2)


def test_extract_flow_output_dtype(mock_model_path, test_frames):
    """Test that extract_flow returns float32."""
    frame1, frame2 = test_frames
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    flow = extractor.extract_flow(frame1, frame2)
    
    # Verify dtype
    assert flow.dtype == np.float32


def test_extract_flow_mismatched_shapes(mock_model_path):
    """Test that extract_flow raises ValueError for mismatched frame shapes."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
    frame2 = np.zeros((32, 32, 3), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="Frame shape mismatch"):
        extractor.extract_flow(frame1, frame2)


def test_extract_flow_invalid_dtype(mock_model_path):
    """Test that extract_flow validates frame dtype."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    frame1 = np.zeros((64, 64, 3), dtype=np.float32)  # Wrong dtype
    frame2 = np.zeros((64, 64, 3), dtype=np.float32)
    
    with pytest.raises(ValueError, match="Expected uint8 dtype"):
        extractor.extract_flow(frame1, frame2)


def test_extract_flow_invalid_channels(mock_model_path):
    """Test that extract_flow validates number of channels."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    frame1 = np.zeros((64, 64), dtype=np.uint8)  # Missing channel dimension
    frame2 = np.zeros((64, 64), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="Expected frames with shape"):
        extractor.extract_flow(frame1, frame2)


# ============================================================================
# TEST BATCH_EXTRACT
# ============================================================================

def test_batch_extract_correct_count(mock_model_path, test_frames):
    """Test that batch_extract returns N-1 flows for N frames."""
    frame1, frame2 = test_frames
    
    # Create a list of frames
    frames = [frame1, frame2, frame1.copy(), frame2.copy()]
    
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    flows = extractor.batch_extract(frames)
    
    # Should have N-1 flows for N frames
    assert len(flows) == len(frames) - 1


def test_batch_extract_consistency(mock_model_path, test_frames):
    """Test that batch_extract is consistent with individual extract_flow calls."""
    frame1, frame2 = test_frames
    frames = [frame1, frame2]
    
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Extract using batch method
    batch_flows = extractor.batch_extract(frames)
    
    # Extract individually
    individual_flow = extractor.extract_flow(frame1, frame2)
    
    # Should have same shape
    assert batch_flows[0].shape == individual_flow.shape
    assert batch_flows[0].dtype == individual_flow.dtype


def test_batch_extract_insufficient_frames(mock_model_path, test_frames):
    """Test that batch_extract raises error with insufficient frames."""
    frame1, _ = test_frames
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Single frame should raise error
    with pytest.raises(ValueError, match="Need at least 2 frames"):
        extractor.batch_extract([frame1])
    
    # Empty list should raise error
    with pytest.raises(ValueError, match="Need at least 2 frames"):
        extractor.batch_extract([])


# ============================================================================
# TEST VISUALIZE_FLOW
# ============================================================================

def test_visualize_flow_output_shape(mock_model_path, test_flow):
    """Test that visualize_flow returns correct shape."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    vis = extractor.visualize_flow(test_flow)
    
    # Should return RGB image with same spatial dimensions
    assert vis.shape == (test_flow.shape[0], test_flow.shape[1], 3)


def test_visualize_flow_output_dtype(mock_model_path, test_flow):
    """Test that visualize_flow returns uint8."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    vis = extractor.visualize_flow(test_flow)
    
    assert vis.dtype == np.uint8


def test_visualize_flow_output_range(mock_model_path, test_flow):
    """Test that visualize_flow output is in valid range."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    vis = extractor.visualize_flow(test_flow)
    
    # RGB values should be in [0, 255]
    assert np.all(vis >= 0)
    assert np.all(vis <= 255)


def test_visualize_flow_invalid_shape(mock_model_path):
    """Test that visualize_flow validates flow shape."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Wrong shape
    invalid_flow = np.zeros((64, 64, 3), dtype=np.float32)
    
    with pytest.raises(ValueError, match="must have shape"):
        extractor.visualize_flow(invalid_flow)


def test_visualize_flow_invalid_dtype(mock_model_path):
    """Test that visualize_flow validates flow dtype."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Wrong dtype
    invalid_flow = np.zeros((64, 64, 2), dtype=np.int32)
    
    with pytest.raises(ValueError, match="must be float32"):
        extractor.visualize_flow(invalid_flow)


# ============================================================================
# TEST COMPUTE_STATISTICS
# ============================================================================

def test_compute_statistics_required_keys(mock_model_path, test_flow):
    """Test that compute_statistics returns required keys."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    stats = extractor.compute_statistics(test_flow)
    
    # Check required keys from MODULE_INTERFACES.md
    required_keys = ['mean_magnitude', 'max_magnitude', 'directional_entropy']
    for key in required_keys:
        assert key in stats, f"Missing required statistic: {key}"


def test_compute_statistics_value_types(mock_model_path, test_flow):
    """Test that compute_statistics returns float values."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    stats = extractor.compute_statistics(test_flow)
    
    # All values should be floats
    for key, value in stats.items():
        assert isinstance(value, float), f"{key} should be float, got {type(value)}"


def test_compute_statistics_non_negative_magnitude(mock_model_path):
    """Test that magnitude statistics are non-negative."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Create flow with known values
    flow = np.random.randn(64, 64, 2).astype(np.float32)
    stats = extractor.compute_statistics(flow)
    
    # Magnitudes should be non-negative
    assert stats['mean_magnitude'] >= 0
    assert stats['max_magnitude'] >= 0


def test_compute_statistics_zero_flow(mock_model_path):
    """Test compute_statistics with zero flow."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Zero flow
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    stats = extractor.compute_statistics(flow)
    
    # Magnitudes should be zero
    assert stats['mean_magnitude'] == 0.0
    assert stats['max_magnitude'] == 0.0


def test_compute_statistics_invalid_shape(mock_model_path):
    """Test that compute_statistics validates flow shape."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    invalid_flow = np.zeros((64, 64), dtype=np.float32)  # Missing channel dimension
    
    with pytest.raises(ValueError, match="must have shape"):
        extractor.compute_statistics(invalid_flow)


# ============================================================================
# TEST UTILITY FUNCTIONS
# ============================================================================

def test_visualize_flow_utility(test_flow):
    """Test standalone visualize_flow utility function."""
    vis = visualize_flow(test_flow)
    
    assert vis.shape == (test_flow.shape[0], test_flow.shape[1], 3)
    assert vis.dtype == np.uint8


def test_compute_flow_statistics_utility(test_flow):
    """Test standalone compute_flow_statistics utility function."""
    stats = compute_flow_statistics(test_flow)
    
    assert 'mean_magnitude' in stats
    assert 'max_magnitude' in stats
    assert 'directional_entropy' in stats
    assert all(isinstance(v, float) for v in stats.values())


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_pipeline(mock_model_path, test_frames):
    """Test complete flow extraction pipeline."""
    frame1, frame2 = test_frames
    
    # Initialize extractor
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Extract flow
    flow = extractor.extract_flow(frame1, frame2)
    
    # Visualize
    vis = extractor.visualize_flow(flow)
    
    # Compute statistics
    stats = extractor.compute_statistics(flow)
    
    # Verify all outputs
    assert flow.shape == (frame1.shape[0], frame1.shape[1], 2)
    assert flow.dtype == np.float32
    assert vis.shape == (frame1.shape[0], frame1.shape[1], 3)
    assert vis.dtype == np.uint8
    assert 'mean_magnitude' in stats
    assert 'max_magnitude' in stats
    assert 'directional_entropy' in stats


def test_batch_pipeline(mock_model_path, test_frames):
    """Test batch extraction pipeline."""
    frame1, frame2 = test_frames
    frames = [frame1, frame2, frame1.copy()]
    
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Batch extract
    flows = extractor.batch_extract(frames)
    
    # Should have 2 flows
    assert len(flows) == 2
    
    # Each flow should be valid
    for flow in flows:
        assert flow.shape == (frame1.shape[0], frame1.shape[1], 2)
        assert flow.dtype == np.float32
        
        # Should be able to visualize and compute stats
        vis = extractor.visualize_flow(flow)
        stats = extractor.compute_statistics(flow)
        
        assert vis.shape == (frame1.shape[0], frame1.shape[1], 3)
        assert 'mean_magnitude' in stats


# ============================================================================
# EDGE CASES
# ============================================================================

def test_large_flow_values(mock_model_path):
    """Test handling of large flow values."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Flow with large values
    flow = np.random.randn(64, 64, 2).astype(np.float32) * 100.0
    
    # Should handle without error
    vis = extractor.visualize_flow(flow)
    stats = extractor.compute_statistics(flow)
    
    assert vis.dtype == np.uint8
    assert stats['max_magnitude'] > 0


def test_directional_entropy_bounds(mock_model_path):
    """Test that directional entropy is bounded."""
    extractor = OpticalFlowExtractor(mock_model_path, device="cpu")
    
    # Uniform random flow
    flow = np.random.randn(64, 64, 2).astype(np.float32)
    stats = extractor.compute_statistics(flow)
    
    # Entropy should be non-negative and bounded by log2(num_bins)
    # With 16 bins, max entropy is log2(16) = 4
    assert stats['directional_entropy'] >= 0
    assert stats['directional_entropy'] <= 4.5  # Allow some margin


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])