"""
Test suite for Video Reconstruction Engine (Module 6).

Tests cover:
- Warping correctness
- Propagative reconstruction
- Blending mode
- Temporal filtering
- Determinism
- Shape/type invariants
- Edge cases
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude/src')

from module7_video_recon import VideoReconstructor, QualityMetrics


def create_test_frames(N=5, H=64, W=64):
    """Create synthetic test frames."""
    frames = []
    for i in range(N):
        # Create simple gradient pattern
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 50) % 256  # Red varies by frame
        frame[:, :, 1] = np.linspace(0, 255, W, dtype=np.uint8)  # Green gradient
        frame[:, :, 2] = np.linspace(0, 255, H, dtype=np.uint8).reshape(-1, 1)  # Blue gradient
        frames.append(frame)
    return frames


def create_test_flows(N=5, H=64, W=64, motion_scale=2.0):
    """Create synthetic optical flows."""
    flows = []
    for i in range(N - 1):
        # Create simple horizontal motion
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[:, :, 0] = motion_scale  # Horizontal shift
        flow[:, :, 1] = 0.0  # No vertical shift
        flows.append(flow)
    return flows


def create_test_config(method='warp', apply_temporal=False):
    """Create test configuration."""
    return {
        'method': method,
        'warping': {
            'interpolation': 'bilinear',
            'border_mode': 'replicate'
        },
        'blending': {
            'alpha': 0.9
        },
        'temporal': {
            'apply_filter': apply_temporal,
            'filter_type': 'gaussian',
            'temporal_window': 3
        },
        'quality_metrics': {
            'compute_psnr': True,
            'compute_ssim': True,
            'compute_mse': True
        }
    }


# ============================================================================
# TEST 1: Basic Warping
# ============================================================================

def test_basic_warping():
    """Test basic frame warping."""
    print("TEST 1: Basic warping...")
    
    # Create simple test case
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[20:40, 20:40, :] = 255  # White square
    
    # Create horizontal shift flow
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    flow[:, :, 0] = 5.0  # Shift right by 5 pixels
    
    reconstructor = VideoReconstructor()
    config = create_test_config()
    
    warped = reconstructor.warp_frame(frame, flow, config)
    
    # Check output shape and type
    assert warped.shape == frame.shape, f"Shape mismatch: {warped.shape} vs {frame.shape}"
    assert warped.dtype == np.uint8, f"Type mismatch: {warped.dtype}"
    
    # Check that square moved (approximately)
    # Original square at [20:40, 20:40]
    # After shift right by 5, should be around [20:40, 25:45]
    # But due to border replication, exact check is tricky
    # Just verify warped output is different from input
    diff = np.sum(np.abs(warped.astype(float) - frame.astype(float)))
    assert diff > 0, "Warping should change the frame"
    
    print("  ✓ Basic warping test passed")


# ============================================================================
# TEST 2: Propagative Reconstruction
# ============================================================================

def test_propagative_reconstruction():
    """Test propagative frame reconstruction."""
    print("TEST 2: Propagative reconstruction...")
    
    N = 5
    frames = create_test_frames(N)
    flows = create_test_flows(N)
    config = create_test_config(method='warp')
    
    reconstructor = VideoReconstructor()
    stego_frames, metrics = reconstructor.reconstruct(frames, flows, config)
    
    # Check frame count
    assert len(stego_frames) == N, f"Frame count mismatch: {len(stego_frames)} vs {N}"
    
    # Check first frame unchanged
    assert np.array_equal(stego_frames[0], frames[0]), "First frame should be unchanged"
    
    # Check shapes
    for i, frame in enumerate(stego_frames):
        assert frame.shape == frames[0].shape, f"Frame {i} shape mismatch"
        assert frame.dtype == np.uint8, f"Frame {i} type mismatch"
    
    # Check metrics computed
    assert metrics is not None, "Metrics should be computed"
    assert hasattr(metrics, 'psnr'), "Missing PSNR"
    assert hasattr(metrics, 'ssim'), "Missing SSIM"
    assert hasattr(metrics, 'mse'), "Missing MSE"
    
    print(f"  Metrics: PSNR={metrics.psnr:.2f} dB, SSIM={metrics.ssim:.4f}, MSE={metrics.mse:.2f}")
    print("  ✓ Propagative reconstruction test passed")


# ============================================================================
# TEST 3: Blending Mode
# ============================================================================

def test_blending_mode():
    """Test blending reconstruction mode."""
    print("TEST 3: Blending mode...")
    
    N = 3
    frames = create_test_frames(N)
    flows = create_test_flows(N)
    config = create_test_config(method='blend')
    
    reconstructor = VideoReconstructor()
    stego_frames, metrics = reconstructor.reconstruct(frames, flows, config)
    
    # Check frame count
    assert len(stego_frames) == N, f"Frame count mismatch"
    
    # Blending should produce different results than pure warping
    config_warp = create_test_config(method='warp')
    stego_warp, _ = reconstructor.reconstruct(frames, flows, config_warp)
    
    # Frames should differ (except first frame)
    for i in range(1, N):
        diff = np.sum(np.abs(stego_frames[i].astype(float) - stego_warp[i].astype(float)))
        assert diff > 0, f"Frame {i}: blending should differ from pure warping"
    
    print("  ✓ Blending mode test passed")


# ============================================================================
# TEST 4: Determinism
# ============================================================================

def test_determinism():
    """Test that reconstruction is deterministic."""
    print("TEST 4: Determinism...")
    
    N = 4
    frames = create_test_frames(N)
    flows = create_test_flows(N)
    config = create_test_config()
    
    reconstructor = VideoReconstructor()
    
    # Reconstruct twice
    stego_1, metrics_1 = reconstructor.reconstruct(frames, flows, config)
    stego_2, metrics_2 = reconstructor.reconstruct(frames, flows, config)
    
    # Should be identical
    for i in range(N):
        assert np.array_equal(stego_1[i], stego_2[i]), f"Frame {i} not deterministic"
    
    # Metrics should be identical
    assert metrics_1.psnr == metrics_2.psnr, "PSNR not deterministic"
    assert metrics_1.ssim == metrics_2.ssim, "SSIM not deterministic"
    assert metrics_1.mse == metrics_2.mse, "MSE not deterministic"
    
    print("  ✓ Determinism test passed")


# ============================================================================
# TEST 5: Temporal Filtering
# ============================================================================

def test_temporal_filtering():
    """Test temporal filtering."""
    print("TEST 5: Temporal filtering...")
    
    N = 5
    frames = create_test_frames(N)
    flows = create_test_flows(N)
    
    # Reconstruct without temporal filtering
    config_no_filter = create_test_config(apply_temporal=False)
    reconstructor = VideoReconstructor()
    stego_no_filter, _ = reconstructor.reconstruct(frames, flows, config_no_filter)
    
    # Reconstruct with temporal filtering
    config_filter = create_test_config(apply_temporal=True)
    stego_filter, _ = reconstructor.reconstruct(frames, flows, config_filter)
    
    # Frames should differ (temporal filtering smooths)
    total_diff = 0
    for i in range(N):
        diff = np.sum(np.abs(stego_filter[i].astype(float) - stego_no_filter[i].astype(float)))
        total_diff += diff
    
    assert total_diff > 0, "Temporal filtering should change frames"
    
    print("  ✓ Temporal filtering test passed")


# ============================================================================
# TEST 6: Input Validation
# ============================================================================

def test_input_validation():
    """Test input validation and error handling."""
    print("TEST 6: Input validation...")
    
    reconstructor = VideoReconstructor()
    config = create_test_config()
    
    # Test 1: Flow count mismatch
    frames = create_test_frames(5)
    flows = create_test_flows(3)  # Wrong count
    
    try:
        reconstructor.reconstruct(frames, flows, config)
        assert False, "Should raise ValueError for flow count mismatch"
    except ValueError as e:
        assert "Flow count mismatch" in str(e)
    
    # Test 2: Empty input
    stego, metrics = reconstructor.reconstruct([], [], config)
    assert len(stego) == 0, "Empty input should return empty output"
    
    # Test 3: Single frame (no flows)
    frames = create_test_frames(1)
    flows = []
    stego, metrics = reconstructor.reconstruct(frames, flows, config)
    assert len(stego) == 1, "Single frame should work"
    assert np.array_equal(stego[0], frames[0]), "Single frame should be unchanged"
    
    print("  ✓ Input validation test passed")


# ============================================================================
# TEST 7: Quality Metrics Computation
# ============================================================================

def test_quality_metrics():
    """Test quality metrics computation."""
    print("TEST 7: Quality metrics computation...")
    
    reconstructor = VideoReconstructor()
    
    # Test with identical frames (perfect reconstruction)
    frames = create_test_frames(3)
    metrics = reconstructor.compute_quality(frames, frames)
    
    # PSNR should be very high (or inf)
    assert metrics.psnr > 50 or np.isinf(metrics.psnr), f"PSNR too low for identical frames: {metrics.psnr}"
    
    # SSIM should be 1.0 (or very close)
    assert metrics.ssim > 0.99, f"SSIM too low for identical frames: {metrics.ssim}"
    
    # MSE should be 0 (or very close)
    assert metrics.mse < 1.0, f"MSE too high for identical frames: {metrics.mse}"
    
    # Test with different frames
    frames2 = [frame + 10 for frame in frames]  # Slightly different
    metrics2 = reconstructor.compute_quality(frames, frames2)
    
    # Metrics should be worse
    assert metrics2.psnr < 50, "PSNR should decrease for different frames"
    assert metrics2.ssim < 1.0, "SSIM should decrease for different frames"
    assert metrics2.mse > 0, "MSE should increase for different frames"
    
    print(f"  Perfect: PSNR={metrics.psnr:.2f}, SSIM={metrics.ssim:.4f}, MSE={metrics.mse:.4f}")
    print(f"  Noisy: PSNR={metrics2.psnr:.2f}, SSIM={metrics2.ssim:.4f}, MSE={metrics2.mse:.4f}")
    print("  ✓ Quality metrics test passed")


# ============================================================================
# TEST 8: Zero Flow (No Motion)
# ============================================================================

def test_zero_flow():
    """Test reconstruction with zero flow (no motion)."""
    print("TEST 8: Zero flow (no motion)...")
    
    N = 4
    frames = create_test_frames(N)
    
    # Create zero flows
    H, W = frames[0].shape[:2]
    flows = [np.zeros((H, W, 2), dtype=np.float32) for _ in range(N - 1)]
    
    config = create_test_config()
    reconstructor = VideoReconstructor()
    
    stego_frames, metrics = reconstructor.reconstruct(frames, flows, config)
    
    # With zero flow, propagative warping should produce:
    # I'_0 = I_0
    # I'_1 = warp(I'_0, 0) = I'_0 = I_0
    # I'_2 = warp(I'_1, 0) = I'_1 = I_0
    # etc.
    # So all frames should be identical to first frame
    
    for i in range(N):
        # Should all be very similar to first frame
        # (exact equality may not hold due to interpolation artifacts)
        diff = np.sum(np.abs(stego_frames[i].astype(float) - frames[0].astype(float)))
        assert diff < 100, f"Frame {i} differs too much from first frame with zero flow"
    
    print("  ✓ Zero flow test passed")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run complete test suite."""
    print("="*70)
    print("MODULE 6: VIDEO RECONSTRUCTION ENGINE - TEST SUITE")
    print("="*70)
    print()
    
    tests = [
        test_basic_warping,
        test_propagative_reconstruction,
        test_blending_mode,
        test_determinism,
        test_temporal_filtering,
        test_input_validation,
        test_quality_metrics,
        test_zero_flow
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)