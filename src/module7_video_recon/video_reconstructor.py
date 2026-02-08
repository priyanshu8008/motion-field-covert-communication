"""
Video Reconstruction Engine (Module 6).

Reconstructs stego video frames from modified optical flow fields.
Uses propagative warping to preserve embedded QIM perturbations.
"""

import numpy as np
from typing import List, Tuple, Optional

from .warping import warp_frame, blend_frames
from .quality_metrics import QualityMetrics, compute_video_quality
from .temporal_filter import apply_temporal_filter


class VideoReconstructor:
    """
    Video Reconstruction Engine.
    
    Implements propagative frame warping:
        I'_0 = I_0
        I'_t = warp(I'_{t-1}, F'_{t-1}) for t = 1..N-1
    
    Supports optional blending and temporal filtering.
    All operations are deterministic and preserve embedded QIM signal.
    """
    
    def __init__(self):
        """Initialize reconstructor (stateless)."""
        pass
    
    def reconstruct(
        self,
        frames: List[np.ndarray],
        flows: List[np.ndarray],
        config: dict
    ) -> Tuple[List[np.ndarray], Optional[QualityMetrics]]:
        """
        Reconstruct video from modified optical flows.
        
        Algorithm (MANDATORY - LOCKED):
            I'_0 = I_0  (first frame unchanged)
            
            For t = 1 to N-1:
                I_warped_t = warp(I'_{t-1}, F'_{t-1})
                
                If blend mode:
                    I'_t = α·I_original_t + (1-α)·I_warped_t
                Else:
                    I'_t = I_warped_t
            
            If temporal filtering enabled:
                Apply temporal filter to all I'_t
        
        Args:
            frames: Original video frames (N frames, H×W×3 uint8)
            flows: Modified optical flows from Module 5 (N-1 flows, H×W×2 float32)
            config: Configuration dict from default_config.yaml
            
        Returns:
            stego_frames: Reconstructed frames (N frames, H×W×3 uint8)
            metrics: Quality metrics (optional, None if disabled)
            
        Raises:
            ValueError: If frame/flow count mismatch or invalid config
            
        Notes:
            - Deterministic: same (frames, flows, config) → same output
            - Preserves embedded QIM perturbations exactly
            - Does NOT modify flows in any way
            - First frame is always unchanged (I'_0 = I_0)
        """
        # Validate inputs
        N = len(frames)
        
        # Handle empty input
        if N == 0:
            return [], None
        
        if len(flows) != N - 1:
            raise ValueError(
                f"Flow count mismatch: expected {N-1} flows for {N} frames, "
                f"got {len(flows)}"
            )
        
        # Extract configuration
        method = config.get('method', 'warp')
        
        # Warping parameters
        warping_config = config.get('warping', {})
        interpolation = warping_config.get('interpolation', 'bilinear')
        border_mode = warping_config.get('border_mode', 'replicate')
        
        # Blending parameters
        blending_config = config.get('blending', {})
        alpha = blending_config.get('alpha', 0.9)
        
        # Temporal filtering parameters
        temporal_config = config.get('temporal', {})
        apply_temporal = temporal_config.get('apply_filter', False)
        filter_type = temporal_config.get('filter_type', 'gaussian')
        temporal_window = temporal_config.get('temporal_window', 3)
        
        # Quality metrics parameters
        metrics_config = config.get('quality_metrics', {})
        compute_metrics = (
            metrics_config.get('compute_psnr', True) or
            metrics_config.get('compute_ssim', True) or
            metrics_config.get('compute_mse', True)
        )
        
        # Initialize reconstructed frames list
        stego_frames = []
        
        # STEP 1: Propagative warping
        
        # First frame unchanged
        stego_frames.append(frames[0].copy())
        
        # Warp subsequent frames
        for t in range(1, N):
            # Get previous reconstructed frame
            prev_frame = stego_frames[t - 1]
            
            # Get flow F'_{t-1}
            flow = flows[t - 1]
            
            # Warp previous reconstructed frame
            warped_frame = warp_frame(
                prev_frame,
                flow,
                interpolation=interpolation,
                border_mode=border_mode
            )
            
            # Apply blending if configured
            if method == "blend":
                # Blend with original frame
                reconstructed_frame = blend_frames(
                    frames[t],
                    warped_frame,
                    alpha
                )
            else:
                # Pure warping (default)
                reconstructed_frame = warped_frame
            
            stego_frames.append(reconstructed_frame)
        
        # STEP 2: Temporal filtering (optional)
        if apply_temporal:
            stego_frames = apply_temporal_filter(
                stego_frames,
                filter_type,
                temporal_window
            )
        
        # STEP 3: Compute quality metrics (optional)
        metrics = None
        if compute_metrics:
            metrics = compute_video_quality(frames, stego_frames)
        
        return stego_frames, metrics
    
    def warp_frame(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        config: dict
    ) -> np.ndarray:
        """
        Warp single frame using optical flow.
        
        Args:
            frame: Source frame (H, W, 3) uint8
            flow: Optical flow field (H, W, 2) float32
            config: Configuration dict
            
        Returns:
            warped_frame: Warped frame (H, W, 3) uint8
            
        Notes:
            - Wrapper for warping.warp_frame with config extraction
            - Used for single-frame operations
        """
        warping_config = config.get('warping', {})
        interpolation = warping_config.get('interpolation', 'bilinear')
        border_mode = warping_config.get('border_mode', 'replicate')
        
        return warp_frame(frame, flow, interpolation, border_mode)
    
    def compute_quality(
        self,
        original_frames: List[np.ndarray],
        reconstructed_frames: List[np.ndarray]
    ) -> QualityMetrics:
        """
        Compute quality metrics comparing original and reconstructed frames.
        
        Args:
            original_frames: Original frames (N, H, W, 3)
            reconstructed_frames: Reconstructed frames (N, H, W, 3)
            
        Returns:
            metrics: Aggregated quality metrics (PSNR, SSIM, MSE)
            
        Notes:
            - Computes per-frame metrics, then aggregates (mean)
            - For evaluation/debugging only
            - Not used by downstream modules
        """
        return compute_video_quality(original_frames, reconstructed_frames)