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
        """
        # Validate inputs
        N = len(frames)
        
        if N == 0:
            return [], None
        
        if len(flows) != N - 1:
            raise ValueError(
                f"Flow count mismatch: expected {N-1} flows for {N} frames, "
                f"got {len(flows)}"
            )
        
        # Extract configuration
        recon_config = config.get("reconstruction", {})
        method = recon_config.get("method", "warp")

        
        # Warping parameters
        warping_config = recon_config.get('warping', {})
        interpolation = warping_config.get('interpolation', 'bilinear')
        border_mode = warping_config.get('border_mode', 'replicate')
        
        # Blending parameters
        blending_config = recon_config.get('blending', {})
        alpha = blending_config.get('alpha', 0.9)
        
        # Temporal filtering parameters
        temporal_config = recon_config.get('temporal', {})
        apply_temporal = temporal_config.get('apply_filter', False)
        filter_type = temporal_config.get('filter_type', 'gaussian')
        temporal_window = temporal_config.get('temporal_window', 3)
        
        # Quality metrics parameters
        metrics_config = recon_config.get('quality_metrics', {})
        compute_metrics = (
            metrics_config.get('compute_psnr', True) or
            metrics_config.get('compute_ssim', True) or
            metrics_config.get('compute_mse', True)
        )

        # 🔑 MINIMAL ADDITION: anchor period (drift control)
        anchor_period = int(config.get("anchor_period", 10))
        anchor_period = max(1, anchor_period)
        
        # Initialize reconstructed frames list
        stego_frames = []
        
        # STEP 1: Propagative warping
        
        # First frame unchanged
        stego_frames.append(frames[0].copy())
        
        # Warp subsequent frames
        for t in range(1, N):

            # 🔒 ANCHOR FRAME RESET (ONLY NEW LOGIC)
            if t % anchor_period == 0:
                stego_frames.append(frames[t].copy())
                continue

            prev_frame = stego_frames[t - 1]
            flow = flows[t - 1]
            
            warped_frame = warp_frame(
                prev_frame,
                flow,
                interpolation=interpolation,
                border_mode=border_mode
            )
            
            reconstructed_frame = blend_frames(
                frames[t],
                warped_frame,
                alpha
            )

            
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
        """
        return compute_video_quality(original_frames, reconstructed_frames)
