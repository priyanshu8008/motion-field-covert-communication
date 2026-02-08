"""
Flow-based frame warping utilities for video reconstruction.

This module implements forward warping using optical flow fields.
Warping is deterministic and preserves embedded QIM perturbations.
"""

import numpy as np
import cv2
from typing import Tuple


def create_sampling_grid(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pixel sampling grid from optical flow field.
    
    Args:
        flow: Flow field (H, W, 2) with flow[:,:,0]=dx, flow[:,:,1]=dy
        
    Returns:
        map_x: X-coordinates for sampling (H, W) float32
        map_y: Y-coordinates for sampling (H, W) float32
        
    Notes:
        - Creates grid where each pixel (i,j) maps to (j+dx, i+dy)
        - Used by cv2.remap for forward warping
        - Deterministic: same flow → same grid
    """
    H, W = flow.shape[:2]
    
    # Create base coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    
    # Apply flow displacement
    # flow[:,:,0] is horizontal displacement (dx)
    # flow[:,:,1] is vertical displacement (dy)
    map_x = (x_coords + flow[:, :, 0]).astype(np.float32)
    map_y = (y_coords + flow[:, :, 1]).astype(np.float32)
    
    return map_x, map_y


def warp_frame(
    frame: np.ndarray,
    flow: np.ndarray,
    interpolation: str = "bilinear",
    border_mode: str = "replicate"
) -> np.ndarray:
    """
    Warp frame using optical flow field (forward warping).
    
    Algorithm:
        For each pixel (x, y):
            new_position = (x + flow[y,x,0], y + flow[y,x,1])
            warped_frame[y, x] = frame[new_position] (interpolated)
    
    Args:
        frame: Source frame (H, W, 3) uint8
        flow: Optical flow field (H, W, 2) float32
        interpolation: "nearest", "bilinear", or "bicubic"
        border_mode: "constant", "replicate", or "reflect"
        
    Returns:
        warped_frame: Warped frame (H, W, 3) uint8
        
    Notes:
        - Uses cv2.remap for efficient grid sampling
        - Deterministic: same (frame, flow, params) → same output
        - Preserves QIM perturbations in flow field
        - Border handling prevents artifacts at frame edges
    """
    # Map interpolation method
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC
    }
    
    # Map border mode
    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT
    }
    
    interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)
    border_flag = border_map.get(border_mode, cv2.BORDER_REPLICATE)
    
    # Create sampling grid from flow
    map_x, map_y = create_sampling_grid(flow)
    
    # Apply warping using remap
    warped_frame = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=interp_flag,
        borderMode=border_flag
    )
    
    return warped_frame


def blend_frames(
    original_frame: np.ndarray,
    warped_frame: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Blend original frame with warped frame.
    
    Formula: I' = α·I_original + (1-α)·I_warped
    
    Args:
        original_frame: Original frame (H, W, 3) uint8
        warped_frame: Warped frame (H, W, 3) uint8
        alpha: Blending weight [0, 1]
               - α=0: fully warped
               - α=1: fully original
               - α=0.9 (default): mostly original, subtle warping
    
    Returns:
        blended_frame: Blended result (H, W, 3) uint8
        
    Notes:
        - Blending is for visual quality only
        - Does NOT affect flow semantics or QIM signal
        - Deterministic
    """
    # Convert to float for blending
    original_float = original_frame.astype(np.float32)
    warped_float = warped_frame.astype(np.float32)
    
    # Blend
    blended_float = alpha * original_float + (1.0 - alpha) * warped_float
    
    # Clip and convert back to uint8
    blended_frame = np.clip(blended_float, 0, 255).astype(np.uint8)
    
    return blended_frame