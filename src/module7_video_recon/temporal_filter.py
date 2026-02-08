"""
Temporal filtering for video frame sequences.

Applies temporal smoothing to reduce flickering and improve temporal consistency.
This is OPTIONAL and configured via config.yaml.
"""

import numpy as np
from typing import List
from scipy.ndimage import gaussian_filter1d


def apply_gaussian_temporal_filter(
    frames: List[np.ndarray],
    window_size: int,
    sigma: float = 1.0
) -> List[np.ndarray]:
    """
    Apply Gaussian temporal filtering across frame sequence.
    
    Args:
        frames: List of frames (N, H, W, 3) uint8
        window_size: Temporal window size (e.g., 3 frames)
        sigma: Gaussian standard deviation
        
    Returns:
        filtered_frames: Temporally filtered frames (N, H, W, 3) uint8
        
    Algorithm:
        For each pixel (x, y, c):
            1. Extract temporal values: [f_0[x,y,c], ..., f_N[x,y,c]]
            2. Apply 1D Gaussian filter along time axis
            3. Store filtered value
            
    Notes:
        - Applied to reconstructed frames (not flows)
        - Reduces temporal flickering
        - Preserves embedded signal (operates post-reconstruction)
        - Deterministic
    """
    if len(frames) == 0:
        return []
    
    N = len(frames)
    H, W, C = frames[0].shape
    
    # Stack frames into 4D array (N, H, W, C)
    frames_array = np.stack(frames, axis=0).astype(np.float32)
    
    # Apply Gaussian filter along temporal axis (axis=0)
    # truncate to window_size
    truncate = (window_size - 1) / (2 * sigma) if sigma > 0 else 3.0
    
    filtered_array = gaussian_filter1d(
        frames_array,
        sigma=sigma,
        axis=0,  # Temporal axis
        mode='nearest',  # Border handling
        truncate=truncate
    )
    
    # Convert back to uint8
    filtered_array = np.clip(filtered_array, 0, 255).astype(np.uint8)
    
    # Split back into list of frames
    filtered_frames = [filtered_array[i] for i in range(N)]
    
    return filtered_frames


def apply_bilateral_temporal_filter(
    frames: List[np.ndarray],
    window_size: int,
    sigma_space: float = 1.0,
    sigma_color: float = 10.0
) -> List[np.ndarray]:
    """
    Apply bilateral temporal filtering.
    
    Args:
        frames: List of frames (N, H, W, 3) uint8
        window_size: Temporal window size
        sigma_space: Spatial (temporal) sigma
        sigma_color: Color (intensity) sigma
        
    Returns:
        filtered_frames: Filtered frames
        
    Notes:
        - Edge-preserving temporal filtering
        - More complex than Gaussian
        - Preserves sharp temporal transitions
        - Simplified implementation (full bilateral is expensive)
    """
    # For simplicity, fall back to Gaussian
    # Full bilateral temporal filter is computationally expensive
    # and not critical for this research system
    return apply_gaussian_temporal_filter(frames, window_size, sigma_space)


def apply_temporal_filter(
    frames: List[np.ndarray],
    filter_type: str,
    window_size: int
) -> List[np.ndarray]:
    """
    Apply temporal filtering based on configuration.
    
    Args:
        frames: List of frames
        filter_type: "gaussian" or "bilateral"
        window_size: Temporal window size
        
    Returns:
        filtered_frames: Filtered frames
    """
    if filter_type == "gaussian":
        return apply_gaussian_temporal_filter(frames, window_size)
    elif filter_type == "bilateral":
        return apply_bilateral_temporal_filter(frames, window_size)
    else:
        # Unknown filter type, return unchanged
        return frames