"""
Frame preprocessing functionality for Motion-Field Covert Communication System.

This module provides functions to normalize and standardize video frames to
consistent size, format, and data type.
"""

from typing import List
import numpy as np
import cv2


# Type alias for Frame
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]


def normalize_frames(frames: List[Frame]) -> List[Frame]:
    """
    Normalize frames to consistent size/format.
    
    This function ensures all frames have:
    - The same dimensions (height, width)
    - RGB color format
    - uint8 data type
    - Values in range [0, 255]
    
    The target dimensions are determined from the first frame.
    All subsequent frames are resized to match.
    
    Args:
        frames: List of frames (possibly different sizes)
        
    Returns:
        normalized_frames: All same size, RGB, uint8
    """
    if len(frames) == 0:
        return []
    
    # Determine target dimensions from first frame
    first_frame = frames[0]
    
    # Validate first frame
    if first_frame.ndim != 3:
        raise ValueError(
            f"Invalid first frame dimensions: {first_frame.ndim}. Expected 3 (H, W, C)."
        )
    
    target_height, target_width, target_channels = first_frame.shape
    
    # Ensure first frame is RGB (3 channels)
    if target_channels != 3:
        raise ValueError(
            f"Invalid number of channels in first frame: {target_channels}. Expected 3."
        )
    
    normalized_frames: List[Frame] = []
    
    for idx, frame in enumerate(frames):
        # Validate frame structure
        if frame.ndim != 3:
            raise ValueError(
                f"Invalid frame dimensions at index {idx}: {frame.ndim}. Expected 3."
            )
        
        height, width, channels = frame.shape
        
        # Handle grayscale images (convert to RGB)
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif channels == 4:
            # Handle RGBA (drop alpha channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif channels != 3:
            raise ValueError(
                f"Unsupported number of channels at index {idx}: {channels}. "
                f"Expected 1, 3, or 4."
            )
        
        # Resize if dimensions don't match target
        if height != target_height or width != target_width:
            frame = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Convert to uint8 if necessary
        if frame.dtype != np.uint8:
            # Normalize to [0, 255] range if needed
            if frame.dtype in [np.float32, np.float64]:
                # Assume float values are in [0, 1] range
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    # Values already in [0, 255] range
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                # For other integer types, clip and convert
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Ensure values are in valid range [0, 255]
        frame = np.clip(frame, 0, 255)
        
        normalized_frames.append(frame)
    
    return normalized_frames