"""
Video loading functionality for Motion-Field Covert Communication System.

This module provides functions to load video files and extract frames with optional
resampling to target FPS and resolution.
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2
import os
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """Metadata for loaded video"""
    fps: int
    width: int
    height: int
    num_frames: int
    duration: float  # seconds
    codec: str
    pixel_format: str


# Type alias for Frame
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]


def load_video(
    path: str,
    fps: Optional[int] = None,
    resolution: Optional[Tuple[int, int]] = None
) -> Tuple[List[Frame], VideoMetadata]:
    """
    Load video from file.
    
    Args:
        path: Path to video file
        fps: Target frame rate (None = keep original)
        resolution: Target (width, height) (None = keep original)
        
    Returns:
        frames: List of RGB frames
        metadata: Video metadata
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format unsupported
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    # Open video file
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {path}. Format may be unsupported.")
    
    # Extract original metadata
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Validate metadata
    if original_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS detected: {original_fps}")
    
    if original_width <= 0 or original_height <= 0:
        cap.release()
        raise ValueError(f"Invalid resolution detected: {original_width}x{original_height}")
    
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid frame count: {total_frames}")
    
    # Decode codec
    codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    # Determine target parameters
    target_fps = fps if fps is not None else int(original_fps)
    target_width, target_height = resolution if resolution is not None else (original_width, original_height)
    
    # Calculate frame sampling if FPS conversion needed
    frame_interval = 1
    if fps is not None and fps != original_fps:
        frame_interval = original_fps / fps
    
    # Read frames
    frames: List[Frame] = []
    frame_idx = 0
    next_frame_to_capture = 0.0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Sample frames based on target FPS
        if fps is not None:
            if frame_idx >= next_frame_to_capture:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if resolution is not None and (frame_rgb.shape[1] != target_width or frame_rgb.shape[0] != target_height):
                    frame_rgb = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                
                # Ensure uint8 type and valid range
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
                
                frames.append(frame_rgb)
                next_frame_to_capture += frame_interval
        else:
            # No FPS conversion, just capture all frames
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if resolution is not None and (frame_rgb.shape[1] != target_width or frame_rgb.shape[0] != target_height):
                frame_rgb = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Ensure uint8 type and valid range
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
            
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    
    # Validate that frames were loaded
    if len(frames) == 0:
        raise ValueError("No frames could be loaded from video file")
    
    # Create metadata with actual loaded parameters
    metadata = VideoMetadata(
        fps=target_fps,
        width=target_width,
        height=target_height,
        num_frames=len(frames),
        duration=len(frames) / target_fps,
        codec=codec_str,
        pixel_format="rgb24"  # We always convert to RGB
    )
    
    return frames, metadata