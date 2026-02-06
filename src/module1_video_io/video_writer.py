"""
Video writing functionality for Motion-Field Covert Communication System.

This module provides functions to write frames to video files with specified
codec and quality parameters.
"""

from typing import List
import numpy as np
import cv2
import os


# Type alias for Frame
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]


def write_video(
    frames: List[Frame],
    path: str,
    fps: int,
    codec: str = "libx264",
    crf: int = 18
) -> None:
    """
    Write frames to video file.
    
    Args:
        frames: List of RGB frames
        path: Output video path
        fps: Frame rate
        codec: Video codec
        crf: Constant rate factor (quality)
        
    Raises:
        ValueError: If frames list is empty or inconsistent shapes
        IOError: If write fails
    """
    # Validate inputs
    if len(frames) == 0:
        raise ValueError("Frames list is empty")
    
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}. Must be positive.")
    
    if crf < 0 or crf > 51:
        raise ValueError(f"Invalid CRF: {crf}. Must be in range [0, 51].")
    
    # Validate frame consistency
    first_frame = frames[0]
    if first_frame.ndim != 3:
        raise ValueError(f"Invalid frame dimensions: {first_frame.ndim}. Expected 3 (H, W, 3).")
    
    if first_frame.shape[2] != 3:
        raise ValueError(f"Invalid number of channels: {first_frame.shape[2]}. Expected 3 (RGB).")
    
    height, width = first_frame.shape[:2]
    
    for idx, frame in enumerate(frames):
        if frame.shape != first_frame.shape:
            raise ValueError(
                f"Inconsistent frame shape at index {idx}: {frame.shape}. "
                f"Expected {first_frame.shape}."
            )
        
        if frame.dtype != np.uint8:
            raise ValueError(
                f"Invalid frame dtype at index {idx}: {frame.dtype}. Expected uint8."
            )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Map codec string to fourcc
    # OpenCV uses VideoWriter with fourcc codes
    # For libx264, we use mp4v fourcc and rely on ffmpeg for actual encoding
    codec_map = {
        "libx264": "mp4v",
        "libx265": "mp4v",
        "h264": "H264",
        "h265": "HEVC",
        "mpeg4": "mp4v",
        "xvid": "XVID",
    }
    
    fourcc_str = codec_map.get(codec.lower(), "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    
    # Create VideoWriter
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise IOError(f"Failed to create video writer for: {path}")
    
    # Write frames
    try:
        for idx, frame in enumerate(frames):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    
    except Exception as e:
        writer.release()
        # Clean up partial file
        if os.path.exists(path):
            os.remove(path)
        raise IOError(f"Error writing video: {str(e)}")
    
    finally:
        writer.release()
    
    # Verify file was created
    if not os.path.exists(path):
        raise IOError(f"Video file was not created: {path}")
    
    # For high-quality encoding with CRF control, we would need to use
    # ffmpeg directly. OpenCV's VideoWriter doesn't expose CRF parameter.
    # If exact CRF control is needed, this should be done via subprocess
    # calling ffmpeg with explicit parameters.
    # For now, we use OpenCV's default quality settings.