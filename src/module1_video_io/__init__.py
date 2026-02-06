"""
Module 1: Video I/O

This module provides video input/output operations for the Motion-Field Covert
Communication System. It handles loading videos, writing videos, and normalizing
frames to consistent formats.
"""

from typing import List, Optional, Tuple
import numpy as np

from .video_loader import load_video as _load_video, VideoMetadata
from .video_writer import write_video as _write_video
from .preprocessing import normalize_frames as _normalize_frames


# Type alias for Frame
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]


class VideoIO:
    """Video input/output operations"""
    
    def load_video(
        self, 
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
        return _load_video(path, fps, resolution)
    
    def write_video(
        self,
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
        _write_video(frames, path, fps, codec, crf)
    
    def normalize_frames(
        self,
        frames: List[Frame]
    ) -> List[Frame]:
        """
        Normalize frames to consistent size/format.
        
        Args:
            frames: List of frames (possibly different sizes)
            
        Returns:
            normalized_frames: All same size, RGB, uint8
        """
        return _normalize_frames(frames)


# Export public interface
__all__ = [
    'VideoIO',
    'VideoMetadata',
    'Frame'
]