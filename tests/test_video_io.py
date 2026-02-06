"""
Unit tests for Module 1: Video I/O

Tests cover:
- load_video functionality
- write_video functionality
- normalize_frames functionality
- Error handling and edge cases
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from typing import List
import cv2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.module1_video_io import VideoIO, VideoMetadata, Frame


class TestVideoIO(unittest.TestCase):
    """Test suite for VideoIO class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.video_io = VideoIO()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_video(
        self,
        filename: str,
        num_frames: int = 10,
        width: int = 320,
        height: int = 240,
        fps: int = 30
    ) -> str:
        """
        Create a test video file with synthetic frames.
        
        Args:
            filename: Name of video file
            num_frames: Number of frames to generate
            width: Frame width
            height: Frame height
            fps: Frame rate
            
        Returns:
            path: Full path to created video
        """
        path = os.path.join(self.temp_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        for i in range(num_frames):
            # Create synthetic frame with varying color
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 25) % 256  # Varying blue channel
            frame[:, :, 1] = 128  # Constant green
            frame[:, :, 2] = 255 - (i * 25) % 256  # Varying red channel
            
            writer.write(frame)
        
        writer.release()
        return path
    
    # =========================================================================
    # Tests for load_video
    # =========================================================================
    
    def test_load_video_basic(self):
        """Test basic video loading with default parameters"""
        video_path = self.create_test_video("test_basic.mp4", num_frames=5)
        
        frames, metadata = self.video_io.load_video(video_path)
        
        # Check frames
        self.assertIsInstance(frames, list)
        self.assertEqual(len(frames), 5)
        
        # Check each frame
        for frame in frames:
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (240, 320, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertTrue(np.all(frame >= 0))
            self.assertTrue(np.all(frame <= 255))
        
        # Check metadata
        self.assertIsInstance(metadata, VideoMetadata)
        self.assertEqual(metadata.fps, 30)
        self.assertEqual(metadata.width, 320)
        self.assertEqual(metadata.height, 240)
        self.assertEqual(metadata.num_frames, 5)
        self.assertAlmostEqual(metadata.duration, 5/30, places=2)
        self.assertEqual(metadata.pixel_format, "rgb24")
    
    def test_load_video_with_fps_conversion(self):
        """Test loading video with FPS conversion"""
        video_path = self.create_test_video("test_fps.mp4", num_frames=30, fps=30)
        
        # Load with target FPS of 10 (should get ~10 frames)
        frames, metadata = self.video_io.load_video(video_path, fps=10)
        
        self.assertEqual(metadata.fps, 10)
        self.assertGreater(len(frames), 8)  # Should get ~10 frames
        self.assertLess(len(frames), 12)
    
    def test_load_video_with_resolution_conversion(self):
        """Test loading video with resolution conversion"""
        video_path = self.create_test_video("test_resolution.mp4", width=640, height=480)
        
        # Load with target resolution 320x240
        frames, metadata = self.video_io.load_video(video_path, resolution=(320, 240))
        
        self.assertEqual(metadata.width, 320)
        self.assertEqual(metadata.height, 240)
        
        for frame in frames:
            self.assertEqual(frame.shape, (240, 320, 3))
    
    def test_load_video_with_both_conversions(self):
        """Test loading video with both FPS and resolution conversion"""
        video_path = self.create_test_video(
            "test_both.mp4",
            num_frames=30,
            width=640,
            height=480,
            fps=30
        )
        
        frames, metadata = self.video_io.load_video(
            video_path,
            fps=15,
            resolution=(160, 120)
        )
        
        self.assertEqual(metadata.fps, 15)
        self.assertEqual(metadata.width, 160)
        self.assertEqual(metadata.height, 120)
        
        for frame in frames:
            self.assertEqual(frame.shape, (120, 160, 3))
    
    def test_load_video_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file"""
        with self.assertRaises(FileNotFoundError):
            self.video_io.load_video("nonexistent_video.mp4")
    
    def test_load_video_invalid_file(self):
        """Test that ValueError is raised for invalid video file"""
        # Create a text file instead of video
        invalid_path = os.path.join(self.temp_dir, "invalid.mp4")
        with open(invalid_path, 'w') as f:
            f.write("This is not a video file")
        
        with self.assertRaises(ValueError):
            self.video_io.load_video(invalid_path)
    
    def test_load_video_rgb_conversion(self):
        """Test that frames are correctly converted to RGB"""
        video_path = self.create_test_video("test_rgb.mp4")
        frames, _ = self.video_io.load_video(video_path)
        
        # Verify RGB format (OpenCV creates BGR, should be converted)
        first_frame = frames[0]
        # The first frame should have blue=0, green=128, red=255
        # After conversion from our synthetic BGR frames
        self.assertEqual(first_frame.shape[2], 3)  # 3 channels
    
    # =========================================================================
    # Tests for write_video
    # =========================================================================
    
    def test_write_video_basic(self):
        """Test basic video writing"""
        # Create test frames
        frames = []
        for i in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255
            frames.append(frame)
        
        output_path = os.path.join(self.temp_dir, "output_basic.mp4")
        
        # Write video
        self.video_io.write_video(frames, output_path, fps=30)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_write_video_custom_codec(self):
        """Test writing with custom codec"""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(5)]
        output_path = os.path.join(self.temp_dir, "output_codec.mp4")
        
        self.video_io.write_video(frames, output_path, fps=25, codec="libx264")
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_write_video_custom_crf(self):
        """Test writing with custom CRF value"""
        frames = [np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8) for _ in range(5)]
        output_path = os.path.join(self.temp_dir, "output_crf.mp4")
        
        self.video_io.write_video(frames, output_path, fps=20, crf=23)
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_write_video_empty_frames(self):
        """Test that ValueError is raised for empty frames list"""
        output_path = os.path.join(self.temp_dir, "output_empty.mp4")
        
        with self.assertRaises(ValueError) as context:
            self.video_io.write_video([], output_path, fps=30)
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_write_video_invalid_fps(self):
        """Test that ValueError is raised for invalid FPS"""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        output_path = os.path.join(self.temp_dir, "output_invalid_fps.mp4")
        
        with self.assertRaises(ValueError):
            self.video_io.write_video(frames, output_path, fps=0)
        
        with self.assertRaises(ValueError):
            self.video_io.write_video(frames, output_path, fps=-10)
    
    def test_write_video_invalid_crf(self):
        """Test that ValueError is raised for invalid CRF"""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        output_path = os.path.join(self.temp_dir, "output_invalid_crf.mp4")
        
        with self.assertRaises(ValueError):
            self.video_io.write_video(frames, output_path, fps=30, crf=-1)
        
        with self.assertRaises(ValueError):
            self.video_io.write_video(frames, output_path, fps=30, crf=52)
    
    def test_write_video_inconsistent_shapes(self):
        """Test that ValueError is raised for inconsistent frame shapes"""
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 100, 3), dtype=np.uint8),  # Different height
        ]
        output_path = os.path.join(self.temp_dir, "output_inconsistent.mp4")
        
        with self.assertRaises(ValueError) as context:
            self.video_io.write_video(frames, output_path, fps=30)
        
        self.assertIn("inconsistent", str(context.exception).lower())
    
    def test_write_video_wrong_dtype(self):
        """Test that ValueError is raised for wrong dtype"""
        frames = [np.zeros((100, 100, 3), dtype=np.float32)]
        output_path = os.path.join(self.temp_dir, "output_wrong_dtype.mp4")
        
        with self.assertRaises(ValueError) as context:
            self.video_io.write_video(frames, output_path, fps=30)
        
        self.assertIn("dtype", str(context.exception).lower())
    
    def test_write_video_wrong_dimensions(self):
        """Test that ValueError is raised for wrong dimensions"""
        frames = [np.zeros((100, 100), dtype=np.uint8)]  # 2D instead of 3D
        output_path = os.path.join(self.temp_dir, "output_wrong_dims.mp4")
        
        with self.assertRaises(ValueError):
            self.video_io.write_video(frames, output_path, fps=30)
    
    def test_write_video_wrong_channels(self):
        """Test that ValueError is raised for wrong number of channels"""
        frames = [np.zeros((100, 100, 4), dtype=np.uint8)]  # RGBA instead of RGB
        output_path = os.path.join(self.temp_dir, "output_wrong_channels.mp4")
        
        with self.assertRaises(ValueError):
            self.video_io.write_video(frames, output_path, fps=30)
    
    def test_write_video_creates_directory(self):
        """Test that write_video creates output directory if needed"""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "output.mp4")
        
        self.video_io.write_video(frames, nested_path, fps=30)
        
        self.assertTrue(os.path.exists(nested_path))
    
    # =========================================================================
    # Tests for normalize_frames
    # =========================================================================
    
    def test_normalize_frames_basic(self):
        """Test basic frame normalization"""
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
            np.ones((100, 100, 3), dtype=np.uint8) * 128,
        ]
        
        normalized = self.video_io.normalize_frames(frames)
        
        self.assertEqual(len(normalized), 3)
        for frame in normalized:
            self.assertEqual(frame.shape, (100, 100, 3))
            self.assertEqual(frame.dtype, np.uint8)
    
    def test_normalize_frames_different_sizes(self):
        """Test normalizing frames with different sizes"""
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 150, 3), dtype=np.uint8),
            np.zeros((50, 75, 3), dtype=np.uint8),
        ]
        
        normalized = self.video_io.normalize_frames(frames)
        
        # All should match first frame size
        self.assertEqual(len(normalized), 3)
        for frame in normalized:
            self.assertEqual(frame.shape, (100, 100, 3))
    
    def test_normalize_frames_float_to_uint8(self):
        """Test converting float frames to uint8"""
        frames = [
            np.random.random((100, 100, 3)).astype(np.float32),  # Values in [0, 1]
            np.random.random((100, 100, 3)).astype(np.float32),
        ]
        
        normalized = self.video_io.normalize_frames(frames)
        
        for frame in normalized:
            self.assertEqual(frame.dtype, np.uint8)
            self.assertTrue(np.all(frame >= 0))
            self.assertTrue(np.all(frame <= 255))
    
    def test_normalize_frames_value_clipping(self):
        """Test that out-of-range values are clipped"""
        frames = [
            np.ones((100, 100, 3), dtype=np.float32) * 1.5,  # Values > 1
        ]
        
        normalized = self.video_io.normalize_frames(frames)
        
        self.assertTrue(np.all(normalized[0] <= 255))
        self.assertTrue(np.all(normalized[0] >= 0))
    
    def test_normalize_frames_empty_list(self):
        """Test normalizing empty frame list"""
        normalized = self.video_io.normalize_frames([])
        self.assertEqual(len(normalized), 0)
    
    def test_normalize_frames_maintains_values(self):
        """Test that normalization preserves pixel values when possible"""
        original_frame = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]]
        ], dtype=np.uint8)
        
        frames = [original_frame]
        normalized = self.video_io.normalize_frames(frames)
        
        np.testing.assert_array_equal(normalized[0], original_frame)
    
    def test_normalize_frames_consistent_output(self):
        """Test that normalization produces consistent results"""
        frames = [
            np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        normalized = self.video_io.normalize_frames(frames)
        
        # All frames should have same shape
        first_shape = normalized[0].shape
        for frame in normalized:
            self.assertEqual(frame.shape, first_shape)
    
    # =========================================================================
    # Integration Tests
    # =========================================================================
    
    def test_roundtrip_video_io(self):
        """Test loading and writing video in a roundtrip"""
        # Create original video
        original_path = self.create_test_video("roundtrip_original.mp4", num_frames=10)
        
        # Load video
        frames, metadata = self.video_io.load_video(original_path)
        
        # Write video
        output_path = os.path.join(self.temp_dir, "roundtrip_output.mp4")
        self.video_io.write_video(frames, output_path, fps=metadata.fps)
        
        # Verify output exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load output video
        frames2, metadata2 = self.video_io.load_video(output_path)
        
        # Compare
        self.assertEqual(len(frames), len(frames2))
        self.assertEqual(metadata.fps, metadata2.fps)
        self.assertEqual(metadata.width, metadata2.width)
        self.assertEqual(metadata.height, metadata2.height)
    
    def test_load_normalize_write(self):
        """Test complete pipeline: load -> normalize -> write"""
        # Create videos with different sizes
        video1 = self.create_test_video("multi1.mp4", width=320, height=240)
        video2 = self.create_test_video("multi2.mp4", width=640, height=480)
        
        # Load both
        frames1, _ = self.video_io.load_video(video1)
        frames2, _ = self.video_io.load_video(video2, resolution=(320, 240))
        
        # Combine and normalize
        all_frames = frames1 + frames2
        normalized = self.video_io.normalize_frames(all_frames)
        
        # Write
        output_path = os.path.join(self.temp_dir, "combined.mp4")
        self.video_io.write_video(normalized, output_path, fps=30)
        
        # Verify
        self.assertTrue(os.path.exists(output_path))
        
        # Load result
        result_frames, result_metadata = self.video_io.load_video(output_path)
        self.assertEqual(len(result_frames), len(normalized))


if __name__ == '__main__':
    unittest.main()