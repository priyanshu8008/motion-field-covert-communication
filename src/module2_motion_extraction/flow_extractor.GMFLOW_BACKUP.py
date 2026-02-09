"""
Optical Flow Extractor

Main interface for dense optical flow extraction using RAFT.
Implements the OpticalFlowExtractor class as specified in MODULE_INTERFACES.md.
"""

import numpy as np
from typing import List, Dict
from .raft_wrapper import RAFTWrapper
from . import flow_utils
from .exceptions import FlowExtractionError


# Type aliases (from MODULE_INTERFACES.md)
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]
FlowField = np.ndarray  # Shape: (H, W, 2), dtype: float32


class OpticalFlowExtractor:
    """
    Dense optical flow extraction using RAFT.
    
    This class provides the main interface for extracting optical flow
    between video frames. It uses the RAFT (Recurrent All-Pairs Field
    Transforms) model for state-of-the-art flow estimation.
    
    All methods perform inference only (no training).
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        """
        Initialize RAFT model.
        
        Args:
            model_path: Path to RAFT pretrained weights (.pth file)
            device: Device for inference ("cuda" or "cpu")
        
        Raises:
            FileNotFoundError: If model weights don't exist
            FlowExtractionError: If model initialization fails
        """
        try:
            self.raft = RAFTWrapper(
                model_path=model_path,
                device=device,
                iters=20,  # Default refinement iterations
                mixed_precision=False
            )
            self.device = device
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model weights not found: {e}")
        except Exception as e:
            raise FlowExtractionError(f"Failed to initialize RAFT model: {e}")
    
    def extract_flow(
        self,
        frame1: Frame,
        frame2: Frame
    ) -> FlowField:
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: First frame (H, W, 3), uint8, RGB
            frame2: Second frame (H, W, 3), uint8, RGB
        
        Returns:
            flow: Flow field (H, W, 2), float32
                  flow[:, :, 0] = horizontal displacement (dx)
                  flow[:, :, 1] = vertical displacement (dy)
        
        Raises:
            ValueError: If frames have different shapes
            FlowExtractionError: If flow extraction fails
        """
        # Validate input shapes
        if frame1.shape != frame2.shape:
            raise ValueError(
                f"Frame shape mismatch: frame1 {frame1.shape} vs frame2 {frame2.shape}"
            )
        
        if len(frame1.shape) != 3 or frame1.shape[2] != 3:
            raise ValueError(
                f"Expected frames with shape (H, W, 3), got {frame1.shape}"
            )
        
        if frame1.dtype != np.uint8 or frame2.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 dtype, got frame1: {frame1.dtype}, frame2: {frame2.dtype}"
            )
        
        try:
            # Preprocess frames to tensors
            img1_tensor, img2_tensor = self.raft.preprocess_frames(frame1, frame2)
            
            # Run RAFT inference
            flow_tensor = self.raft.infer(img1_tensor, img2_tensor)
            
            # Convert to numpy flow field
            flow = self.raft.postprocess_flow(flow_tensor)
            
            # Verify output shape and dtype
            assert flow.shape[:2] == frame1.shape[:2], "Flow spatial dimensions mismatch"
            assert flow.shape[2] == 2, "Flow must have 2 channels (dx, dy)"
            assert flow.dtype == np.float32, "Flow must be float32"
            
            return flow
            
        except Exception as e:
            raise FlowExtractionError(f"Flow extraction failed: {e}")
    
    def batch_extract(
        self,
        frames: List[Frame]
    ) -> List[FlowField]:
        """
        Extract flow for all consecutive frame pairs.
        
        For N frames, this computes N-1 flow fields:
            flow[0] = flow(frames[0] -> frames[1])
            flow[1] = flow(frames[1] -> frames[2])
            ...
            flow[N-2] = flow(frames[N-2] -> frames[N-1])
        
        Args:
            frames: List of N frames, each (H, W, 3), uint8
        
        Returns:
            flows: List of (N-1) flow fields, each (H, W, 2), float32
        
        Raises:
            ValueError: If frames list is empty or has less than 2 frames
            FlowExtractionError: If any flow extraction fails
        """
        if len(frames) < 2:
            raise ValueError(
                f"Need at least 2 frames for flow extraction, got {len(frames)}"
            )
        
        flows = []
        
        for i in range(len(frames) - 1):
            try:
                flow = self.extract_flow(frames[i], frames[i + 1])
                flows.append(flow)
            except Exception as e:
                raise FlowExtractionError(
                    f"Failed to extract flow between frames {i} and {i+1}: {e}"
                )
        
        return flows
    
    def visualize_flow(
        self,
        flow: FlowField
    ) -> Frame:
        """
        Visualize flow as HSV color-coded image.
        
        Color encoding:
            - Hue: Flow direction (angle)
            - Saturation: Flow magnitude
            - Value: Maximum brightness
        
        Args:
            flow: Flow field (H, W, 2), float32
        
        Returns:
            visualization: RGB image (H, W, 3), uint8, range [0, 255]
        
        Raises:
            ValueError: If flow has incorrect shape or dtype
        """
        if not isinstance(flow, np.ndarray):
            raise ValueError(f"Flow must be numpy array, got {type(flow)}")
        
        if flow.dtype != np.float32:
            raise ValueError(f"Flow must be float32, got {flow.dtype}")
        
        if len(flow.shape) != 3 or flow.shape[2] != 2:
            raise ValueError(
                f"Flow must have shape (H, W, 2), got {flow.shape}"
            )
        
        return flow_utils.visualize_flow(flow)
    
    def compute_statistics(
        self,
        flow: FlowField
    ) -> Dict[str, float]:
        """
        Compute flow statistics.
        
        Args:
            flow: Flow field (H, W, 2), float32
        
        Returns:
            stats: Dictionary containing:
                - 'mean_magnitude': Mean flow magnitude
                - 'max_magnitude': Maximum flow magnitude
                - 'directional_entropy': Entropy of flow directions
                - Additional statistics (median, std dev, etc.)
        
        Raises:
            ValueError: If flow has incorrect shape or dtype
        """
        if not isinstance(flow, np.ndarray):
            raise ValueError(f"Flow must be numpy array, got {type(flow)}")
        
        if flow.dtype != np.float32:
            raise ValueError(f"Flow must be float32, got {flow.dtype}")
        
        if len(flow.shape) != 3 or flow.shape[2] != 2:
            raise ValueError(
                f"Flow must have shape (H, W, 2), got {flow.shape}"
            )
        
        return flow_utils.compute_flow_statistics(flow)