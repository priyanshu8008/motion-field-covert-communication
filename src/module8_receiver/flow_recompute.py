"""
Flow Recomputation Wrapper

Wraps Module 2 (OpticalFlowExtractor) to recompute optical flow from stego frames.
This is a thin wrapper that ensures consistent flow extraction on the receiver side.
"""

import numpy as np
from typing import List, Optional

# Type aliases
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8
FlowField = np.ndarray  # Shape: (H, W, 2), dtype: float32


class FlowRecomputeWrapper:
    """
    Wrapper for optical flow recomputation on receiver side.
    
    Uses Module 2's OpticalFlowExtractor to extract flow from stego frames
    with the same preprocessing and parameters as the encoder.
    """
    
    def __init__(self, config: dict):
        """
        Initialize flow extractor wrapper.
        
        Args:
            config: Configuration dictionary containing:
                - optical_flow.model: Flow model name (e.g., "raft")
                - optical_flow.weights_path: Path to model weights
                - optical_flow.device: "cuda" or "cpu"
                - optical_flow.preprocessing: Preprocessing parameters
        """
        self.config = config
        self.flow_config = config.get('optical_flow', {})
        self.preprocessing_config = self.flow_config.get('preprocessing', {})
        
        # Extract preprocessing parameters
        self.normalize = self.preprocessing_config.get('normalize', True)
        self.max_flow_magnitude = self.preprocessing_config.get('max_flow_magnitude', 100.0)
        
        # Note: Actual Module 2 initialization would happen here
        # For now, this is a stub that will be replaced when Module 2 is available
        self._flow_extractor = None
        
    def extract_flow(self, frame1: Frame, frame2: Frame) -> FlowField:
        """
        Extract optical flow between two consecutive frames.
        
        This method applies the same preprocessing as the encoder to ensure
        deterministic flow extraction.
        
        Args:
            frame1: First frame (H, W, 3), dtype: uint8
            frame2: Second frame (H, W, 3), dtype: uint8
            
        Returns:
            flow: Flow field (H, W, 2), dtype: float32
                  flow[:,:,0] = horizontal displacement (dx)
                  flow[:,:,1] = vertical displacement (dy)
                  
        Raises:
            ValueError: If frames have different shapes or invalid format
        """
        # Validate inputs
        if frame1.shape != frame2.shape:
            raise ValueError(
                f"Frame shape mismatch: {frame1.shape} vs {frame2.shape}"
            )
        
        if frame1.ndim != 3 or frame1.shape[2] != 3:
            raise ValueError(
                f"Expected RGB frames with shape (H, W, 3), got {frame1.shape}"
            )
        
        # TODO: Call Module 2's extract_flow method
        # For now, return a placeholder (zero flow)
        # In actual implementation, this would be:
        # flow = self._flow_extractor.extract_flow(frame1, frame2)
        
        H, W = frame1.shape[:2]
        flow = np.zeros((H, W, 2), dtype=np.float32)
        
        # Apply preprocessing (same as encoder)
        flow = self._preprocess_flow(flow)
        
        return flow
    
    def batch_extract(self, frames: List[Frame]) -> List[FlowField]:
        """
        Extract flow for all consecutive frame pairs.
        
        Args:
            frames: List of N frames
            
        Returns:
            flows: List of (N-1) flow fields
            
        Raises:
            ValueError: If frames list is empty or has < 2 frames
        """
        if len(frames) < 2:
            raise ValueError(
                f"Need at least 2 frames for flow extraction, got {len(frames)}"
            )
        
        flows = []
        for i in range(len(frames) - 1):
            flow = self.extract_flow(frames[i], frames[i + 1])
            flows.append(flow)
        
        return flows
    
    def _preprocess_flow(self, flow: FlowField) -> FlowField:
        """
        Apply preprocessing to flow field (same as encoder).
        
        Args:
            flow: Raw flow field
            
        Returns:
            preprocessed_flow: Flow after preprocessing
        """
        if self.normalize:
            # Apply max magnitude clipping (remove outliers)
            magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            mask = magnitude > self.max_flow_magnitude
            
            if np.any(mask):
                # Scale down vectors exceeding max magnitude
                scale_factor = self.max_flow_magnitude / np.maximum(magnitude, 1e-8)
                flow[:, :, 0] = np.where(mask, flow[:, :, 0] * scale_factor, flow[:, :, 0])
                flow[:, :, 1] = np.where(mask, flow[:, :, 1] * scale_factor, flow[:, :, 1])
        
        return flow