"""
Region Selector

Deterministically selects regions/pixels for bit extraction.
Uses the same logic as encoder to ensure synchronization.
"""

import numpy as np
from typing import List, Tuple

# Type alias
FlowField = np.ndarray  # Shape: (H, W, 2), dtype: float32
EmbeddingMap = np.ndarray  # Shape: (H, W), dtype: bool


class RegionSelector:
    """
    Selects regions for bit extraction in deterministic manner.
    
    Works in conjunction with CapacityEstimator to produce the same
    pixel selection as the encoder.
    """
    
    def __init__(self, config: dict):
        """
        Initialize region selector.
        
        Args:
            config: Configuration dictionary (same as CapacityEstimator)
        """
        self.config = config
        self.modulation_config = config.get('modulation', {})
        self.selection_config = self.modulation_config.get('selection', {})
        
        # Extract parameters
        self.motion_threshold = self.selection_config.get('motion_threshold', 1.0)
        self.spatial_distribution = self.selection_config.get('spatial_distribution', 'uniform')
    
    def get_extraction_vectors(
        self,
        flow: FlowField,
        embedding_map: EmbeddingMap
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get motion vectors to extract bits from.
        
        Args:
            flow: Flow field (H, W, 2)
            embedding_map: Boolean mask indicating which pixels were used (H, W)
            
        Returns:
            vectors: Motion vectors at selected pixels (N, 2)
            positions: Pixel coordinates of selected vectors (N, 2) as (row, col)
        """
        # Get positions where embedding_map is True
        positions = np.argwhere(embedding_map)  # Shape: (N, 2)
        
        if len(positions) == 0:
            # No pixels selected
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=int)
        
        # Extract vectors at these positions
        vectors = flow[positions[:, 0], positions[:, 1]]  # Shape: (N, 2)
        
        return vectors, positions
    
    def get_extraction_order(
        self,
        positions: np.ndarray,
        flow: FlowField
    ) -> np.ndarray:
        """
        Determine the order in which to extract bits from selected pixels.
        
        This MUST match the encoder's embedding order for correct synchronization.
        The order is determined by sorting pixels in raster scan order (top-to-bottom,
        left-to-right) to ensure determinism.
        
        Args:
            positions: Pixel coordinates (N, 2) as (row, col)
            flow: Flow field (H, W, 2) - used for magnitude-based ordering if needed
            
        Returns:
            indices: Permutation indices defining extraction order
        """
        if len(positions) == 0:
            return np.array([], dtype=int)
        
        # Sort in raster scan order: row-major (top-to-bottom, left-to-right)
        # This ensures deterministic ordering that matches encoder
        
        # Convert positions to linear indices
        H, W = flow.shape[:2]
        linear_indices = positions[:, 0] * W + positions[:, 1]
        
        # Sort by linear index
        sorted_order = np.argsort(linear_indices)
        
        return sorted_order