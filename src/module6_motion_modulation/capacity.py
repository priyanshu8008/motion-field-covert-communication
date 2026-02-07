"""
Capacity estimation for motion-field embedding.

Determines how many bits can be embedded in a flow field based on
motion characteristics and configuration constraints.
"""

import numpy as np
from typing import Dict


def compute_capacity(
    flow: np.ndarray,
    motion_threshold: float,
    use_high_motion_regions: bool = True,
    spatial_distribution: str = "uniform"
) -> int:
    """
    Estimate embedding capacity for a flow field.
    
    Capacity is determined by:
    1. Number of motion vectors exceeding motion_threshold
    2. Spatial distribution strategy (uniform, random, adaptive)
    3. High-motion region preference (if enabled)
    
    Args:
        flow: Flow field (H, W, 2)
        motion_threshold: Minimum motion magnitude to embed (pixels)
        use_high_motion_regions: Prefer high-motion areas
        spatial_distribution: "uniform", "random", or "adaptive"
        
    Returns:
        capacity: Number of bits that can be embedded
        
    Notes:
        - Capacity is dynamic per frame (not fixed allocation)
        - Static frames may have capacity = 0
        - Deterministic for same inputs
    """
    H, W, _ = flow.shape
    
    # Compute motion magnitudes
    magnitudes = np.linalg.norm(flow, axis=2)  # (H, W)
    
    # Find vectors exceeding threshold
    sufficient_motion = magnitudes >= motion_threshold
    
    # Count available locations
    capacity = np.sum(sufficient_motion)
    
    return int(capacity)


def create_embedding_map(
    flow: np.ndarray,
    num_bits: int,
    motion_threshold: float,
    use_high_motion_regions: bool = True,
    spatial_distribution: str = "uniform"
) -> np.ndarray:
    """
    Create embedding location map for a flow field.
    
    Selects which motion vectors to use for embedding based on:
    - Motion magnitude (must exceed threshold)
    - Spatial distribution strategy
    - High-motion preference
    
    Args:
        flow: Flow field (H, W, 2)
        num_bits: Number of bits to embed
        motion_threshold: Minimum motion magnitude (pixels)
        use_high_motion_regions: Prefer high-motion areas
        spatial_distribution: Selection strategy
        
    Returns:
        embedding_map: Boolean array (H, W) indicating embedding locations
        
    Notes:
        - deterministic for same inputs
        - Returns map with exactly num_bits True values (if capacity allows)
        - If num_bits > capacity, uses all available locations
    """
    H, W, _ = flow.shape
    
    # Compute motion magnitudes
    magnitudes = np.linalg.norm(flow, axis=2)  # (H, W)
    
    # Find vectors exceeding threshold
    sufficient_motion = magnitudes >= motion_threshold
    available_locations = np.argwhere(sufficient_motion)  # (N, 2)
    
    # Check capacity
    capacity = len(available_locations)
    if capacity == 0:
        return np.zeros((H, W), dtype=bool)
    
    # Limit to requested bits
    n_embed = min(num_bits, capacity)
    
    # Select locations based on strategy
    if spatial_distribution == "uniform":
        # Simple: take first N in raster-scan order
        selected_locations = available_locations[:n_embed]
        
    elif spatial_distribution == "adaptive" or use_high_motion_regions:
        # Sort by magnitude (descending) and take top N
        location_magnitudes = magnitudes[available_locations[:, 0], 
                                         available_locations[:, 1]]
        sorted_indices = np.argsort(location_magnitudes)[::-1]  # Descending
        selected_locations = available_locations[sorted_indices[:n_embed]]
        
    else:
        # Default: uniform
        selected_locations = available_locations[:n_embed]
    
    # Create map
    embedding_map = np.zeros((H, W), dtype=bool)
    embedding_map[selected_locations[:, 0], selected_locations[:, 1]] = True
    
    return embedding_map


def estimate_embedding_statistics(
    original_flow: np.ndarray,
    modified_flow: np.ndarray,
    embedding_map: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistics about the embedding operation.
    
    Args:
        original_flow: Original flow field (H, W, 2)
        modified_flow: Modified flow field (H, W, 2)
        embedding_map: Where bits were embedded (H, W)
        
    Returns:
        stats: Dictionary with:
            - num_vectors_modified: Number of modified vectors
            - avg_perturbation: Average L2 norm of perturbation
            - max_perturbation: Maximum L2 norm
            - avg_magnitude_change: Average magnitude change
    """
    # Compute perturbations only at embedding locations
    embed_indices = np.argwhere(embedding_map)
    
    if len(embed_indices) == 0:
        return {
            'num_vectors_modified': 0,
            'avg_perturbation': 0.0,
            'max_perturbation': 0.0,
            'avg_magnitude_change': 0.0
        }
    
    # Extract vectors
    orig_vectors = original_flow[embed_indices[:, 0], embed_indices[:, 1]]
    mod_vectors = modified_flow[embed_indices[:, 0], embed_indices[:, 1]]
    
    # Compute perturbations
    perturbations = np.linalg.norm(mod_vectors - orig_vectors, axis=1)
    
    # Compute magnitude changes
    orig_mags = np.linalg.norm(orig_vectors, axis=1)
    mod_mags = np.linalg.norm(mod_vectors, axis=1)
    mag_changes = np.abs(mod_mags - orig_mags)
    
    return {
        'num_vectors_modified': len(embed_indices),
        'avg_perturbation': float(np.mean(perturbations)),
        'max_perturbation': float(np.max(perturbations)),
        'avg_magnitude_change': float(np.mean(mag_changes))
    }