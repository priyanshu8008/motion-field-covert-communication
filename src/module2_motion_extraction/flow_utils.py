"""
Optical Flow Utilities

Utility functions for optical flow visualization and statistical analysis.
"""

import numpy as np
import cv2
from typing import Dict


def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """
    Visualize optical flow using HSV color coding.
    
    Color encoding:
        - Hue: Flow direction (angle)
        - Saturation: Flow magnitude (normalized)
        - Value: Maximum brightness (255)
    
    Args:
        flow: Flow field (H, W, 2), float32
              flow[:, :, 0] = horizontal displacement (dx)
              flow[:, :, 1] = vertical displacement (dy)
    
    Returns:
        visualization: RGB image (H, W, 3), uint8, range [0, 255]
    
    Raises:
        ValueError: If flow has incorrect shape
    """
    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise ValueError(
            f"Expected flow shape (H, W, 2), got {flow.shape}"
        )
    
    h, w = flow.shape[:2]
    
    # Compute flow magnitude and angle
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Hue: encode direction (angle in radians to degrees, normalized to [0, 180])
    # OpenCV hue range is [0, 180] for uint8
    hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    
    # Saturation: encode magnitude (normalized to [0, 255])
    # Normalize by maximum magnitude (with small epsilon to avoid division by zero)
    max_mag = np.max(magnitude)
    if max_mag > 1e-6:
        hsv[:, :, 1] = np.clip(magnitude / max_mag * 255, 0, 255).astype(np.uint8)
    else:
        hsv[:, :, 1] = 0
    
    # Value: constant maximum brightness
    hsv[:, :, 2] = 255
    
    # Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def compute_flow_statistics(flow: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical properties of optical flow field.
    
    Args:
        flow: Flow field (H, W, 2), float32
    
    Returns:
        stats: Dictionary containing:
            - 'mean_magnitude': Mean flow magnitude
            - 'max_magnitude': Maximum flow magnitude
            - 'directional_entropy': Entropy of flow directions
            - 'median_magnitude': Median flow magnitude
            - 'std_magnitude': Standard deviation of magnitude
            - 'mean_dx': Mean horizontal displacement
            - 'mean_dy': Mean vertical displacement
            - 'std_dx': Std dev of horizontal displacement
            - 'std_dy': Std dev of vertical displacement
    
    Raises:
        ValueError: If flow has incorrect shape
    """
    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise ValueError(
            f"Expected flow shape (H, W, 2), got {flow.shape}"
        )
    
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    
    # Compute magnitude
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Compute angle (direction)
    angle = np.arctan2(dy, dx)
    
    # Basic magnitude statistics
    mean_magnitude = float(np.mean(magnitude))
    max_magnitude = float(np.max(magnitude))
    median_magnitude = float(np.median(magnitude))
    std_magnitude = float(np.std(magnitude))
    
    # Displacement statistics
    mean_dx = float(np.mean(dx))
    mean_dy = float(np.mean(dy))
    std_dx = float(np.std(dx))
    std_dy = float(np.std(dy))
    
    # Directional entropy
    directional_entropy = _compute_directional_entropy(angle)
    
    stats = {
        'mean_magnitude': mean_magnitude,
        'max_magnitude': max_magnitude,
        'median_magnitude': median_magnitude,
        'std_magnitude': std_magnitude,
        'mean_dx': mean_dx,
        'mean_dy': mean_dy,
        'std_dx': std_dx,
        'std_dy': std_dy,
        'directional_entropy': directional_entropy,
    }
    
    return stats


def _compute_directional_entropy(angles: np.ndarray, num_bins: int = 16) -> float:
    """
    Compute entropy of flow direction distribution.
    
    Args:
        angles: Flow angles in radians (H, W)
        num_bins: Number of bins for histogram
    
    Returns:
        entropy: Shannon entropy of direction distribution
    """
    # Flatten angles
    angles_flat = angles.flatten()
    
    # Create histogram of directions
    # Angles are in range [-π, π], normalize to [0, 2π]
    angles_normalized = (angles_flat + np.pi) % (2 * np.pi)
    
    hist, _ = np.histogram(angles_normalized, bins=num_bins, range=(0, 2 * np.pi))
    
    # Normalize histogram to get probabilities
    hist = hist.astype(np.float64)
    hist_sum = np.sum(hist)
    
    if hist_sum > 0:
        prob = hist / hist_sum
        
        # Compute Shannon entropy: H = -Σ p(x) log2(p(x))
        # Filter out zero probabilities to avoid log(0)
        prob_nonzero = prob[prob > 0]
        entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero))
    else:
        entropy = 0.0
    
    return float(entropy)


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of flow vectors.
    
    Args:
        flow: Flow field (H, W, 2)
    
    Returns:
        magnitude: Flow magnitude (H, W)
    """
    return np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)


def flow_angle(flow: np.ndarray) -> np.ndarray:
    """
    Compute angle of flow vectors.
    
    Args:
        flow: Flow field (H, W, 2)
    
    Returns:
        angle: Flow angle in radians (H, W), range [-π, π]
    """
    return np.arctan2(flow[:, :, 1], flow[:, :, 0])


def apply_flow_colorwheel(flow: np.ndarray, max_flow: float = None) -> np.ndarray:
    """
    Apply Middlebury color wheel to flow field.
    
    This is an alternative visualization method using the standard
    Middlebury flow color coding.
    
    Args:
        flow: Flow field (H, W, 2)
        max_flow: Maximum flow for normalization (None = auto)
    
    Returns:
        rgb: RGB visualization (H, W, 3), uint8
    """
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    
    # Compute magnitude and angle
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Auto-scale if not provided
    if max_flow is None:
        max_flow = np.max(magnitude)
        if max_flow < 1e-6:
            max_flow = 1.0
    
    # Normalize magnitude
    magnitude_norm = np.clip(magnitude / max_flow, 0, 1)
    
    # Create color wheel visualization
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    
    # Hue from angle
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)
    
    # Saturation from magnitude
    hsv[:, :, 1] = magnitude_norm
    
    # Value constant
    hsv[:, :, 2] = 1.0
    
    # Convert to RGB (OpenCV requires uint8 for HSV)
    hsv_uint8 = (hsv * 255).astype(np.uint8)
    hsv_uint8[:, :, 0] = (hsv[:, :, 0] * 180).astype(np.uint8)  # Hue is [0, 180]
    
    rgb = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)
    
    return rgb