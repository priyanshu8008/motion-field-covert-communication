"""
Video quality metrics computation.

Computes PSNR, SSIM, and MSE for evaluating reconstruction quality.
These metrics are for analysis/debugging only - not required by downstream modules.
"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Video quality metrics."""
    psnr: float  # Peak Signal-to-Noise Ratio (dB)
    ssim: float  # Structural Similarity Index
    mse: float   # Mean Squared Error


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two frames.
    
    Args:
        original: Original frame (H, W, 3) uint8
        reconstructed: Reconstructed frame (H, W, 3) uint8
        
    Returns:
        mse: Mean squared error (lower is better)
        
    Formula:
        MSE = mean((original - reconstructed)^2)
    """
    # Convert to float
    orig_float = original.astype(np.float64)
    recon_float = reconstructed.astype(np.float64)
    
    # Compute squared error
    squared_error = (orig_float - recon_float) ** 2
    
    # Mean over all pixels and channels
    mse = np.mean(squared_error)
    
    return float(mse)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        original: Original frame (H, W, 3) uint8
        reconstructed: Reconstructed frame (H, W, 3) uint8
        
    Returns:
        psnr: PSNR in dB (higher is better, typically 20-50 dB)
        
    Formula:
        PSNR = 10 * log10(MAX^2 / MSE)
        where MAX = 255 for uint8 images
        
    Special cases:
        - If MSE = 0 (identical frames): returns inf
        - If MSE is very small: clamps to avoid numerical issues
    """
    mse = compute_mse(original, reconstructed)
    
    # Handle perfect reconstruction
    if mse < 1e-10:
        return float('inf')
    
    # Compute PSNR
    # MAX = 255 for uint8 images
    max_pixel_value = 255.0
    psnr = 10.0 * np.log10((max_pixel_value ** 2) / mse)
    
    return float(psnr)


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        original: Original frame (H, W, 3) uint8
        reconstructed: Reconstructed frame (H, W, 3) uint8
        
    Returns:
        ssim: SSIM value in [0, 1] (higher is better, 1 = identical)
        
    Notes:
        - Computes SSIM per channel, then averages
        - Uses simplified implementation without skimage dependency
        - Constants: C1=(0.01*255)^2, C2=(0.03*255)^2
        
    Formula:
        SSIM(x,y) = (2*μx*μy + C1)(2*σxy + C2) / ((μx^2 + μy^2 + C1)(σx^2 + σy^2 + C2))
    """
    # Convert to float
    orig_float = original.astype(np.float64)
    recon_float = reconstructed.astype(np.float64)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Compute per channel
    ssim_channels = []
    
    for channel in range(3):  # R, G, B
        x = orig_float[:, :, channel]
        y = recon_float[:, :, channel]
        
        # Compute means
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        
        # Compute variances
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        
        # Compute covariance
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        
        # Compute SSIM
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        
        ssim_channel = numerator / denominator
        ssim_channels.append(ssim_channel)
    
    # Average across channels
    ssim = np.mean(ssim_channels)
    
    return float(ssim)


def compute_video_quality(
    original_frames: List[np.ndarray],
    reconstructed_frames: List[np.ndarray]
) -> QualityMetrics:
    """
    Compute aggregated quality metrics for video sequence.
    
    Args:
        original_frames: List of original frames (N, H, W, 3)
        reconstructed_frames: List of reconstructed frames (N, H, W, 3)
        
    Returns:
        metrics: Aggregated QualityMetrics (mean across all frames)
        
    Notes:
        - Computes metrics per frame, then averages
        - Deterministic
        - Used for evaluation/debugging only
    """
    if len(original_frames) != len(reconstructed_frames):
        raise ValueError(
            f"Frame count mismatch: {len(original_frames)} vs {len(reconstructed_frames)}"
        )
    
    if len(original_frames) == 0:
        return QualityMetrics(psnr=0.0, ssim=0.0, mse=0.0)
    
    # Compute metrics for each frame
    psnr_values = []
    ssim_values = []
    mse_values = []
    
    for orig, recon in zip(original_frames, reconstructed_frames):
        # Skip if shapes don't match
        if orig.shape != recon.shape:
            continue
        
        mse = compute_mse(orig, recon)
        psnr = compute_psnr(orig, recon)
        ssim = compute_ssim(orig, recon)
        
        # Filter out inf values for PSNR averaging
        if not np.isinf(psnr):
            psnr_values.append(psnr)
        ssim_values.append(ssim)
        mse_values.append(mse)
    
    # Aggregate (mean)
    avg_psnr = np.mean(psnr_values) if psnr_values else float('inf')
    avg_ssim = np.mean(ssim_values) if ssim_values else 1.0
    avg_mse = np.mean(mse_values) if mse_values else 0.0
    
    return QualityMetrics(
        psnr=float(avg_psnr),
        ssim=float(avg_ssim),
        mse=float(avg_mse)
    )