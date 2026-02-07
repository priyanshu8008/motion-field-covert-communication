"""
Constraint enforcement for motion-field modulation.

Implements perceptual and smoothness constraints to ensure modified
motion fields remain plausible and undetectable.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple


def enforce_smoothness_constraint(
    flow: np.ndarray,
    kernel_size: int,
    sigma: float
) -> np.ndarray:
    """
    Enforce spatial smoothness via Gaussian filtering.
    
    Args:
        flow: Flow field (H, W, 2)
        kernel_size: Gaussian kernel size (must be odd)
        sigma: Gaussian standard deviation
        
    Returns:
        smoothed_flow: Spatially smoothed flow field
        
    Notes:
        - Applies Gaussian filter to each component (dx, dy)
        - Reduces high-frequency artifacts
        - Makes embedding less detectable
    """
    
    smoothed_flow = np.zeros_like(flow)
    
    # Apply Gaussian filter to each component
    smoothed_flow[:, :, 0] = gaussian_filter(
        flow[:, :, 0], 
        sigma=sigma,
        truncate=(kernel_size - 1) / (2 * sigma)
    )
    smoothed_flow[:, :, 1] = gaussian_filter(
        flow[:, :, 1], 
        sigma=sigma,
        truncate=(kernel_size - 1) / (2 * sigma)
    )
    
    return smoothed_flow


def enforce_magnitude_bounds(
    original_flow: np.ndarray,
    modified_flow: np.ndarray,
    min_ratio: float,
    max_ratio: float
) -> np.ndarray:
    """
    Enforce magnitude bounds: m' ∈ [min_ratio * m, max_ratio * m].
    
    Args:
        original_flow: Original flow field (H, W, 2)
        modified_flow: Modified flow field (H, W, 2)
        min_ratio: Minimum magnitude ratio (e.g., 0.8)
        max_ratio: Maximum magnitude ratio (e.g., 1.2)
        
    Returns:
        constrained_flow: Flow with magnitude bounds enforced
        
    Notes:
        - Clips magnitudes that violate bounds
        - Preserves direction of modified vectors
    """
    H, W, _ = original_flow.shape
    constrained_flow = modified_flow.copy()
    
    # Compute magnitudes
    orig_mags = np.linalg.norm(original_flow, axis=2)  # (H, W)
    mod_mags = np.linalg.norm(modified_flow, axis=2)    # (H, W)
    
    # Compute bounds
    min_mags = orig_mags * min_ratio
    max_mags = orig_mags * max_ratio
    
    # Find violations
    too_small = mod_mags < min_mags
    too_large = mod_mags > max_mags
    
    # Clip magnitudes (preserve direction)
    for i in range(H):
        for j in range(W):
            if too_small[i, j] and mod_mags[i, j] > 1e-6:
                # Scale up to minimum
                scale = min_mags[i, j] / mod_mags[i, j]
                constrained_flow[i, j] = modified_flow[i, j] * scale
                
            elif too_large[i, j] and mod_mags[i, j] > 1e-6:
                # Scale down to maximum
                scale = max_mags[i, j] / mod_mags[i, j]
                constrained_flow[i, j] = modified_flow[i, j] * scale
    
    return constrained_flow


def enforce_perceptual_limit(
    original_flow: np.ndarray,
    modified_flow: np.ndarray,
    max_perturbation: float
) -> np.ndarray:
    """
    Enforce perceptual limit: ||v' - v||_∞ <= max_perturbation.
    
    Args:
        original_flow: Original flow field (H, W, 2)
        modified_flow: Modified flow field (H, W, 2)
        max_perturbation: Maximum L-infinity norm (pixels)
        
    Returns:
        constrained_flow: Flow with perceptual limit enforced
        
    Notes:
        - Clips perturbations exceeding max_perturbation
        - Uses L-infinity norm (max absolute difference)
    """
    H, W, _ = original_flow.shape
    constrained_flow = modified_flow.copy()
    
    # Compute perturbation vectors
    perturbations = modified_flow - original_flow  # (H, W, 2)
    
    # Compute L-infinity norm (max of |dx|, |dy|)
    l_inf_norms = np.max(np.abs(perturbations), axis=2)  # (H, W)
    
    # Find violations
    violations = l_inf_norms > max_perturbation
    
    # Clip perturbations
    for i in range(H):
        for j in range(W):
            if violations[i, j]:
                # Scale perturbation to max_perturbation
                scale = max_perturbation / l_inf_norms[i, j]
                perturbation = perturbations[i, j] * scale
                constrained_flow[i, j] = original_flow[i, j] + perturbation
    
    return constrained_flow


def requantize_to_qim(
    flow: np.ndarray,
    embedding_map: np.ndarray,
    delta: float,
    original_embedded_flow: np.ndarray = None
) -> np.ndarray:
    """
    Re-quantize flow magnitudes to valid QIM values.
    
    After applying constraints, magnitudes may no longer decode correctly.
    This function restores each embedded vector to its original QIM quantization
    level by extracting the bit from the original embedded flow and re-embedding it.
    
    Args:
        flow: Flow field (H, W, 2) after constraints
        embedding_map: Boolean array (H, W) indicating embedding locations
        delta: Quantization step size
        original_embedded_flow: Flow immediately after QIM (REQUIRED for correct operation)
        
    Returns:
        requantized_flow: Flow with vectors restored to original QIM levels
        
    Notes:
        - Only re-quantizes vectors in embedding_map
        - Extracts bit from original_embedded_flow and re-embeds it
        - Preserves original embedded directions
        - Ensures correct bit extraction after constraints
    """
    requantized_flow = flow.copy()
    H, W, _ = flow.shape
    
    # Get embedding locations
    embed_indices = np.argwhere(embedding_map)
    
    if original_embedded_flow is None:
        # Fallback: snap to nearest level (less reliable)
        for row, col in embed_indices:
            v = flow[row, col]
            m = np.linalg.norm(v)
            
            if m < 1e-6:
                continue
            
            # Find nearest QIM quantization level
            q = np.round(m / delta)
            m0 = q * delta
            m1 = (q + 0.5) * delta
            
            # Snap to nearest
            if abs(m - m0) < abs(m - m1):
                m_new = m0
            else:
                m_new = m1
            
            # Scale to new magnitude
            if m > 1e-6:
                requantized_flow[row, col] = (v / m) * m_new
    else:
        # Correct approach: extract bit from original embedded flow,
        # then re-embed using original direction and magnitude
        from .qim_core import qim_extract_bit
        
        for row, col in embed_indices:
            v_original = original_embedded_flow[row, col]
            
            # Extract the bit that was originally embedded
            bit = qim_extract_bit(v_original, delta, decision_boundary=0.25)
            
            # Get original direction and magnitude
            m_orig = np.linalg.norm(v_original)
            if m_orig > 1e-6:
                direction = v_original / m_orig
            else:
                direction = np.array([1.0, 0.0])
            
            # Compute QIM magnitude for this bit
            q = np.round(m_orig / delta)
            if bit == 0:
                m_qim = q * delta
            else:
                m_qim = (q + 0.5) * delta
            
            # Restore original QIM vector
            requantized_flow[row, col] = direction * m_qim
    
    return requantized_flow


def enforce_all_constraints(
    original_flow: np.ndarray,
    modified_flow: np.ndarray,
    config: dict,
    embedding_map: np.ndarray = None,
    original_embedded_flow: np.ndarray = None
) -> np.ndarray:
    """
    Apply all configured constraints to modified flow.
    
    Args:
        original_flow: Original flow field (H, W, 2)
        modified_flow: Modified flow field (H, W, 2)
        config: Configuration dictionary with constraint parameters
        embedding_map: Optional embedding map for re-quantization
        original_embedded_flow: Flow immediately after QIM (before constraints)
        
    Returns:
        constrained_flow: Flow with all constraints enforced
        
    Notes:
        - Applies constraints in order: smoothness → magnitude → perceptual → re-quantize
        - Each constraint is optional (based on config flags)
        - Re-quantization preserves original embedded directions
        - Deterministic: same inputs → same outputs
    """
    constrained_flow = modified_flow.copy()
    
    # 1. Smoothness constraint
    if config.get('enforce_smoothness', False):
        constrained_flow = enforce_smoothness_constraint(
            constrained_flow,
            kernel_size=config.get('smoothness_kernel_size', 5),
            sigma=config.get('smoothness_sigma', 1.0)
        )
    
    # 2. Magnitude bounds
    if config.get('enforce_magnitude_bounds', False):
        constrained_flow = enforce_magnitude_bounds(
            original_flow,
            constrained_flow,
            min_ratio=config.get('min_magnitude_ratio', 0.8),
            max_ratio=config.get('max_magnitude_ratio', 1.2)
        )
    
    # 3. Perceptual limit (L-infinity)
    if config.get('enforce_perceptual_limit', False):
        constrained_flow = enforce_perceptual_limit(
            original_flow,
            constrained_flow,
            max_perturbation=config.get('max_l_infinity_norm', 1.0)
        )
    
    # 4. Re-quantize to ensure correct extraction
    #    This is CRITICAL for QIM to work after constraints
    #    Preserves original embedded directions
    if embedding_map is not None and config.get('requantize_after_constraints', True):
        delta = config.get('quantization_step', 2.0)
        constrained_flow = requantize_to_qim(
            constrained_flow, 
            embedding_map, 
            delta,
            original_embedded_flow
        )
    
    return constrained_flow


def compute_constraint_violations(
    original_flow: np.ndarray,
    modified_flow: np.ndarray,
    config: dict
) -> dict:
    """
    Compute statistics on constraint violations.
    
    Args:
        original_flow: Original flow field (H, W, 2)
        modified_flow: Modified flow field (H, W, 2)
        config: Configuration dictionary
        
    Returns:
        violations: Dictionary with:
            - magnitude_violations: Number of magnitude bound violations
            - perceptual_violations: Number of L-inf violations
            - max_l_inf_perturbation: Maximum L-inf perturbation
            - max_magnitude_ratio: Maximum magnitude ratio
    """
    H, W, _ = original_flow.shape
    
    # Compute magnitudes
    orig_mags = np.linalg.norm(original_flow, axis=2)
    mod_mags = np.linalg.norm(modified_flow, axis=2)
    
    # Magnitude ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_ratios = mod_mags / orig_mags
        mag_ratios = np.nan_to_num(mag_ratios, nan=1.0, posinf=1.0)
    
    # Magnitude violations
    min_ratio = config.get('min_magnitude_ratio', 0.8)
    max_ratio = config.get('max_magnitude_ratio', 1.2)
    mag_violations = np.sum((mag_ratios < min_ratio) | (mag_ratios > max_ratio))
    
    # L-infinity violations
    perturbations = modified_flow - original_flow
    l_inf_norms = np.max(np.abs(perturbations), axis=2)
    max_l_inf = config.get('max_l_infinity_norm', 1.0)
    perceptual_violations = np.sum(l_inf_norms > max_l_inf)
    
    return {
        'magnitude_violations': int(mag_violations),
        'perceptual_violations': int(perceptual_violations),
        'max_l_inf_perturbation': float(np.max(l_inf_norms)),
        'max_magnitude_ratio': float(np.max(mag_ratios))
    }