"""
Motion-Field Modulation Engine (Architectural Module 5).

This is the CORE module that embeds encrypted data into optical flow fields
using Quantization Index Modulation (QIM).
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from .qim_core import (
    qim_embed_flow,
    qim_extract_flow,
    bits_to_bytes,
    bytes_to_bits
)
from .capacity import (
    compute_capacity,
    create_embedding_map,
    estimate_embedding_statistics
)
from .constraints import (
    enforce_all_constraints,
    compute_constraint_violations
)


@dataclass
class EmbeddingMetadata:
    """Metadata about embedding operation."""
    bits_embedded: int
    bits_requested: int
    embedding_rate: float  # bits embedded / bits requested
    num_vectors_modified: int
    total_vectors_available: int
    avg_perturbation: float  # Average ||v' - v||
    max_perturbation: float


class MotionFieldModulator:
    """
    Motion-Field Modulation Engine using QIM.
    
    This class provides the interface specified in MODULE_INTERFACES.md:
    - embed_bits(flow, payload, config) -> (modified_flow, metadata)
    - extract_bits(flow, config, num_bits) -> bytes
    - compute_capacity(flow, config) -> int
    - enforce_constraints(original_flow, modified_flow, config) -> FlowField
    
    Notes:
        - Operates ONLY on FlowField arrays (H×W×2 float32)
        - Does NOT handle cryptography, ECC, or video I/O
        - Deterministic: same inputs → same outputs
        - Preserves motion direction; modifies magnitude only
    """
    
    def __init__(self):
        """Initialize modulator (no state required)."""
        pass
    
    def embed_bits(
        self,
        flow: np.ndarray,
        payload: bytes,
        config: dict
    ) -> Tuple[np.ndarray, EmbeddingMetadata]:
        """
        Embed encrypted payload into flow field.
        
        Args:
            flow: Original flow field (H, W, 2), dtype float32
            payload: Encrypted data to embed (opaque bytes)
            config: Configuration dictionary from default_config.yaml
            
        Returns:
            modified_flow: Flow field with embedded data (H, W, 2)
            metadata: EmbeddingMetadata with operation statistics
            
        Raises:
            ValueError: If payload exceeds available capacity
            
        Algorithm:
            1. Compute embedding capacity
            2. Convert payload to bits
            3. Create embedding location map
            4. Embed bits using QIM
            5. Enforce constraints
            6. Compute metadata
            
        Notes:
            - Deterministic: same (flow, payload, config) → same output
            - Does NOT introduce randomness
            - Skips frames with insufficient motion (capacity = 0)
        """
        # Extract configuration parameters
        delta = config.get('quantization_step', 2.0)
        motion_threshold = config.get('motion_threshold', 1.0)
        use_high_motion = config.get('use_high_motion_regions', True)
        spatial_dist = config.get('spatial_distribution', 'uniform')
        max_payload_bits = config.get('max_payload_bits', 4096)
        
        # 1. Compute capacity
        capacity = self.compute_capacity(flow, config)
        
        # Convert payload to bits
        bits = bytes_to_bits(payload)
        bits_requested = len(bits)
        
        # Check capacity
        if bits_requested > max_payload_bits:
            raise ValueError(
                f"Payload ({bits_requested} bits) exceeds max_payload_bits "
                f"({max_payload_bits})"
            )
        
        if capacity == 0:
            # No motion - return original flow
            metadata = EmbeddingMetadata(
                bits_embedded=0,
                bits_requested=bits_requested,
                embedding_rate=0.0,
                num_vectors_modified=0,
                total_vectors_available=capacity,
                avg_perturbation=0.0,
                max_perturbation=0.0
            )
            return flow.copy(), metadata
        
        # Limit to available capacity
        bits_to_embed = min(bits_requested, capacity)
        bits = bits[:bits_to_embed]
        
        # 2. Create embedding map
        embedding_map = create_embedding_map(
            flow,
            num_bits=bits_to_embed,
            motion_threshold=motion_threshold,
            use_high_motion_regions=use_high_motion,
            spatial_distribution=spatial_dist
        )
        
        # 3. Embed bits using QIM
        modified_flow = qim_embed_flow(flow, bits, embedding_map, delta)
        
        # Store a copy of the embedded flow (before constraints) for re-quantization
        embedded_flow_copy = modified_flow.copy()
        
        # 4. Enforce constraints (with re-quantization using original embedded directions)
        modified_flow = self.enforce_constraints(
            flow, modified_flow, config, embedding_map, embedded_flow_copy
        )
        
        # 5. Compute metadata
        stats = estimate_embedding_statistics(flow, modified_flow, embedding_map)
        
        metadata = EmbeddingMetadata(
            bits_embedded=bits_to_embed,
            bits_requested=bits_requested,
            embedding_rate=bits_to_embed / bits_requested if bits_requested > 0 else 0.0,
            num_vectors_modified=stats['num_vectors_modified'],
            total_vectors_available=capacity,
            avg_perturbation=stats['avg_perturbation'],
            max_perturbation=stats['max_perturbation']
        )
        
        return modified_flow, metadata
    
    def extract_bits(
        self,
        flow: np.ndarray,
        config: dict,
        num_bits: int
    ) -> bytes:
        """
        Extract embedded bits from flow field.
        
        Args:
            flow: Flow field with embedded data (H, W, 2)
            config: Configuration dictionary
            num_bits: Number of bits to extract (REQUIRED - no implicit inference)
            
        Returns:
            payload: Extracted data as bytes
            
        Notes:
            - num_bits MUST be specified explicitly (contract from MODULE_INTERFACES.md)
            - Extracts from same locations as embedding (deterministic order)
            - Returns zero-padded bytes if num_bits not multiple of 8
        """
        # Extract configuration
        delta = config.get('quantization_step', 2.0)
        decision_boundary = config.get('decision_boundary', 0.25)
        motion_threshold = config.get('motion_threshold', 1.0)
        use_high_motion = config.get('use_high_motion_regions', True)
        spatial_dist = config.get('spatial_distribution', 'uniform')
        
        # Create embedding map (same as transmitter)
        embedding_map = create_embedding_map(
            flow,
            num_bits=num_bits,
            motion_threshold=motion_threshold,
            use_high_motion_regions=use_high_motion,
            spatial_distribution=spatial_dist
        )
        
        # Extract bits using QIM
        bits = qim_extract_flow(
            flow,
            embedding_map,
            delta,
            decision_boundary,
            num_bits
        )
        
        # Convert to bytes
        payload = bits_to_bytes(bits)
        
        return payload
    
    def compute_capacity(
        self,
        flow: np.ndarray,
        config: dict
    ) -> int:
        """
        Estimate embedding capacity for flow field.
        
        Args:
            flow: Flow field (H, W, 2)
            config: Configuration dictionary
            
        Returns:
            capacity: Maximum bits that can be embedded
            
        Notes:
            - Capacity is dynamic (depends on motion content)
            - Static scenes may have capacity = 0
            - Deterministic for same inputs
        """
        motion_threshold = config.get('motion_threshold', 1.0)
        use_high_motion = config.get('use_high_motion_regions', True)
        spatial_dist = config.get('spatial_distribution', 'uniform')
        
        capacity = compute_capacity(
            flow,
            motion_threshold,
            use_high_motion,
            spatial_dist
        )
        
        return capacity
    
    def enforce_constraints(
        self,
        original_flow: np.ndarray,
        modified_flow: np.ndarray,
        config: dict,
        embedding_map: np.ndarray = None,
        original_embedded_flow: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply perceptual and smoothness constraints to modified flow.
        
        Args:
            original_flow: Original flow field (H, W, 2)
            modified_flow: Modified flow field (H, W, 2)
            config: Configuration dictionary
            embedding_map: Optional embedding map for re-quantization
            original_embedded_flow: Flow immediately after QIM (before constraints)
            
        Returns:
            constrained_flow: Flow with constraints enforced
            
        Notes:
            - Applies constraints from default_config.yaml
            - Order: smoothness → magnitude bounds → perceptual limit → re-quantize
            - Re-quantization preserves original embedded directions
            - Deterministic
        """
        # Build constraint config
        constraint_config = {
            'enforce_smoothness': config.get('enforce_smoothness', True),
            'smoothness_kernel_size': config.get('smoothness_kernel_size', 5),
            'smoothness_sigma': config.get('smoothness_sigma', 1.0),
            'enforce_magnitude_bounds': config.get('enforce_magnitude_bounds', True),
            'min_magnitude_ratio': config.get('min_magnitude_ratio', 0.8),
            'max_magnitude_ratio': config.get('max_magnitude_ratio', 1.2),
            'enforce_perceptual_limit': config.get('enforce_perceptual_limit', True),
            'max_l_infinity_norm': config.get('max_l_infinity_norm', 1.0),
            'requantize_after_constraints': config.get('requantize_after_constraints', True),
            'quantization_step': config.get('quantization_step', 2.0)
        }
        
        constrained_flow = enforce_all_constraints(
            original_flow,
            modified_flow,
            constraint_config,
            embedding_map,
            original_embedded_flow
        )
        
        return constrained_flow
    
    def compute_constraint_violations(
        self,
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
            violations: Dictionary with violation statistics
        """
        constraint_config = {
            'min_magnitude_ratio': config.get('min_magnitude_ratio', 0.8),
            'max_magnitude_ratio': config.get('max_magnitude_ratio', 1.2),
            'max_l_infinity_norm': config.get('max_l_infinity_norm', 1.0)
        }
        
        return compute_constraint_violations(
            original_flow,
            modified_flow,
            constraint_config
        )