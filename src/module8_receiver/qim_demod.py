"""
QIM Demodulator

Implements Quantization Index Modulation (QIM) bit extraction.
This is the EXACT inverse of the encoder's QIM embedding.

QIM Extraction Algorithm (from clarification resolution):
    1. m = ||v|| (motion magnitude)
    2. q = round(m / Δ)
    3. frac = (m / Δ) - q
    4. bit = 0 if |frac| < decision_boundary else 1

Where:
    - Δ = quantization_step (from config)
    - decision_boundary (from config, default 0.25)
"""

import numpy as np
from typing import List, Tuple

# Type alias
FlowField = np.ndarray  # Shape: (H, W, 2), dtype: float32


class QIMDemodulator:
    """
    Quantization Index Modulation (QIM) demodulator for bit extraction.
    
    Implements the exact inverse of the encoder's QIM embedding algorithm.
    """
    
    def __init__(self, config: dict):
        """
        Initialize QIM demodulator.
        
        Args:
            config: Configuration dictionary containing:
                - modulation.embedding.quantization_step: QIM step size (Δ)
                - modulation.demodulation.decision_boundary: Threshold for bit decision
                - modulation.demodulation.use_soft_decisions: Hard vs soft decisions
        """
        self.config = config
        self.modulation_config = config.get('modulation', {})
        self.embedding_config = self.modulation_config.get('embedding', {})
        self.demodulation_config = self.modulation_config.get('demodulation', {})
        
        # Extract QIM parameters
        self.quantization_step = self.embedding_config.get('quantization_step', 2.0)
        self.decision_boundary = self.demodulation_config.get('decision_boundary', 0.25)
        self.use_soft_decisions = self.demodulation_config.get('use_soft_decisions', False)
        
        # Validate parameters
        if self.quantization_step <= 0:
            raise ValueError(
                f"quantization_step must be positive, got {self.quantization_step}"
            )
        
        if not (0 < self.decision_boundary < 0.5):
            raise ValueError(
                f"decision_boundary must be in (0, 0.5), got {self.decision_boundary}"
            )
    
    def extract_bit(self, vector: np.ndarray) -> int:
        """
        Extract one bit from a motion vector using QIM.
        
        Algorithm (EXACT inverse of encoder):
            1. m = ||v|| = sqrt(vx^2 + vy^2)
            2. q = round(m / Δ)
            3. frac = (m / Δ) - q
            4. bit = 0 if |frac| < decision_boundary else 1
        
        Args:
            vector: Motion vector (2,) as [dx, dy]
            
        Returns:
            bit: Extracted bit (0 or 1)
        """
        # Compute magnitude
        magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
        
        # Avoid division by zero
        if magnitude < 1e-8:
            # Zero motion → default to bit 0
            return 0
        
        # Compute quantization index
        q = np.round(magnitude / self.quantization_step)
        
        # Compute fractional part
        frac = (magnitude / self.quantization_step) - q
        
        # Decision rule (using ABSOLUTE VALUE as per clarification)
        bit = 0 if abs(frac) < self.decision_boundary else 1
        
        return int(bit)
    
    def extract_bits(self, vectors: np.ndarray) -> np.ndarray:
        """
        Extract bits from multiple motion vectors.
        
        Args:
            vectors: Motion vectors (N, 2)
            
        Returns:
            bits: Extracted bits (N,) as 0/1 integers
        """
        if len(vectors) == 0:
            return np.array([], dtype=int)
        
        # Compute magnitudes for all vectors
        magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        
        # Compute quantization indices
        q = np.round(magnitudes / self.quantization_step)
        
        # Compute fractional parts
        frac = (magnitudes / self.quantization_step) - q
        
        # Apply decision rule (using absolute value)
        bits = np.where(np.abs(frac) < self.decision_boundary, 0, 1)
        
        return bits.astype(int)
    
    def extract_soft_bits(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract soft bits (for potential use with soft-decision ECC).
        
        Currently not used (use_soft_decisions = false by default).
        Included for future extension.
        
        Args:
            vectors: Motion vectors (N, 2)
            
        Returns:
            hard_bits: Hard bit decisions (N,) as 0/1 integers
            soft_metrics: Soft reliability metrics (N,) as floats in [0, 1]
                         Higher value = more confident in bit=1
        """
        if len(vectors) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        
        # Compute magnitudes
        magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        
        # Compute quantization indices
        q = np.round(magnitudes / self.quantization_step)
        
        # Compute fractional parts
        frac = (magnitudes / self.quantization_step) - q
        
        # Hard decisions
        hard_bits = np.where(np.abs(frac) < self.decision_boundary, 0, 1)
        
        # Soft metrics: distance from decision boundary
        # |frac| close to 0 → confident 0
        # |frac| close to 0.5 → confident 1
        soft_metrics = np.abs(frac)  # In range [0, 0.5]
        
        # Normalize to [0, 1] where:
        # 0 = very confident bit=0
        # 1 = very confident bit=1
        soft_metrics = soft_metrics / 0.5  # Now in [0, 1]
        
        return hard_bits.astype(int), soft_metrics.astype(float)
    
    def extract_from_flow(
        self,
        flow: FlowField,
        embedding_map: np.ndarray
    ) -> np.ndarray:
        """
        Extract bits from a flow field using an embedding map.
        
        Args:
            flow: Flow field (H, W, 2)
            embedding_map: Boolean mask indicating which pixels to extract from
            
        Returns:
            bits: Extracted bits in raster scan order (N,)
        """
        # Get positions where embedding_map is True
        positions = np.argwhere(embedding_map)
        
        if len(positions) == 0:
            return np.array([], dtype=int)
        
        # Sort positions in raster scan order for determinism
        H, W = flow.shape[:2]
        linear_indices = positions[:, 0] * W + positions[:, 1]
        sorted_order = np.argsort(linear_indices)
        sorted_positions = positions[sorted_order]
        
        # Extract vectors at these positions
        vectors = flow[sorted_positions[:, 0], sorted_positions[:, 1]]
        
        # Extract bits
        bits = self.extract_bits(vectors)
        
        return bits