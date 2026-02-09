"""
Feature Extraction for Motion-Field Steganalysis

This module implements handcrafted statistical feature extraction from optical 
flow sequences for detecting hidden data embedded in motion fields.

Author: Motion Covert Communication System
Module: 8 (Steganalysis Attacker)
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FlowFeatureExtractor:
    """
    Extracts baseline handcrafted statistical features from optical flow sequences.
    
    Features capture statistical anomalies introduced by motion-field modulation:
    - Motion magnitude distribution (histogram)
    - Motion direction distribution (histogram)
    - First-order statistics (mean, variance)
    - Temporal consistency (frame-to-frame variations)
    - Spatial coherence (local gradient smoothness)
    
    Output is a fixed-length feature vector suitable for classical ML classifiers.
    """
    
    def __init__(
        self,
        magnitude_bins: int = 32,
        direction_bins: int = 16,
        max_magnitude: float = 100.0
    ):
        """
        Initialize feature extractor.
        
        Args:
            magnitude_bins: Number of histogram bins for motion magnitude (default: 32)
            direction_bins: Number of histogram bins for motion direction (default: 16)
            max_magnitude: Maximum magnitude for histogram clipping (default: 100.0 pixels)
        """
        self.magnitude_bins = magnitude_bins
        self.direction_bins = direction_bins
        self.max_magnitude = max_magnitude
        
        # Feature dimensionality (computed once)
        self._feature_dim = (
            magnitude_bins +    # Magnitude histogram
            direction_bins +    # Direction histogram
            2 +                 # Magnitude mean + variance
            2 +                 # Direction mean + variance
            1 +                 # Temporal consistency
            1                   # Spatial coherence
        )
        
        logger.info(
            f"Initialized FlowFeatureExtractor: "
            f"{self._feature_dim}-D features "
            f"({magnitude_bins} mag bins, {direction_bins} dir bins)"
        )
    
    @property
    def feature_dim(self) -> int:
        """Return total feature dimensionality."""
        return self._feature_dim
    
    def extract_features(self, flow_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Extract complete feature vector from a flow sequence.
        
        Args:
            flow_sequence: List of optical flow fields
                          Each element has shape (H, W, 2) with dtype float32
                          [:,:,0] = horizontal displacement (dx)
                          [:,:,1] = vertical displacement (dy)
        
        Returns:
            feature_vector: 1D numpy array of shape (D,) where D = self.feature_dim
                           Features are ordered as follows:
                           [0:32]      → Magnitude histogram (32)
                           [32:48]     → Direction histogram (16)
                           [48:50]     → Magnitude statistics (mean, var)
                           [50:52]     → Direction statistics (mean, var)
                           [52]        → Temporal consistency
                           [53]        → Spatial coherence
        
        Raises:
            ValueError: If flow_sequence is empty or has invalid shape
        """
        if not flow_sequence:
            raise ValueError("Flow sequence cannot be empty")
        
        # Validate flow field shapes
        for i, flow in enumerate(flow_sequence):
            if flow.ndim != 3 or flow.shape[2] != 2:
                raise ValueError(
                    f"Flow field {i} has invalid shape {flow.shape}. "
                    f"Expected (H, W, 2)"
                )
        
        # Extract feature groups
        mag_hist = self._magnitude_histogram(flow_sequence)
        dir_hist = self._direction_histogram(flow_sequence)
        mag_stats = self._magnitude_statistics(flow_sequence)
        dir_stats = self._direction_statistics(flow_sequence)
        temporal_feat = self._temporal_consistency(flow_sequence)
        spatial_feat = self._spatial_coherence(flow_sequence)
        
        # Concatenate all features in fixed order
        features = np.concatenate([
            mag_hist,       # [0:32]
            dir_hist,       # [32:48]
            mag_stats,      # [48:50]
            dir_stats,      # [50:52]
            temporal_feat,  # [52]
            spatial_feat    # [53]
        ])
        
        assert features.shape == (self._feature_dim,), \
            f"Feature dimension mismatch: got {features.shape}, expected ({self._feature_dim},)"
        
        return features.astype(np.float32)
    
    # =========================================================================
    # HELPER METHODS: Magnitude and Direction Computation
    # =========================================================================
    
    def _compute_magnitude_direction(
        self, flow: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude and direction from flow field.
        
        Args:
            flow: Flow field of shape (H, W, 2)
        
        Returns:
            magnitude: Flow magnitude array of shape (H, W)
            direction: Flow direction array of shape (H, W) in radians [-π, π]
        """
        u = flow[:, :, 0]  # Horizontal component
        v = flow[:, :, 1]  # Vertical component
        
        magnitude = np.sqrt(u**2 + v**2)
        direction = np.arctan2(v, u)  # Range: [-π, π]
        
        return magnitude, direction
    
    # =========================================================================
    # FEATURE GROUP 1: Magnitude Histogram (32-D)
    # =========================================================================
    
    def _magnitude_histogram(self, flow_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute normalized histogram of motion magnitudes across all frames.
        
        Rationale:
            Embedding in motion fields perturbs magnitude distribution.
            Clean videos have characteristic magnitude distributions (e.g., 
            heavy-tailed for natural motion). Embedding introduces anomalies.
        
        Args:
            flow_sequence: List of flow fields
        
        Returns:
            hist: Normalized histogram of shape (magnitude_bins,)
        """
        # Accumulate all magnitudes from all frames
        all_magnitudes = []
        for flow in flow_sequence:
            mag, _ = self._compute_magnitude_direction(flow)
            all_magnitudes.append(mag.flatten())
        
        all_magnitudes = np.concatenate(all_magnitudes)
        
        # Compute histogram with uniform bins from 0 to max_magnitude
        hist, _ = np.histogram(
            all_magnitudes,
            bins=self.magnitude_bins,
            range=(0, self.max_magnitude),
            density=True  # Normalize to probability density
        )
        
        # Additional normalization to ensure sum = 1 (convert density to probability)
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    # =========================================================================
    # FEATURE GROUP 2: Direction Histogram (16-D)
    # =========================================================================
    
    def _direction_histogram(self, flow_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute normalized histogram of motion directions across all frames.
        
        Rationale:
            Natural motion has directional biases (e.g., camera pans, object 
            trajectories). Embedding may disrupt these patterns by introducing
            random directional perturbations.
        
        Args:
            flow_sequence: List of flow fields
        
        Returns:
            hist: Normalized histogram of shape (direction_bins,)
        """
        # Accumulate all directions from all frames
        all_directions = []
        for flow in flow_sequence:
            _, dir_ = self._compute_magnitude_direction(flow)
            all_directions.append(dir_.flatten())
        
        all_directions = np.concatenate(all_directions)
        
        # Compute histogram over full angular range [-π, π]
        hist, _ = np.histogram(
            all_directions,
            bins=self.direction_bins,
            range=(-np.pi, np.pi),
            density=True
        )
        
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    # =========================================================================
    # FEATURE GROUP 3: Magnitude Statistics (2-D)
    # =========================================================================
    
    def _magnitude_statistics(self, flow_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute mean and variance of motion magnitudes.
        
        Rationale:
            Embedding shifts the first and second moments of magnitude 
            distribution. Mean captures overall motion intensity, variance 
            captures magnitude spread.
        
        Args:
            flow_sequence: List of flow fields
        
        Returns:
            stats: Array [mean_mag, var_mag] of shape (2,)
        """
        all_magnitudes = []
        for flow in flow_sequence:
            mag, _ = self._compute_magnitude_direction(flow)
            all_magnitudes.append(mag.flatten())
        
        all_magnitudes = np.concatenate(all_magnitudes)
        
        mean_mag = np.mean(all_magnitudes)
        var_mag = np.var(all_magnitudes)
        
        return np.array([mean_mag, var_mag])
    
    # =========================================================================
    # FEATURE GROUP 4: Direction Statistics (2-D)
    # =========================================================================
    
    def _direction_statistics(self, flow_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute mean and variance of motion directions.
        
        Rationale:
            Captures directional bias and spread. Note: For circular data, 
            standard mean/variance are approximations. More sophisticated 
            circular statistics could be used in future work.
        
        Args:
            flow_sequence: List of flow fields
        
        Returns:
            stats: Array [mean_dir, var_dir] of shape (2,)
        """
        all_directions = []
        for flow in flow_sequence:
            _, dir_ = self._compute_magnitude_direction(flow)
            all_directions.append(dir_.flatten())
        
        all_directions = np.concatenate(all_directions)
        
        # Simple linear mean/variance (not circular statistics)
        mean_dir = np.mean(all_directions)
        var_dir = np.var(all_directions)
        
        return np.array([mean_dir, var_dir])
    
    # =========================================================================
    # FEATURE GROUP 5: Temporal Consistency (1-D)
    # =========================================================================
    
    def _temporal_consistency(self, flow_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute mean L2 norm of frame-to-frame flow differences.
        
        Rationale:
            Natural motion exhibits temporal smoothness. Embedding introduces
            frame-to-frame variations that violate temporal coherence.
            High temporal inconsistency indicates potential stego content.
        
        Formula:
            temporal_consistency = mean_{t} ||F_{t+1} - F_t||_2
        
        Args:
            flow_sequence: List of flow fields
        
        Returns:
            consistency: Array [mean_temporal_diff] of shape (1,)
        """
        if len(flow_sequence) < 2:
            # Single frame: no temporal comparison possible
            return np.array([0.0])
        
        temporal_diffs = []
        for i in range(len(flow_sequence) - 1):
            # Compute difference between consecutive flows
            diff = flow_sequence[i+1] - flow_sequence[i]
            
            # L2 norm of the difference (Frobenius norm for matrix)
            l2_norm = np.linalg.norm(diff)
            
            temporal_diffs.append(l2_norm)
        
        mean_temporal_diff = np.mean(temporal_diffs)
        
        return np.array([mean_temporal_diff])
    
    # =========================================================================
    # FEATURE GROUP 6: Spatial Coherence (1-D)
    # =========================================================================
    
    def _spatial_coherence(self, flow_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute mean local gradient magnitude of flow fields.
        
        Rationale:
            Natural motion is spatially smooth within objects/regions. 
            Embedding may create high-frequency artifacts (sharp transitions)
            that violate spatial coherence. High gradients indicate roughness.
        
        Formula:
            spatial_coherence = mean_{x,y,t} ||∇F(x,y,t)||
        
        Args:
            flow_sequence: List of flow fields
        
        Returns:
            coherence: Array [mean_gradient_magnitude] of shape (1,)
        """
        all_gradients = []
        
        for flow in flow_sequence:
            u = flow[:, :, 0]  # Horizontal component
            v = flow[:, :, 1]  # Vertical component
            
            # Compute spatial gradients using finite differences
            # Horizontal gradients (∂/∂x)
            grad_u_x = np.abs(u[:, 1:] - u[:, :-1])
            grad_v_x = np.abs(v[:, 1:] - v[:, :-1])
            
            # Vertical gradients (∂/∂y)
            grad_u_y = np.abs(u[1:, :] - u[:-1, :])
            grad_v_y = np.abs(v[1:, :] - v[:-1, :])
            
            # Accumulate all gradient magnitudes
            all_gradients.extend(grad_u_x.flatten())
            all_gradients.extend(grad_u_y.flatten())
            all_gradients.extend(grad_v_x.flatten())
            all_gradients.extend(grad_v_y.flatten())
        
        # Mean gradient magnitude across all locations and frames
        mean_gradient = np.mean(all_gradients)
        
        return np.array([mean_gradient])


# =============================================================================
# SELF-TEST FUNCTION
# =============================================================================

def self_test():
    """
    Self-test for FlowFeatureExtractor.
    
    Tests:
    1. Feature extraction on synthetic flow
    2. Feature dimensionality verification
    3. Deterministic output
    4. Edge cases (single frame, zero flow)
    """
    print("="*70)
    print("FlowFeatureExtractor Self-Test")
    print("="*70)
    
    # Test 1: Basic extraction
    print("\n[Test 1] Basic feature extraction on synthetic flow sequence")
    extractor = FlowFeatureExtractor(magnitude_bins=32, direction_bins=16)
    
    # Create synthetic flow sequence (5 frames, 64x64 resolution)
    np.random.seed(42)
    flow_sequence = []
    for i in range(5):
        # Random flow with some structure
        u = np.random.randn(64, 64).astype(np.float32) * 2.0
        v = np.random.randn(64, 64).astype(np.float32) * 2.0
        flow = np.stack([u, v], axis=2)
        flow_sequence.append(flow)
    
    features = extractor.extract_features(flow_sequence)
    print(f"✓ Feature vector shape: {features.shape}")
    print(f"✓ Expected dimensionality: {extractor.feature_dim}")
    assert features.shape == (54,), f"Expected (54,), got {features.shape}"
    
    # Test 2: Feature breakdown
    print("\n[Test 2] Feature component verification")
    print(f"  Magnitude histogram [0:32]:   {features[0:32].shape}")
    print(f"  Direction histogram [32:48]:  {features[32:48].shape}")
    print(f"  Magnitude stats [48:50]:      {features[48:50]}")
    print(f"  Direction stats [50:52]:      {features[50:52]}")
    print(f"  Temporal consistency [52]:    {features[52]:.4f}")
    print(f"  Spatial coherence [53]:       {features[53]:.4f}")
    
    # Verify histogram normalization
    mag_hist_sum = features[0:32].sum()
    dir_hist_sum = features[32:48].sum()
    print(f"\n  Magnitude histogram sum: {mag_hist_sum:.6f} (should ≈ 1.0)")
    print(f"  Direction histogram sum: {dir_hist_sum:.6f} (should ≈ 1.0)")
    assert np.isclose(mag_hist_sum, 1.0, atol=1e-5), "Magnitude histogram not normalized"
    assert np.isclose(dir_hist_sum, 1.0, atol=1e-5), "Direction histogram not normalized"
    
    # Test 3: Determinism
    print("\n[Test 3] Deterministic output")
    features_repeat = extractor.extract_features(flow_sequence)
    assert np.allclose(features, features_repeat), "Features are not deterministic"
    print("✓ Features are deterministic")
    
    # Test 4: Single frame (edge case)
    print("\n[Test 4] Single frame edge case")
    single_flow = [flow_sequence[0]]
    features_single = extractor.extract_features(single_flow)
    print(f"✓ Single frame feature shape: {features_single.shape}")
    print(f"  Temporal consistency (should be 0.0): {features_single[52]:.4f}")
    assert features_single[52] == 0.0, "Single frame should have zero temporal consistency"
    
    # Test 5: Zero flow (edge case)
    print("\n[Test 5] Zero flow edge case")
    zero_flow = [np.zeros((64, 64, 2), dtype=np.float32) for _ in range(3)]
    features_zero = extractor.extract_features(zero_flow)
    print(f"✓ Zero flow feature shape: {features_zero.shape}")
    print(f"  Magnitude mean: {features_zero[48]:.4f} (should be 0.0)")
    print(f"  Spatial coherence: {features_zero[53]:.4f} (should be 0.0)")
    assert features_zero[48] == 0.0, "Zero flow should have zero magnitude mean"
    assert features_zero[53] == 0.0, "Zero flow should have zero spatial coherence"
    
    # Test 6: Different flow sequences should give different features
    print("\n[Test 6] Discriminative power")
    flow_sequence_2 = []
    for i in range(5):
        u = np.random.randn(64, 64).astype(np.float32) * 5.0  # Higher magnitude
        v = np.random.randn(64, 64).astype(np.float32) * 5.0
        flow = np.stack([u, v], axis=2)
        flow_sequence_2.append(flow)
    
    features_2 = extractor.extract_features(flow_sequence_2)
    l2_distance = np.linalg.norm(features - features_2)
    print(f"✓ L2 distance between different flows: {l2_distance:.4f}")
    assert l2_distance > 1.0, "Features should differ for different flow sequences"
    
    print("\n" + "="*70)
    print("All self-tests PASSED ✓")
    print("="*70)


if __name__ == "__main__":
    # Run self-test when executed directly
    self_test()