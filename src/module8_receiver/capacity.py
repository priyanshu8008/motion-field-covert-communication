"""
Capacity Estimator

Deterministically computes embedding capacity from optical flow fields.
MUST produce identical results to encoder's capacity computation.
"""

import numpy as np

# Type alias
FlowField = np.ndarray  # Shape: (H, W, 2), dtype: float32


class CapacityEstimator:
    """
    Computes embedding capacity from flow fields.

    This class mirrors the encoder-side (Module 5) capacity logic
    exactly for deterministic synchronization.
    """

    def __init__(self, config: dict):
        """
        Initialize capacity estimator.

        Args:
            config: Full configuration dictionary (default_config.yaml)
        """
        self.config = config

        modulation = config.get("modulation", {})
        selection = modulation.get("selection", {})
        embedding = modulation.get("embedding", {})

        # LOCKED parameters (must match encoder)
        self.motion_threshold: float = selection.get("motion_threshold", 1.0)
        self.spatial_distribution: str = selection.get(
            "spatial_distribution", "uniform"
        )
        self.use_high_motion: bool = selection.get(
            "use_high_motion_regions", True
        )
        self.max_payload_bits: int = embedding.get(
            "max_payload_bits", 4096
        )

        if self.spatial_distribution not in ("uniform", "adaptive"):
            raise ValueError(
                f"Invalid spatial_distribution: {self.spatial_distribution}"
            )

    # ------------------------------------------------------------------
    # CORE CAPACITY FUNCTION (TESTED + LOCKED)
    # ------------------------------------------------------------------
    def compute_capacity(self, flow: FlowField) -> int:
        """
    Compute embedding capacity for a single flow field.

    LOCKED RECEIVER RULES (per tests & architecture):
    1. Motion magnitude = L2 norm
    2. If MEAN motion < motion_threshold → capacity = 0
    3. Otherwise, capacity = number of pixels where ||v|| >= motion_threshold
    4. Capacity is capped at max_payload_bits
    5. Deterministic, stateless
    """

        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(f"Invalid flow shape {flow.shape}, expected (H, W, 2)")

        # L2 motion magnitude
        magnitude = np.linalg.norm(flow, axis=2)

        # 🔒 FRAME-LEVEL GATE (MEAN, NOT MAX)
        if float(np.mean(magnitude)) < self.motion_threshold:
            return 0

        # Pixel-level capacity
        capacity = int(np.sum(magnitude >= self.motion_threshold))

        # Encoder-symmetric cap
        capacity = min(capacity, self.max_payload_bits)

        return capacity




    # ------------------------------------------------------------------
    # EMBEDDING MAP (USED BY RECEIVER FOR BIT EXTRACTION)
    # ------------------------------------------------------------------
    def compute_embedding_map(self, flow: FlowField, num_bits: int) -> np.ndarray:
        """
        Create deterministic embedding/extraction map.

        Args:
            flow: Flow field (H, W, 2)
            num_bits: Number of bits to extract

        Returns:
            embedding_map: Boolean mask (H, W)
        """
        H, W = flow.shape[:2]
        magnitude = np.linalg.norm(flow, axis=2)

        valid_mask = magnitude >= self.motion_threshold

        if np.sum(valid_mask) == 0 or num_bits == 0:
            return np.zeros((H, W), dtype=bool)

        if self.spatial_distribution == "uniform":
            return self._select_pixels_deterministic(
                valid_mask, magnitude, num_bits
            )

        if self.spatial_distribution == "adaptive":
            if self.use_high_motion:
                threshold = np.median(magnitude[valid_mask])
                high_motion_mask = (magnitude >= threshold) & valid_mask
                return self._select_pixels_deterministic(
                    high_motion_mask, magnitude, num_bits
                )
            else:
                return self._select_pixels_deterministic(
                    valid_mask, magnitude, num_bits
                )

        raise RuntimeError("Unreachable")

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _select_pixels_deterministic(
        self,
        mask: np.ndarray,
        magnitude: np.ndarray,
        num_pixels: int
    ) -> np.ndarray:
        """
        Deterministically select top-N pixels by magnitude.
        """
        H, W = mask.shape
        indices = np.argwhere(mask)

        if len(indices) == 0:
            return np.zeros((H, W), dtype=bool)

        mags = magnitude[mask]
        order = np.argsort(mags)[::-1][:num_pixels]

        selection_map = np.zeros((H, W), dtype=bool)
        chosen = indices[order]
        selection_map[chosen[:, 0], chosen[:, 1]] = True

        return selection_map
