"""
Video Reconstruction Engine (Architectural Module 6).

Reconstructs stego video frames from modified optical flow fields using
propagative warping to preserve embedded QIM perturbations.

Public API:
    - VideoReconstructor: Main reconstruction class
    - QualityMetrics: Quality metrics dataclass
"""

from .video_reconstructor import VideoReconstructor
from .quality_metrics import QualityMetrics

__all__ = [
    'VideoReconstructor',
    'QualityMetrics'
]

__version__ = '1.0.0'