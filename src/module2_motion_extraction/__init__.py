"""
Module 2: Motion Field Extraction

Dense optical flow extraction using RAFT (Recurrent All-Pairs Field Transforms).
This module performs inference-only optical flow computation between video frames.
"""

from .flow_extractor import OpticalFlowExtractor
from .flow_utils import visualize_flow, compute_flow_statistics
from .exceptions import FlowExtractionError


__all__ = [
    'OpticalFlowExtractor',
    'FlowExtractionError',
    'visualize_flow',
    'compute_flow_statistics',
]