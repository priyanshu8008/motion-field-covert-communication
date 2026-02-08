"""
Receiver Engine

Main orchestrator for the receiver/extraction pipeline.
Coordinates all subcomponents to extract hidden data from stego videos.

Pipeline:
    Stego Frames
    → Optical Flow Recomputation (Module 2 wrapper)
    → Capacity Computation (deterministic)
    → Region Selection (deterministic)
    → QIM Demodulation
    → Bitstream Aggregation
    → RAW bitstream output (NO ECC, NO crypto)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import time

from .flow_recompute import FlowRecomputeWrapper
from .capacity import CapacityEstimator
from .region_selection import RegionSelector
from .qim_demod import QIMDemodulator
from .bitstream import BitstreamAggregator

# Type aliases
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8
FlowField = np.ndarray  # Shape: (H, W, 2), dtype: float32


class ReceiverEngine:
    """
    Main receiver/extraction engine.
    
    Extracts embedded bitstream from stego video frames using deterministic
    QIM demodulation. Does NOT perform ECC decoding or cryptographic operations.
    """
    
    def __init__(self, config: dict):
        """
        Initialize receiver engine.
        
        Args:
            config: Configuration dictionary (from default_config.yaml)
        """
        self.config = config
        
        # Initialize subcomponents
        self.flow_recompute = FlowRecomputeWrapper(config)
        self.capacity_estimator = CapacityEstimator(config)
        self.region_selector = RegionSelector(config)
        self.qim_demod = QIMDemodulator(config)
        self.bitstream_aggregator = BitstreamAggregator(config)
        
        # Configuration flags
        self.verbose = config.get('system', {}).get('verbose', True)
        self.debug_mode = config.get('system', {}).get('debug_mode', False)
    
    def extract(
        self,
        frames: List[Frame],
        config: Optional[dict] = None
    ) -> bytes:
        """
        Extract embedded bitstream from stego video frames.
        
        This is the main public interface for the receiver.
        
        Args:
            frames: List of stego video frames (N frames, each (H, W, 3), uint8)
            config: Optional config override (uses self.config if None)
            
        Returns:
            bitstream: Raw extracted bits as bytes (NO ECC decoding, NO decryption)
            
        Raises:
            ValueError: If frames list is empty or invalid
        """
        if config is not None:
            # Use provided config (for testing)
            self.config = config
            self._reinitialize_components()
        
        # Validate input
        if len(frames) < 2:
            raise ValueError(
                f"Need at least 2 frames for extraction, got {len(frames)}"
            )
        
        # Execute extraction pipeline
        bitstream, _ = self._extract_internal(frames, collect_metadata=False)
        
        return bitstream
    
    def extract_with_metadata(
        self,
        frames: List[Frame],
        config: Optional[dict] = None
    ) -> Tuple[bytes, Dict[str, any]]:
        """
        Extract bitstream and collect detailed metadata.
        
        Useful for debugging and analysis.
        
        Args:
            frames: List of stego video frames
            config: Optional config override
            
        Returns:
            bitstream: Raw extracted bits as bytes
            metadata: Dictionary containing:
                - total_bits: Total bits extracted
                - num_frames_processed: Number of frames
                - num_frames_skipped: Frames with zero capacity
                - bits_per_frame: List of bit counts
                - extraction_time: Total time in seconds
                - per_frame_stats: Detailed per-frame statistics
        """
        if config is not None:
            self.config = config
            self._reinitialize_components()
        
        # Validate input
        if len(frames) < 2:
            raise ValueError(
                f"Need at least 2 frames for extraction, got {len(frames)}"
            )
        
        # Execute extraction with metadata collection
        bitstream, metadata = self._extract_internal(frames, collect_metadata=True)
        
        return bitstream, metadata
    
    def _extract_internal(
        self,
        frames: List[Frame],
        collect_metadata: bool
    ) -> Tuple[bytes, Optional[Dict[str, any]]]:
        """
        Internal extraction implementation.
        
        Args:
            frames: List of frames
            collect_metadata: Whether to collect detailed statistics
            
        Returns:
            bitstream: Extracted bits as bytes
            metadata: Metadata dict (or None if collect_metadata=False)
        """
        start_time = time.time()
        
        # Step 1: Recompute optical flow from stego frames
        if self.verbose:
            print(f"[Receiver] Step 1: Extracting optical flow from {len(frames)} frames...")
        
        flows = self.flow_recompute.batch_extract(frames)
        
        if self.verbose:
            print(f"[Receiver] Extracted {len(flows)} flow fields")
        
        # Step 2: Extract bits from each flow field
        if self.verbose:
            print(f"[Receiver] Step 2: Demodulating bits from flow fields...")
        
        per_frame_bits = []
        per_frame_stats = []
        
        for frame_idx, flow in enumerate(flows):
            # Compute capacity (deterministic)
            capacity = self.capacity_estimator.compute_capacity(flow)
            
            if capacity == 0:
                # Skip frame (insufficient motion)
                per_frame_bits.append(np.array([], dtype=int))
                
                if collect_metadata:
                    per_frame_stats.append({
                        'frame_idx': frame_idx,
                        'capacity': 0,
                        'bits_extracted': 0,
                        'skipped': True
                    })
                continue
            
            # Create embedding map (deterministic)
            embedding_map = self.capacity_estimator.compute_embedding_map(flow, capacity)
            
            # Extract bits using QIM
            bits = self.qim_demod.extract_from_flow(flow, embedding_map)
            
            per_frame_bits.append(bits)
            
            if collect_metadata:
                per_frame_stats.append({
                    'frame_idx': frame_idx,
                    'capacity': capacity,
                    'bits_extracted': len(bits),
                    'skipped': False
                })
            
            if self.verbose and frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}: extracted {len(bits)} bits")
        
        # Step 3: Aggregate bits into output bitstream
        if self.verbose:
            print(f"[Receiver] Step 3: Aggregating bitstream...")
        
        if collect_metadata:
            bitstream, agg_metadata = self.bitstream_aggregator.aggregate_with_metadata(
                per_frame_bits
            )
        else:
            bitstream = self.bitstream_aggregator.aggregate(per_frame_bits)
            agg_metadata = None
        
        extraction_time = time.time() - start_time
        
        if self.verbose:
            total_bits = len(bitstream) * 8 if not collect_metadata else agg_metadata['total_bits']
            print(f"[Receiver] Extraction complete!")
            print(f"  Total bits extracted: {total_bits}")
            print(f"  Output size: {len(bitstream)} bytes")
            print(f"  Time: {extraction_time:.2f}s")
        
        # Build metadata if requested
        metadata = None
        if collect_metadata:
            metadata = {
                'total_bits': agg_metadata['total_bits'],
                'total_bytes': len(bitstream),
                'num_frames_processed': len(flows),
                'num_frames_skipped': agg_metadata['skipped_frames'],
                'bits_per_frame': agg_metadata['bits_per_frame'],
                'avg_bits_per_frame': agg_metadata['avg_bits_per_frame'],
                'extraction_time_seconds': extraction_time,
                'per_frame_stats': per_frame_stats,
            }
        
        return bitstream, metadata
    
    def _reinitialize_components(self):
        """Reinitialize all components with updated config."""
        self.flow_recompute = FlowRecomputeWrapper(self.config)
        self.capacity_estimator = CapacityEstimator(self.config)
        self.region_selector = RegionSelector(self.config)
        self.qim_demod = QIMDemodulator(self.config)
        self.bitstream_aggregator = BitstreamAggregator(self.config)