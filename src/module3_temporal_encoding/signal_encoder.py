"""
Main temporal signal encoder class.
"""

import numpy as np
from typing import Optional
import yaml
import os

from .exceptions import TemporalEncodingError
from .encoding_utils import (
    validate_flow_input,
    compute_flow_magnitude,
    compute_flow_direction,
    check_signal_validity,
    compute_signal_statistics
)
from .temporal_filters import apply_filter
from .quantization import quantize_signal


class TemporalSignalEncoder:
    """
    Encoder for converting optical flow sequences to 1D temporal signals.
    
    This class provides deterministic, stable encoding of optical flow data
    into temporal signals suitable for downstream analysis and processing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the temporal signal encoder.
        
        Args:
            config_path: Path to configuration YAML file.
                        If None, uses default configuration.
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to config file or None
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Use default config from same directory
            default_config_path = os.path.join(
                os.path.dirname(__file__),
                "default_config.yaml"
            )
            if os.path.exists(default_config_path):
                config_path = default_config_path
            else:
                # Fallback to hardcoded defaults
                return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception:
            # If config loading fails, use hardcoded defaults
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """
        Get hardcoded default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "default_method": "magnitude_mean",
            "default_filter": None,
            "quantization": {
                "enabled": True,
                "levels": 256
            },
            "filters": {
                "moving_average": {
                    "window": 5
                },
                "exponential_smoothing": {
                    "alpha": 0.3
                }
            }
        }
    
    def encode(
        self,
        flow_sequence: np.ndarray,
        *,
        method: str,
        filter: Optional[str] = None,
        quantize: bool = True
    ) -> dict:
        """
        Encode optical flow sequence into temporal signal.
        
        Args:
            flow_sequence: Optical flow array
                          Shape: (T, H, W, 2) or (H, W, 2)
                          dtype: float32
            method: Aggregation method
                   Options: 'magnitude_mean', 'magnitude_sum', 'directional_projection'
            filter: Optional temporal filter
                   Options: 'moving_average', 'exponential_smoothing', None
            quantize: Whether to quantize the signal
            
        Returns:
            Dictionary with keys:
                - "signal": np.ndarray of shape (T,), dtype float32
                - "quantized_signal": np.ndarray of shape (T,) or None
                - "metadata": dict with statistics and configuration
                
        Raises:
            TemporalEncodingError: If inputs are invalid or processing fails
        """
        # Validate and normalize input
        flow_normalized, num_frames = validate_flow_input(flow_sequence)
        
        # Compute temporal signal based on method
        signal = self._compute_signal(flow_normalized, method)
        
        # Apply temporal filter if specified
        if filter is not None:
            signal = self._apply_temporal_filter(signal, filter)
        
        # Quantize signal if requested
        quantized_signal = None
        if quantize:
            quantized_signal = self._quantize_signal(signal)
        
        # Compute metadata
        metadata = self._build_metadata(signal, quantized_signal, method, filter, quantize)
        
        return {
            "signal": signal,
            "quantized_signal": quantized_signal,
            "metadata": metadata
        }
    
    def _compute_signal(self, flow: np.ndarray, method: str) -> np.ndarray:
        """
        Compute temporal signal from flow using specified method.
        
        Args:
            flow: Normalized flow array of shape (T, H, W, 2)
            method: Aggregation method
            
        Returns:
            Temporal signal of shape (T,)
            
        Raises:
            TemporalEncodingError: If method is unsupported
        """
        if method == "magnitude_mean":
            return self._magnitude_mean(flow)
        elif method == "magnitude_sum":
            return self._magnitude_sum(flow)
        elif method == "directional_projection":
            return self._directional_projection(flow)
        else:
            raise TemporalEncodingError(
                f"Unsupported method: {method}. "
                f"Supported: 'magnitude_mean', 'magnitude_sum', 'directional_projection'"
            )
    
    def _magnitude_mean(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute mean magnitude per frame.
        
        Args:
            flow: Flow array of shape (T, H, W, 2)
            
        Returns:
            Signal of shape (T,) containing mean magnitude per frame
        """
        T = flow.shape[0]
        signal = np.zeros(T, dtype=np.float32)
        
        for t in range(T):
            magnitude = compute_flow_magnitude(flow[t])
            signal[t] = np.mean(magnitude)
        
        check_signal_validity(signal, "magnitude_mean signal")
        return signal
    
    def _magnitude_sum(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute sum of magnitudes per frame.
        
        Args:
            flow: Flow array of shape (T, H, W, 2)
            
        Returns:
            Signal of shape (T,) containing total magnitude per frame
        """
        T = flow.shape[0]
        signal = np.zeros(T, dtype=np.float32)
        
        for t in range(T):
            magnitude = compute_flow_magnitude(flow[t])
            signal[t] = np.sum(magnitude)
        
        check_signal_validity(signal, "magnitude_sum signal")
        return signal
    
    def _directional_projection(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute directional projection signal.
        
        Projects flow vectors onto a reference direction (horizontal: [1, 0])
        and computes mean projection per frame.
        
        Args:
            flow: Flow array of shape (T, H, W, 2)
            
        Returns:
            Signal of shape (T,) containing mean directional projection per frame
        """
        T = flow.shape[0]
        signal = np.zeros(T, dtype=np.float32)
        
        # Reference direction: horizontal (right)
        reference_direction = np.array([1.0, 0.0], dtype=np.float32)
        
        for t in range(T):
            # Compute dot product with reference direction
            # flow[t] has shape (H, W, 2)
            projection = np.dot(flow[t], reference_direction)
            signal[t] = np.mean(projection)
        
        check_signal_validity(signal, "directional_projection signal")
        return signal
    
    def _apply_temporal_filter(self, signal: np.ndarray, filter_type: str) -> np.ndarray:
        """
        Apply temporal filter to signal.
        
        Args:
            signal: Input signal of shape (T,)
            filter_type: Type of filter to apply
            
        Returns:
            Filtered signal of shape (T,)
        """
        filter_params = self.config.get("filters", {}).get(filter_type, {})
        return apply_filter(signal, filter_type, filter_params)
    
    def _quantize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Quantize continuous signal.
        
        Args:
            signal: Continuous signal of shape (T,)
            
        Returns:
            Quantized signal
        """
        quantization_config = self.config.get("quantization", {})
        levels = quantization_config.get("levels", 256)
        return quantize_signal(signal, levels)
    
    def _build_metadata(
        self,
        signal: np.ndarray,
        quantized_signal: Optional[np.ndarray],
        method: str,
        filter_type: Optional[str],
        quantize: bool
    ) -> dict:
        """
        Build metadata dictionary.
        
        Args:
            signal: Continuous signal
            quantized_signal: Quantized signal or None
            method: Aggregation method used
            filter_type: Filter type used or None
            quantize: Whether quantization was applied
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "method": method,
            "filter": filter_type,
            "quantize": quantize,
            "signal_statistics": compute_signal_statistics(signal)
        }
        
        if quantized_signal is not None:
            quantization_config = self.config.get("quantization", {})
            metadata["quantization"] = {
                "levels": quantization_config.get("levels", 256),
                "dtype": str(quantized_signal.dtype),
                "min": int(np.min(quantized_signal)),
                "max": int(np.max(quantized_signal)),
                "unique_values": int(len(np.unique(quantized_signal)))
            }
        
        return metadata