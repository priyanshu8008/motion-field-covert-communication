"""
Custom exceptions for Module 3: Temporal Signal Encoding.
"""


class TemporalEncodingError(Exception):
    """
    Exception raised for errors in temporal signal encoding operations.
    
    This includes:
    - Invalid input shapes or dtypes
    - Empty or malformed inputs
    - NaN/Inf values in computation
    - Unsupported methods or filters
    - Configuration errors
    """
    pass