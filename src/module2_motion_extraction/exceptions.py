"""
Module 2 Exceptions

Custom exceptions for motion field extraction.
"""


class FlowExtractionError(Exception):
    """
    Exception raised when optical flow extraction fails.
    
    This can occur due to:
    - RAFT model implementation not found
    - RAFT model inference failure
    - Invalid input tensor shapes
    - CUDA out of memory
    - Model loading errors
    """
    pass