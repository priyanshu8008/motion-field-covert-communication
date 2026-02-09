"""
Module 8: Steganalysis Attacker

This module provides tools for detecting hidden data in motion fields using
statistical analysis and machine learning.

Components:
    - feature_extractor: Handcrafted statistical features from optical flow
    - classical_detector: Binary classifier using Logistic Regression or Linear SVM
    - evaluation: (Future) Comprehensive ROC/AUC evaluation and plotting
"""

from .feature_extractor import FlowFeatureExtractor
from .classical_detector import ClassicalSteganalysisDetector, train_detector

__all__ = [
    'FlowFeatureExtractor',
    'ClassicalSteganalysisDetector',
    'train_detector'
]