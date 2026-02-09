"""
Classical Machine Learning Detector for Motion-Field Steganalysis

This module implements a baseline binary classifier using traditional ML algorithms
(Logistic Regression or Linear SVM) to distinguish between clean and stego videos
based on handcrafted flow features.

Author: Motion Covert Communication System
Module: 8 (Steganalysis Attacker)
"""

import numpy as np
from typing import Tuple, Dict, Optional, Literal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


class ClassicalSteganalysisDetector:
    """
    Classical machine learning detector for steganalysis.
    
    Uses linear classifiers (Logistic Regression or Linear SVM) to detect
    hidden data in motion fields based on statistical feature vectors.
    
    The detector learns to distinguish between:
    - Class 0: Clean videos (no hidden data)
    - Class 1: Stego videos (contain hidden data)
    """
    
    def __init__(
        self,
        model_type: Literal["logistic", "svm"] = "logistic",
        random_seed: int = 42,
        normalize_features: bool = True,
        **model_kwargs
    ):
        """
        Initialize classical steganalysis detector.
        
        Args:
            model_type: Classifier type - "logistic" for Logistic Regression,
                       "svm" for Linear SVM (default: "logistic")
            random_seed: Random seed for reproducibility (default: 42)
            normalize_features: Whether to normalize features using StandardScaler
                               (default: True, recommended)
            **model_kwargs: Additional arguments passed to the classifier
                           For logistic: max_iter, C, penalty, etc.
                           For svm: C, max_iter, etc.
        """
        self.model_type = model_type
        self.random_seed = random_seed
        self.normalize_features = normalize_features
        
        # Initialize classifier
        if model_type == "logistic":
            # Logistic Regression with L2 regularization
            default_kwargs = {
                'random_state': random_seed,
                'max_iter': 1000,
                'C': 1.0,  # Inverse regularization strength
                'penalty': 'l2',
                'solver': 'lbfgs'
            }
            default_kwargs.update(model_kwargs)
            self.classifier = LogisticRegression(**default_kwargs)
            
        elif model_type == "svm":
            # Linear SVM (uses hinge loss)
            default_kwargs = {
                'random_state': random_seed,
                'max_iter': 1000,
                'C': 1.0,
                'dual': False  # Use primal formulation for n_samples > n_features
            }
            default_kwargs.update(model_kwargs)
            self.classifier = LinearSVC(**default_kwargs)
            
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Must be 'logistic' or 'svm'"
            )
        
        # Feature normalizer (fitted during training)
        self.scaler = StandardScaler() if normalize_features else None
        
        # Training state
        self.is_trained = False
        self.decision_threshold = 0.5  # Default threshold for binary classification
        
        logger.info(
            f"Initialized {model_type.upper()} detector "
            f"(normalize={normalize_features}, seed={random_seed})"
        )
    
    def train(
        self,
        X_clean: np.ndarray,
        X_stego: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the detector on labeled feature vectors.
        
        Args:
            X_clean: Clean video features, shape (N_clean, D)
            X_stego: Stego video features, shape (N_stego, D)
            test_size: Fraction of data to use for testing (default: 0.2)
            verbose: Print training statistics (default: True)
        
        Returns:
            metrics: Dictionary containing training metrics:
                    - train_accuracy: Accuracy on training set
                    - test_accuracy: Accuracy on test set
                    - train_size: Number of training samples
                    - test_size: Number of test samples
        
        Raises:
            ValueError: If input arrays have incompatible shapes
        """
        # Validate inputs
        if X_clean.ndim != 2 or X_stego.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays, got X_clean.shape={X_clean.shape}, "
                f"X_stego.shape={X_stego.shape}"
            )
        
        if X_clean.shape[1] != X_stego.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: "
                f"X_clean has {X_clean.shape[1]} features, "
                f"X_stego has {X_stego.shape[1]} features"
            )
        
        # Combine datasets and create labels
        # Label 0 = clean, Label 1 = stego
        X = np.vstack([X_clean, X_stego])
        y = np.hstack([
            np.zeros(len(X_clean), dtype=np.int32),  # Clean = 0
            np.ones(len(X_stego), dtype=np.int32)     # Stego = 1
        ])
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y  # Preserve class distribution
        )
        
        # Normalize features if enabled
        if self.normalize_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on train and test sets
        y_train_pred = self.classifier.predict(X_train)
        y_test_pred = self.classifier.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        self.is_trained = True
        
        # Compute additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary', zero_division=0
        )
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        if verbose:
            logger.info("Training complete:")
            logger.info(f"  Train accuracy: {train_accuracy:.4f}")
            logger.info(f"  Test accuracy:  {test_accuracy:.4f}")
            logger.info(f"  Test precision: {precision:.4f}")
            logger.info(f"  Test recall:    {recall:.4f}")
            logger.info(f"  Test F1:        {f1:.4f}")
            logger.info(f"  Train samples:  {len(X_train)}")
            logger.info(f"  Test samples:   {len(X_test)}")
        
        return metrics
    
    def predict(
        self,
        features: np.ndarray,
        return_score: bool = False
    ) -> np.ndarray:
        """
        Predict binary labels for feature vectors.
        
        Args:
            features: Feature vectors, shape (N, D) or (D,)
            return_score: If True, return (labels, scores) tuple (default: False)
        
        Returns:
            labels: Binary predictions (0=clean, 1=stego), shape (N,)
            scores: (Optional) Detection scores, shape (N,)
        
        Raises:
            RuntimeError: If detector is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before prediction")
        
        # Handle single feature vector
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize if enabled
        if self.normalize_features:
            features = self.scaler.transform(features)
        
        # Predict labels
        labels = self.classifier.predict(features)
        
        if return_score:
            scores = self.predict_proba(features)
            return labels, scores
        
        return labels
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict probability or decision function score for stego class.
        
        For Logistic Regression: Returns P(stego | features)
        For Linear SVM: Returns decision function value (distance to hyperplane)
        
        Args:
            features: Feature vectors, shape (N, D) or (D,)
        
        Returns:
            scores: Detection scores, shape (N,)
                   For logistic: probability in [0, 1]
                   For SVM: unbounded decision function value
        
        Raises:
            RuntimeError: If detector is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before prediction")
        
        # Handle single feature vector
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize if enabled
        if self.normalize_features:
            features = self.scaler.transform(features)
        
        # Get scores
        if self.model_type == "logistic":
            # Return probability for class 1 (stego)
            scores = self.classifier.predict_proba(features)[:, 1]
        else:  # SVM
            # Return decision function (signed distance to hyperplane)
            scores = self.classifier.decision_function(features)
        
        return scores
    
    def set_decision_threshold(
        self,
        threshold: float,
        target_fpr: Optional[float] = None,
        X_clean: Optional[np.ndarray] = None,
        X_stego: Optional[np.ndarray] = None
    ) -> float:
        """
        Set the decision threshold for binary classification.
        
        Two modes:
        1. Manual: Set threshold directly
        2. Automatic: Find threshold that achieves target False Positive Rate
        
        Args:
            threshold: Manual threshold value (used if target_fpr is None)
            target_fpr: Target false positive rate (e.g., 0.01 for 1% FPR)
            X_clean: Clean features (required if target_fpr is set)
            X_stego: Stego features (required if target_fpr is set)
        
        Returns:
            threshold: The threshold that was set
        
        Raises:
            ValueError: If target_fpr is set but validation data not provided
        """
        if target_fpr is not None:
            # Automatic threshold selection based on target FPR
            if X_clean is None or X_stego is None:
                raise ValueError(
                    "X_clean and X_stego must be provided when target_fpr is set"
                )
            
            # Compute ROC curve
            fpr, tpr, thresholds = self.compute_roc_curve(X_clean, X_stego)
            
            # Find threshold closest to target FPR
            idx = np.argmin(np.abs(fpr - target_fpr))
            threshold = thresholds[idx]
            actual_fpr = fpr[idx]
            actual_tpr = tpr[idx]
            
            logger.info(
                f"Set threshold={threshold:.4f} for target FPR={target_fpr:.4f} "
                f"(actual FPR={actual_fpr:.4f}, TPR={actual_tpr:.4f})"
            )
        
        self.decision_threshold = threshold
        return threshold
    
    def compute_roc_curve(
        self,
        X_clean: np.ndarray,
        X_stego: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve on validation data.
        
        Args:
            X_clean: Clean video features, shape (N_clean, D)
            X_stego: Stego video features, shape (N_stego, D)
        
        Returns:
            fpr: False positive rates, shape (T,)
            tpr: True positive rates, shape (T,)
            thresholds: Decision thresholds, shape (T,)
        
        Raises:
            RuntimeError: If detector is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before ROC computation")
        
        # Combine datasets and create labels
        X = np.vstack([X_clean, X_stego])
        y_true = np.hstack([
            np.zeros(len(X_clean), dtype=np.int32),
            np.ones(len(X_stego), dtype=np.int32)
        ])
        
        # Get detection scores
        y_scores = self.predict_proba(X)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        return fpr, tpr, thresholds
    
    def compute_roc_auc(
        self,
        X_clean: np.ndarray,
        X_stego: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute ROC-AUC and related metrics.
        
        Args:
            X_clean: Clean video features, shape (N_clean, D)
            X_stego: Stego video features, shape (N_stego, D)
            verbose: Print evaluation results (default: True)
        
        Returns:
            results: Dictionary containing:
                    - auc: Area under ROC curve
                    - tpr_at_1_fpr: True positive rate at 1% FPR
                    - tpr_at_5_fpr: True positive rate at 5% FPR
                    - eer: Equal error rate (where FPR = 1 - TPR)
        """
        # Compute ROC curve
        fpr, tpr, thresholds = self.compute_roc_curve(X_clean, X_stego)
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)
        
        # Find TPR at specific FPR values
        tpr_at_1_fpr = tpr[np.argmin(np.abs(fpr - 0.01))]
        tpr_at_5_fpr = tpr[np.argmin(np.abs(fpr - 0.05))]
        
        # Find Equal Error Rate (EER): point where FPR = FNR = 1 - TPR
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        
        results = {
            'auc': roc_auc,
            'tpr_at_1_fpr': tpr_at_1_fpr,
            'tpr_at_5_fpr': tpr_at_5_fpr,
            'eer': eer
        }
        
        if verbose:
            logger.info("ROC-AUC Evaluation:")
            logger.info(f"  AUC:              {roc_auc:.4f}")
            logger.info(f"  TPR @ 1% FPR:     {tpr_at_1_fpr:.4f}")
            logger.info(f"  TPR @ 5% FPR:     {tpr_at_5_fpr:.4f}")
            logger.info(f"  Equal Error Rate: {eer:.4f}")
        
        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def train_detector(
    X_clean: np.ndarray,
    X_stego: np.ndarray,
    model_type: Literal["logistic", "svm"] = "logistic",
    test_size: float = 0.2,
    random_seed: int = 42,
    verbose: bool = True
) -> ClassicalSteganalysisDetector:
    """
    Convenience function to train a detector.
    
    Args:
        X_clean: Clean video features, shape (N_clean, D)
        X_stego: Stego video features, shape (N_stego, D)
        model_type: "logistic" or "svm" (default: "logistic")
        test_size: Fraction for test set (default: 0.2)
        random_seed: Random seed (default: 42)
        verbose: Print training info (default: True)
    
    Returns:
        detector: Trained ClassicalSteganalysisDetector
    """
    detector = ClassicalSteganalysisDetector(
        model_type=model_type,
        random_seed=random_seed,
        normalize_features=True
    )
    
    detector.train(X_clean, X_stego, test_size=test_size, verbose=verbose)
    
    return detector


# =============================================================================
# SYNTHETIC DEMO
# =============================================================================

def synthetic_demo():
    """
    Demonstrate detector on synthetic clean vs noisy features.
    
    This demo simulates:
    - Clean features: Gaussian distribution N(0, 1)
    - Stego features: Shifted Gaussian N(0.5, 1.2) to simulate embedding artifacts
    
    The shift represents subtle statistical changes introduced by embedding.
    """
    print("="*70)
    print("Classical Steganalysis Detector - Synthetic Demo")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulate clean and stego features
    print("\n[1] Generating synthetic features...")
    n_clean = 200
    n_stego = 200
    n_features = 54  # Matches FlowFeatureExtractor output
    
    # Clean features: standard normal
    X_clean = np.random.randn(n_clean, n_features).astype(np.float32)
    
    # Stego features: shifted and scaled normal (simulates embedding artifacts)
    # Mean shift = 0.5, std scale = 1.2
    X_stego = (np.random.randn(n_stego, n_features) * 1.2 + 0.5).astype(np.float32)
    
    print(f"  Clean features: {X_clean.shape} (mean={X_clean.mean():.3f})")
    print(f"  Stego features: {X_stego.shape} (mean={X_stego.mean():.3f})")
    
    # Test 1: Logistic Regression
    print("\n[2] Training Logistic Regression detector...")
    detector_lr = ClassicalSteganalysisDetector(
        model_type="logistic",
        random_seed=42,
        normalize_features=True
    )
    
    metrics_lr = detector_lr.train(
        X_clean, X_stego,
        test_size=0.2,
        verbose=False
    )
    
    print(f"  Train accuracy: {metrics_lr['train_accuracy']:.4f}")
    print(f"  Test accuracy:  {metrics_lr['test_accuracy']:.4f}")
    
    # Test 2: Linear SVM
    print("\n[3] Training Linear SVM detector...")
    detector_svm = ClassicalSteganalysisDetector(
        model_type="svm",
        random_seed=42,
        normalize_features=True
    )
    
    metrics_svm = detector_svm.train(
        X_clean, X_stego,
        test_size=0.2,
        verbose=False
    )
    
    print(f"  Train accuracy: {metrics_svm['train_accuracy']:.4f}")
    print(f"  Test accuracy:  {metrics_svm['test_accuracy']:.4f}")
    
    # Test 3: ROC-AUC evaluation
    print("\n[4] Computing ROC-AUC for Logistic Regression...")
    
    # Generate fresh test data
    X_clean_test = np.random.randn(100, n_features).astype(np.float32)
    X_stego_test = (np.random.randn(100, n_features) * 1.2 + 0.5).astype(np.float32)
    
    results = detector_lr.compute_roc_auc(
        X_clean_test, X_stego_test,
        verbose=False
    )
    
    print(f"  AUC:              {results['auc']:.4f}")
    print(f"  TPR @ 1% FPR:     {results['tpr_at_1_fpr']:.4f}")
    print(f"  TPR @ 5% FPR:     {results['tpr_at_5_fpr']:.4f}")
    print(f"  Equal Error Rate: {results['eer']:.4f}")
    
    # Test 4: Individual predictions
    print("\n[5] Testing individual predictions...")
    
    # Test on single clean sample
    clean_sample = X_clean_test[0:1]
    label_clean, score_clean = detector_lr.predict(clean_sample, return_score=True)
    print(f"  Clean sample: label={label_clean[0]}, score={score_clean[0]:.4f}")
    
    # Test on single stego sample
    stego_sample = X_stego_test[0:1]
    label_stego, score_stego = detector_lr.predict(stego_sample, return_score=True)
    print(f"  Stego sample: label={label_stego[0]}, score={score_stego[0]:.4f}")
    
    # Test 5: Decision threshold tuning
    print("\n[6] Testing threshold tuning for target FPR...")
    
    # Find threshold for 1% FPR
    threshold = detector_lr.set_decision_threshold(
        threshold=0.5,
        target_fpr=0.01,
        X_clean=X_clean_test,
        X_stego=X_stego_test
    )
    
    print(f"  Threshold set to: {threshold:.4f}")
    
    # Test 6: Model comparison
    print("\n[7] Model comparison (Logistic vs SVM)...")
    
    results_svm = detector_svm.compute_roc_auc(
        X_clean_test, X_stego_test,
        verbose=False
    )
    
    print(f"  Logistic AUC: {results['auc']:.4f}")
    print(f"  SVM AUC:      {results_svm['auc']:.4f}")
    print(f"  Winner:       {'Logistic' if results['auc'] > results_svm['auc'] else 'SVM'}")
    
    print("\n" + "="*70)
    print("Demo complete ✓")
    print("="*70)
    print("\nKey Observations:")
    print("1. Both models achieve high accuracy on synthetic data (easy task)")
    print("2. AUC > 0.9 indicates strong separability of shifted distributions")
    print("3. Real stego detection is harder (expect AUC 0.6-0.8)")
    print("4. Threshold tuning allows trading off FPR vs TPR")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Run synthetic demo
    synthetic_demo()