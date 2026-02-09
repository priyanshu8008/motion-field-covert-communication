#!/usr/bin/env python3
"""
End-to-End Steganalysis Evaluation Script

This script evaluates the performance of the classical steganalysis detector
on a dataset of clean and stego videos. It performs the complete pipeline:
1. Load videos from disk
2. Extract optical flow fields
3. Extract handcrafted features
4. Train binary classifier
5. Evaluate ROC-AUC performance
6. Save results and plots

Author: Motion Covert Communication System
Module: 8 (Steganalysis Attacker - Evaluation)
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None):
        if desc:
            logging.info(desc)
        return iterable

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Module 8 components (steganalysis)
from module8_steganalysis import FlowFeatureExtractor, ClassicalSteganalysisDetector

# Import Module 1 (Video I/O) and Module 2 (Optical Flow)
# These should already be implemented in the project
try:
    from module1_video_io import VideoLoader
    MODULE1_AVAILABLE = True
except ImportError:
    MODULE1_AVAILABLE = False
    logging.warning("Module 1 (Video I/O) not available - using mock")

try:
    from module2_motion_extraction import OpticalFlowExtractor
    MODULE2_AVAILABLE = True
except ImportError:
    MODULE2_AVAILABLE = False
    logging.warning("Module 2 (Optical Flow) not available - using mock")


# =============================================================================
# MOCK IMPLEMENTATIONS (for testing when modules not available)
# =============================================================================

class MockVideoLoader:
    """Mock video loader for testing when Module 1 is not available."""
    
    def load_video(self, path: str) -> Tuple[List[np.ndarray], Dict]:
        """Load video and return random frames for testing."""
        logging.warning(f"MOCK: Loading video from {path}")
        # Generate 10 random frames (64x64 RGB)
        frames = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        metadata = {'fps': 30, 'width': 64, 'height': 64, 'num_frames': 10}
        return frames, metadata


class MockOpticalFlowExtractor:
    """Mock flow extractor for testing when Module 2 is not available."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        logging.warning("MOCK: Initialized optical flow extractor")
    
    def batch_extract(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Generate random flow fields for testing."""
        logging.warning(f"MOCK: Extracting flow from {len(frames)} frames")
        h, w = frames[0].shape[:2]
        # Generate N-1 flow fields
        flows = [
            np.random.randn(h, w, 2).astype(np.float32) * 2.0
            for _ in range(len(frames) - 1)
        ]
        return flows


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(verbose: bool = True):
    """Configure logging for the evaluation script."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# =============================================================================
# VIDEO PROCESSING PIPELINE
# =============================================================================

def load_videos_from_directory(
    video_dir: str,
    max_videos: int = None
) -> List[str]:
    """
    Scan directory for video files.
    
    Args:
        video_dir: Path to directory containing videos
        max_videos: Maximum number of videos to load (None = all)
    
    Returns:
        video_paths: List of absolute paths to video files
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Directory not found: {video_dir}")
    
    # Supported video extensions
    extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Find all video files
    video_paths = []
    for ext in extensions:
        video_paths.extend(video_dir.glob(f'*{ext}'))
        video_paths.extend(video_dir.glob(f'*{ext.upper()}'))
    
    video_paths = sorted([str(p) for p in video_paths])
    
    if max_videos is not None:
        video_paths = video_paths[:max_videos]
    
    logging.info(f"Found {len(video_paths)} videos in {video_dir}")
    
    return video_paths


def extract_flows_from_video(
    video_path: str,
    flow_extractor: Any,  # OpticalFlowExtractor or Mock
    video_loader: Any  # VideoLoader or Mock
) -> List[np.ndarray]:
    """
    Extract optical flow from a single video.
    
    Pipeline:
    1. Load video frames using Module 1
    2. Extract flow fields using Module 2
    
    Args:
        video_path: Path to video file
        flow_extractor: OpticalFlowExtractor instance (Module 2)
        video_loader: VideoLoader instance (Module 1)
    
    Returns:
        flows: List of flow fields, each of shape (H, W, 2)
    """
    # Stage 1: Load video frames
    frames, metadata = video_loader.load_video(video_path)
    logging.debug(
        f"  Loaded {len(frames)} frames "
        f"({metadata.get('width', '?')}x{metadata.get('height', '?')})"
    )
    
    # Stage 2: Extract optical flow
    flows = flow_extractor.batch_extract(frames)
    logging.debug(f"  Extracted {len(flows)} flow fields")
    
    return flows


def extract_features_from_flows(
    flows: List[np.ndarray],
    feature_extractor: FlowFeatureExtractor
) -> np.ndarray:
    """
    Extract statistical features from optical flow sequence.
    
    Uses Module 8 feature extractor to compute 54-D feature vector.
    
    Args:
        flows: List of flow fields
        feature_extractor: FlowFeatureExtractor instance (Module 8)
    
    Returns:
        features: Feature vector of shape (54,)
    """
    features = feature_extractor.extract_features(flows)
    return features


def process_video_dataset(
    video_paths: List[str],
    flow_extractor: Any,  # OpticalFlowExtractor or Mock
    video_loader: Any,  # VideoLoader or Mock
    feature_extractor: FlowFeatureExtractor,
    label: int,
    label_name: str
) -> np.ndarray:
    """
    Process a dataset of videos and extract features.
    
    Complete pipeline per video:
    1. Load frames (Module 1)
    2. Extract flow (Module 2)
    3. Extract features (Module 8)
    
    Args:
        video_paths: List of video file paths
        flow_extractor: OpticalFlowExtractor instance
        video_loader: VideoLoader instance
        feature_extractor: FlowFeatureExtractor instance
        label: Integer label for this dataset (0=clean, 1=stego)
        label_name: Human-readable label name
    
    Returns:
        X: Feature matrix of shape (N_videos, 54)
    """
    logging.info(f"Processing {len(video_paths)} {label_name} videos...")
    
    features_list = []
    
    for video_path in tqdm(video_paths, desc=f"Extracting {label_name}"):
        try:
            # Extract flows from video
            flows = extract_flows_from_video(
                video_path,
                flow_extractor,
                video_loader
            )
            
            # Skip videos with insufficient motion
            if len(flows) == 0:
                logging.warning(f"  Skipping {video_path}: no flow fields")
                continue
            
            # Extract features
            features = extract_features_from_flows(flows, feature_extractor)
            features_list.append(features)
            
        except Exception as e:
            logging.error(f"  Error processing {video_path}: {e}")
            continue
    
    if len(features_list) == 0:
        raise ValueError(f"No valid {label_name} videos processed")
    
    X = np.vstack(features_list)
    logging.info(f"  Extracted features: {X.shape}")
    
    return X


# =============================================================================
# EVALUATION AND VISUALIZATION
# =============================================================================

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    output_path: str
):
    """
    Plot and save ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under curve
        output_path: Path to save PNG file
    """
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Formatting
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Steganalysis ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved ROC curve to {output_path}")


def save_results_csv(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_path: str
):
    """
    Save detection results to CSV.
    
    Args:
        y_true: Ground truth labels (0=clean, 1=stego)
        y_scores: Detection scores
        output_path: Path to save CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'true_label', 'score'])
        
        for i, (label, score) in enumerate(zip(y_true, y_scores)):
            writer.writerow([i, int(label), float(score)])
    
    logging.info(f"Saved results to {output_path}")


def save_metrics_json(
    metrics: Dict,
    output_path: str
):
    """
    Save evaluation metrics to JSON.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Saved metrics to {output_path}")


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_steganalysis(
    clean_video_dir: str,
    stego_video_dir: str,
    config: Dict,
    output_dir: str,
    max_videos_per_class: int = None
) -> Dict:
    """
    Run complete steganalysis evaluation.
    
    Pipeline:
    1. Load clean and stego videos
    2. Extract optical flow (Module 2)
    3. Extract features (Module 8)
    4. Train detector (Module 8)
    5. Evaluate on test set
    6. Compute ROC-AUC
    7. Save results
    
    Args:
        clean_video_dir: Directory containing clean videos
        stego_video_dir: Directory containing stego videos
        config: Configuration dictionary
        output_dir: Directory to save results
        max_videos_per_class: Limit number of videos (None = all)
    
    Returns:
        results: Dictionary containing evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # STAGE 1: INITIALIZE MODULES
    # =========================================================================
    
    logging.info("="*70)
    logging.info("STAGE 1: Initializing modules")
    logging.info("="*70)
    
    # Initialize Module 1 (Video I/O)
    if MODULE1_AVAILABLE:
        video_loader = VideoLoader()
    else:
        video_loader = MockVideoLoader()
        logging.warning("Using MOCK video loader (Module 1 not available)")
    
    # Initialize Module 2 (Optical Flow)
    if MODULE2_AVAILABLE:
        flow_config = config.get('optical_flow', {})
        model_path = flow_config.get('weights_path', 'models/raft/raft-things.pth')
        device = flow_config.get('device', 'cpu')
        flow_extractor = OpticalFlowExtractor(model_path, device)
    else:
        flow_extractor = MockOpticalFlowExtractor('mock_path', 'cpu')
        logging.warning("Using MOCK flow extractor (Module 2 not available)")
    
    # Initialize Module 8 (Feature Extractor)
    steg_config = config.get('steganalysis', {}).get('features', {})
    magnitude_bins = steg_config.get('magnitude_bins', 32)
    direction_bins = steg_config.get('direction_bins', 16)
    
    feature_extractor = FlowFeatureExtractor(
        magnitude_bins=magnitude_bins,
        direction_bins=direction_bins
    )
    logging.info(f"Initialized feature extractor ({feature_extractor.feature_dim}-D)")
    
    # =========================================================================
    # STAGE 2: LOAD VIDEOS
    # =========================================================================
    
    logging.info("")
    logging.info("="*70)
    logging.info("STAGE 2: Loading videos")
    logging.info("="*70)
    
    clean_paths = load_videos_from_directory(clean_video_dir, max_videos_per_class)
    stego_paths = load_videos_from_directory(stego_video_dir, max_videos_per_class)
    
    if len(clean_paths) == 0 or len(stego_paths) == 0:
        raise ValueError("Need at least one video in each class")
    
    # =========================================================================
    # STAGE 3: EXTRACT FEATURES
    # =========================================================================
    
    logging.info("")
    logging.info("="*70)
    logging.info("STAGE 3: Extracting features from videos")
    logging.info("="*70)
    
    # Process clean videos
    X_clean = process_video_dataset(
        clean_paths,
        flow_extractor,
        video_loader,
        feature_extractor,
        label=0,
        label_name="clean"
    )
    
    # Process stego videos
    X_stego = process_video_dataset(
        stego_paths,
        flow_extractor,
        video_loader,
        feature_extractor,
        label=1,
        label_name="stego"
    )
    
    logging.info(f"Total dataset: {len(X_clean)} clean, {len(X_stego)} stego")
    
    # =========================================================================
    # STAGE 4: TRAIN DETECTOR
    # =========================================================================
    
    logging.info("")
    logging.info("="*70)
    logging.info("STAGE 4: Training steganalysis detector")
    logging.info("="*70)
    
    detector_config = config.get('steganalysis', {}).get('detector', {})
    model_type = detector_config.get('architecture', 'logistic')
    if model_type == 'mlp':
        model_type = 'logistic'  # Map MLP to logistic for classical detector
    
    random_seed = config.get('system', {}).get('random_seed', 42)
    test_size = config.get('steganalysis', {}).get('training', {}).get(
        'validation_split', 0.2
    )
    
    detector = ClassicalSteganalysisDetector(
        model_type=model_type,
        random_seed=random_seed,
        normalize_features=True
    )
    
    # Train with explicit train/test split
    train_metrics = detector.train(
        X_clean, X_stego,
        test_size=test_size,
        verbose=True
    )
    
    # =========================================================================
    # STAGE 5: EVALUATE ON TEST SET
    # =========================================================================
    
    logging.info("")
    logging.info("="*70)
    logging.info("STAGE 5: Evaluating on test set")
    logging.info("="*70)
    
    # Create fresh test set (20% held out)
    from sklearn.model_selection import train_test_split
    
    # Combine and split
    X = np.vstack([X_clean, X_stego])
    y = np.hstack([
        np.zeros(len(X_clean), dtype=np.int32),
        np.ones(len(X_stego), dtype=np.int32)
    ])
    
    # Deterministic split
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )
    
    logging.info(f"Test set: {len(X_test)} samples ({np.sum(y_test)} stego)")
    
    # Get predictions
    y_pred = detector.predict(X_test)
    y_scores = detector.predict_proba(X_test)
    
    # Compute accuracy
    from sklearn.metrics import accuracy_score, classification_report
    test_accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Clean', 'Stego'],
        digits=4
    ))
    
    # =========================================================================
    # STAGE 6: COMPUTE ROC-AUC
    # =========================================================================
    
    logging.info("")
    logging.info("="*70)
    logging.info("STAGE 6: Computing ROC curve and AUC")
    logging.info("="*70)
    
    # Recompute on full test set for ROC
    X_clean_test = X_test[y_test == 0]
    X_stego_test = X_test[y_test == 1]
    
    roc_metrics = detector.compute_roc_auc(
        X_clean_test, X_stego_test,
        verbose=True
    )
    
    # Get ROC curve data
    fpr, tpr, thresholds = detector.compute_roc_curve(
        X_clean_test, X_stego_test
    )
    
    # =========================================================================
    # STAGE 7: SAVE RESULTS
    # =========================================================================
    
    logging.info("")
    logging.info("="*70)
    logging.info("STAGE 7: Saving results")
    logging.info("="*70)
    
    # Save ROC curve plot
    roc_plot_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(fpr, tpr, roc_metrics['auc'], roc_plot_path)
    
    # Save predictions CSV
    csv_path = os.path.join(output_dir, 'predictions.csv')
    save_results_csv(y_test, y_scores, csv_path)
    
    # Compile all metrics
    all_metrics = {
        'test_accuracy': float(test_accuracy),
        'train_metrics': {k: float(v) for k, v in train_metrics.items()},
        'roc_metrics': {k: float(v) for k, v in roc_metrics.items()},
        'dataset': {
            'n_clean_total': int(len(X_clean)),
            'n_stego_total': int(len(X_stego)),
            'n_test': int(len(X_test)),
            'test_split': float(test_size)
        },
        'config': {
            'model_type': model_type,
            'random_seed': int(random_seed),
            'feature_dim': int(feature_extractor.feature_dim)
        }
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    save_metrics_json(all_metrics, metrics_path)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    logging.info("")
    logging.info("="*70)
    logging.info("EVALUATION COMPLETE")
    logging.info("="*70)
    logging.info(f"AUC: {roc_metrics['auc']:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"TPR @ 1% FPR: {roc_metrics['tpr_at_1_fpr']:.4f}")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("="*70)
    
    return all_metrics


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate steganalysis detector on clean vs stego videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python evaluate_steganalysis.py \\
      --clean-dir data/videos/clean \\
      --stego-dir data/videos/stego \\
      --output-dir results/steganalysis
  
  # With custom config
  python evaluate_steganalysis.py \\
      --clean-dir data/videos/clean \\
      --stego-dir data/videos/stego \\
      --config config/default_config.yaml \\
      --output-dir results/experiment1
  
  # Limit number of videos (for quick testing)
  python evaluate_steganalysis.py \\
      --clean-dir data/videos/clean \\
      --stego-dir data/videos/stego \\
      --max-videos 10 \\
      --output-dir results/test
        """
    )
    
    parser.add_argument(
        '--clean-dir',
        type=str,
        required=True,
        help='Directory containing clean videos'
    )
    
    parser.add_argument(
        '--stego-dir',
        type=str,
        required=True,
        help='Directory containing stego videos'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration YAML file (default: config/default_config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/steganalysis',
        help='Directory to save results (default: results/steganalysis)'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum videos per class (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        config: Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
    else:
        logging.warning(f"Config file not found: {config_path}, using defaults")
        config = {}
    
    return config


# =============================================================================
# EXPERIMENTAL VALIDITY NOTES
# =============================================================================

"""
EXPERIMENTAL VALIDITY CONSIDERATIONS:

1. **Data Leakage Prevention:**
   - Explicit train/test split with stratification
   - Deterministic random seed ensures reproducibility
   - Test set is completely held out during training

2. **Statistical Significance:**
   - Minimum sample requirement: 20+ videos per class for meaningful AUC
   - Stratified split preserves class distribution
   - Standard 80/20 train/test split

3. **Generalization Testing:**
   - Test videos are unseen during training
   - ROC curve computed on independent test set
   - Multiple metrics (AUC, accuracy, TPR@FPR) provide comprehensive view

4. **Reproducibility:**
   - Fixed random seed (default: 42)
   - Deterministic feature extraction
   - Versioned configuration files

5. **Limitations:**
   - Assumes clean and stego videos are from same source distribution
   - Does not test cross-dataset generalization
   - Linear models may underestimate detection difficulty
   - No confidence intervals (would require bootstrap resampling)

6. **Threats to Validity:**
   - Small sample size (< 20 videos per class) → unstable AUC estimates
   - Imbalanced classes → biased accuracy, use AUC as primary metric
   - Video diversity → results may not generalize to different scenes/motions
   - Compression effects → should match stego video compression level

7. **Best Practices Applied:**
   - Feature normalization (StandardScaler)
   - Balanced training set (equal clean/stego samples)
   - Multiple evaluation metrics
   - Clear separation of concerns (modules 1, 2, 8)
   - Extensive logging for debugging
"""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for evaluation script."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logging.info("Steganalysis Evaluation Script")
    logging.info(f"Clean videos: {args.clean_dir}")
    logging.info(f"Stego videos: {args.stego_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Run evaluation
    try:
        results = evaluate_steganalysis(
            clean_video_dir=args.clean_dir,
            stego_video_dir=args.stego_dir,
            config=config,
            output_dir=args.output_dir,
            max_videos_per_class=args.max_videos
        )
        
        # Print summary
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"AUC:                    {results['roc_metrics']['auc']:.4f}")
        print(f"Test Accuracy:          {results['test_accuracy']:.4f}")
        print(f"TPR @ 1% FPR:           {results['roc_metrics']['tpr_at_1_fpr']:.4f}")
        print(f"TPR @ 5% FPR:           {results['roc_metrics']['tpr_at_5_fpr']:.4f}")
        print(f"Equal Error Rate:       {results['roc_metrics']['eer']:.4f}")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())