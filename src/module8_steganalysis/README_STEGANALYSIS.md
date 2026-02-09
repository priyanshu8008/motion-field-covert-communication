# Steganalysis Evaluation Script

## Overview

`evaluate_steganalysis.py` is an end-to-end evaluation script for the classical steganalysis detector. It processes clean and stego videos, extracts motion-field features, trains a detector, and computes ROC-AUC metrics.

## Usage

### Basic Command

```bash
python evaluate_steganalysis.py \
    --clean-dir data/videos/clean \
    --stego-dir data/videos/stego \
    --output-dir results/steganalysis
```

### With Custom Configuration

```bash
python evaluate_steganalysis.py \
    --clean-dir data/videos/clean \
    --stego-dir data/videos/stego \
    --config config/default_config.yaml \
    --output-dir results/experiment1 \
    --verbose
```

### Quick Test (Limited Videos)

```bash
python evaluate_steganalysis.py \
    --clean-dir data/videos/clean \
    --stego-dir data/videos/stego \
    --max-videos 10 \
    --output-dir results/test
```

## Arguments

- `--clean-dir`: Directory containing clean (cover) videos [REQUIRED]
- `--stego-dir`: Directory containing stego videos [REQUIRED]
- `--config`: Path to YAML configuration file (default: `config/default_config.yaml`)
- `--output-dir`: Directory to save results (default: `results/steganalysis`)
- `--max-videos`: Maximum videos per class (default: all videos)
- `--verbose`: Enable detailed logging

## Pipeline Stages

The script performs 7 stages:

1. **Initialize Modules**: Load Module 1 (Video I/O), Module 2 (Optical Flow), Module 8 (Features)
2. **Load Videos**: Scan directories for video files
3. **Extract Features**: Process each video → flow fields → 54-D feature vector
4. **Train Detector**: Train Logistic Regression or Linear SVM with 80/20 split
5. **Evaluate**: Compute predictions on test set
6. **ROC-AUC**: Calculate ROC curve and AUC metrics
7. **Save Results**: Export CSV, JSON, and PNG files

## Output Files

The script creates three files in `--output-dir`:

### 1. `predictions.csv`
Detection scores for each test sample:
```csv
sample_id,true_label,score
0,1,0.8523
1,0,0.1234
...
```

### 2. `metrics.json`
Comprehensive evaluation metrics:
```json
{
  "test_accuracy": 0.8750,
  "train_metrics": {...},
  "roc_metrics": {
    "auc": 0.7823,
    "tpr_at_1_fpr": 0.3200,
    "tpr_at_5_fpr": 0.6100,
    "eer": 0.1850
  },
  "dataset": {...},
  "config": {...}
}
```

### 3. `roc_curve.png`
ROC curve visualization with AUC annotation

## Configuration

The script uses `default_config.yaml` for parameters. Key settings:

```yaml
steganalysis:
  features:
    magnitude_bins: 32      # Histogram bins for magnitude
    direction_bins: 16      # Histogram bins for direction
  
  detector:
    architecture: "logistic"  # "logistic" or "svm"
  
  training:
    validation_split: 0.2   # Test set fraction

system:
  random_seed: 42           # For reproducibility
```

## Requirements

### Python Packages
- numpy
- scikit-learn
- matplotlib
- pyyaml

### Project Modules
- Module 1: Video I/O (`module1_video_io`)
- Module 2: Optical Flow (`module2_motion_extraction`)
- Module 8: Steganalysis (`module8_steganalysis`)

**Note**: If Module 1 or Module 2 are not available, the script uses mock implementations for testing.

## Expected Performance

### Baseline (Handcrafted Features + Linear Model)

| Embedding Strength | Expected AUC | Interpretation |
|-------------------|--------------|----------------|
| Low (ε ≤ 0.5)     | 0.55-0.65    | Near random guessing |
| Medium (ε = 0.5-1.0) | 0.65-0.75 | Moderate detection |
| High (ε > 1.0)    | 0.75-0.85    | Clear artifacts |

### Minimum Sample Requirements

- **Minimum**: 20 videos per class (40 total)
- **Recommended**: 50+ videos per class (100+ total)
- **Optimal**: 100+ videos per class (200+ total)

Smaller datasets will produce unstable AUC estimates.

## Experimental Validity

### Strengths
✓ Deterministic train/test split (reproducible)
✓ Stratified sampling (preserves class balance)
✓ Feature normalization (prevents scale bias)
✓ Multiple metrics (AUC, accuracy, TPR@FPR)
✓ Independent test set (no data leakage)

### Limitations
⚠ No cross-validation (single split)
⚠ No confidence intervals
⚠ Assumes i.i.d. data (same distribution)
⚠ Linear model may underestimate detectability

### Best Practices
1. Use ≥50 videos per class
2. Match compression levels (clean vs stego)
3. Use diverse video content (scenes, motion)
4. Report multiple metrics (not just accuracy)
5. Set random seed for reproducibility

## Interpreting Results

### AUC Score
- **0.50**: Random guessing (detector fails)
- **0.60-0.70**: Weak detection (baseline expected)
- **0.70-0.80**: Moderate detection
- **0.80-0.90**: Strong detection (high payload or weak embedding)
- **> 0.90**: Very strong detection (embedding is easily detectable)

### TPR @ 1% FPR
Critical metric for operational deployment:
- **< 0.20**: Embedding is secure (hard to detect with low false alarms)
- **0.20-0.50**: Moderate security risk
- **> 0.50**: High security risk (easily detected)

## Troubleshooting

### Error: "Directory not found"
Check that `--clean-dir` and `--stego-dir` paths exist and contain video files.

### Error: "No valid videos processed"
- Check video file extensions (.mp4, .avi, .mov, .mkv)
- Verify videos are readable and not corrupted
- Check Module 1 (VideoLoader) is working

### Warning: "Using MOCK video loader"
Module 1 or 2 not found. Install required modules or use mock for testing.

### Low AUC (< 0.55)
- Insufficient sample size (need more videos)
- Embedding is too weak (ε too small)
- Clean/stego videos are too similar
- Features may not capture embedding artifacts

### High AUC (> 0.90)
- Embedding is easily detectable (too strong)
- Possible data leakage (check train/test split)
- Features are very discriminative (good for attacker)

## Example Output

```
======================================================================
EVALUATION COMPLETE
======================================================================
AUC: 0.7234
Test Accuracy: 0.7500
TPR @ 1% FPR: 0.3100
Results saved to: results/steganalysis
======================================================================
```

## Contact

For issues or questions about the steganalysis evaluation:
- Check Module 8 documentation
- Review ARCHITECTURE.md for system overview
- Verify Module 1 and Module 2 are properly installed