import numpy as np
from src.module8_steganalysis.classical_detector import ClassicalSteganalysisDetector

# -----------------------------
# Deterministic seed
# -----------------------------
np.random.seed(42)

# -----------------------------
# Synthetic feature generation
# -----------------------------
N = 200          # samples per class
D = 54           # feature dimension

# Clean videos: baseline distribution
X_clean = np.random.normal(loc=0.0, scale=1.0, size=(N, D))

# Stego videos: slightly shifted statistics (simulates embedding artifacts)
X_stego = np.random.normal(loc=0.4, scale=1.0, size=(N, D))

# -----------------------------
# Train detector
# -----------------------------
detector = ClassicalSteganalysisDetector(
    model_type="logistic",   # or "svm"
    random_seed=42
)

detector.train(X_clean, X_stego)

# -----------------------------
# Evaluate ROC / AUC
# -----------------------------
metrics = detector.compute_roc_auc(X_clean, X_stego)

print("\n=== Detector Evaluation ===")
print(f"AUC            : {metrics['auc']:.4f}")
print(f"TPR @ 1% FPR   : {metrics['tpr_at_1_fpr']:.4f}")
print(f"TPR @ 5% FPR   : {metrics['tpr_at_5_fpr']:.4f}")
print(f"EER            : {metrics['eer']:.4f}")

# -----------------------------
# Sanity assertions
# -----------------------------
assert metrics["auc"] > 0.8, "Detector failed to learn separable distributions"

print("\n✅ Classical detector sanity test PASSED")
