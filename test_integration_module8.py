import numpy as np
from pathlib import Path

from src.module2_motion_extraction.flow_extractor import OpticalFlowExtractor
from src.module8_steganalysis.feature_extractor import FlowFeatureExtractor

# --------------------------------------------------
# Resolve RAFT model path (PROJECT-RELATIVE)
# --------------------------------------------------
RAFT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "src"
    / "module2_motion_extraction"
    / "models"
    / "raft-things.pth"
)

assert RAFT_MODEL_PATH.exists(), f"RAFT model not found: {RAFT_MODEL_PATH}"

# --------------------------------------------------
# Create synthetic frames (SMALL to avoid OOM)
# --------------------------------------------------
H, W = 64, 64
frame1 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
frame2 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

# --------------------------------------------------
# Extract optical flow (Module 2)
# --------------------------------------------------
flow_extractor = OpticalFlowExtractor(
    model_path=str(RAFT_MODEL_PATH),
    device="cpu"
)

flow = flow_extractor.extract_flow(frame1, frame2)

# --------------------------------------------------
# Extract steganalysis features (Module 8)
# --------------------------------------------------
extractor = FlowFeatureExtractor()
features = extractor.extract_features([flow])


print("Flow shape:", flow.shape)
print("Feature shape:", features.shape)
