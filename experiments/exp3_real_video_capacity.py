import csv
import yaml
import cv2
import numpy as np

from pathlib import Path

from module1_video_io.video_loader import load_video
from module2_motion_extraction.flow_extractor import OpticalFlowExtractor
from module8_receiver.capacity import CapacityEstimator

def downscale_frames(frames, max_size=512):
    """
    Downscale frames so max(H, W) <= max_size
    Preserves aspect ratio.
    """
    out = []
    for f in frames:
        h, w = f.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            f = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.append(f)
    return out


# --------------------------------------------------
# Load config (canonical method)
# --------------------------------------------------
CONFIG_PATH = Path("default_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --------------------------------------------------
# Paths
# --------------------------------------------------
VIDEO_PATH = Path("tests/assets/test_motion_video.mp4")
RAFT_MODEL_PATH = "src/module2_motion_extraction/models/raft-things.pth"

# --------------------------------------------------
# Load video
# --------------------------------------------------
frames, metadata = load_video(VIDEO_PATH)

# CPU-safe limits
frames = frames[:8]
frames = downscale_frames(frames, 512)
# --------------------------------------------------
# Optical flow
# --------------------------------------------------
flow_extractor = OpticalFlowExtractor(
    model_path=RAFT_MODEL_PATH,
    device="cpu"
)

flows = flow_extractor.batch_extract(frames)

# --------------------------------------------------
# Capacity estimation
# --------------------------------------------------
estimator = CapacityEstimator(config)

results = []
for i, flow in enumerate(flows):
    cap = estimator.compute_capacity(flow)
    results.append((i, cap))

# --------------------------------------------------
# Save CSV
# --------------------------------------------------
OUT = Path("experiments/results_real_video_capacity.csv")
OUT.parent.mkdir(exist_ok=True)

with open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "capacity_bits"])
    writer.writerows(results)

print(f"Saved capacity results → {OUT}")
