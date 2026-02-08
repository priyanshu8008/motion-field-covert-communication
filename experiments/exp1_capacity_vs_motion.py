import numpy as np
import yaml
from pathlib import Path

from module8_receiver.capacity import CapacityEstimator


# --------------------------------------------------
# Load global config (YAML → dict)
# --------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[1] / "default_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --------------------------------------------------
# Capacity experiment
# --------------------------------------------------
estimator = CapacityEstimator(config)

H, W = 64, 64
magnitudes = [0.5, 1.0, 2.0, 5.0, 10.0]

print("motion_magnitude,capacity")

for mag in magnitudes:
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[:, :, 0] = mag  # horizontal motion

    capacity = estimator.compute_capacity(flow)
    print(f"{mag},{capacity}")
