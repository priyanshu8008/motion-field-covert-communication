import numpy as np
import yaml
from pathlib import Path

from module6_motion_modulation.modulator import MotionFieldModulator
from module5_ecc.encoder import ecc_encode
from module5_ecc.decoder import ecc_decode
from module8_receiver.capacity import CapacityEstimator   # ✅ FIXED IMPORT

# --------------------------------------------------
# Load config
# --------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[1] / "default_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --------------------------------------------------
# 1. Synthetic high-motion flow
# --------------------------------------------------
H, W = 64, 64
flow = np.zeros((H, W, 2), dtype=np.float32)

# Strong uniform horizontal motion
flow[:, :, 0] = 5.0

# --------------------------------------------------
# 2. Capacity check
# --------------------------------------------------
capacity = CapacityEstimator(config).compute_capacity(flow)
assert capacity > 0, "Synthetic flow must have non-zero capacity"

# --------------------------------------------------
# 3. Payload → ECC encode
# --------------------------------------------------
payload = b"HELLO_COVERT"
encoded_bits = ecc_encode(payload, config=config)

# --------------------------------------------------
# 4. Embed bits (Module 6)
# --------------------------------------------------
modulator = MotionFieldModulator()

mod_flow, meta = modulator.embed_bits(
    flow=flow,
    payload=encoded_bits,
    config=config
)

print(f"Bits embedded: {meta.bits_embedded}")
assert meta.bits_embedded > 0

# --------------------------------------------------
# 5. Extract bits (synthetic path)
# --------------------------------------------------
extracted_bits = modulator.extract_bits(
    flow=mod_flow,
    num_bits=meta.bits_embedded,
    config=config
)

assert len(extracted_bits) > 0

# --------------------------------------------------
# 6. ECC decode
# --------------------------------------------------
recovered = ecc_decode(extracted_bits, config=config)

# --------------------------------------------------
# 7. Verification
# --------------------------------------------------
print("Original :", payload)
print("Recovered:", recovered)

assert recovered == payload
print("✅ Synthetic roundtrip SUCCESS")
