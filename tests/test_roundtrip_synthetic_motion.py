import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).resolve().parents[1] / "default_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


import numpy as np

from module6_motion_modulation.modulator import MotionFieldModulator
from module8_receiver.capacity import CapacityEstimator
from module5_ecc.encoder import ecc_encode
from module5_ecc.decoder import ecc_decode



def test_synthetic_motion_roundtrip():
    """
    POSITIVE TEST:
    High, uniform synthetic motion → embed → extract → ECC decode
    """

    config = load_config()

    # ------------------------------------------------------
    # 1. Create synthetic high-motion flow
    # ------------------------------------------------------
    H, W = 64, 64
    flow = np.zeros((H, W, 2), dtype=np.float32)

    # Strong horizontal motion everywhere
    flow[:, :, 0] = 5.0

    # ------------------------------------------------------
    # 2. Check capacity is non-zero
    # ------------------------------------------------------
    estimator = CapacityEstimator(config)
    capacity = estimator.compute_capacity(flow)

    assert capacity > 0, "Synthetic flow must have capacity"

    # ------------------------------------------------------
    # 3. Encode payload
    # ------------------------------------------------------
    payload = b"HELLO"
    ecc_bits = ecc_encode(payload, config=config)

    # ------------------------------------------------------
    # 4. Embed bits
    # ------------------------------------------------------
    modulator = MotionFieldModulator()
    mod_flow, meta = modulator.embed_bits(
        flow=flow,
        payload=ecc_bits,
        config=config
    )

    assert meta.bits_embedded > 0

    # ------------------------------------------------------
    # 5. Extract bits
    # ------------------------------------------------------
    extracted_bits = modulator.extract_bits(
        flow=mod_flow,
        num_bits=meta.bits_embedded,
        config=config
    )

    # ------------------------------------------------------
    # 6. ECC decode
    # ------------------------------------------------------
    recovered = ecc_decode(extracted_bits, config=config)

    # ------------------------------------------------------
    # 7. Final assertion
    # ------------------------------------------------------
    assert recovered == payload
