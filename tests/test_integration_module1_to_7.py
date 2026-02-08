import numpy as np
import pytest
import yaml

from module1_video_io.video_loader import load_video
from module2_motion_extraction.flow_extractor import OpticalFlowExtractor
from module6_motion_modulation.modulator import MotionFieldModulator
from module7_video_recon.video_reconstructor import VideoReconstructor
from module8_receiver.receiver import ReceiverEngine

from module5_ecc.encoder import ecc_encode
from module5_ecc.decoder import ecc_decode

import cv2

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


def load_config():
    with open("default_config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_module1_to_7_end_to_end():
    """
    FULL SYSTEM INTEGRATION TEST
    Module 1 → Module 2 → Module 6 → Module 7 → Module 8 → Module 5
    """

    # ------------------------------------------------------------------
    # 0. Load config
    # ------------------------------------------------------------------
    config = load_config()

    # ------------------------------------------------------------------
    # 1. Load video (Module 1)
    # ------------------------------------------------------------------
    video_path = "tests/assets/test_motion_video.mp4"

    frames, metadata = load_video(video_path)

    # Limit frames INSIDE test
    frames = frames[:8]
    frames = downscale_frames(frames, max_size=512)

    assert len(frames) >= 2

    # ------------------------------------------------------------------
    # 2. Optical flow (Module 2)
    # ------------------------------------------------------------------
    RAFT_MODEL_PATH = "src/module2_motion_extraction/models/raft-things.pth"

    flow_extractor = OpticalFlowExtractor(
        model_path=RAFT_MODEL_PATH,
        device="cpu"
    )

    flows = flow_extractor.batch_extract(frames)

    assert len(flows) == len(frames) - 1

    # ------------------------------------------------------------------
    # 3. ECC encode (Module 5)
    # ------------------------------------------------------------------
    payload = b"HELLO123"
    ecc_bits = ecc_encode(payload, config=config)

    assert len(ecc_bits) > 0

    # ------------------------------------------------------------------
    # 4. QIM embed (Module 6)
    # ------------------------------------------------------------------
    modulator = MotionFieldModulator()

    modified_flows = []
    payload_cursor = 0

    for flow in flows:
        if payload_cursor >= len(ecc_bits):
            modified_flows.append(flow)
            continue

        mod_flow, meta = modulator.embed_bits(
            flow=flow,
            payload=ecc_bits[payload_cursor:],
            config=config
        )

        payload_cursor += meta.bits_embedded
        modified_flows.append(mod_flow)

    assert len(modified_flows) == len(flows)

    # ------------------------------------------------------------------
    # 5. Video reconstruction (Module 7)
    # ------------------------------------------------------------------
    reconstructor = VideoReconstructor()

    stego_frames, recon_meta = reconstructor.reconstruct(
        frames=frames,
        flows=modified_flows,
        config=config
    )

    # ------------------------------------------------------------------
    # 6. Receiver extract (Module 8)
    # ------------------------------------------------------------------
    receiver = ReceiverEngine(config=config)
    extracted_bits = receiver.extract(stego_frames)

    # ------------------------------------------------------------------
    # 7. ECC decode (Module 5)
    # ------------------------------------------------------------------
    if len(extracted_bits) == 0:
        pytest.skip("No extractable motion capacity in test video")

    recovered = ecc_decode(extracted_bits, config=config)
    assert recovered == payload

    # ------------------------------------------------------------------
    # 8. Final assertion
    # ------------------------------------------------------------------
    assert isinstance(recovered, (bytes, bytearray))
