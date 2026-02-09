import numpy as np
import yaml
from pathlib import Path

from module1_video_io.video_loader import load_video
from module1_video_io.video_writer import write_video
from module2_motion_extraction.flow_extractor import OpticalFlowExtractor
from module6_motion_modulation.modulator import MotionFieldModulator
from module7_video_recon.video_reconstructor import VideoReconstructor
from module5_ecc.encoder import ecc_encode


# --------------------------------------------------
# Load config
# --------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[1] / "default_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

assert config is not None, "Config failed to load"


# --------------------------------------------------
# Paths
# --------------------------------------------------
CLEAN_VIDEO = Path("data/videos/clean/clean_01.mp4")

STEGO_VIDEO = (
    Path(__file__).resolve().parents[1]
    / "data/videos/stego/stego_01.mp4"
)

STEGO_VIDEO.parent.mkdir(parents=True, exist_ok=True)

RAFT_MODEL = Path(
    "src/module2_motion_extraction/models/raft-things.pth"
)


# --------------------------------------------------
# Load clean video
# --------------------------------------------------
frames, video_meta = load_video(str(CLEAN_VIDEO))
assert len(frames) >= 2, "Need at least 2 frames"

# 🔎 OPTIONAL sanity run (first time only)
# frames = frames[:60]   # uncomment for quick correctness test


# --------------------------------------------------
# Extract optical flow (FULL RESOLUTION)
# --------------------------------------------------
flow_extractor = OpticalFlowExtractor(
    model_path=str(RAFT_MODEL),
    device="cpu"   # correctness baseline
)

flows = flow_extractor.batch_extract(frames)
assert len(flows) == len(frames) - 1


# --------------------------------------------------
# Payload → ECC
# --------------------------------------------------
payload = b"TEST"
encoded_bits = ecc_encode(payload, config=config)


# --------------------------------------------------
# Embed bits into motion fields
# --------------------------------------------------
modulator = MotionFieldModulator()
mod_flows = []

bit_offset = 0

for flow in flows:
    if bit_offset >= len(encoded_bits):
        mod_flows.append(flow)
        continue

    mod_flow, embed_meta = modulator.embed_bits(
        flow=flow,
        payload=encoded_bits[bit_offset:],
        config=config
    )

    bit_offset += embed_meta.bits_embedded
    mod_flows.append(mod_flow)

print(f"Bits embedded: {bit_offset}")


# --------------------------------------------------
# Reconstruct stego frames
# --------------------------------------------------
reconstructor = VideoReconstructor()

stego_frames, recon_meta = reconstructor.reconstruct(
    frames=frames,
    flows=mod_flows,
    config=config
)


# --------------------------------------------------
# Write output video
# --------------------------------------------------
print("Frames in clean:", len(frames))
print("Frames in stego:", len(stego_frames))

write_video(
    stego_frames,
    str(STEGO_VIDEO),
    fps=video_meta.fps
)

print("✅ Stego video generated:", STEGO_VIDEO.resolve())
