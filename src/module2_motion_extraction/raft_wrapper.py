"""
RAFT Model Wrapper

Wrapper for the RAFT (Recurrent All-Pairs Field Transforms) optical flow model.
Handles model initialization, weight loading, and inference.
"""

import os
import sys
from typing import Tuple, Optional

import numpy as np
import torch

from .exceptions import FlowExtractionError


# -----------------------------------------------------------------------------
# Deterministic RAFT import (OFFICIAL IMPLEMENTATION ONLY)
# -----------------------------------------------------------------------------

RAFT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../RAFT")
)

if not os.path.exists(RAFT_ROOT):
    raise FlowExtractionError(
        "RAFT repository not found.\n"
        "Expected location: project_root/RAFT\n"
        "Clone from: https://github.com/princeton-vl/RAFT"
    )

RAFT_CORE = os.path.join(RAFT_ROOT, "core")

if RAFT_CORE not in sys.path:
    sys.path.insert(0, RAFT_CORE)

try:
    from raft import RAFT
except Exception as e:
    raise FlowExtractionError(
        "Failed to import RAFT from official repository."
    ) from e


# -----------------------------------------------------------------------------
# RAFT Wrapper
# -----------------------------------------------------------------------------

class RAFTWrapper:
    """
    Wrapper for RAFT optical flow model.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        small: bool = False,
        iters: int = 20,
        mixed_precision: bool = False
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RAFT model weights not found at: {model_path}"
            )

        # --------------------------------------------------
        # Device + CPU performance tuning
        # --------------------------------------------------
        if device == "cpu":
            torch.set_num_threads(os.cpu_count())
            torch.set_num_interop_threads(os.cpu_count())

        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        # 🔥 CRITICAL SPEED FIX:
        # RAFT on CPU must use fewer iterations
        if self.device.type == "cpu":
            self.iters = min(iters, 8)   # <= THIS SAVES YOU
        else:
            self.iters = iters

        self.mixed_precision = mixed_precision

        from argparse import Namespace
        args = Namespace(
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=False,
            dropout=0
        )

        try:
            self.model = RAFT(args)
        except Exception as e:
            raise FlowExtractionError(
                f"Failed to initialize RAFT model: {e}"
            ) from e

        # --------------------------------------------------
        # Load pretrained weights
        # --------------------------------------------------
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                checkpoint = (
                    checkpoint.get("state_dict")
                    or checkpoint.get("model")
                    or checkpoint
                )

            self.model.load_state_dict(checkpoint, strict=False)

        except Exception as e:
            raise FlowExtractionError(
                f"Failed to load RAFT weights: {e}"
            ) from e

        self.model.to(self.device)
        self.model.eval()


    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def infer(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None
    ) -> torch.Tensor:

        if iters is None:
            iters = self.iters

        if image1.device != self.device:
            image1 = image1.to(self.device, non_blocking=True)
            image2 = image2.to(self.device, non_blocking=True)

        try:
            if self.mixed_precision and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    flow = self.model(image1, image2, iters=iters, test_mode=True)
            else:
                flow = self.model(image1, image2, iters=iters, test_mode=True)

            if isinstance(flow, (list, tuple)):
                flow = flow[-1]

            return flow

        except Exception as e:
            raise FlowExtractionError(
                f"RAFT inference failed: {e}"
            ) from e


    # -------------------------------------------------------------------------
    # Pre/Post-processing
    # -------------------------------------------------------------------------

    def preprocess_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if frame1.shape != frame2.shape:
            raise ValueError(
                f"Frame shape mismatch: {frame1.shape} vs {frame2.shape}"
            )

        img1 = torch.from_numpy(
            np.transpose(frame1.astype(np.float32), (2, 0, 1))
        ).unsqueeze(0)

        img2 = torch.from_numpy(
            np.transpose(frame2.astype(np.float32), (2, 0, 1))
        ).unsqueeze(0)

        return img1, img2


    def postprocess_flow(
        self,
        flow_tensor: torch.Tensor
    ) -> np.ndarray:

        flow = flow_tensor.cpu().numpy()

        if flow.ndim == 4:
            flow = flow[0]

        return np.transpose(flow, (1, 2, 0)).astype(np.float32)
