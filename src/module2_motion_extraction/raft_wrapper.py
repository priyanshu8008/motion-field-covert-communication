"""
RAFT Model Wrapper

Wrapper for the RAFT (Recurrent All-Pairs Field Transforms) optical flow model.
Handles model initialization, weight loading, and inference.

Reference:
Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
(ECCV 2020)
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
    from raft import RAFT  # this now resolves to RAFT/core/raft.py
except Exception as e:
    raise FlowExtractionError(
        "Failed to import RAFT from official repository.\n"
        "Ensure RAFT/core is on PYTHONPATH and dependencies are installed."
    ) from e



# -----------------------------------------------------------------------------
# RAFT Wrapper
# -----------------------------------------------------------------------------

class RAFTWrapper:
    """
    Wrapper for RAFT optical flow model.

    Handles model loading, device placement, and inference.
    Assumes the official RAFT implementation is available.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        small: bool = False,
        iters: int = 20,
        mixed_precision: bool = False
    ):
        """
        Initialize RAFT model.

        Args:
            model_path: Path to pretrained RAFT weights (.pth)
            device: "cuda" or "cpu"
            small: Use RAFT-small variant
            iters: Number of refinement iterations
            mixed_precision: Enable AMP inference

        Raises:
            FileNotFoundError: If weights file is missing
            FlowExtractionError: If model initialization fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RAFT model weights not found at: {model_path}"
            )

        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.iters = iters
        self.mixed_precision = mixed_precision

        # Construct RAFT args object (official pattern)
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

        # Load pretrained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint = checkpoint["model"]

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

    @torch.no_grad()
    def infer(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform optical flow inference.

        Args:
            image1: (B, 3, H, W), float32, range [0, 255]
            image2: (B, 3, H, W), float32, range [0, 255]
            iters: Override iteration count

        Returns:
            flow: (B, 2, H, W), float32
        """
        if iters is None:
            iters = self.iters

        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        try:
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    flow = self.model(image1, image2, iters=iters, test_mode=True)
            else:
                flow = self.model(image1, image2, iters=iters, test_mode=True)

            # RAFT returns a list of predictions — take final
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
        """
        Convert RGB uint8 frames to RAFT tensors.

        Args:
            frame1, frame2: (H, W, 3), uint8

        Returns:
            img1, img2: (1, 3, H, W), float32
        """
        if frame1.shape != frame2.shape:
            raise ValueError(
                f"Frame shape mismatch: {frame1.shape} vs {frame2.shape}"
            )

        if frame1.ndim != 3 or frame1.shape[2] != 3:
            raise ValueError(
                f"Expected RGB frame (H, W, 3), got {frame1.shape}"
            )

        img1 = frame1.astype(np.float32)
        img2 = frame2.astype(np.float32)

        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        img1 = torch.from_numpy(img1).unsqueeze(0)
        img2 = torch.from_numpy(img2).unsqueeze(0)

        return img1, img2

    def postprocess_flow(
        self,
        flow_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Convert RAFT output to numpy flow field.

        Args:
            flow_tensor: (1, 2, H, W) or (B, 2, H, W)

        Returns:
            flow: (H, W, 2), float32
        """
        flow = flow_tensor.detach().cpu().numpy()

        if flow.ndim == 4:
            flow = flow[0]

        flow = np.transpose(flow, (1, 2, 0)).astype(np.float32)
        return flow
