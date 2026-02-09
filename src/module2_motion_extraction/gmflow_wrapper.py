"""
GMFlow Wrapper

Thin wrapper to provide a RAFT-compatible interface
for GMFlow optical flow estimation.
"""

import torch
import numpy as np
from gmflow.gmflow import GMFlow
from .exceptions import FlowExtractionError


class GMFlowWrapper:
    """
    Wrapper around GMFlow model for dense optical flow inference.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        try:
            self.device = device

            # Initialize GMFlow model
            self.model = GMFlow(
                feature_channels=128,
                num_scales=1,
                upsample_factor=8,
                num_head=1,
                attention_type="swin",
                ffn_dim_expansion=4,
                num_transformer_layers=6
            )

            # Load weights
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

            self.model.to(device)
            self.model.eval()

        except Exception as e:
            raise FlowExtractionError(f"Failed to initialize GMFlow: {e}")

    @torch.no_grad()
    def infer(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow using GMFlow.

        Args:
            frame1, frame2: uint8 numpy arrays (H, W, 3)

        Returns:
            flow: float32 numpy array (H, W, 2)
        """
        try:
            # Convert to torch tensor
            img1 = torch.from_numpy(frame1).permute(2, 0, 1).float()[None] / 255.0
            img2 = torch.from_numpy(frame2).permute(2, 0, 1).float()[None] / 255.0

            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            # GMFlow inference
            results = self.model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1])

            flow = results["flow_preds"][-1][0]  # (2, H, W)

            flow = flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            return flow

        except Exception as e:
            raise FlowExtractionError(f"GMFlow inference failed: {e}")
