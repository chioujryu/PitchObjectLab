from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class BallReIDExtractor:
    """
    Lightweight appearance feature extractor for ball re-identification.

    Uses MobileNetV3-Small as the backbone (pretrained on ImageNet).
    The final classification head is replaced with a projection layer that
    produces a normalised embedding of configurable dimension.

    Input: BGR frame (numpy uint8) + list of xyxy boxes.
    Output: (N, embedding_dim) float32 numpy array.
    """

    def __init__(
        self,
        device: torch.device,
        embedding_dim: int = 256,
        half: bool = False,
    ) -> None:
        self.device = device
        self.embedding_dim = embedding_dim
        self.half = half and device.type == "cuda"
        self._model = self._build_model()
        self._transform = T.Compose([
            T.Resize((64, 64), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        import torch.nn.functional as F
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # MobileNetV3-Small: backbone.features output channels = 576
        feature_extractor = backbone.features
        pool = nn.AdaptiveAvgPool2d(1)
        proj = nn.Linear(576, self.embedding_dim, bias=False)
        norm = nn.LayerNorm(self.embedding_dim)

        class _EmbedNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = feature_extractor
                self.pool = pool
                self.proj = proj
                self.norm = norm

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x).flatten(1)
                x = self.proj(x)
                x = self.norm(x)
                x = F.normalize(x, dim=1)
                return x

        model = _EmbedNet().to(self.device)
        if self.half:
            model = model.half()
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(self, frame_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalised embeddings for `boxes` (N×4 xyxy) in `frame_bgr`.
        Returns (N, embedding_dim) float32 array.
        """
        import cv2
        from PIL import Image

        if len(boxes) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = frame_rgb.shape[:2]
        tensors: list[torch.Tensor] = []
        for box in boxes:
            x1, y1, x2, y2 = (
                int(max(box[0], 0)), int(max(box[1], 0)),
                int(min(box[2], W)), int(min(box[3], H)),
            )
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                crop = frame_rgb[y1:y2, x1:x2]
            pil = Image.fromarray(crop)
            t = self._transform(pil)
            tensors.append(t)

        batch = torch.stack(tensors).to(self.device)
        if self.half:
            batch = batch.half()
        embeddings = self._model(batch)
        return embeddings.float().cpu().numpy()

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        a: (M, D), b: (N, D)  →  returns (M, N)
        Assumes embeddings are already L2-normalised.
        """
        return a @ b.T
