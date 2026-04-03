"""
YOLOFM (2024): Adapted for FLAME Segmentation.

Based on: "YOLOFM: an improved fire and smoke object detection algorithm" (2024)

Adaptation for segmentation:
- FocalNext Backbone with FocalNextBlock for efficient multi-scale feature extraction
- QAHARep-FPN Neck with QARepVGGB blocks and transpose operations
- Lightweight decoder head for 1-channel binary segmentation mask
- Focal-SIoU Loss for training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class FocalNextBlock(nn.Module):
    """FocalNext block: focal depth-wise convolution + channel attention."""

    def __init__(self, channels: int, focal_level: int = 3):
        super().__init__()
        self.focal_level = focal_level

        # Focal depth-wise conv (multiple dilations)
        self.focal_convs = nn.ModuleList()
        for i in range(focal_level):
            dilation = 2 ** i
            self.focal_convs.append(
                nn.Conv2d(
                    channels, channels, kernel_size=3, padding=dilation,
                    dilation=dilation, groups=channels, bias=False
                )
            )

        self.bn = nn.BatchNorm2d(channels)
        self.silu = nn.SiLU(inplace=True)

        # Channel attention (squeeze-and-excitation)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply focal convolutions and aggregate
        out = torch.zeros_like(x)
        for fc in self.focal_convs:
            out = out + fc(x)

        out = self.bn(out)
        out = self.silu(out)

        # Channel attention
        attn = self.se(out)
        out = out * attn

        return out + x


class FocalNextBackbone(nn.Module):
    """FocalNext backbone for efficient feature extraction."""

    def __init__(self, in_channels: int = 3, out_channels: List[int] = None):
        super().__init__()
        if out_channels is None:
            out_channels = [64, 128, 256]

        c1, c2, c3 = out_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Stage 1: 256 -> 128
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            FocalNextBlock(c1, focal_level=2),
            FocalNextBlock(c1, focal_level=2),
        )

        # Stage 2: 128 -> 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            FocalNextBlock(c2, focal_level=2),
            FocalNextBlock(c2, focal_level=2),
        )

        # Stage 3: 64 -> 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            FocalNextBlock(c3, focal_level=3),
            FocalNextBlock(c3, focal_level=3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stem(x)
        p3 = self.layer1(stem)       # 256 -> 128
        p4 = self.layer2(p3)         # 128 -> 64
        p5 = self.layer3(p4)         # 64 -> 32
        return p3, p4, p5


class QARepVGGB(nn.Module):
    """QARepVGGB: Quality-Aware Reparameterizable block inspired by VGG-B design."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        # Main conv path
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Skip path
        self.skip = nn.Identity() if channels > 0 else None

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return self.act(out + x)


class QAHARepFPN(nn.Module):
    """QAHARep-FPN: Quality-Aware Hierarchical Attention Reparameterizable FPN."""

    def __init__(self, channels: List[int]):
        super().__init__()
        c1, c2, c3 = channels  # [64, 128, 256]

        # Lateral layers (reduce to uniform channel)
        self.lateral3 = nn.Conv2d(c1, c3, kernel_size=1)
        self.lateral4 = nn.Conv2d(c2, c3, kernel_size=1)
        self.lateral5 = nn.Conv2d(c3, c3, kernel_size=1)

        # Top-down pathways
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # FPN blocks with QARepVGGB
        self.fpn_p5 = QARepVGGB(c3)
        self.fpn_p4 = QARepVGGB(c3)
        self.fpn_p3 = QARepVGGB(c3)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lateral conversion
        lat3 = self.lateral3(p3)  # 128
        lat4 = self.lateral4(p4)  # 64
        lat5 = self.lateral5(p5)  # 32

        # Top-down
        f5 = self.fpn_p5(lat5)
        f4_up = self.upsample(f5)
        f4 = self.fpn_p4(lat4 + f4_up)

        f3_up = self.upsample(f4)
        f3 = self.fpn_p3(lat3 + f3_up)

        return f3, f4, f5


class SegmentationHead(nn.Module):
    """Lightweight segmentation head for FLAME."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class YOLOFM2024(nn.Module):
    """
    YOLOFM (2024) for FLAME Segmentation.

    Components:
    - FocalNext backbone
    - QAHARep-FPN neck
    - Lightweight segmentation head
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        self.backbone = FocalNextBackbone(in_channels=in_channels, out_channels=[64, 128, 256])
        self.fpn = QAHARepFPN(channels=[64, 128, 256])
        self.head = SegmentationHead(in_channels=256, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # FPN
        f3, f4, f5 = self.fpn(p3, p4, p5)

        # Use finest feature for segmentation (f3, 128x128)
        out = F.interpolate(f3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        out = self.head(out)

        return out

    def get_parameter_count(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


def create_yolofm_2024(in_channels: int = 3, out_channels: int = 1) -> YOLOFM2024:
    """Factory for YOLOFM (2024)."""
    return YOLOFM2024(in_channels=in_channels, out_channels=out_channels)


# Focal-SIoU Loss
class FocalSIoULoss(nn.Module):
    """
    Focal-SIoU Loss: combines Focal loss for hard negatives + SIoU for boundary quality.
    
    SIoU emphasizes spatial alignment via center distance and shape asymmetry.
    Focal weight increases loss for hard-to-classify samples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def siou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Spatial IoU Loss."""
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # Intersection and union
        inter = (pred * target).sum(dim=[2, 3])
        union = (pred + target - pred * target).sum(dim=[2, 3])
        iou = inter / (union + self.eps)

        # Bounding boxes
        pred_box = self._get_bbox_coords(pred)
        target_box = self._get_bbox_coords(target)

        # Center distance
        pred_center = (pred_box[:, [0, 1]] + pred_box[:, [2, 3]]) / 2.0
        target_center = (target_box[:, [0, 1]] + target_box[:, [2, 3]]) / 2.0
        center_dist = torch.sqrt(((pred_center - target_center) ** 2).sum(dim=1) + self.eps)

        # Box diagonals
        pred_diag = torch.sqrt(((pred_box[:, [2, 3]] - pred_box[:, [0, 1]]) ** 2).sum(dim=1) + self.eps)
        target_diag = torch.sqrt(((target_box[:, [2, 3]] - target_box[:, [0, 1]]) ** 2).sum(dim=1) + self.eps)
        max_diag = torch.max(pred_diag, target_diag)

        # Shape asymmetry
        pred_w = pred_box[:, 2] - pred_box[:, 0] + self.eps
        pred_h = pred_box[:, 3] - pred_box[:, 1] + self.eps
        target_w = target_box[:, 2] - target_box[:, 0] + self.eps
        target_h = target_box[:, 3] - target_box[:, 1] + self.eps
        shape = torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)

        # SIoU
        siou = iou - (center_dist ** 2 / (max_diag ** 2 + self.eps)) - (shape / torch.pi)
        return 1.0 - siou

    def _get_bbox_coords(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract bounding box coordinates from mask."""
        B = mask.shape[0]
        coords = []
        for i in range(B):
            m = mask[i].squeeze()
            if m.sum() == 0:
                coords.append(torch.tensor([0.0, 0.0, 1.0, 1.0], device=mask.device))
            else:
                y_idx, x_idx = torch.where(m > 0.5)
                coords.append(torch.tensor(
                    [x_idx.min().float(), y_idx.min().float(),
                     x_idx.max().float(), y_idx.max().float()],
                    device=mask.device
                ))
        return torch.stack(coords)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal-SIoU Loss."""
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # BCE as base
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        bce = bce.mean(dim=[2, 3])  # (B, 1)

        # Focal weight: (1 - p_t)^gamma
        p_t = torch.where(target.mean(dim=[2, 3]) > 0.5, pred.mean(dim=[2, 3]), 1 - pred.mean(dim=[2, 3]))
        focal_weight = (1 - p_t) ** self.gamma

        # SIoU
        siou = self.siou_loss(pred, target)

        # Combine
        loss = self.alpha * focal_weight * bce.squeeze() + (1 - self.alpha) * siou
        return loss.mean()


if __name__ == "__main__":
    model = create_yolofm_2024()
    stats = model.get_parameter_count()
    print("YOLOFM (2024)")
    print(f"Total params: {stats['total_parameters']:,}")
    print(f"Model size: {stats['model_size_mb']:.2f} MB")

    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
