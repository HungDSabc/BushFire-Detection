"""
ThermalFusion-Net (2023): SOTA baseline for wildfire segmentation on FLAME.

Design goals:
- Thermal-prioritized dual-branch encoder (RGB + thermal cue)
- Multi-scale residual fusion at 1/4, 1/8, and 1/16 resolutions
- Lightweight decoder for edge-friendly benchmarking
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv -> BN -> SiLU block used across the network."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUnit(nn.Module):
    """Simple residual unit with two 3x3 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)


class DownBlock(nn.Module):
    """Downsample block with stride-2 stem followed by residual refinement."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2)
        self.refine = ResidualUnit(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.refine(x)


class ResidualFusionBlock(nn.Module):
    """Residual additive fusion where thermal branch modulates RGB features."""

    def __init__(self, channels: int):
        super().__init__()
        self.thermal_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = ResidualUnit(channels)

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        thermal_mod = self.thermal_proj(thermal_feat)
        alpha = self.gate(torch.cat([rgb_feat, thermal_mod], dim=1))
        fused = rgb_feat + alpha * thermal_mod
        return self.refine(fused)


class UpBlock(nn.Module):
    """Upsample + skip fusion block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3, stride=1),
            ResidualUnit(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ThermalFusionNet2023(nn.Module):
    """
    ThermalFusion-Net (2023) with multi-scale residual fusion.

    Input:
    - rgb: normalized RGB tensor (B, 3, H, W)

    Internal thermal cue:
    - derived from de-normalized RGB emphasizing red/luminance to mimic thermal saliency.

    Multi-scale fusion levels:
    - 1/4, 1/8, 1/16
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 12):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"ThermalFusionNet2023 expects RGB input with 3 channels, got {in_channels}")

        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        # RGB encoder (spatial branch)
        self.rgb_stem = ConvBNAct(3, c1, kernel_size=3, stride=1)
        self.rgb_down1 = DownBlock(c1, c2)   # 1/2
        self.rgb_down2 = DownBlock(c2, c3)   # 1/4
        self.rgb_down3 = DownBlock(c3, c4)   # 1/8
        self.rgb_down4 = DownBlock(c4, c4)   # 1/16

        # Thermal encoder (cue branch)
        self.th_stem = ConvBNAct(1, c1, kernel_size=3, stride=1)
        self.th_down1 = DownBlock(c1, c2)    # 1/2
        self.th_down2 = DownBlock(c2, c3)    # 1/4
        self.th_down3 = DownBlock(c3, c4)    # 1/8
        self.th_down4 = DownBlock(c4, c4)    # 1/16

        # Multi-scale fusion (required levels)
        self.fuse_1_4 = ResidualFusionBlock(c3)
        self.fuse_1_8 = ResidualFusionBlock(c4)
        self.fuse_1_16 = ResidualFusionBlock(c4)

        # Decoder
        self.up_1 = UpBlock(c4, c4, c3)  # 1/16 -> 1/8
        self.up_2 = UpBlock(c3, c3, c2)  # 1/8 -> 1/4
        self.up_3 = UpBlock(c2, c2, c1)  # 1/4 -> 1/2

        self.head = nn.Sequential(
            ConvBNAct(c1, c1, kernel_size=3, stride=1),
            nn.Conv2d(c1, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _thermal_from_rgb(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        """Create thermal-prioritized cue map from normalized RGB tensor."""
        mean = rgb_norm.new_tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        std = rgb_norm.new_tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        rgb = rgb_norm * std + mean
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        # Bias toward red channel + luminance for fire-intensity emphasis.
        thermal = 0.7 * r + 0.3 * luma
        return torch.clamp(thermal, 0.0, 1.0)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        thermal = self._thermal_from_rgb(rgb)

        # RGB branch
        r1 = self.rgb_stem(rgb)           # 1x
        r2 = self.rgb_down1(r1)           # 1/2
        r3 = self.rgb_down2(r2)           # 1/4
        r4 = self.rgb_down3(r3)           # 1/8
        r5 = self.rgb_down4(r4)           # 1/16

        # Thermal branch
        t1 = self.th_stem(thermal)        # 1x
        t2 = self.th_down1(t1)            # 1/2
        t3 = self.th_down2(t2)            # 1/4
        t4 = self.th_down3(t3)            # 1/8
        t5 = self.th_down4(t4)            # 1/16

        # Multi-scale residual fusion
        f3 = self.fuse_1_4(r3, t3)
        f4 = self.fuse_1_8(r4, t4)
        f5 = self.fuse_1_16(r5, t5)

        # Decode
        x = self.up_1(f5, f4)
        x = self.up_2(x, f3)
        x = self.up_3(x, r2)
        x = F.interpolate(x, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)

    def get_parameter_count(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


def create_thermalfusion_net_2023(
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 12,
) -> ThermalFusionNet2023:
    """Factory for ThermalFusion-Net (2023)."""
    return ThermalFusionNet2023(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


if __name__ == "__main__":
    model = create_thermalfusion_net_2023()
    stats = model.get_parameter_count()
    print("ThermalFusion-Net (2023)")
    print(f"Total parameters: {stats['total_parameters']:,}")
    print(f"Model size: {stats['model_size_mb']:.2f} MB")

    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")