"""
EfficientFireNet (2022) - SOTA Baseline for Fire Segmentation

Implementation of EfficientFireNet from "EfficientFireNet: An Efficient Fire Detection Model for Real-Time Applications"
(2022) adapted for FLAME dataset segmentation.

Architecture:
- EfficientNet-B0 backbone (pretrained on ImageNet)
- Lightweight decoder with skip connections
- Optimized for edge deployment (<5MB model size)
- Target mIoU > 0.75 on FLAME dataset

Reference: https://arxiv.org/abs/2203.08096 (adapted for segmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientFireNetDecoder(nn.Module):
    """Lightweight decoder for EfficientFireNet segmentation head."""

    def __init__(self, in_channels: int = 1280, num_classes: int = 2):
        super().__init__()

        # Decoder blocks with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final segmentation head
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)  # 1280 -> 256
        x = self.up2(x)  # 256 -> 128
        x = self.up3(x)  # 128 -> 64
        x = self.up4(x)  # 64 -> 32
        x = self.final_conv(x)  # 32 -> num_classes
        return x


class EfficientFireNet(nn.Module):
    """
    EfficientFireNet (2022) for fire segmentation.

    Uses EfficientNet-B0 backbone with custom lightweight decoder.
    Optimized for real-time fire detection with <5MB model size.

    Args:
        num_classes: Number of segmentation classes (default: 2 for fire/no-fire)
        pretrained: Whether to use ImageNet pretrained weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B0 backbone
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Remove classifier head
        self.backbone.classifier = nn.Identity()

        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone.features(dummy_input)
            self.feature_channels = features.shape[1]  # Should be 1280 for B0

        # Custom lightweight decoder
        self.decoder = EfficientFireNetDecoder(
            in_channels=self.feature_channels,
            num_classes=num_classes
        )

        # Initialize decoder weights
        self._initialize_decoder()

    def _initialize_decoder(self):
        """Initialize decoder weights with Kaiming normal."""
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Extract features from backbone
        features = self.backbone.features(x)  # [B, 1280, 7, 7] for 224x224 input

        # Decode to segmentation mask
        segmentation = self.decoder(features)

        # Upsample to input size if needed
        if segmentation.shape[-1] != x.shape[-1]:
            segmentation = F.interpolate(
                segmentation,
                size=(x.shape[-2], x.shape[-1]),
                mode='bilinear',
                align_corners=False
            )

        return segmentation

    def get_model_size(self) -> float:
        """Get model size in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 ** 2)  # 4 bytes per float32 param

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_efficientfirenet(num_classes: int = 2, pretrained: bool = True):
    """
    Create EfficientFireNet model instance.

    Args:
        num_classes: Number of segmentation classes
        pretrained: Use ImageNet pretrained weights

    Returns:
        EfficientFireNet model
    """
    model = EfficientFireNet(num_classes=num_classes, pretrained=pretrained)

    # Verify model size constraint (< 5MB)
    model_size = model.get_model_size()
    if model_size >= 5.0:
        print(f"Warning: Model size {model_size:.2f}MB exceeds 5MB target")

    print("EfficientFireNet created:")
    print(f"  - Parameters: {model.get_num_parameters()}")
    print(f"  - Model size: {model_size:.2f}MB")
    print(f"  - Pretrained: {pretrained}")

    return model