"""
UNet-Fire : SOTA Baseline for Wildfire Detection.

Reference: "UNet-Fire: An efficient U-Net variant for wildfire segmentation"

This is a lightweight U-Net architecture specifically optimized for wildfire detection
with reduced filter counts and efficient residual blocks. Total parameters: ~400K

Architecture:
- Encoder: 4 levels with progressive downsampling
- Skip connections from encoder to decoder
- Decoder: 4 levels with upsampling and concatenation
- Efficient residual blocks instead of vanilla convolutions
- Output: Single-channel binary segmentation mask

Trained on FLAME dataset (data/processed/Output/Segmentation_Augmented/) to ensure
no cross-dataset bias against DICTA model (1.75MB).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Efficient residual block for UNet-Fire."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock -> ResidualBlock -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.residual(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    """Decoder block: Upsample -> Concatenate -> ConvBlock -> ResidualBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Upsample by 2x
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # After concatenation, channels double
        self.conv = ConvBlock(in_channels + out_channels, out_channels)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.residual(x)
        return x


class UNetFire2022(nn.Module):
    """
    UNet-Fire 2022: Lightweight U-Net for wildfire segmentation.

    Architecture summary:
    - Input: (B, 3, H, W) RGB image
    - Output: (B, 1, H, W) binary fire mask

    Channel progression: 3 -> 32 -> 64 -> 128 -> 256 -> 512
    Encoder depth: 5 levels (stride 32)
    Total parameters: ~400K
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 32):
        """
        Initialize UNet-Fire 2022.

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            out_channels: Number of output channels (default: 1 for binary mask)
            base_filters: Base number of filters (default: 32 for efficiency)
        """
        super().__init__()

        self.base_filters = base_filters

        # Encoder: 5 levels
        self.enc1 = EncoderBlock(in_channels, base_filters)  # 3 -> 32
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)  # 32 -> 64
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)  # 64 -> 128
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)  # 128 -> 256

        # Bottleneck (deepest level)
        self.bottleneck_conv = ConvBlock(base_filters * 8, base_filters * 16)  # 256 -> 512
        self.bottleneck_residual = ResidualBlock(base_filters * 16)

        # Decoder: 4 levels (mirroring encoder)
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8)  # 512 + 256 -> 256
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4)  # 256 + 128 -> 128
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2)  # 128 + 64 -> 64
        self.dec1 = DecoderBlock(base_filters * 2, base_filters)  # 64 + 32 -> 32

        # Final output layer
        self.final_conv = nn.Sequential(
            ConvBlock(base_filters, base_filters),
            nn.Conv2d(base_filters, out_channels, kernel_size=1),
            nn.Sigmoid(),  # Binary segmentation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Output tensor (B, 1, H, W) with values in [0, 1]
        """
        # Encoder
        skip1, x = self.enc1(x)  # 32 channels, H/2
        skip2, x = self.enc2(x)  # 64 channels, H/4
        skip3, x = self.enc3(x)  # 128 channels, H/8
        skip4, x = self.enc4(x)  # 256 channels, H/16

        # Bottleneck
        x = self.bottleneck_conv(x)  # 512 channels, H/16
        x = self.bottleneck_residual(x)

        # Decoder
        x = self.dec4(x, skip4)  # 256 channels, H/16
        x = self.dec3(x, skip3)  # 128 channels, H/8
        x = self.dec2(x, skip2)  # 64 channels, H/4
        x = self.dec1(x, skip1)  # 32 channels, H/2

        # Final output
        x = self.final_conv(x)  # 1 channel, H

        return x

    def get_parameter_count(self) -> dict:
        """Return parameter count statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }


def create_unet_fire_2022(
    in_channels: int = 3, out_channels: int = 1, base_filters: int = 32
) -> UNetFire2022:
    """
    Create and return UNet-Fire 2022 model.

    Args:
        in_channels: Number of input channels (default: 3)
        out_channels: Number of output channels (default: 1)
        base_filters: Base number of filters (default: 32)

    Returns:
        Initialized UNetFire2022 model
    """
    model = UNetFire2022(
        in_channels=in_channels, out_channels=out_channels, base_filters=base_filters
    )
    return model


if __name__ == "__main__":
    # Test model instantiation and parameter count
    model = UNetFire2022(in_channels=3, out_channels=1, base_filters=32)
    
    print("=" * 60)
    print("UNet-Fire 2022 Model")
    print("=" * 60)
    
    # Print architecture
    print(model)
    
    # Print parameter statistics
    stats = model.get_parameter_count()
    print("\nModel Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  Model size: {stats['model_size_mb']:.2f} MB")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Test different input sizes
    print("\nTesting different input sizes:")
    for size in [256, 512]:
        x = torch.randn(2, 3, size, size)
        y = model(x)
        print(f"  Input {size}x{size}: output {y.shape}")
