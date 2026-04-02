"""
Segmentation-oriented models for BushFire-Detection.

This module provides the ultra-light shared-backbone model used by DICTA
training/inference scripts and tests.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class UltraOptimizedFireNet(nn.Module):
    """Ultra-optimized dual-head model for edge fire detection."""

    def __init__(self, num_classes: int = 2, num_seg_classes: int = 2):
        super().__init__()

        backbone = models.mobilenet_v3_small(weights=None)

        # Adapt first conv to 4-channel RGBT input.
        original_conv = backbone.features[0][0]
        self.backbone_conv1 = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize with Kaiming; caller can load trained checkpoint.
        nn.init.kaiming_normal_(self.backbone_conv1.weight, mode="fan_out", nonlinearity="relu")

        backbone.features[0][0] = self.backbone_conv1

        # Truncated backbone for low-latency edge inference.
        self.backbone = nn.Sequential(*backbone.features[:8])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            test_input = torch.randn(1, 4, 224, 224)
            feature_dim = self.backbone(test_input).shape[1]

        cls_hidden = 48
        seg_channels = 24

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cls_hidden, num_classes),
        )

        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, seg_channels, kernel_size=16, stride=16, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(seg_channels, num_seg_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)

        cls_features = self.avgpool(features)
        cls_features = torch.flatten(cls_features, 1)
        classification = self.classifier(cls_features)

        segmentation = self.seg_head(features)

        return {
            "classification": classification,
            "segmentation": segmentation,
        }

    def forward_classification_only(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        cls_features = self.avgpool(features)
        cls_features = torch.flatten(cls_features, 1)
        return self.classifier(cls_features)

    def get_model_size(self) -> float:
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 ** 2)
