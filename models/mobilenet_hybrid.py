"""
FLAME Hybrid Fire Detection - MobileNet Shared Backbone Architecture
BDS DSC312 - Computer Vision with Multi-modal Models & Analytics

Lightweight shared-backbone architecture for real-time fire detection
optimized for DICTA 2026 Challenge requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, Tuple, Optional, List

class SharedBackbone(nn.Module):
    """
    Shared MobileNetV3 backbone for both classification and segmentation tasks.
    Optimized for edge deployment with minimal computational overhead.
    """
    
    def __init__(self, pretrained: bool = True, width_mult: float = 1.0):
        super(SharedBackbone, self).__init__()
        
        # Load MobileNetV3-Large as backbone
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Remove classifier to use as feature extractor
        self.features = self.backbone.features
        
        # Feature dimensions for different scales
        self.feature_dims = {
            'low': 24,      # Early features (spatial detail)
            'mid': 80,      # Mid-level features  
            'high': 960     # High-level features (semantic)
        }
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from shared backbone.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of feature maps at different scales
        """
        features = {}
        
        # Extract features at different scales
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Store features at key points
            if i == 3:  # After first inverted residual block
                features['low'] = x
            elif i == 6:  # After mid-level blocks
                features['mid'] = x
                
        features['high'] = x  # Final high-level features
        
        return features

class ClassificationHead(nn.Module):
    """
    Lightweight classification head for fire/no-fire detection.
    Designed for maximum precision with minimal parameters.
    """
    
    def __init__(self, input_dim: int = 960, num_classes: int = 2, dropout: float = 0.2):
        super(ClassificationHead, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Minimal classifier for efficiency
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify fire/no-fire from high-level features.
        
        Args:
            features: High-level feature map [B, C, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        x = self.global_pool(features)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class SegmentationHead(nn.Module):
    """
    Lightweight segmentation head for fire region localization.
    Uses multi-scale feature fusion for accurate boundaries.
    """
    
    def __init__(self, feature_dims: Dict[str, int], num_classes: int = 2):
        super(SegmentationHead, self).__init__()
        
        self.feature_dims = feature_dims
        
        # Upsampling layers for multi-scale fusion
        self.upsample_high = nn.Sequential(
            nn.Conv2d(feature_dims['high'], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.upsample_mid = nn.Sequential(
            nn.Conv2d(feature_dims['mid'], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upsample_low = nn.Sequential(
            nn.Conv2d(feature_dims['low'], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + 128 + 64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation output
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, features: Dict[str, torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate segmentation mask from multi-scale features.
        
        Args:
            features: Multi-scale feature dictionary
            target_size: Target output size (H, W)
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Process high-level features
        high_feat = self.upsample_high(features['high'])
        
        # Resize to match mid-level features
        mid_size = features['mid'].shape[2:]
        high_feat = F.interpolate(high_feat, size=mid_size, mode='bilinear', align_corners=False)
        
        # Process mid-level features
        mid_feat = self.upsample_mid(features['mid'])
        
        # Process low-level features
        low_feat = self.upsample_low(features['low'])
        
        # Resize all to low-level size for fusion
        low_size = features['low'].shape[2:]
        high_feat = F.interpolate(high_feat, size=low_size, mode='bilinear', align_corners=False)
        mid_feat = F.interpolate(mid_feat, size=low_size, mode='bilinear', align_corners=False)
        
        # Fuse features
        fused = torch.cat([high_feat, mid_feat, low_feat], dim=1)
        fused = self.fusion(fused)
        
        # Generate segmentation
        seg_logits = self.seg_head(fused)
        
        # Resize to target size
        seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
        
        return seg_logits

class FLAMEHybridModel(nn.Module):
    """
    Complete FLAME Hybrid model with shared backbone architecture.
    Optimized for DICTA 2026 Challenge requirements.
    
    Key Features:
    - Shared MobileNetV3 backbone for efficiency
    - Dual-task learning (classification + segmentation)
    - Lightweight design for edge deployment
    - Multi-scale feature fusion
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.2):
        super(FLAMEHybridModel, self).__init__()
        
        # Shared backbone
        self.backbone = SharedBackbone(pretrained=pretrained)
        
        # Task-specific heads
        self.classification_head = ClassificationHead(
            input_dim=self.backbone.feature_dims['high'],
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.segmentation_head = SegmentationHead(
            feature_dims=self.backbone.feature_dims,
            num_classes=num_classes
        )
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared backbone and task heads.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing classification and segmentation outputs
        """
        # Extract features from shared backbone
        features = self.backbone(x)
        
        # Classification output
        cls_logits = self.classification_head(features['high'])
        
        # Segmentation output
        target_size = (x.shape[2], x.shape[3])
        seg_logits = self.segmentation_head(features, target_size)
        
        outputs = {
            'classification': cls_logits,
            'segmentation': seg_logits
        }
        
        if return_features:
            outputs['features'] = features
            
        return outputs
    
    def get_model_size(self) -> Dict[str, float]:
        """Calculate model size and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in MB (4 bytes per float32 parameter)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb
        }

def create_flame_hybrid_model(config: Optional[Dict] = None) -> FLAMEHybridModel:
    """
    Factory function to create FLAME Hybrid model with configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured FLAME Hybrid model
    """
    if config is None:
        config = {
            'num_classes': 2,
            'pretrained': True,
            'dropout': 0.2
        }
    
    model = FLAMEHybridModel(**config)
    
    # Print model info
    model_info = model.get_model_size()
    print(f"FLAME Hybrid Model Created:")
    print(f"  Total Parameters: {model_info['total_params']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_params']:,}")
    print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
    
    return model

def main():
    """Demo function for model testing."""
    print("🔥 FLAME Hybrid Fire Detection Model")
    print("=" * 50)
    
    # Create model
    model = create_flame_hybrid_model()
    model.eval()
    
    # Test with dummy input
    batch_size = 2
    height, width = 224, 224
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    print(f"\nTesting with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        outputs = model(dummy_input)
        
        print(f"Classification output shape: {outputs['classification'].shape}")
        print(f"Segmentation output shape: {outputs['segmentation'].shape}")
        
        # Test predictions
        cls_probs = torch.softmax(outputs['classification'], dim=1)
        seg_probs = torch.softmax(outputs['segmentation'], dim=1)
        
        print(f"Classification probabilities: {cls_probs[0].numpy()}")
        print(f"Segmentation shape: {seg_probs.shape}")
    
    print("\n✅ Model test completed successfully!")

if __name__ == "__main__":
    main()