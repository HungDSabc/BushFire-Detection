"""
FLAME Hybrid Fire Detection - Loss Functions
BDS DSC312 - Computer Vision with Multi-modal Models & Analytics

Specialized loss functions for multi-task fire detection training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fire detection.
    Particularly useful for hard negative mining in fire/no-fire classification.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [B, C] or [B, C, H, W]
            targets: Ground truth labels [B] or [B, H, W]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Particularly effective for fire region segmentation with class imbalance.
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Predicted logits [B, C, H, W]
            targets: Ground truth masks [B, H, W]
            
        Returns:
            Dice loss value
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Flatten tensors
        inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Compute Dice coefficient
        intersection = (inputs_flat * targets_flat).sum(dim=2)
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss

class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation: Cross-Entropy + Dice Loss.
    Balances pixel-wise accuracy and region overlap.
    """
    
    def __init__(self, 
                 ce_weight: float = 0.5, 
                 dice_weight: float = 0.5,
                 focal_gamma: float = 2.0,
                 use_focal: bool = True):
        super(CombinedSegmentationLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        
        if use_focal:
            self.ce_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined segmentation loss.
        
        Args:
            inputs: Predicted logits [B, C, H, W]
            targets: Ground truth masks [B, H, W]
            
        Returns:
            Dictionary with individual and total losses
        """
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss
        }

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint classification and segmentation training.
    Balances both tasks with learnable or fixed weights.
    """
    
    def __init__(self, 
                 cls_weight: float = 1.0,
                 seg_weight: float = 1.0,
                 learnable_weights: bool = False,
                 use_focal_cls: bool = True,
                 use_focal_seg: bool = True):
        super(MultiTaskLoss, self).__init__()
        
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Learnable task weights (uncertainty weighting)
            self.log_cls_weight = nn.Parameter(torch.tensor(0.0))
            self.log_seg_weight = nn.Parameter(torch.tensor(0.0))
        else:
            self.cls_weight = cls_weight
            self.seg_weight = seg_weight
        
        # Classification loss
        if use_focal_cls:
            self.cls_loss_fn = FocalLoss(gamma=2.0)
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()
        
        # Segmentation loss
        self.seg_loss_fn = CombinedSegmentationLoss(use_focal=use_focal_seg)
        
    def forward(self, 
                cls_pred: torch.Tensor, 
                seg_pred: torch.Tensor,
                cls_target: torch.Tensor, 
                seg_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            cls_pred: Classification predictions [B, num_classes]
            seg_pred: Segmentation predictions [B, num_classes, H, W]
            cls_target: Classification targets [B]
            seg_target: Segmentation targets [B, H, W]
            
        Returns:
            Dictionary with individual and total losses
        """
        # Compute individual losses
        cls_loss = self.cls_loss_fn(cls_pred, cls_target)
        seg_losses = self.seg_loss_fn(seg_pred, seg_target)
        seg_loss = seg_losses['total']
        
        if self.learnable_weights:
            # Uncertainty-based weighting (Kendall et al.)
            cls_weight = torch.exp(-self.log_cls_weight)
            seg_weight = torch.exp(-self.log_seg_weight)
            
            total_loss = (cls_weight * cls_loss + self.log_cls_weight + 
                         seg_weight * seg_loss + self.log_seg_weight)
        else:
            total_loss = self.cls_weight * cls_loss + self.seg_weight * seg_loss
            cls_weight = self.cls_weight
            seg_weight = self.seg_weight
        
        return {
            'total': total_loss,
            'cls_loss': cls_loss,
            'seg_loss': seg_loss,
            'seg_ce_loss': seg_losses['ce_loss'],
            'seg_dice_loss': seg_losses['dice_loss'],
            'cls_weight': cls_weight,
            'seg_weight': seg_weight
        }

class ConsistencyLoss(nn.Module):
    """
    Consistency loss between classification and segmentation predictions.
    Ensures that classification matches segmentation-derived predictions.
    """
    
    def __init__(self, temperature: float = 1.0):
        super(ConsistencyLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, 
                cls_pred: torch.Tensor, 
                seg_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            cls_pred: Classification predictions [B, num_classes]
            seg_pred: Segmentation predictions [B, num_classes, H, W]
            
        Returns:
            Consistency loss value
        """
        # Derive classification from segmentation (global average pooling)
        seg_derived_cls = F.adaptive_avg_pool2d(seg_pred, 1).squeeze(-1).squeeze(-1)
        
        # Apply temperature scaling
        cls_pred_scaled = cls_pred / self.temperature
        seg_derived_scaled = seg_derived_cls / self.temperature
        
        # KL divergence between predictions
        cls_prob = F.log_softmax(cls_pred_scaled, dim=1)
        seg_prob = F.softmax(seg_derived_scaled, dim=1)
        
        consistency_loss = F.kl_div(cls_prob, seg_prob, reduction='batchmean')
        
        return consistency_loss

class FLAMELoss(nn.Module):
    """
    Complete FLAME loss combining all components for optimal training.
    """
    
    def __init__(self, 
                 cls_weight: float = 1.0,
                 seg_weight: float = 1.0,
                 consistency_weight: float = 0.1,
                 learnable_weights: bool = False):
        super(FLAMELoss, self).__init__()
        
        self.consistency_weight = consistency_weight
        
        self.multitask_loss = MultiTaskLoss(
            cls_weight=cls_weight,
            seg_weight=seg_weight,
            learnable_weights=learnable_weights
        )
        
        self.consistency_loss = ConsistencyLoss()
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute complete FLAME loss.
        
        Args:
            outputs: Model outputs with 'classification' and 'segmentation' keys
            targets: Ground truth with 'classification' and 'segmentation' keys
            
        Returns:
            Dictionary with all loss components
        """
        # Multi-task loss
        mt_losses = self.multitask_loss(
            outputs['classification'], 
            outputs['segmentation'],
            targets['classification'], 
            targets['segmentation']
        )
        
        # Consistency loss
        consistency_loss = self.consistency_loss(
            outputs['classification'], 
            outputs['segmentation']
        )
        
        # Total loss
        total_loss = mt_losses['total'] + self.consistency_weight * consistency_loss
        
        # Combine all losses
        all_losses = mt_losses.copy()
        all_losses.update({
            'consistency_loss': consistency_loss,
            'total': total_loss
        })
        
        return all_losses

def create_loss_function(config: Optional[Dict] = None) -> FLAMELoss:
    """
    Factory function to create FLAME loss with configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Configured FLAME loss function
    """
    if config is None:
        config = {
            'cls_weight': 1.0,
            'seg_weight': 1.0,
            'consistency_weight': 0.1,
            'learnable_weights': False
        }
    
    return FLAMELoss(**config)

def main():
    """Demo function for loss testing."""
    print("🔥 FLAME Loss Functions Test")
    print("=" * 40)
    
    # Create dummy data
    batch_size = 4
    num_classes = 2
    height, width = 64, 64
    
    # Dummy predictions
    cls_pred = torch.randn(batch_size, num_classes)
    seg_pred = torch.randn(batch_size, num_classes, height, width)
    
    # Dummy targets
    cls_target = torch.randint(0, num_classes, (batch_size,))
    seg_target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test loss function
    loss_fn = create_loss_function()
    
    outputs = {
        'classification': cls_pred,
        'segmentation': seg_pred
    }
    
    targets = {
        'classification': cls_target,
        'segmentation': seg_target
    }
    
    losses = loss_fn(outputs, targets)
    
    print("Loss Components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n✅ Loss function test completed!")

if __name__ == "__main__":
    main()