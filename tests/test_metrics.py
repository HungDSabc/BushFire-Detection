"""
Unit Tests for DICTA Metrics
==============================
Validate metric calculations (mIoU, Fire IoU, latency, etc).

Run with: pytest tests/test_metrics.py
"""

import torch
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMIoUCalculation:
    """Test mIoU (mean Intersection over Union) calculation."""
    
    def test_perfect_prediction(self):
        """Test mIoU = 1.0 for perfect predictions."""
        batch_size, height, width = 2, 32, 32
        
        # Perfect segmentation (all correct)
        pred_mask = torch.zeros(batch_size, 2, height, width)
        target_mask = torch.zeros(batch_size, height, width, dtype=torch.long)
        
        # Set half to fire class
        pred_mask[:, 1, :height//2, :] = 10.0  # High confidence fire
        target_mask[:, :height//2, :] = 1  # Target is fire
        
        miou = calculate_miou(pred_mask, target_mask)
        assert miou > 0.95, f"Expected mIoU close to 1.0, got {miou}"
        print(f"✅ Perfect prediction mIoU: {miou:.4f}")
    
    def test_random_predictions(self):
        """Test mIoU for random predictions is reasonable."""
        batch_size, height, width = 4, 32, 32
        
        pred_mask = torch.randn(batch_size, 2, height, width)
        target_mask = torch.randint(0, 2, (batch_size, height, width))
        
        miou = calculate_miou(pred_mask, target_mask)
        assert 0.0 <= miou <= 1.0, f"mIoU out of bounds: {miou}"
        print(f"✅ Random prediction mIoU: {miou:.4f}")
    
    def test_miou_symmetry(self):
        """Test that mIoU calculation is consistent."""
        batch_size, height, width = 2, 32, 32
        
        pred_mask = torch.randn(batch_size, 2, height, width)
        target_mask = torch.randint(0, 2, (batch_size, height, width))
        
        miou1 = calculate_miou(pred_mask.clone(), target_mask.clone())
        miou2 = calculate_miou(pred_mask.clone(), target_mask.clone())
        
        assert miou1 == miou2, "mIoU calculation not deterministic"
        print(f"✅ mIoU calculation deterministic: {miou1:.4f}")
    
    def test_fire_iou_calculation(self):
        """Test Fire class IoU calculation."""
        batch_size, height, width = 2, 32, 32
        
        pred_mask = torch.zeros(batch_size, 2, height, width)
        target_mask = torch.zeros(batch_size, height, width, dtype=torch.long)
        
        # 50% fire
        pred_mask[:, 1, :height//2, :] = 10.0
        target_mask[:, :height//2, :] = 1
        
        fire_iou = calculate_fire_iou(pred_mask, target_mask)
        assert fire_iou > 0.8, f"Expected high Fire IoU, got {fire_iou}"
        print(f"✅ Fire class IoU: {fire_iou:.4f}")


class TestLatencyMetrics:
    """Test latency benchmark calculations."""
    
    def test_latency_stats(self):
        """Test latency statistics calculation."""
        # Simulate latency measurements (in ms)
        latencies = torch.tensor([
            10.2, 11.5, 9.8, 10.1, 10.5,  # Batch 1
            12.1, 11.8, 10.2, 11.0, 10.9   # Batch 2
        ])
        
        stats = calculate_latency_stats(latencies)
        
        assert 'mean' in stats
        assert 'p50' in stats
        assert 'p95' in stats
        assert 'p99' in stats
        assert 'max' in stats
        
        # Verify statistics are reasonable
        assert stats['p50'] <= stats['p95'] <= stats['p99']
        assert stats['mean'] > 0
        print(f"✅ Latency stats: mean={stats['mean']:.2f}ms, p95={stats['p95']:.2f}ms")
    
    def test_frames_per_second(self):
        """Test FPS calculation from latency."""
        mean_latency_ms = 10.0  # 10 ms per frame
        
        fps = 1000.0 / mean_latency_ms
        
        assert abs(fps - 100.0) < 0.1, f"Expected 100 FPS, got {fps}"
        print(f"✅ FPS calculation: {fps:.1f} FPS for {mean_latency_ms}ms latency")


class TestClassificationMetrics:
    """Test classification metrics."""
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        batch_size = 100
        
        # Perfect predictions
        pred = torch.ones(batch_size, 2)
        pred[:, 0] = 0
        target = torch.ones(batch_size, dtype=torch.long)
        
        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
        assert accuracy == 1.0, f"Expected accuracy 1.0, got {accuracy}"
        print(f"✅ Perfect predictions accuracy: {accuracy:.4f}")
    
    def test_fp_fn_calculation(self):
        """Test False Positive / False Negative calculation."""
        batch_size = 100
        
        # Mix of correct and incorrect predictions
        pred = torch.randn(batch_size, 2)
        target = torch.randint(0, 2, (batch_size,))
        
        pred_class = pred.argmax(dim=1)
        
        tp = ((pred_class == 1) & (target == 1)).sum().item()
        fp = ((pred_class == 1) & (target == 0)).sum().item()
        fn = ((pred_class == 0) & (target == 1)).sum().item()
        tn = ((pred_class == 0) & (target == 0)).sum().item()
        
        assert tp + fp + fn + tn == batch_size
        print(f"✅ Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")


class TestSegmentationMetrics:
    """Test segmentation-specific metrics."""
    
    def test_pixel_accuracy(self):
        """Test pixel-level accuracy."""
        height, width = 64, 64
        
        # Perfect segmentation
        pred = torch.ones(1, 2, height, width)
        pred[:, 0] = 0
        target = torch.ones(1, height, width, dtype=torch.long)
        
        pred_mask = pred.argmax(dim=1)
        accuracy = (pred_mask == target).float().mean().item()
        
        assert accuracy == 1.0, f"Expected pixel accuracy 1.0, got {accuracy}"
        print(f"✅ Perfect segmentation pixel accuracy: {accuracy:.4f}")
    
    def test_miou_edge_cases(self):
        """Test mIoU with edge cases."""
        # All black (no fire in prediction or target)
        pred = torch.zeros(1, 2, 32, 32)
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        
        miou = calculate_miou(pred, target)
        assert 0 <= miou <= 1, f"mIoU out of valid range: {miou}"
        print(f"✅ All-black segmentation mIoU: {miou:.4f}")
        
        # All fire
        pred = torch.zeros(1, 2, 32, 32)
        pred[:, 1] = 1
        target = torch.ones(1, 32, 32, dtype=torch.long)
        
        miou = calculate_miou(pred, target)
        assert 0.95 < miou <= 1.0, f"Expected high mIoU, got {miou}"
        print(f"✅ All-fire segmentation mIoU: {miou:.4f}")


# Helper functions (mimicking actual metric calculations)

def calculate_miou(pred_mask, target_mask, num_classes=2):
    """Calculate mean Intersection over Union."""
    pred_mask = pred_mask.argmax(dim=1)
    
    ious = []
    for class_id in range(num_classes):
        pred_pixels = pred_mask == class_id
        target_pixels = target_mask == class_id
        
        intersection = (pred_pixels & target_pixels).sum().float()
        union = (pred_pixels | target_pixels).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return np.mean(ious)


def calculate_fire_iou(pred_mask, target_mask):
    """Calculate Fire class IoU specifically."""
    pred_mask = pred_mask.argmax(dim=1)
    
    fire_pred = pred_mask == 1
    fire_target = target_mask == 1
    
    intersection = (fire_pred & fire_target).sum().float()
    union = (fire_pred | fire_target).sum().float()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def calculate_latency_stats(latencies):
    """Calculate latency statistics."""
    latencies_np = latencies.numpy() if isinstance(latencies, torch.Tensor) else latencies
    
    return {
        'mean': np.mean(latencies_np),
        'std': np.std(latencies_np),
        'min': np.min(latencies_np),
        'p50': np.percentile(latencies_np, 50),
        'p95': np.percentile(latencies_np, 95),
        'p99': np.percentile(latencies_np, 99),
        'max': np.max(latencies_np),
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
