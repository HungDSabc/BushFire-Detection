"""
Unit Tests for DICTA Model Architecture
========================================
Validate model architectures, forward pass, and output shapes.

Run with: pytest tests/test_models.py
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestUltraOptimizedFireNet:
    """Test UltraOptimizedFireNet architecture."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        from models.unet_segmentation import UltraOptimizedFireNet
        return UltraOptimizedFireNet()
    
    def test_model_creation(self, model):
        """Test model instantiation."""
        assert model is not None
        print(f"✅ Model created successfully")
    
    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # Create dummy input [batch_size=1, channels=4 (RGBT), height=224, width=224]
        dummy_input = torch.randn(1, 4, 224, 224, device=device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        assert isinstance(outputs, dict), "Output should be dictionary"
        assert 'classification' in outputs, "Missing 'classification' key"
        assert 'segmentation' in outputs, "Missing 'segmentation' key"
        print(f"✅ Forward pass successful")
    
    def test_classification_output_shape(self, model):
        """Test classification output shape."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        dummy_input = torch.randn(1, 4, 224, 224, device=device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        cls_output = outputs['classification']
        assert cls_output.shape == (1, 2), f"Expected (1, 2), got {cls_output.shape}"
        print(f"✅ Classification output shape correct: {cls_output.shape}")
    
    def test_segmentation_output_shape(self, model):
        """Test segmentation output shape."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        dummy_input = torch.randn(1, 4, 224, 224, device=device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        seg_output = outputs['segmentation']
        # Output channels=2 (no_fire, fire) at full resolution
        assert seg_output.shape[1] == 2, f"Expected 2 channels, got {seg_output.shape[1]}"
        print(f"✅ Segmentation output shape correct: {seg_output.shape}")
    
    def test_model_parameters(self, model):
        """Test model parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
        # UltraOptimized should have < 1M parameters
        assert total_params < 1_000_000, f"Too many parameters: {total_params}"
    
    def test_model_eval_mode(self, model):
        """Test model switches to eval mode correctly."""
        model.eval()
        
        for module in model.modules():
            if hasattr(module, 'training'):
                assert not module.training, "Model should be in eval mode"
        
        print(f"✅ Model eval mode set correctly")
    
    def test_batch_processing(self, model):
        """Test processing multiple samples in batch."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 4, 224, 224, device=device)
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            assert outputs['classification'].shape[0] == batch_size
            assert outputs['segmentation'].shape[0] == batch_size
            print(f"✅ Batch size {batch_size} processed correctly")
    
    def test_gradient_flow(self, model):
        """Test that gradients flow through model."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.train()
        
        dummy_input = torch.randn(2, 4, 224, 224, device=device, requires_grad=True)
        
        outputs = model(dummy_input)
        loss = outputs['classification'].sum() + outputs['segmentation'].sum()
        loss.backward()
        
        # Check if gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients flowing through model"
        print(f"✅ Gradients flowing correctly through model")
    
    def test_model_device_transfer(self, model):
        """Test model transfer between devices."""
        # Test transfer to CPU
        model_cpu = model.cpu()
        dummy_input = torch.randn(1, 4, 224, 224)
        
        with torch.no_grad():
            outputs = model_cpu(dummy_input)
        
        assert outputs['classification'].device.type == 'cpu'
        print(f"✅ Model transfers to CPU successfully")
        
        # Test transfer to CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            dummy_input = torch.randn(1, 4, 224, 224, device='cuda')
            
            with torch.no_grad():
                outputs = model_cuda(dummy_input)
            
            assert outputs['classification'].device.type == 'cuda'
            print(f"✅ Model transfers to CUDA successfully")


class TestSharedBackboneFireNet:
    """Test SharedBackboneFireNet architecture."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        from models.mobilenet_hybrid import SharedBackboneFireNet
        return SharedBackboneFireNet()
    
    def test_model_creation(self, model):
        """Test model instantiation."""
        assert model is not None
        print(f"✅ SharedBackbone model created successfully")
    
    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        dummy_input = torch.randn(1, 4, 224, 224, device=device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        assert 'classification' in outputs
        assert 'segmentation' in outputs
        print(f"✅ SharedBackbone forward pass successful")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
