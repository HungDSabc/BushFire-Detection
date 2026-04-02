"""
Inference Pipeline for DICTA Fire Detection
==========================================
Load trained model and perform inference on images.

Usage:
    python scripts/inference_pipeline.py --model best_dicta_model.pth --image test.jpg
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms as T
import json
from datetime import datetime


class InferencePipeline:
    """Complete inference pipeline for DICTA model."""
    
    def __init__(self, model_path, device='auto', config_path=None):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or 'auto'
            config_path: Path to config YAML (optional)
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.model = None
        self.config = self._load_config(config_path)
        
        self._load_model()
        
    def _setup_device(self, device):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_config(self, config_path):
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_model(self):
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine model class from checkpoint or config
        from models.unet_segmentation import UltraOptimizedFireNet
        self.model = UltraOptimizedFireNet()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def preprocess_image(self, image_path, size=224):
        """Preprocess image to RGBT 4-channel input."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load RGB image
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        rgb_tensor = transform(image)  # [3, H, W]
        
        # Generate thermal channel (proxy)
        rgb_denorm = rgb_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                     torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = rgb_denorm.clamp(0, 1)
        
        thermal = 0.299 * rgb_denorm[0] + 0.587 * rgb_denorm[1] + 0.114 * rgb_denorm[2]
        thermal = thermal.unsqueeze(0)
        thermal = (thermal - 0.5) / 0.5
        
        # Combine RGBT
        rgbt = torch.cat([rgb_tensor, thermal], dim=0)  # [4, H, W]
        
        return rgbt, rgb_denorm.permute(1, 2, 0).numpy()
    
    @torch.no_grad()
    def predict(self, image_path, return_segmentation=True):
        """
        Predict on single image.
        
        Returns:
            dict with classification and segmentation results
        """
        rgbt, rgb_denorm = self.preprocess_image(image_path)
        rgbt = rgbt.unsqueeze(0).to(self.device)  # [1, 4, H, W]
        
        outputs = self.model(rgbt)
        
        # Classification
        cls_logits = outputs['classification']  # [1, 2]
        cls_probs = torch.softmax(cls_logits, dim=1)
        cls_pred = torch.argmax(cls_probs, dim=1).item()
        cls_confidence = cls_probs[0, cls_pred].item()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_path': str(image_path),
            'classification': {
                'is_fire': bool(cls_pred == 1),
                'confidence': float(cls_confidence),
                'class_probs': {
                    'no_fire': float(cls_probs[0, 0].item()),
                    'fire': float(cls_probs[0, 1].item())
                }
            }
        }
        
        # Segmentation (if requested)
        if return_segmentation:
            seg_logits = outputs['segmentation']  # [1, 2, H, W]
            seg_probs = torch.softmax(seg_logits, dim=1)
            seg_pred = torch.argmax(seg_probs, dim=1)[0].cpu().numpy()  # [H, W]
            
            fire_confidence = seg_probs[0, 1].cpu().numpy()  # [H, W]
            fire_pixels = (seg_pred == 1).sum()
            total_pixels = seg_pred.size
            
            results['segmentation'] = {
                'mask': seg_pred.tolist(),
                'fire_confidence_map': fire_confidence.tolist(),
                'fire_pixels': int(fire_pixels),
                'total_pixels': int(total_pixels),
                'fire_ratio': float(fire_pixels / total_pixels)
            }
        
        return results
    
    def batch_predict(self, image_dir, output_json=None):
        """Predict on all images in directory."""
        image_dir = Path(image_dir)
        results = []
        
        for image_file in sorted(image_dir.glob('*.*')):
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    result = self.predict(image_file)
                    results.append(result)
                    print(f"✅ {image_file.name}: {result['classification']['class_probs']}")
                except Exception as e:
                    print(f"❌ {image_file.name}: {e}")
        
        # Save results if requested
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to {output_json}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='DICTA Fire Detection Inference')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--image', help='Path to single image')
    parser.add_argument('--image-dir', help='Path to image directory')
    parser.add_argument('--output-json', help='Path to save results JSON')
    parser.add_argument('--device', default='auto', help='Device: cuda/cpu/auto')
    parser.add_argument('--no-segmentation', action='store_true', help='Skip segmentation output')
    parser.add_argument('--config', help='Path to config YAML')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InferencePipeline(args.model, args.device, args.config)
    
    # Single image inference
    if args.image:
        result = pipeline.predict(args.image, not args.no_segmentation)
        print("\n📊 INFERENCE RESULT:")
        print(json.dumps(result, indent=2))
        
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump([result], f, indent=2)
            print(f"\n✅ Result saved to {args.output_json}")
    
    # Batch inference
    elif args.image_dir:
        results = pipeline.batch_predict(args.image_dir, args.output_json)
        print(f"\n✅ Processed {len(results)} images")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
