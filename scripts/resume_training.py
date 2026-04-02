"""
Resume Training from Checkpoint
================================
Continue training from a saved checkpoint.

Usage:
    python scripts/resume_training.py --checkpoint best_dicta_model.pth --epochs 10
"""

import torch
import argparse
import yaml
from pathlib import Path


def resume_training(checkpoint_path, config_path=None, additional_epochs=10):
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        config_path: Path to training config (uses saved config if not provided)
        additional_epochs: Additional epochs to train
    """
    print(f"📦 Resuming training from checkpoint...")
    print(f"   Checkpoint: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n📋 Checkpoint Information:")
    print(f"   Epoch: {checkpoint.get('epoch', '?')}")
    print(f"   Best mIoU: {checkpoint.get('best_miou', '?'):.4f}")
    print(f"   Best Fire IoU: {checkpoint.get('best_fire_iou', '?'):.4f}")
    
    # Load config from checkpoint or file
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   Config: (from checkpoint)")
    elif config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"   Config: {config_path}")
    else:
        raise ValueError("Config not found in checkpoint and --config not provided")
    
    # Load model and optimizer state
    print(f"\n🧠 Model Information:")
    model_state_dict = checkpoint['model_state_dict']
    print(f"   Layers: {len(model_state_dict)}")
    print(f"   Total parameters: {sum(v.numel() for v in model_state_dict.values()):,}")
    
    # Prepare for continued training
    print(f"\n⚙️  Resuming Configuration:")
    print(f"   Additional epochs: {additional_epochs}")
    print(f"   Learning rate: {checkpoint['optimizer_state_dict']['param_groups'][0]['lr']}")
    
    # Save resume checkpoint
    resume_checkpoint = {
        'epoch': checkpoint.get('epoch', 0) + 1,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': checkpoint['optimizer_state_dict'],
        'best_miou': checkpoint.get('best_miou', 0.0),
        'best_fire_iou': checkpoint.get('best_fire_iou', 0.0),
        'history': checkpoint.get('history', {}),
        'config': config,
        'additional_epochs': additional_epochs,
    }
    
    resume_path = checkpoint_path.parent / f"{checkpoint_path.stem}_resume.pth"
    torch.save(resume_checkpoint, resume_path)
    print(f"\n✅ Resume checkpoint prepared: {resume_path}")
    
    print(f"\n📝 Note:")
    print(f"   Run the DICTA training notebook and update config to load from:")
    print(f"   {resume_path}")
    print(f"\n   Or use in a Python script:")
    print(f"   >>> checkpoint = torch.load('{resume_path}')")
    print(f"   >>> model.load_state_dict(checkpoint['model_state_dict'])")
    print(f"   >>> optimizer.load_state_dict(checkpoint['optimizer_state_dict'])")
    
    return resume_path


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--config', help='Path to training config YAML')
    parser.add_argument('--epochs', type=int, default=10, help='Additional epochs to train')
    
    args = parser.parse_args()
    
    resume_training(args.checkpoint, args.config, args.epochs)


if __name__ == '__main__':
    main()
