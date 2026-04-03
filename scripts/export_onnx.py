"""
Export PyTorch Model to ONNX Format
===================================
Convert trained MSBW-Net model to ONNX for edge deployment.

Usage:
    python scripts/export_onnx.py --model best_msbw-net_model.pth --output model.onnx
"""

import torch
import argparse
from pathlib import Path


def export_to_onnx(model_path, output_path, input_shape=(1, 4, 224, 224), 
                   opset_version=14, optimize=True):
    """
    Export PyTorch model to ONNX.
    
    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape [B, C, H, W]
        opset_version: ONNX opset version (13, 14, 16, etc.)
        optimize: Whether to optimize the ONNX model
    """
    print(f"📦 Exporting PyTorch model to ONNX...")
    print(f"   Input model: {model_path}")
    print(f"   Output model: {output_path}")
    print(f"   Input shape: {input_shape}")
    print(f"   ONNX opset: {opset_version}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Reconstruct model
    from models.unet_segmentation import UltraOptimizedFireNet
    model = UltraOptimizedFireNet()
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    input_names = ['rgbt_input']
    output_names = ['classification', 'segmentation']
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        dynamic_axes={
            'rgbt_input': {0: 'batch_size'},
            'classification': {0: 'batch_size'},
            'segmentation': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported successfully!")
    print(f"   File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verified successfully")
    except ImportError:
        print(f"⚠️  ONNX verification skipped (onnx not installed)")
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
    
    # Optional: Optimize ONNX model
    if optimize:
        try:
            from onnxruntime.transformers import optimizer
            print(f"🔧 Optimizing ONNX model...")
            # Optimization for transformers/attention models
            # This is optional and may not apply to all architectures
        except ImportError:
            print(f"⚠️  ONNX Runtime optimizer not available")


def quantize_onnx(onnx_path, output_path=None, quantization_type='dynamic'):
    """
    Quantize ONNX model for further size reduction.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save quantized model
        quantization_type: 'dynamic' or 'static'
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
        
        if output_path is None:
            output_path = str(onnx_path).replace('.onnx', '_quantized.onnx')
        
        print(f"🔧 Quantizing ONNX model ({quantization_type})...")
        
        if quantization_type == 'dynamic':
            quantize_dynamic(str(onnx_path), output_path, weight_type=QuantType.QInt8)
        else:
            # Static quantization requires calibration data
            print(f"❌ Static quantization requires calibration data - skipped")
            return
        
        print(f"✅ Quantization successful!")
        print(f"   Original size: {Path(onnx_path).stat().st_size / (1024**2):.1f} MB")
        print(f"   Quantized size: {Path(output_path).stat().st_size / (1024**2):.1f} MB")
        
        return output_path
    
    except ImportError:
        print(f"❌ ONNX Runtime quantization tools not available")
        print(f"   Install with: pip install onnxruntime[optimization]")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model', required=True, help='Path to trained PyTorch model')
    parser.add_argument('--output', required=True, help='Path to save ONNX model')
    parser.add_argument('--input-shape', nargs=4, type=int, default=[1, 4, 224, 224],
                       help='Input shape [batch, channels, height, width]')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version')
    parser.add_argument('--quantize', action='store_true', help='Quantize ONNX model')
    parser.add_argument('--no-optimize', action='store_true', help='Skip optimization')
    
    args = parser.parse_args()
    
    # Export to ONNX
    export_to_onnx(
        args.model,
        args.output,
        input_shape=tuple(args.input_shape),
        opset_version=args.opset,
        optimize=not args.no_optimize
    )
    
    # Quantize if requested
    if args.quantize:
        quantize_onnx(args.output)


if __name__ == '__main__':
    main()
