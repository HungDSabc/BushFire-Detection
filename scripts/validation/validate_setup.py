#!/usr/bin/env python3
"""
FLAME Hybrid Fire Detection - Setup Validation Script
BDS DSC312 - Computer Vision with Multi-modal Models & Analytics

This script validates that the environment is properly set up and all
required dependencies are available.
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print(" Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_required_packages():
    """Check if all required packages are installed."""
    print("\n Checking required packages...")
    
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scikit-learn', 'sklearn'),
        ('opencv-python', 'cv2'),
        ('pillow', 'PIL'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    warning_packages = []
    
    for package_name, import_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"   ✅ {package_name}")
        except ImportError as e:
            print(f"   ❌ {package_name} - Not found")
            missing_packages.append(package_name)
        except Exception as e:
            error_msg = str(e)
            if "numpy" in error_msg.lower() and "dtype size changed" in error_msg.lower():
                print(f"   ⚠️ {package_name} - NumPy compatibility issue")
                warning_packages.append(package_name)
            else:
                print(f"   ⚠️ {package_name} - Import error: {str(e)[:50]}...")
                warning_packages.append(package_name)
    
    if warning_packages:
        print(f"\n   ⚠️ Warning: Some packages have compatibility issues but are installed")
        print(f"   💡 Consider: pip install numpy<2.0.0 to fix compatibility")
    
    return len(missing_packages) == 0, missing_packages

def check_project_structure():
    """Check if project structure is correct."""
    print("\n Checking project structure...")
    
    required_paths = [
        'models/mobilenet_hybrid.py',
        'models/loss_functions.py',
        'models/unet_segmentation.py',
        'notebooks/02_training/DICTA_Shared_Backbone.ipynb',
        'notebooks/03_analysis/SOTA_Benchmarking.ipynb',
        'data/processed/stage2_results_table.csv',
        'docs/DICTA_BENCHMARK_PROTOCOL.md',
        'requirements.txt',
        'LICENSE'
    ]
    
    missing_paths = []
    
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"   ✅ {path_str}")
        else:
            print(f"   ❌ {path_str} - Not found")
            missing_paths.append(path_str)
    
    return len(missing_paths) == 0, missing_paths

def check_torch_functionality():
    """Check if PyTorch is working correctly."""
    print("\n🔥 Checking PyTorch functionality...")
    
    try:
        import torch
        
        # Check basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print(f"   ✅ Basic tensor operations working")
        
        # Check device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   ✅ Device: {device}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ℹ️ CUDA not available - using CPU")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PyTorch error: {str(e)[:100]}...")
        return False

def main():
    """Main validation function."""
    print("🔥 FLAME Hybrid Fire Detection - Setup Validation")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
    
    # Check required packages
    packages_ok, missing_packages = check_required_packages()
    if not packages_ok:
        all_checks_passed = False
    
    # Check project structure
    structure_ok, missing_paths = check_project_structure()
    if not structure_ok:
        all_checks_passed = False
    
    # Check PyTorch functionality
    if not check_torch_functionality():
        all_checks_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED - Environment is ready!")
        print("\n📋 Next steps:")
        print("   1. Run notebooks in order (see EXECUTION_ORDER.md)")
        print("   2. Start with EDA: notebooks/01_eda/EDA_FLAME_FireDetection.ipynb")
        print("   3. Train models: notebooks/02_training/DICTA_Shared_Backbone.ipynb")
        print("   4. Analyze results: notebooks/03_analysis/SOTA_Benchmarking.ipynb")
    else:
        print("⚠️ SOME CHECKS FAILED - Please review the issues above")
        
        if missing_packages:
            print(f"\n📦 Install missing packages:")
            print(f"   pip install {' '.join(missing_packages)}")
        
        if missing_paths:
            print(f"\n📁 Missing files/directories:")
            for path in missing_paths:
                print(f"   - {path}")
        
        print(f"\n💡 Common fixes:")
        print(f"   - NumPy compatibility: pip install numpy<2.0.0")
        print(f"   - Reinstall packages: pip install --upgrade --force-reinstall matplotlib pandas seaborn scikit-learn")
        print(f"   - Fresh environment: Create new venv and install requirements.txt")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)