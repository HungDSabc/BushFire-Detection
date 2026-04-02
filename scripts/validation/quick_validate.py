#!/usr/bin/env python3
"""
FLAME Hybrid Fire Detection - Quick Validation Script
Simple validation without importing problematic packages
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_pip_packages():
    """Check if packages are installed via pip list."""
    print("\n📦 Checking installed packages...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'opencv-python',
        'pillow',
        'tqdm'
    ]
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True)
        installed_packages = result.stdout.lower()
        
        missing_packages = []
        for package in required_packages:
            package_check = package.replace('-', '_').lower()  # Handle package name variations
            if package_check in installed_packages or package.lower() in installed_packages:
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package} - Not found")
                missing_packages.append(package)
        
        return len(missing_packages) == 0, missing_packages
        
    except subprocess.CalledProcessError:
        print("   ⚠️ Could not check pip packages")
        return False, []

def check_torch_basic():
    """Basic PyTorch check without importing other packages."""
    print("\n🔥 Checking PyTorch...")
    
    try:
        import torch
        
        # Basic test
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        
        print(f"   ✅ PyTorch {torch.__version__} working")
        print(f"   ✅ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PyTorch error: {str(e)[:50]}...")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 Checking project structure...")
    
    required_paths = [
        'models/mobilenet_hybrid.py',
        'models/loss_functions.py',
        'notebooks/02_training/DICTA_Shared_Backbone.ipynb',
        'notebooks/03_analysis/SOTA_Benchmarking.ipynb',
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

def main():
    """Main validation function."""
    print("🔥 FLAME Hybrid Fire Detection - Quick Validation")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
    
    # Check pip packages
    packages_ok, missing_packages = check_pip_packages()
    if not packages_ok:
        all_checks_passed = False
    
    # Check project structure
    structure_ok, missing_paths = check_project_structure()
    if not structure_ok:
        all_checks_passed = False
    
    # Check PyTorch
    if not check_torch_basic():
        all_checks_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 BASIC CHECKS PASSED - Environment looks good!")
        print("\n📋 Next steps:")
        print("   1. Try running: jupyter lab")
        print("   2. Open: notebooks/01_eda/EDA_FLAME_FireDetection.ipynb")
        print("   3. If you encounter import errors, run: python validate_setup.py")
    else:
        print("⚠️ SOME CHECKS FAILED")
        
        if missing_packages:
            print(f"\n📦 Install missing packages:")
            print(f"   pip install {' '.join(missing_packages)}")
        
        if missing_paths:
            print(f"\n📁 Missing files - check project structure")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)