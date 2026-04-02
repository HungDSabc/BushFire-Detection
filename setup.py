"""
FLAME Hybrid Fire Detection Setup
BDS DSC312 - Computer Vision with Multi-modal Models & Analytics
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="flame-hybrid-fire-detection",
    version="1.0.0",
    author="BDS DSC312 Team",
    author_email="your.email@example.com",
    description="Lightweight Shared-Backbone Architecture for Real-time Fire Detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/HungDSabc/BushFire-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "flame-train=models.mobilenet_hybrid:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="fire detection, computer vision, deep learning, pytorch, edge computing, DICTA",
    project_urls={
        "Bug Reports": "https://github.com/HungDSabc/BushFire-Detection/issues",
        "Source": "https://github.com/HungDSabc/BushFire-Detection",
        "Documentation": "https://github.com/HungDSabc/BushFire-Detection/blob/main/README.md",
    },
)