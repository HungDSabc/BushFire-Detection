# Repository Structure Guide

This document describes the professionalized layout for the MSBW-Net-focused workflow.

## Top-Level Layout

- `README.md`: Main project overview.
- `config/`: Training and evaluation configuration files.
- `models/`: Model architectures and loss definitions.
- `scripts/`: Operational scripts for data prep, inference, export, and validation.
- `notebooks/`: Active analysis and training notebooks.
- `docs/`: Technical documentation, guides, and submission artifacts.
- `archive/`: Legacy and experimental code (not part of primary submission).
- `data/`, `frames/`: Datasets and processed outputs.
- `tests/`: Unit tests for model and metric checks.

## Notebook Status

- `notebooks/01_eda/`: Exploratory analysis notebooks ✅ Completed
- `notebooks/02_training/`: Active training notebooks (MSBW-Net primary + legacy references) ✅ Primary complete
- `notebooks/03_analysis/`: Benchmarking and analysis notebooks ✅ Complete
  - **Legacy Notebooks** (reference only): `FLAME_Hybrid_Training.ipynb`, `EfficientNetB0_4ch.ipynb`, `Segmentation_Training.ipynb`

## Script Organization

- `scripts/data/`: Data preprocessing and dataset utilities.
- `scripts/validation/`: Environment and setup validation scripts.
- `scripts/`: Production-facing scripts (`inference_pipeline.py`, `export_onnx.py`, etc.).

## Documentation Organization

- `docs/guides/`: Execution order and workflow guides.
- `docs/submission/`: Submission-facing notes and exports.
- `docs/project/`: Project management and restructuring history.
- `docs/assets/eda/`: EDA figures used by docs.
Structure & Policy

The `archive/` folder (created April 4, 2026) contains legacy and experimental code:

- `archive/README.md`: Complete inventory of archived items and reactivation guide
- `archive/notebooks/`: Documentation for legacy training notebooks (actual files stay in `notebooks/02_training/`)
- `archive/scripts/`: Documentation for optional/experimental utility scripts
- `archive/models/`: Documentation for baseline model implementations (used in SOTA benchmarking)

**Archive Policy**:
- ✅ Legacy items remain FUNCTIONAL in their original locations
- ✅ NOT part of DICTA 2026 primary submission workflow
- ✅ Marked clearly as "Legacy Reference Only" in documentation
- ✅ Can be reactivated for future experiments without restoration

## Notebook Lifecycle

- **Active Notebooks**: EDA, MSBW-Net training, SOTA benchmarking, complete pipeline analysis ✅
- **Legacy Notebooks**: FLAME_Hybrid, EfficientNetB0_4ch, Segmentation_Training (reference only)

## Naming Conventions

- Keep notebook names action-oriented (for example: `SOTA_Benchmarking.ipynb`).
- Keep script names task-oriented (for example: `benchmark_latency.py`).
- Avoid duplicate copies at root; place files in their functional folder.
