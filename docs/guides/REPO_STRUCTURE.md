# Repository Structure Guide

This document describes the professionalized layout for the DICTA-focused workflow.

## Top-Level Layout

- `README.md`: Main project overview.
- `config/`: Training and evaluation configuration files.
- `models/`: Model architectures and loss definitions.
- `scripts/`: Operational scripts for data prep, inference, export, and validation.
- `notebooks/`: Active analysis and training notebooks.
- `docs/`: Technical documentation, guides, and submission artifacts.
- `archive/`: Legacy notebooks retained for reference only.
- `data/`, `frames/`: Datasets and processed outputs.
- `tests/`: Unit tests for model and metric checks.

## Notebook Organization

- `notebooks/01_eda/`: Exploratory analysis notebooks.
- `notebooks/02_training/`: Active training notebooks used in DICTA workflow.
- `notebooks/03_analysis/`: Benchmarking and analysis notebooks.
- `notebooks/04_experiments/`: Optional experiments.

## Script Organization

- `scripts/data/`: Data preprocessing and dataset utilities.
- `scripts/validation/`: Environment and setup validation scripts.
- `scripts/`: Production-facing scripts (`inference_pipeline.py`, `export_onnx.py`, etc.).

## Documentation Organization

- `docs/guides/`: Execution order and workflow guides.
- `docs/submission/`: Submission-facing notes and exports.
- `docs/project/`: Project management and restructuring history.
- `docs/assets/eda/`: EDA figures used by docs.

## Archive Policy

- Legacy notebooks are moved to `archive/notebooks/`.
- Archived files are not used as DICTA evidence.
- New development should not target archived files.

## Naming Conventions

- Keep notebook names action-oriented (for example: `SOTA_Benchmarking.ipynb`).
- Keep script names task-oriented (for example: `benchmark_latency.py`).
- Avoid duplicate copies at root; place files in their functional folder.
