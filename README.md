# BushFire Detection

BushFire Detection is a computer-vision project for wildfire classification and segmentation.
The repository is organized around a DICTA-focused shared-backbone workflow designed for reproducible training and edge deployment.

## What This Repo Contains

- Multi-task fire pipeline (classification + segmentation).
- Notebook-first experimentation and analysis workflow.
- Script-based data preprocessing, validation, inference, and ONNX export.
- Config and documentation for repeatable experiments.

## Quick Start

### 1. Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Sanity check

```bash
python -c "import torch; print(torch.__version__)"
python -c "from models.unet_segmentation import UltraOptimizedFireNet; print('Model import OK')"
```

### 3. Run the main workflow

Open Jupyter and execute notebooks in order:

1. [notebooks/01_eda/EDA_FLAME_FireDetection.ipynb](notebooks/01_eda/EDA_FLAME_FireDetection.ipynb)
2. [notebooks/02_training/DICTA_Shared_Backbone.ipynb](notebooks/02_training/DICTA_Shared_Backbone.ipynb)
3. [notebooks/02_training/Ablation_Studies.ipynb](notebooks/02_training/Ablation_Studies.ipynb)
4. [notebooks/03_analysis/Complete_Pipeline_Analysis.ipynb](notebooks/03_analysis/Complete_Pipeline_Analysis.ipynb)
5. [notebooks/03_analysis/SOTA_Benchmarking.ipynb](notebooks/03_analysis/SOTA_Benchmarking.ipynb)

## Project Structure

```text
BushFire-Detection/
├── config/                  # YAML configs
├── data/                    # processed outputs and artifacts
├── docs/                    # architecture, benchmarking, deployment docs
├── frames/                  # raw frame-level dataset layout
├── models/                  # model definitions and checkpoints
├── notebooks/               # EDA, training, analysis, experiments
├── scripts/                 # CLI scripts for data, validation, export, inference
├── tests/                   # unit tests
├── README.md
├── requirements.txt
└── setup.py
```

## Key Folders

- [config/](config): experiment and runtime configuration.
- [data/](data): generated outputs. Keep large artifacts here, not at root.
- [docs/](docs): technical references and workflow guides.
- [frames/](frames): raw dataset structure used by notebooks/scripts.
- [models/](models): architectures and saved checkpoints in [models/trained/](models/trained).
- [scripts/](scripts): operational scripts.

## Script Entry Points

- [scripts/data/preprocessing_pipeline.py](scripts/data/preprocessing_pipeline.py): preprocessing pipeline.
- [scripts/data/rgb_thermal_dataset.py](scripts/data/rgb_thermal_dataset.py): dataset utilities.
- [scripts/data/boreal_subset_c_preprocessing.py](scripts/data/boreal_subset_c_preprocessing.py): Subset-C preprocessing.
- [scripts/validation/validate_setup.py](scripts/validation/validate_setup.py): environment validation.
- [scripts/validation/quick_validate.py](scripts/validation/quick_validate.py): quick checks.
- [scripts/validation/check_overfitting_subset_c.py](scripts/validation/check_overfitting_subset_c.py): overfitting diagnostics.
- [scripts/inference_pipeline.py](scripts/inference_pipeline.py): inference path.
- [scripts/export_onnx.py](scripts/export_onnx.py): ONNX export.
- [scripts/resume_training.py](scripts/resume_training.py): resume training jobs.

Tip: run each script with `--help` first to inspect expected arguments.

## Legacy Notebooks

The following notebooks are kept as references and are not the primary DICTA path:

- [notebooks/02_training/FLAME_Hybrid_Training.ipynb](notebooks/02_training/FLAME_Hybrid_Training.ipynb)
- [notebooks/02_training/EfficientNetB0_4ch.ipynb](notebooks/02_training/EfficientNetB0_4ch.ipynb)
- [notebooks/02_training/Segmentation_Training.ipynb](notebooks/02_training/Segmentation_Training.ipynb)

## Where Outputs Should Go

- Metrics, tables, and plots: [data/processed/Output/](data/processed/Output)
- Trained checkpoints: [models/trained/](models/trained)
- Documentation assets: [docs/assets/](docs/assets)

## Troubleshooting

- Import errors: activate `.venv` and reinstall with `pip install -r requirements.txt`.
- Notebook kernel mismatch: select the `.venv` Python interpreter in Jupyter.
- CUDA mismatch: install the correct PyTorch build for your GPU/driver.
- Missing files in notebook runs: verify paths under [frames/](frames) and [data/](data).

## Additional Documentation

- [docs/guides/EXECUTION_ORDER.md](docs/guides/EXECUTION_ORDER.md)
- [docs/guides/REPO_STRUCTURE.md](docs/guides/REPO_STRUCTURE.md)
- [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)
- [docs/EDGE_DEPLOYMENT.md](docs/EDGE_DEPLOYMENT.md)
- [docs/SOTA_COMPARISON.md](docs/SOTA_COMPARISON.md)
