# BushFire-Detection

BushFire-Detection is a computer vision project for wildfire detection with two tasks:
- Classification (fire vs. no fire)
- Segmentation (fire region masks)

The current primary direction is an MSBW-Net shared-backbone pipeline with SOTA baseline benchmarking.

## 1) Project Scope

Repository goals:
- Train and evaluate multi-task models (classification + segmentation).
- Benchmark against SOTA baselines with consistent evaluation settings.
- Support inference and edge deployment (including ONNX export).
- Keep experiments reproducible through notebooks, scripts, and documentation.

## 2) Quick Setup

Requirements:
- Python 3.8+
- pip

### Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional package install

```bash
pip install -e .
```

### Sanity check

```bash
python -c "import torch; print('Torch:', torch.__version__)"
python -c "from models.unet_segmentation import UltraOptimizedFireNet; print('Model import OK')"
```

## 3) Main Workflow (Recommended)

Run notebooks in this order for the primary pipeline:

1. `notebooks/01_eda/EDA_FLAME_FireDetection.ipynb`
2. `notebooks/02_training/MSBW-Net_Shared_Backbone.ipynb`
3. `notebooks/03_analysis/SOTA_Benchmarking.ipynb`
4. `notebooks/03_analysis/Complete_Pipeline_Analysis.ipynb`
5. `notebooks/02_training/Ablation_Studies.ipynb` (optional ablation analysis)

## 4) SOTA Training and Baselines (Active)

SOTA training notebooks:
- `notebooks/02_training/SOTA_CrossModalFire_Training.ipynb`
- `notebooks/02_training/SOTA_EfficientFireNet_Training.ipynb`
- `notebooks/02_training/SOTA_ThermalFusionNet_Training.ipynb`
- `notebooks/02_training/SOTA_YOLOFM_Training.ipynb`

Active baseline implementations:
- `models/sota_baselines/`

Current baseline files:
- `models/sota_baselines/crossmodal_fire_2024.py`
- `models/sota_baselines/efficientfirenet_2022.py`
- `models/sota_baselines/thermalfusion_net_2023.py`
- `models/sota_baselines/unet_fire_2022.py`
- `models/sota_baselines/yolofm_2024.py`

Note:
- The SOTA training notebooks and `models/sota_baselines/` are active benchmark components and are not archived.

## 5) Repository Structure

```text
BushFire-Detection/
|-- archive/                  # Legacy notebooks kept for reference
|-- config/                   # YAML configurations
|-- data/                     # Data and processed outputs
|-- docs/                     # Technical documentation and guides
|-- frames/                   # Frame-level dataset structure
|-- models/                   # Model definitions and checkpoints
|-- notebooks/                # EDA, training, and analysis notebooks (active)
|-- scripts/                  # Preprocess, validation, inference, export scripts
|-- tests/                    # Unit tests
|-- requirements.txt
|-- setup.py
`-- README.md
```

## 6) Archive Policy

`archive/` stores legacy/reference content.

Current legacy notebooks in `archive/notebooks/`:
- `FLAME_Hybrid_Training.ipynb`
- `EfficientNetB0_4ch.ipynb`
- `Segmentation_Training.ipynb`

These notebooks are retained for reference and are not part of the primary submission workflow.

See details in:
- `archive/README.md`

## 7) Main Scripts

### Data
- `scripts/data/preprocessing_pipeline.py`
- `scripts/data/rgb_thermal_dataset.py`
- `scripts/data/boreal_subset_c_preprocessing.py`
- `scripts/data/flame_dataset.py`

### Validation
- `scripts/validation/validate_setup.py`
- `scripts/validation/quick_validate.py`
- `scripts/validation/check_overfitting_subset_c.py`

### Inference / Export / Training Utility
- `scripts/inference_pipeline.py`
- `scripts/export_onnx.py`
- `scripts/resume_training.py`

### Quick CLI usage examples

Run single-image inference:

```bash
python scripts/inference_pipeline.py --model models/trained/msbw-net_shared_backbone/best_msbw_model.pth --image path/to/image.jpg --output-json outputs/prediction.json
```

Run folder inference:

```bash
python scripts/inference_pipeline.py --model models/trained/msbw-net_shared_backbone/best_msbw_model.pth --image-dir path/to/images --output-json outputs/batch_predictions.json
```

Export ONNX:

```bash
python scripts/export_onnx.py --model models/trained/msbw-net_shared_backbone/best_msbw_model.pth --output models/trained/msbw-net_shared_backbone/msbw_net.onnx --input-shape 1 4 224 224 --quantize
```

Resume training from checkpoint:

```bash
python scripts/resume_training.py --checkpoint models/trained/msbw-net_shared_backbone/best_msbw_model.pth --epochs 10
```

## 8) Models and Checkpoints

Model definitions:
- `models/unet_segmentation.py`
- `models/mobilenet_hybrid.py`
- `models/loss_functions.py`

Training artifacts and checkpoints:
- `models/trained/`

## 9) Tests

- `tests/test_models.py`
- `tests/test_metrics.py`

Run tests:

```bash
pytest -q
```

## 10) Additional Documentation

- `docs/guides/EXECUTION_ORDER.md`
- `docs/guides/REPO_STRUCTURE.md`
- `docs/MODEL_ARCHITECTURE.md`
- `docs/EDGE_DEPLOYMENT.md`
- `docs/SOTA_COMPARISON.md`
- `docs/MSBW-NET_BENCHMARK_PROTOCOL.md`
- `docs/COMPLETE_PIPELINE_ANALYSIS.md`

## 11) Notes
- For SOTA comparison, use benchmark notebooks in `notebooks/03_analysis/` with models from `models/sota_baselines/`.
- For benchmark reporting format and evaluation details, follow `docs/MSBW-NET_BENCHMARK_PROTOCOL.md`.
