# BushFire-Detection


BushFire-Detection is a computer vision project for wildfire detection, supporting:
- **Classification** ([fire vs. no fire](notebooks/02_training/MSBW-Net_Shared_Backbone.ipynb))
- **Segmentation** ([fire region masks](notebooks/02_training/Segmentation_Training.ipynb))

The main pipeline is based on an [MSBW-Net shared-backbone](notebooks/02_training/MSBW-Net_Shared_Backbone.ipynb) with SOTA baseline benchmarking. All code, models, and data structure are organized for reproducibility and extensibility.

## 1) Project Scope


**Repository goals:**
- Train and evaluate multi-task models ([classification + segmentation](notebooks/02_training/MSBW-Net_Shared_Backbone.ipynb)).
- Benchmark against [SOTA baselines](models/sota_baselines/) with consistent evaluation settings.
- Support [inference and edge deployment](docs/EDGE_DEPLOYMENT.md) (including [ONNX export](scripts/export_onnx.py)).
- Ensure experiments are reproducible via [notebooks](notebooks/), [scripts](scripts/), and [documentation](docs/).

## 2) Quick Setup


**Requirements:**
- [Python 3.8+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/)

### Create environment


```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r [requirements.txt](requirements.txt)
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


**Recommended notebook execution order:**
1. [EDA_FLAME_FireDetection.ipynb](notebooks/01_eda/EDA_FLAME_FireDetection.ipynb) – Exploratory data analysis
2. [MSBW-Net_Shared_Backbone.ipynb](notebooks/02_training/MSBW-Net_Shared_Backbone.ipynb) – Main training pipeline
3. [SOTA_Benchmarking.ipynb](notebooks/03_analysis/SOTA_Benchmarking.ipynb) – SOTA benchmarking
4. [Complete_Pipeline_Analysis.ipynb](notebooks/03_analysis/Complete_Pipeline_Analysis.ipynb) – Full pipeline analysis
5. [Ablation_Studies.ipynb](notebooks/02_training/Ablation_Studies.ipynb) – Optional ablation analysis

## 4) SOTA Training and Baselines (Active)


**SOTA training notebooks:**
- [SOTA_CrossModalFire_Training.ipynb](notebooks/02_training/SOTA_CrossModalFire_Training.ipynb)
- [SOTA_EfficientFireNet_Training.ipynb](notebooks/02_training/SOTA_EfficientFireNet_Training.ipynb)
- [SOTA_ThermalFusionNet_Training.ipynb](notebooks/02_training/SOTA_ThermalFusionNet_Training.ipynb)

**Active baseline implementations:**

**Current SOTA Baselines and Models Used for Benchmarking:**

- **UNet (Custom, Lightweight):** Ultra-lightweight U-Net for wildfire segmentation, optimized for edge deployment and 4-channel (RGBT) input.
- **EfficientNet-B0:** SOTA baseline using EfficientNet-B0 with a lightweight decoder, designed for real-time fire segmentation and edge devices.
- **MobileNetV3 Small:** MobileNetV3 Small backbone with dual-head for classification and segmentation, adapted for 4-channel input and low-latency edge inference.
- **Xception:** Xception-based segmentation model for benchmarking against deeper architectures.
- [thermalfusion_net_2023.py](models/sota_baselines/thermalfusion_net_2023.py): ThermalFusion-Net 2023 – Dual-branch RGB+thermal fusion
- [efficientfirenet_2022.py](models/sota_baselines/efficientfirenet_2022.py): EfficientFireNet 2022 – EfficientNet-B0 backbone, edge-optimized
- [crossmodal_fire_2024.py](models/sota_baselines/crossmodal_fire_2024.py): CrossModalFire 2024 – (placeholder, implementation WIP)

Tất cả các baseline và model trên đều được sử dụng trong pipeline benchmark SOTA hiện tại. Xem chi tiết kiến trúc và tham khảo trong từng file. Để so sánh SOTA, xem thêm [SOTA_COMPARISON.md](docs/SOTA_COMPARISON.md).

**Note:**
Các notebook huấn luyện SOTA và thư mục [models/sota_baselines/](models/sota_baselines/) là thành phần benchmark chính và không bị lưu trữ vào archive.

## 5) Repository Structure


```text
BushFire-Detection/
|-- [archive/](archive/)                  # Legacy notebooks kept for reference
|-- [config/](config/)                    # YAML configurations
|-- [data/](data/)                        # Data and processed outputs
|-- [docs/](docs/)                        # Technical documentation and guides
|-- [frames/](frames/)                    # Frame-level dataset structure
|-- [models/](models/)                    # Model definitions and checkpoints
|-- [notebooks/](notebooks/)              # EDA, training, and analysis notebooks (active)
|-- [scripts/](scripts/)                  # Preprocess, validation, inference, export scripts
|-- [tests/](tests/)                      # Unit tests
|-- [requirements.txt](requirements.txt)
|-- [setup.py](setup.py)
`-- [README.md](README.md)
```

## 6) Archive Policy


The [archive/](archive/) folder stores legacy/reference content.

**Current legacy notebooks in [archive/notebooks/](archive/notebooks/):**
- [FLAME_Hybrid_Training.ipynb](archive/notebooks/FLAME_Hybrid_Training.ipynb)
- [EfficientNetB0_4ch.ipynb](archive/notebooks/EfficientNetB0_4ch.ipynb)
- [Segmentation_Training.ipynb](archive/notebooks/Segmentation_Training.ipynb)

These notebooks are retained for reference and are not part of the primary submission workflow.

See details in [archive/README.md](archive/README.md)

## 7) Main Scripts


**Data scripts:**
- [preprocessing_pipeline.py](scripts/data/preprocessing_pipeline.py)
- [rgb_thermal_dataset.py](scripts/data/rgb_thermal_dataset.py)
- [boreal_subset_c_preprocessing.py](scripts/data/boreal_subset_c_preprocessing.py)
- [flame_dataset.py](scripts/data/flame_dataset.py)

**Validation scripts:**
- [validate_setup.py](scripts/validation/validate_setup.py)
- [quick_validate.py](scripts/validation/quick_validate.py)
- [check_overfitting_subset_c.py](scripts/validation/check_overfitting_subset_c.py)

**Inference / Export / Training Utility:**
- [inference_pipeline.py](scripts/inference_pipeline.py)
- [export_onnx.py](scripts/export_onnx.py)
- [resume_training.py](scripts/resume_training.py)

### Quick CLI usage examples


**Quick CLI usage examples:**

*Run single-image inference:*
```bash
python scripts/inference_pipeline.py --model models/trained/msbw-net_shared_backbone/best_msbw_model.pth --image path/to/image.jpg --output-json outputs/prediction.json
```

*Run folder inference:*
```bash
python scripts/inference_pipeline.py --model models/trained/msbw-net_shared_backbone/best_msbw_model.pth --image-dir path/to/images --output-json outputs/batch_predictions.json
```

*Export ONNX:*
```bash
python scripts/export_onnx.py --model models/trained/msbw-net_shared_backbone/best_msbw_model.pth --output models/trained/msbw-net_shared_backbone/msbw_net.onnx --input-shape 1 4 224 224 --quantize
```

*Resume training from checkpoint:*
```bash
python scripts/resume_training.py --checkpoint models/trained/msbw-net_shared_backbone/best_msbw_model.pth --epochs 10
```

## 8) Models and Checkpoints


**Model definitions:**
- [unet_segmentation.py](models/unet_segmentation.py)
- [mobilenet_hybrid.py](models/mobilenet_hybrid.py)
- [loss_functions.py](models/loss_functions.py)

**Training artifacts and checkpoints:**
- [models/trained/](models/trained/)

## 9) Tests


- [test_models.py](tests/test_models.py)
- [test_metrics.py](tests/test_metrics.py)

Run all tests:
```bash
pytest -q
```

## 10) Additional Documentation


- [EXECUTION_ORDER.md](docs/guides/EXECUTION_ORDER.md)
- [REPO_STRUCTURE.md](docs/guides/REPO_STRUCTURE.md)
- [MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)
- [EDGE_DEPLOYMENT.md](docs/EDGE_DEPLOYMENT.md)
- [SOTA_COMPARISON.md](docs/SOTA_COMPARISON.md)
- [MSBW-NET_BENCHMARK_PROTOCOL.md](docs/MSBW-NET_BENCHMARK_PROTOCOL.md)
- [COMPLETE_PIPELINE_ANALYSIS.md](docs/COMPLETE_PIPELINE_ANALYSIS.md)

## 11) Notes

- For SOTA comparison, use benchmark notebooks in [notebooks/03_analysis/](notebooks/03_analysis/) with models from [models/sota_baselines/](models/sota_baselines/).
- For benchmark reporting format and evaluation details, follow [MSBW-NET_BENCHMARK_PROTOCOL.md](docs/MSBW-NET_BENCHMARK_PROTOCOL.md).

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
You are free to use, modify, and distribute this code for academic and commercial purposes, provided that the original copyright
and license notice are retained.
