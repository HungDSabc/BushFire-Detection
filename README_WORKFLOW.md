# BushFire Detection - Complete Workflow Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Project Structure](#project-structure)
4. [Complete Workflow Guide](#complete-workflow-guide)
5. [Detailed Step Instructions](#detailed-step-instructions)
6. [Output Files Explanation](#output-files-explanation)
7. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements a **deep learning pipeline for fire detection** using:
- **Classification Models**: Detect whether an image contains fire or not
  - EfficientNet-B0 (high accuracy, ~1 hour training)
  - MobileNetV3-Small (lightweight, faster)
  - Xception (balance of performance and speed)
- **Segmentation Models**: Localize fire regions in images
  - U-Net architecture for pixel-level fire localization

The pipeline is designed to work on bush fire detection datasets with the following workflow:
1. **Data Preprocessing** - Image augmentation and normalization
2. **Exploratory Data Analysis (EDA)** - Understand data distribution
3. **Model Training** - Train classification and segmentation models
4. **Inference & Testing** - Use trained models for predictions

---

## Prerequisites & Installation

### System Requirements
- **Python**: 3.8 or higher (tested with Python 3.11)
- **CUDA** (Optional): For GPU acceleration (NVIDIA GPU recommended)
- **RAM**: Minimum 8GB (16GB+ recommended for efficient training)
- **Storage**: ~10GB free space for datasets and models

### Installation Steps

#### Step 1: Clone/Navigate to Project Directory
```bash
cd "Computer Vision class/BushFire-Detection"
```

#### Step 2: Create Virtual Environment
```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate it:
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies Installed:**
- `torch` (2.0+) - Deep learning framework
- `torchvision` (0.15+) - Computer vision utilities
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `matplotlib` & `seaborn` - Visualization
- `Pillow` - Image processing
- `tqdm` - Progress bars
- `timm` - Vision model library (for Xception)
- `jupyter` - For running notebooks

### Quick Install Command
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn Pillow tqdm timm jupyter
```

---

## Project Structure

```
BushFire-Detection/
│
├── README_WORKFLOW.md                      # This file - complete workflow guide
├── README.md                               # Original brief description
├── preprocessing_pipeline.py               # Preprocessing script
│
├── EDA_FLAME_FireDetection.ipynb          # Exploratory Data Analysis notebook
├── FLAME_Pipeline.ipynb                   # Main training pipeline notebook
│
├── frames/                                 # RAW DATA FOLDER (provided separately)
│   ├── Segmentation/
│   │   ├── Data/
│   │   │   ├── Images/                     # Raw segmentation images
│   │   │   └── masks/Masks/                # Ground truth segmentation masks
│   │   ├── Test/Test/
│   │   │   ├── Fire/                       # Test images with fire
│   │   │   └── No_Fire/                    # Test images without fire
│   │   └── Training/Training/
│   │       ├── Fire/                       # Training images with fire
│   │       └── No_Fire/                    # Training images without fire
│   └── Test/
│       ├── Fire/                           # Test classification images with fire
│       └── No_Fire/                        # Test classification images without fire
│
├── Output/                                 # GENERATED OUTPUTS (created during training)
│   ├── Classification/
│   │   ├── best_efficientnet.pth          # Best EfficientNet model weights
│   │   ├── best_mobilenetv3_small.pth     # Best MobileNet model weights
│   │   └── best_xception.pth              # Best Xception model weights
│   ├── Segmentation/
│   │   └── best_unet.pth                  # Best U-Net model weights
│   ├── preprocessed_by_class/
│   │   ├── Fire/                          # Preprocessed fire images
│   │   └── No_Fire/                        # Preprocessed non-fire images
│   ├── preprocessed_all/
│   │   └── transform_pipeline.pt           # Transforms snapshot
│   └── Segmentation_Augmented/
│       ├── Images/                        # Augmented segmentation images
│       └── Masks/                         # Augmented segmentation masks
│
└── .venv/                                  # Virtual environment (auto-created)
```

### Understanding Data Organization

**Raw Data Location**: `frames/` folder contains all original images
- **Classification Data**: Images labeled as Fire or No_Fire
- **Segmentation Data**: Images + pixel-level fire masks
- **Test Data**: Separate test set for final evaluation

**Preprocessed Data Location**: `Output/preprocessed_*` folders
- Images are resized to 224x224 pixels
- Normalized using ImageNet mean/std values
- Augmented with random rotations, crops, and color shifts (training only)

---

## Complete Workflow Guide

### Quick Reference: Running the Full Pipeline

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Optional: Preprocess data (if not already done)
python preprocessing_pipeline.py --input frames/Segmentation/Training/Training/Fire --output Output/preprocessed_by_class/Fire --train

# 3. Open Jupyter and run notebooks in order
jupyter notebook

# Then in browser, open:
# - EDA_FLAME_FireDetection.ipynb (understand data)
# - FLAME_Pipeline.ipynb (train models)
```

### High-Level Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: DATA PREPARATION                                    │
│ - Check raw data in frames/ folder                          │
│ - Verify both Fire and No_Fire classes exist                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: PREPROCESSING (Optional - may be pre-done)          │
│ - Run preprocessing_pipeline.py                             │
│ - Creates normalized, augmented images in Output/           │
│ - Applies 224x224 resize + ImageNet normalization           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: EXPLORATORY DATA ANALYSIS (EDA)                     │
│ - Open EDA_FLAME_FireDetection.ipynb                        │
│ - Visualize class distribution                              │
│ - Check image statistics                                    │
│ - Understand dataset imbalance                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: CLASSIFICATION MODEL TRAINING (FLAME_Pipeline.ipynb)│
│ - Cell 1-4: Setup & Data Loading                            │
│ - Cell 5: Segmentation Setup                                │
│ - Cell 6: Train EfficientNet-B0 (~1 hour)                   │
│ - Cell 6B: Load Shared Helpers                              │
│ - Cell 6C: Train MobileNetV3-Small (~15 minutes, optional)  │
│ - Cell 6D: Load Xception Helper                             │
│ - Cell 6E: Train Xception (~30 minutes, optional)           │
│ - Cell 7+: Train Segmentation Models                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: TEST & GENERATE RESULTS                             │
│ - Evaluate models on test set                               │
│ - Generate accuracy/confusion matrices                      │
│ - Create performance comparison plots                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Step Instructions

### STEP 1: Data Preparation

#### What This Step Does
Verifies your raw data is properly organized before training.

#### Low-Level Details
- Raw images are in `frames/` folder
- Two classes: **Fire** (positive class) and **No_Fire** (negative class)
- Training and test sets must be separated
- Typical dataset: ~4000-10000 images per class

#### Instructions

1. **Check if raw data exists:**
   ```bash
   # List Fire images
   dir frames\Segmentation\Training\Training\Fire
   
   # List No_Fire images
   dir frames\Segmentation\Training\Training\No_Fire
   ```

2. **Verify folder structure:**
   - ✅ `frames/Segmentation/Training/Training/Fire/` - should have .jpg or .png files
   - ✅ `frames/Segmentation/Training/Training/No_Fire/` - should have .jpg or .png files
   - ✅ `frames/Test/Fire/` - should have test Fire images
   - ✅ `frames/Test/No_Fire/` - should have test No_Fire images

3. **Count images (optional, verify data size):**
   ```bash
   # PowerShell - Show image count
   (Get-ChildItem -Path "frames\Segmentation\Training\Training\Fire" -Filter *.jpg -Recurse).Count
   (Get-ChildItem -Path "frames\Segmentation\Training\Training\No_Fire" -Filter *.jpg -Recurse).Count
   ```

---

### STEP 2: Preprocessing

#### What This Step Does
1. Loads original images (variable sizes and formats)
2. Resizes all images to 224×224 pixels (standard for deep learning)
3. Applies data augmentation (training set only):
   - Random crops and horizontal/vertical flips
   - Color brightness/contrast/saturation adjustments
4. Normalizes pixel values using ImageNet statistics
5. Saves preprocessed images to `Output/preprocessed_by_class/`

#### Why Preprocessing Matters
- **Standardization**: All images same size (CNN requirement)
- **Augmentation**: Increases training data diversity, improves model generalization
- **Normalization**: Scales pixel values [0,255] → [-2,2], stabilizes training
- **Efficiency**: Pre-processed images load faster during training

#### Low-Level Processing Details

**Transform Pipeline for Training Images:**
```
Original Image (any size, 0-255)
    ↓
1. Resize to 256×256 pixels (preserves aspect ratio)
2. RandomResizedCrop (224×224, scale 0.8-1.0)
   - Creates random crops of varying zoom levels
3. RandomHorizontalFlip (50% probability)
   - Horizontally mirror image
4. RandomVerticalFlip (20% probability)
   - Vertically mirror image
5. ColorJitter (random color adjustments)
   - Brightness: ±30%, Contrast: ±30%
   - Saturation: ±20%, Hue: ±5%
6. Convert to Tensor (numpy array → torch tensor)
7. Normalize with ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
    ↓
Final Output: 224×224 tensor, normalized [-2, 2] range
```

**Transform Pipeline for Validation Images:**
```
Original Image
    ↓
1. Resize to 256×256 pixels
2. CenterCrop (224×224)
   - Takes center crop (deterministic, no random)
3. Convert to Tensor
4. Normalize with ImageNet statistics
    ↓
Final Output: 224×224 tensor, normalized
```

#### Instructions to Run Preprocessing

**Option 1: Preprocess via Python Script**

```bash
# Activate virtual environment first
.venv\Scripts\activate

# Preprocess Fire class (training with augmentation)
python preprocessing_pipeline.py ^
  --input "frames\Segmentation\Training\Training\Fire" ^
  --output "Output\preprocessed_by_class\Fire" ^
  --train

# Preprocess No_Fire class (training with augmentation)
python preprocessing_pipeline.py ^
  --input "frames\Segmentation\Training\Training\No_Fire" ^
  --output "Output\preprocessed_by_class\No_Fire" ^
  --train

# Optional: Preprocess test set (validation - no augmentation)
python preprocessing_pipeline.py ^
  --input "frames\Test\Fire" ^
  --output "Output\preprocessed_test\Fire" ^
  --val
```

**Expected Output:**
```
Processed 250/1000 from Fire...
Processed 500/1000 from Fire...
...
Saved 1000 preprocessed samples to: C:\...\Output\preprocessed_by_class\Fire
Skipped 0 unreadable images in: C:\...\frames\Segmentation\Training\Training\Fire
```

**Option 2: Preprocess via Jupyter Notebook**

If preprocessing cells already exist in FLAME_Pipeline.ipynb:
1. Open `FLAME_Pipeline.ipynb`
2. Find and run preprocessing cells (usually Cell 3-4)
3. Wait for completion message

#### Verification
Check that preprocessed images were created:
```bash
dir Output\preprocessed_by_class\Fire
# Should see files like: image_preproc_0001.png, image_preproc_0002.png, etc.
```

---

### STEP 3: Exploratory Data Analysis (EDA)

#### What This Step Does
- Visualizes the distribution of fire vs. non-fire images
- Plots sample images from each class
- Analyzes pixel value statistics
- Identifies class imbalance
- Shows image size distributions

#### Why EDA Matters
- **Understands imbalance**: Fire vs. No_Fire class ratio
- **Detects issues**: Broken images, missing data
- **Guides preprocessing**: Informs augmentation strategies
- **Sets baselines**: Random guess accuracy ≈ 1/(number of classes)

#### Instructions

1. **Open Jupyter Notebook:**
   ```bash
   # Ensure virtual environment is activated
   .venv\Scripts\activate
   
   # Start Jupyter
   jupyter notebook
   ```

2. **In browser (usually http://localhost:8888):**
   - Click: `EDA_FLAME_FireDetection.ipynb`

3. **Run all cells in order:**
   - **Cell 1:** Load libraries (numpy, torch, PIL, matplotlib)
   - **Cell 2:** Set configuration (image size, paths, device)
   - **Cell 3:** Load dataset from `Output\preprocessed_by_class\`
   - **Cell 4:** Print class distribution
     - Output example: `Fire: 4000 images (50%), No_Fire: 4000 images (50%)`
   - **Cell 5:** Visualize sample images
     - Shows grid of Fire and No_Fire images side-by-side
   - **Cell 6:** Plot class imbalance bar chart
   - **Cell 7:** Analyze pixel statistics per class
   - **Cell 8+:** Additional visualizations

4. **Interpret Results:**
   - **Class Balance**: Ideally 50/50, but often imbalanced (e.g., 80% Fire, 20% No_Fire)
   - **Sample Images**: Visually check for quality, artifacts, or labeling errors
   - **Pixel Statistics**: Fire images typically have higher red channel intensity

#### Expected Output
```
Class Distribution:
Fire: 4200 images (51.2%)
No_Fire: 4000 images (48.8%)

Mean pixel values:
Fire class: R=125.3, G=98.4, B=76.2
No_Fire class: R=105.2, G=110.3, B=115.1
```

---

### STEP 4: Classification Model Training

#### What This Step Does
Trains deep learning models to classify images as Fire or No_Fire using **transfer learning**:
1. Loads pre-trained model from ImageNet (already learned to recognize objects)
2. Replaces final layer to output 2 classes (Fire/No_Fire)
3. Freezes early layers, trains only final classifier (Phase 1)
4. Optionally unfreezes and fine-tunes entire network (Phase 2)

#### Why Transfer Learning
- **Faster training**: Pre-trained features don't need learning from scratch
- **Better accuracy**: Leverages millions of hours of ImageNet pre-training
- **Fewer data requirements**: Works with smaller datasets

#### Available Models

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **EfficientNet-B0** | Slow (~1hr) | Highest ⭐ | 200MB | Production, high accuracy |
| **MobileNetV3-Small** | Fast (~15min) | Good | 50MB | Mobile/edge devices |
| **Xception** | Medium (~30min) | High ⭐⭐ | 120MB | Balance of speed & accuracy |

#### Low-Level Training Process

**Phase 1: Head-Only Training (10 epochs)**
```
For each epoch:
  ├─ For each batch of 32 images:
  │  ├─ Forward pass: Image → Frozen backbone → NEW classifier head
  │  ├─ Compute loss: Cross-entropy loss with class weights
  │  │  (adjusts for Fire/No_Fire imbalance)
  │  ├─ Backward pass: Update ONLY head weights
  │  └─ Optimizer: Adam (lr=0.001)
  ├─ Validation: Evaluate on val set, track loss & accuracy
  └─ Scheduler: If loss doesn't improve for 3 epochs, reduce LR by 50%

Output: Checkpoint with best validation accuracy
```

**Phase 2: Fine-Tuning (optional, remaining epochs)**
```
For each epoch:
  ├─ Unfreeze top layers of backbone
  ├─ For each batch:
  │  ├─ Forward pass: Image → Partially-frozen backbone → updated head
  │  ├─ Compute loss: Cross-entropy with class weights
  │  ├─ Backward pass: Update both head AND unfrozen backbone
  │  └─ Optimizer: Adam (lr=0.00001) - much lower to avoid catastrophic forgetting
  ├─ Validation: Evaluate on val set
  └─ Scheduler: ReduceLROnPlateau

Output: Final model checkpoint
```

**Loss Function: CrossEntropyLoss with Class Weights**
```
When dataset is imbalanced (e.g., 80% Fire, 20% No_Fire):
- Fire class weight: 1/0.8 = 0.786
- No_Fire class weight: 1/0.2 = 1.373
- Incorrect No_Fire predictions penalized 1.373x more
- Prevents model from just predicting "Fire" for everything
```

#### Instructions for Training

1. **Open FLAME_Pipeline.ipynb:**
   ```bash
   # Ensure .venv is activated
   .venv\Scripts\activate
   
   # Start Jupyter
   jupyter notebook
   
   # Click: FLAME_Pipeline.ipynb
   ```

2. **Run Setup Cells (1-4):**
   - **Cell 1:** Import libraries, check PyTorch version
     - Output: `PyTorch 2.10.0, Device: cpu or cuda`
   - **Cell 2:** Configure mode and device
     - Output: `Mode='Train', Device=cpu`
   - **Cell 3:** Load datasets from `Output/preprocessed_by_class/`
     - Output: `Train: 31500 images, Val: 7875 images`
   - **Cell 4:** Load classification dataloaders
     - Output: `Class weights: [0.786, 1.373]`

3. **Train EfficientNet-B0 (Cell 6) - Main Model:**
   - **Click:** Cell 6 (Cell Id: #VSC-19f7457a)
   - **Click:** Run button (▶️) or press Shift+Enter
   - **Wait:** ~1 hour for training to complete
   - **Monitor:** Watch progress bar show Epoch 1/1, batch progress
   - **Output:**
     ```
     Epoch 1/1:
     Training: Loss=0.450, Acc=85.2%
     Validation: Loss=0.382, Acc=87.5%
     Checkpoint saved: Output\Classification\best_efficientnet.pth
     ```
   - **Result:** Model weights saved, training curves plot generated

4. **Optional: Train MobileNetV3-Small (Cell 6C) - Lightweight:**
   - **Prerequisites:** Must run Cell 6B (Shared Helpers) first
   - **Click:** Cell 6B first
     - Output: `Shared training helpers loaded...`
   - **Click:** Cell 6C
     - Wait: ~15 minutes
     - Output: Model checkpoint + comparison plot vs EfficientNet
   - **Use Case:** Mobile apps, edge devices, faster inference

5. **Optional: Train Xception (Cell 6E) - Balanced:**
   - **Prerequisites:** Run Cell 6D (Xception Helper) first
   - **Click:** Cell 6D
     - Output: `Xception helper loaded...`
   - **Click:** Cell 6E
     - Wait: ~30 minutes
     - Output: Model checkpoint + all-models comparison plot (3 models)
   - **Use Case:** Best balance of accuracy and speed

#### Choosing Which Model to Train

- **Want best accuracy?** → Train only **EfficientNet** (Cell 6) - ~1 hour
- **Want multiple options?** → Train all three (Cell 6 → 6B+6C → 6D+6E) - ~2 hours
- **Want fast inference?** → Use **MobileNet** (Cell 6C) - ~15 min training
- **Want balance?** → Use **Xception** (Cell 6E) - ~30 min training

#### Understanding Checkpoints
- Models automatically save "best" weights based on validation accuracy
- Saved in: `Output/Classification/best_{model_name}.pth`
- These are binary files containing:
  - Model architecture information
  - All layer weights (millions of numbers)
  - Training metadata

---

### STEP 5: Segmentation Model Training

#### What This Step Does
Trains U-Net model to create pixel-level fire masks (segmentation), not just classify image as a whole.

#### Output Interpretation
- **Classification**: "Fire or No_Fire?" (binary answer per image)
- **Segmentation**: "Which pixels are fire?" (binary mask per pixel)

#### Instructions

(Located in FLAME_Pipeline.ipynb, Cells 7+)

1. **Prerequisites:**
   - Segmentation training data must be in: `frames/Segmentation/Training/Training/`
   - Images folder: original images
   - Masks folder: ground truth fire masks

2. **Run Segmentation Setup (Cell 7-8):**
   - Loads segmentation datasets
   - Creates dataloaders for training

3. **Run Segmentation Training (Cell 9+):**
   - Similar to classification but outputs pixel masks
   - Saves best checkpoint: `Output/Segmentation/best_unet.pth`

---

### STEP 6: Testing & Inference

#### What This Step Does
Uses trained models to make predictions on new, unseen images.

#### Low-Level Inference Process
```
Input: New image (any size)
   ↓
1. Load trained model weights from checkpoint
2. Preprocess image: resize, normalize (same as training)
3. Forward pass: Image → Model → Probabilities
   - Output: [P(Fire), P(No_Fire)]
   - Example: [0.85, 0.15] = 85% confidence it's Fire
4. Apply threshold: If P(Fire) > 0.5, predict "Fire"
   ↓
Output: Prediction ("Fire" or "No_Fire") + confidence score
```

#### Instructions to Test Model

1. **In FLAME_Pipeline.ipynb, find Test/Inference cells:**
   - Look for cells titled "Evaluate on Test Set"
   - Or "Generate Predictions"

2. **Run inference cells:**
   - Loads best checkpoint automatically
   - Tests on `frames/Test/Fire/` and `frames/Test/No_Fire/`
   - Generates:
     - Accuracy percentage
     - Confusion matrix (True Positives, False Negatives, etc.)
     - ROC curve plot

3. **Interpret Results:**
   ```
   Test Accuracy: 92.3%
   
   Confusion Matrix:
                 Predicted
                 Fire  No_Fire
   Actual Fire    185    15      (93.7% correctly identified)
          No_Fire  8     192     (96.0% correctly identified)
   
   - True Positive Rate: 93.7% (Fire images correctly detected)
   - False Positive Rate: 4% (No_Fire incorrectly detected as Fire)
   ```

---

## Output Files Explanation

### Classification Outputs

| File | Location | Contents | Purpose |
|------|----------|----------|---------|
| `best_efficientnet.pth` | `Output/Classification/` | Model weights | EfficientNet predictions |
| `best_mobilenetv3_small.pth` | `Output/Classification/` | Model weights | MobileNet predictions |
| `best_xception.pth` | `Output/Classification/` | Model weights | Xception predictions |
| `clf_training_curves_*.png` | Saved during training | Loss & accuracy plots | Monitor training progress |
| `clf_val_acc_comparison_*.png` | Saved during training | Accuracy bar charts | Compare model performance |

### Segmentation Outputs

| File | Location | Contents | Purpose |
|------|----------|----------|---------|
| `best_unet.pth` | `Output/Segmentation/` | Model weights | Pixel-level fire masks |
| `seg_training_curves_*.png` | Saved during training | Loss plots | Monitor training progress |

### Preprocessed Data Outputs

| Folder | Purpose | Contents |
|--------|---------|----------|
| `Output/preprocessed_by_class/Fire/` | Training data | 224×224 augmented Fire images |
| `Output/preprocessed_by_class/No_Fire/` | Training data | 224×224 augmented No_Fire images |
| `Output/preprocessed_all/` | Combined | All preprocessed images together |

### Intermediate Data

| File | Location | Purpose |
|------|----------|---------|
| `transform_pipeline.pt` | `Output/preprocessed_all/` | Saved transforms snapshot |

---

## Troubleshooting

### Issue 1: ModuleNotFoundError (Missing Package)
```
Error: ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Activate environment
.venv\Scripts\activate

# Reinstall packages
pip install torch torchvision

# For notebooks, also run:
pip install jupyter
```

### Issue 2: Permission Denied Error on Windows
```
Error: Permission denied: 'C:\path\to\Output\...'
```

**Solution:**
- Close all file explorers viewing the `Output/` folder
- Try again from Jupyter cell or terminal

### Issue 3: GPU Memory Error
```
Error: CUDA out of memory...
```

**Solution:**
- Reduce batch size in configuration (default 32, try 16)
- Use CPU instead: `DEVICE = 'cpu'` in Cell 2
- Restart Jupyter kernel and clear memory: Kernel → Restart

### Issue 4: No Data Found Error
```
Error: Dataset not found at Output/preprocessed_by_class/
```

**Solutions:**
1. Run preprocessing step (Step 2) first
2. Verify raw data exists in `frames/` folder
3. Check folder path spelling (Windows paths are case-insensitive but must match structure)

### Issue 5: Training Very Slow or Freezes
```
Epoch 1/1 stuck, no progress for 1+ minutes
```

**Causes & Solutions:**
- **CPU training**: Normal (slow), be patient or use GPU
- **Notebook memory leak**: Restart kernel (Kernel → Restart)
- **Disk I/O issue**: Move `Output/` to faster disk (SSD preferred)

### Issue 6: Model Loading Error
```
Error: The model weights are not compatible
```

**Solution:**
- Delete old checkpoint: `Output/Classification/best_*.pth`
- Retrain model from scratch

### Issue 7: Port Already in Use (When Starting Jupyter)
```
Error: Address already in use
```

**Solution:**
```bash
# Specify different port
jupyter notebook --port 8889
```

---

## Quick Reference Commands

### Virtual Environment
```bash
# Create
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Deactivate
deactivate

# Install dependencies
pip install -r requirements.txt
```

### Jupyter
```bash
# Start
jupyter notebook

# Stop: Press Ctrl+C in terminal

# Access: http://localhost:8888
```

### Preprocessing
```bash
# Train augmentation
python preprocessing_pipeline.py --input "frames/Segmentation/Training/Training/Fire" --output "Output/preprocessed_by_class/Fire" --train

# Validation (no augmentation)
python preprocessing_pipeline.py --input "frames/Segmentation/Training/Training/Fire" --output "Output/preprocessed_by_class/Fire" --val
```

### File System (Windows PowerShell)
```bash
# Check folder structure
Get-ChildItem -Recurse -Directory

# Count files
(Get-ChildItem -Recurse -File -Filter "*.jpg").Count

# Delete preprocessed data (to retrain)
Remove-Item -Recurse -Force "Output/preprocessed_by_class"

# Delete models (to retrain)
Remove-Item "Output/Classification/*.pth"
```

---

## Performance Expectations

### Training Time
- **EfficientNet-B0**: 45-60 minutes (1 epoch)
- **MobileNetV3-Small**: 10-15 minutes (1 epoch)
- **Xception**: 20-30 minutes (1 epoch)
- **U-Net Segmentation**: 2-4 hours (5-10 epochs)

### Accuracy Targets
- **Classification**: 85-95% on test set (depends on data quality)
- **Segmentation**: 80-92% pixel accuracy (harder than classification)

### Hardware Impact
- **GPU (NVIDIA)**: 2-5x faster training
- **CPU Only**: Slower but fully functional
- **RAM**: 8GB minimum, 16GB recommended

---

## Advanced Usage

### Using Trained Models for Predictions
```python
import torch
from pathlib import Path

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('Output/Classification/best_efficientnet.pth')
model.to(device)
model.eval()

# Load and preprocess image
from PIL import Image
img = Image.open('new_image.jpg').convert('RGB')
# ... apply same preprocessing as training ...

# Predict
with torch.no_grad():
    output = model(img.unsqueeze(0))  # Add batch dimension
    prob = torch.softmax(output, dim=1)[0]  # Get probabilities
    
print(f"Fire confidence: {prob[0]:.2%}")
print(f"No_Fire confidence: {prob[1]:.2%}")
```

### Modifying Training Hyperparameters
In FLAME_Pipeline.ipynb Cell 2 (Configuration), edit:
```python
CFG = {
    'clf_epochs': 1,           # Number of training epochs
    'batch_size': 32,          # Images per batch (reduce if OOM)
    'clf_lr': 0.001,          # Learning rate (smaller = slower, more stable)
    'img_size': 224,          # CNN input size
    'seed': 42,               # Random seed for reproducibility
}
```

---

## Getting Help

1. **Check logs**: Look for error messages with file paths and line numbers
2. **Search issues**: Google the error message
3. **Documentation**: Check official PyTorch/torchvision docs
4. **Rerun from scratch**: Delete `Output/`, restart kernel, run all cells again

---

## Dataset Citation

If you use this pipeline, cite the FLAME dataset:
- [FLAME: Free Large All-Weather Monitoring of Eruptions (or similar dataset)]

---

## Summary

**To run the complete fire detection pipeline:**

1. ✅ Setup: Create `.venv` and install packages
2. ✅ Prepare: Verify data in `frames/` folder
3. ✅ Preprocess: Run `preprocessing_pipeline.py` or notebook cells
4. ✅ Analyze: Run `EDA_FLAME_FireDetection.ipynb`
5. ✅ Train: Run `FLAME_Pipeline.ipynb` classification cells
6. ✅ Test: Evaluate on test set and generate results
7. ✅ Deploy: Use trained `.pth` checkpoints for predictions

**Typical workflow time:** 2-4 hours total (1hr training, rest is data processing/analysis)

---

**Last Updated**: March 2026 | For issues, check troubleshooting section above.
