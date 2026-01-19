# Label Refinement for Object Detection with Noisy Labels

> **📄 Paper Status**: This repository contains the official implementation of our paper submitted to **IJCAI-ECAI 2026** (currently under review).
>
> **📌 Notes:**
> - Detailed experimental results and analysis will be updated after the review process.
> - We are continuously refactoring this codebase to contribute to the broader **Automated Label Refinement** research community.

---

## Motivation

<p align="center">
  <img width="400" alt="Motivation" src="https://github.com/user-attachments/assets/37b2b05b-14b9-4eda-afd5-7a716244c5d3" />
</p>
<p align="center"><b>Figure 1.</b> Common problems caused by noisy bounding box labels and improvements after refinement.</p>

Object detection models are highly sensitive to the quality of training labels. As shown in Figure 1, noisy bounding box annotations lead to critical issues:

- **Misdetection**: Incorrectly sized boxes cause the model to learn inaccurate object boundaries, resulting in false positives (e.g., detecting parts of buildings as vehicles).
- **Misclassification**: Loose or shifted boxes that include surrounding context confuse the classifier (e.g., an elephant labeled as rhinoceros).
- **Overlapping predictions**: Inconsistent box sizes during training lead to redundant, overlapping detections at inference time.

Training with refined labels significantly reduces these issues, producing cleaner and more accurate predictions.

---

## Proposed Method: ReBox

<p align="center">
  <img width="600" alt="ReBox" src="https://github.com/user-attachments/assets/322acc66-2940-442a-a5a3-741b5466bb56" />
</p>
<p align="center"><b>Figure 2.</b> Overview of the ReBox label refinement pipeline.</p>

We propose **ReBox**, a learning-based label refinement framework that corrects noisy bounding box annotations. The pipeline consists of two stages:

- **Stage A: Candidate Generation and Preprocessing**  
  Given a noisy anchor box, we generate a pool of candidate boxes through inverse noise modeling, isotropic scaling, and boundary perturbation. Each candidate (along with image context) is cropped and encoded via a CNN backbone.

- **Stage B: Candidate Scoring and Refinement**  
  A Transformer encoder processes all candidate features jointly, enabling cross-candidate comparison. A scoring head predicts quality scores for each candidate, and the highest-scoring box is selected as the refined annotation.

This approach effectively recovers accurate bounding boxes from various types of label noise, improving downstream object detection performance.

---

## Experimental Results
<p align="center">
  <img width="400" alt="Qualitative Results" src="https://github.com/user-attachments/assets/6bf908c7-0124-4448-92b7-53031b354d99" />
</p>
<p align="center"><b>Figure 3.</b> Qualitative comparison of label refinement methods: (a) Original ground truth, (b) Noisy labels, (c) ReBox (Ours), (d) SAM.</p>

The figure above shows qualitative comparisons across different scenarios. ReBox successfully recovers bounding boxes close to the original annotations, while SAM sometimes fails to capture the correct object boundaries, especially for objects with ambiguous edges (e.g., signatures).

---

## Overview

This repository provides a complete pipeline for **object detection label refinement** using ReBox and SAM (Segment Anything Model). The pipeline handles noisy bounding box labels and refines them to improve object detection performance.

The pipeline consists of 7 main components executed sequentially:

| Step | File | Description |
|------|------|-------------|
| 0 | `0.Data_setting_(ultralytics).py` | Download datasets using Ultralytics |
| 1 | `1.Data_check_and_noise_insection.py` | Inspect datasets and inject label noise |
| 2 | `2.object_detection.ipynb` | Train baseline object detection models |
| 3 | `3.1.Label_refinement_*_Final.ipynb` | Train ReBox label refinement model |
| 4 | `4.SAM_model_label_refine.ipynb` | SAM-based label refinement (comparison) |
| 5 | `5.refine_object_detection_*_Final.ipynb` | Train detection with refined labels |
| 6 | `6.visualization_code.ipynb` | Visualize and analyze results |

---

## Datasets

We evaluate our method on 9 diverse object detection datasets spanning various domains:

<p align="center">
  <img width="400" alt="Dataset Statistics" src="https://github.com/user-attachments/assets/969aa49e-9410-4dd7-9e2c-2dc2e5073c6b" />
</p>
<p align="center"><b>Table 1.</b> Dataset statistics used in our experiments.</p>

The datasets cover a wide range of applications including autonomous driving (PASCAL VOC, Kitti), household objects (Home-objects), construction sites, wildlife, medical imaging (Brain-tumor, BCCD, Medical-pills), and document analysis (Signature).

**Dataset Sources:**
- **BCCD**: Available at [Kaggle BCCD Dataset](https://www.kaggle.com/datasets/konstantinazov/bccd-dataset)
- **All other datasets**: Available through [Ultralytics Datasets](https://docs.ultralytics.com/datasets/)

---

## Requirements

### Core Dependencies

```bash
# PyTorch (CUDA recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ultralytics YOLO
pip install ultralytics

# Core packages
pip install numpy pandas matplotlib seaborn pillow tqdm opencv-python

# For ReBox model
pip install timm  # For DenseNet backbone

# For SAM refinement (Step 4)
pip install segment-anything
# Or clone: git clone https://github.com/facebookresearch/segment-anything.git
```

### Hardware Requirements
- GPU with at least 8GB VRAM (recommended: 16GB+)
- 50GB+ disk space for datasets and checkpoints

---

## Data Structure

### Initial Structure (After Step 0-1)

```
/datasets/
├── coco8/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/                    # Original (clean) labels
│   │   ├── train/
│   │   └── val/
│   ├── labels_uniform_scaling_0.6/          # Uniform scaling noise (factor=0.6)
│   ├── labels_uniform_scaling_0.7/          # Uniform scaling noise (factor=0.7)
│   ├── ...
│   ├── labels_boundary_jitter_3/             # Boundary jitter noise (pattern=3)
│   ├── labels_boundary_jitter_4/             # Boundary jitter noise (pattern=4)
│   └── ...
├── VOC/
│   ├── images/
│   │   ├── train2012/
│   │   └── val2012/
│   ├── labels/
│   └── labels_uniform_scaling_*/labels_boundary_jitter_*/
├── VisDrone/
└── ...
```

### After Refinement (Step 3-4)

```
/experiments_ablation(...)/
├── weights/                       # ReBox model checkpoints
│   ├── coco8/
│   │   └── baseline_both_31_*/
│   │       └── best.pt
│   └── VOC/
├── refines/                       # Refined labels output
│   ├── seed42/
│   │   ├── coco8/
│   │   │   └── <case_id>/
│   │   │       ├── labels_uniform_scaling_0.6/
│   │   │       │   ├── train/
│   │   │       │   └── val/
│   │   │       └── labels_boundary_jitter_3/
│   │   └── VOC/
│   └── seed123/
└── _orchestrator_summary/
    └── summary_*.csv
```

---

## Pipeline Execution

### Step 0: Dataset Download

```bash
python 0.Data_setting_(ultralytics).py --save-dir /path/to/datasets
```

**What it does:**
- Downloads object detection datasets via Ultralytics API
- Supports: COCO, VOC, VisDrone, xView, SKU-110K, etc.
- Automatically handles YAML naming variations

**Configuration:**
```python
# In build_target_candidates()
base = [
    "coco8.yaml",      # Small test dataset
    "voc.yaml",        # Pascal VOC
    "VisDrone.yaml",   # Drone imagery
    # ... add more as needed
]
```

---

### Step 1: Data Inspection & Noise Injection

```bash
python 1.Data_check_and_noise_insection.py
```

**What it does:**
1. Inspects all datasets under `/datasets`
2. Reports train/val image counts, class distributions
3. Generates noisy labels:
   - **Uniform scaling noise**: Randomly scales bbox width/height (factors: 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4)
   - **Boundary jitter noise**: Randomly perturbs bbox sides (patterns: 3, 4, 5, 6, 7)
4. Saves noise check visualizations

**Configuration:**
```python
load_dir = "/home/ISW/project/datasets"
NOISE_MODE = "both"        # "isotropic" | "borderwise" | "both"
NOISE_SEED = 42
OVERWRITE_NOISE = False
GENERATE_FOR_ALL_DATASETS = True
```

**Output:**
- `labels_uniform_scaling_{S}/` folders with scaled noisy labels
- `labels_boundary_jitter_{K}/` folders with side-perturbed labels
- `_noise_reports/noise_check/` visualization images

---

### Step 2: Baseline Object Detection Training

Open and run `2.object_detection.ipynb`

**What it does:**
1. Trains YOLOv8 on original labels (baseline)
2. Trains YOLOv8 on each noise case
3. Records mAP metrics for comparison

**Key Configuration:**
```python
TRAIN_USE_ORIGINAL = True
TRAIN_USE_UNIFORM_SCALING_NOISE = True
TRAIN_USE_BOUNDARY_JITTER_NOISE = True
CLASS_MODES = ["multiclass"]  # or ["multiclass", "object_only"]
TARGET_DATASETS = None  # None = all datasets
```

---

### Step 3: ReBox Label Refinement Training (Core)

Open and run `3.1.Label_refinement_(uniform_scaling_boundary_jitter_noise_start=noise)-(n)_Final.ipynb`

**What it does:**
1. **Cell 1**: Dataset discovery and statistics
2. **Cell 2**: ReBox model definition
   - DenseNet121 backbone for feature extraction
   - Transformer encoder for candidate ranking
   - Supports ListMLE, Monotone Hinge, MSE losses
3. **Cell 3**: Training orchestrator with experiment cases
4. **Cell 4**: Inference - refine noisy labels using trained model

**ReBox Architecture:**
```
Input: Noisy bbox + Image context
    ↓
[Candidate Generation]
    - Anchor (original noisy bbox)
    - Inverse candidates (analytical noise inversion)
    - Isotropic resizing candidates (17 scale factors)
    - Random border-wise perturbation candidates (10 perturbations)
    ↓
[Feature Extraction] DenseNet121
    ↓
[Transformer Encoder]
    ↓
[Ranking Score] → Select best candidate
    ↓
Output: Refined bbox
```

**Experiment Cases (CaseSpec):**
```python
CASE_SPECS_DEFAULT = [
    # Baseline: 31 candidates (anchor + inverse + scale + side)
    CaseSpec(
        case_name="baseline_both_31_...",
        cand_mode="both",
        max_candidates=60,
        num_border_perturb=10,
        include_inverse=True,
    ),
    # Ablation: 15 candidates (half)
    CaseSpec(case_name="exp1_both_15_...", max_candidates=15, ...),
    # Scale-only: 15 candidates
    CaseSpec(case_name="exp2_isotropic_only_15_...", cand_mode="isotropic_only", ...),
    # Side-only: 15 candidates
    CaseSpec(case_name="exp3_borderwise_only_15_...", cand_mode="borderwise_only", ...),
]
```

**Key Parameters:**
```python
n_data = 100              # Training samples per noise case
SEEDS = [42, 123, 456]    # Random seeds for reproducibility
IMG_SIZE = 224            # Crop size for candidates
EPOCHS = 1                # Training epochs
LOSS_TYPE = "listmle"     # "listmle" | "mono" | "mse"
```

---

### Step 4: SAM-based Label Refinement (Comparison)

Open and run `4.SAM_model_label_refine.ipynb`

**What it does:**
- Uses Segment Anything Model (SAM) for bbox refinement
- Box prompt → Mask → Refined bbox
- Provides comparison baseline for ReBox

**Configuration:**
```python
SAM_MODEL_TYPE = "vit_h"
SAM_CKPT_PATH = "/path/to/sam_vit_h_4b8939.pth"
TARGET_NOISE_DIRS = ["labels_uniform_scaling_*", "labels_boundary_jitter_*"]
```

---

### Step 5: Detection with Refined Labels

Choose the appropriate notebook:
- `5.refine_object_detection_Final.ipynb` - Direct refined label training
- `5.refine_object_detection_proposed(n)_Final.ipynb` - ReBox refined labels
- `5.refine_object_detection_sam_Final.ipynb` - SAM refined labels

**What it does:**
1. Loads refined labels from Step 3/4
2. Trains YOLOv8 with refined labels
3. Evaluates on original (clean) labels
4. Compares with baseline (noisy label training)

**Key Metrics:**
- mAP50, mAP50-95
- Delta improvement over noisy baseline
- Per-class precision/recall

---

### Step 6: Visualization & Analysis

Open and run `6.visualization_code.ipynb`

**What it does:**
1. Loads all experiment results
2. Visualizes:
   - Original vs Noisy vs Refined bbox comparisons
   - mAP improvement charts
   - Per-dataset performance breakdown
3. Generates publication-ready figures

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-repo/label-refinement.git
cd label-refinement

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
python 0.Data_setting_(ultralytics).py --save-dir ./datasets

# 4. Generate noisy labels
python 1.Data_check_and_noise_insection.py

# 5. Run notebooks in order (2 → 3 → 4 → 5 → 6)
jupyter notebook
```

---

## Project Module

The pipeline requires custom modules in `PROJECT_MODULE_DIR`:

```
/Project_Module/
├── ultra_det_loader.py    # Dataset loading utilities
├── noisy_insection.py     # Noise injection functions
└── ...
```

**Key Functions:**
```python
from ultra_det_loader import (
    inspect_det_datasets,
    build_dataset,
    build_dataloader,
)
from noisy_insection import (
    generate_noisy_labels,
    UNIFORM_SCALING_FACTORS,      # [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
    JITTER_PATTERNS,     # [3, 4, 5, 6, 7]
)
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{rebox2026ijcai,
  title={ReBox: Learning-based Label Refinement for Object Detection with Noisy Annotations},
  author={Your Name},
  booktitle={Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year={2026},
  note={Under Review}
}
```

> **Note**: The citation will be updated with the official proceedings information upon acceptance.

---

## License

This project is licensed under the MIT License.
