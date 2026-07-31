# Wafer Map Defect Pattern Classification

**Author:** Puvvada Sravan
**Domain:** Semiconductor Process Diagnostics + Applied Machine Learning

## Overview

A CNN-based image classification system that automatically identifies
defect patterns on semiconductor wafer maps — a core problem in fab
yield engineering. Built on 5+ years of hands-on experience in
semiconductor process diagnostics and optical inspection (Applied
Materials, supporting 5nm/3nm HVM ramp-ups and ML-based defect filtering
across TSMC, Samsung, and Hynix), this project applies deep learning to
a task directly aligned with real production inspection-to-classification
pipelines used at semicap companies.

## Dataset

**WM-811K Wafer Map Dataset** — 811,457 real wafer maps from
semiconductor fabs; 172,950 labeled across 9 classes (8 defect types +
"none"). [Kaggle source](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

| Property | Value |
|---|---|
| Total wafers | 811,457 |
| Labeled wafers used | 172,950 |
| Classes | Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none |
| Class imbalance ratio | ~989:1 (most common vs. rarest class) |
| Input format | Variable-size 2D die grids, resized to 32×32 |

## Approach

- **Model:** Custom 2-layer CNN (PyTorch) — see `src/model.py`
- **Preprocessing:** Nearest-neighbor resizing to 32×32 (preserves
  categorical pass/fail/blank values), outlier filtering on degenerate
  wafer map shapes
- **Class imbalance handling:** Three techniques implemented and
  benchmarked — loss reweighting (two variants) and data-level
  oversampling with rotation augmentation
- **Evaluation:** Precision, recall, F1 per class; macro and weighted
  averages; confusion matrices

## Results Summary

| Metric | Baseline | Weighted Loss (raw) | Weighted Loss (sqrt) | Oversampling + Augmentation |
|---|---|---|---|---|
| Overall accuracy | **96.0%** | 81.4% | 93.8% | 88.4% |
| Macro-avg F1 | **0.751** | 0.611 | 0.611 | 0.700 |
| Macro-avg recall | 0.76 | 0.75 | 0.68 | **0.827** |
| Scratch recall (rarest class) | 0.012 | 0.201 | 0.000 | **0.402** |
| "none" recall (dominant class) | 0.984 | 0.828 | 0.979 | 0.891 |

**Selected model: Oversampling + Rotation Augmentation** — achieves the
most balanced detection across all 9 classes, with every class reaching
meaningful recall (≥0.40, most ≥0.80). Prioritizes recall over raw
accuracy, reflecting the real-world cost asymmetry in fab defect triage,
where a missed defect is materially more costly than a false alarm
routed for engineer review.

### Per-Class Performance — Final Model (Oversampling + Augmentation)

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Center | 891 | 0.649 | 0.951 | 0.771 |
| Donut | 115 | 0.755 | 0.939 | 0.837 |
| Edge-Loc | 988 | 0.396 | 0.806 | 0.531 |
| Edge-Ring | 1,941 | 0.939 | 0.930 | 0.935 |
| Loc | 717 | 0.366 | 0.637 | 0.465 |
| Near-full | 22 | 0.875 | 0.955 | 0.913 |
| Random | 182 | 0.650 | 0.929 | 0.765 |
| Scratch | 244 | 0.085 | 0.402 | 0.141 |
| none | 29,489 | 0.991 | 0.891 | 0.938 |

Full classification reports and confusion matrices for all four models
are available in `reports/`.

## Project Structure

```
wafer-defect-classification/
├── data/                   # raw + processed data (not committed to git)
├── notebooks/
│   └── 01_eda.ipynb        # exploratory data analysis
├── src/
│   ├── data_loader.py      # load & preprocess WM-811K
│   ├── dataset.py          # PyTorch Dataset class (resize, normalize, augmentation)
│   ├── model.py             # CNN architecture
│   ├── train.py             # training loop (baseline / weighted / oversampled variants)
│   └── evaluate.py          # evaluation + confusion matrix generation
├── models/                 # saved model checkpoints (not committed)
├── reports/
│   ├── classification_report*.txt
│   └── figures/             # EDA plots, confusion matrices
├── requirements.txt
└── README.md
```

## Tech Stack

Python, PyTorch, pandas, NumPy, scikit-learn, OpenCV, Matplotlib/Seaborn

## Key Technical Decisions

- **Nearest-neighbor interpolation** (not bilinear/bicubic) for resizing
  — preserves categorical die values instead of blending them into
  meaningless intermediate values
- **Rotation-based augmentation** — chosen because wafer defect patterns
  are rotation-invariant (a Scratch pattern remains a Scratch pattern at
  any angle), unlike augmentations that could alter class meaning
- **Best-epoch checkpointing** — saves the model from whichever epoch
  achieved the highest validation accuracy, rather than the final epoch,
  since oversampled training shows epoch-to-epoch variance
- **Data-level rebalancing over loss-level reweighting** — two loss
  reweighting schemes were benchmarked and found to either
  over-correct (precision collapse) or under-correct (near-zero recall
  on the rarest class); oversampling with augmentation produced the most
  balanced result across all classes

## How to Run

```bash
pip install -r requirements.txt
cd src
python train.py       # trains model, saves best checkpoint to models/
python evaluate.py    # generates classification report + confusion matrix
```
