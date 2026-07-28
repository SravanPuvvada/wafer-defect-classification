# Wafer Map Defect Pattern Classification

**Author:** Puvvada Sravan
**Domain:** Semiconductor Process Diagnostics + Applied Machine Learning

## Project Motivation

With 5+ years in semiconductor process diagnostics and optical inspection at
Applied Materials — supporting 5nm/3nm HVM ramp-ups and ML-based defect
filtering across TSMC, Samsung, and Hynix — this project applies deep
learning (CNNs) to automatically classify wafer map defect patterns, a
core problem in fab yield engineering.

The goal is to reproduce, in a self-contained portfolio project, the kind
of defect-pattern-recognition system used in real fab yield analysis:
taking raw wafer bin maps and automatically classifying the defect
signature (edge-ring, center, scratch, donut, etc.) that a process
engineer would otherwise triage manually.

## Dataset

**WM-811K Wafer Map dataset** — ~811,000 wafer maps collected from real
semiconductor fabs, with ~172,000 labeled with one of 8 known defect
pattern classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch,
Near-full) plus a "none" (no pattern) class. Publicly available on Kaggle.

Source: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

## Problem Framing

- **Type:** Multi-class image classification
- **Input:** Wafer bin map (2D array of die pass/fail/pattern values,
  variable size per wafer)
- **Output:** One of 9 classes (8 defect types + "none")
- **Key challenge:** Severe class imbalance (most wafers are "none" or a
  few dominant classes; some defect types have very few examples) —
  this mirrors real fab data and is worth discussing explicitly in the
  writeup, since handling it well is itself a signal of applied maturity.

## Project Structure

```
wafer-defect-classification/
├── data/                   # raw + processed data (not committed to git)
├── notebooks/
│   └── 01_eda.ipynb        # exploratory data analysis
├── src/
│   ├── data_loader.py      # load & preprocess WM-811K
│   ├── dataset.py          # PyTorch Dataset class
│   ├── model.py            # CNN architecture
│   ├── train.py            # training loop
│   └── evaluate.py         # evaluation + confusion matrix
├── models/                 # saved model checkpoints (not committed)
├── reports/
│   └── figures/            # EDA plots, confusion matrices, etc.
├── requirements.txt
└── README.md
```

## Roadmap / Milestones

- [X] **M1 (this weekend):** Environment set up, dataset downloaded, EDA
      complete — class distribution, sample wafer visualizations, image
      size/shape analysis
- [ ] **M2:** Baseline CNN trained, confusion matrix + accuracy/F1 logged
- [ ] **M3:** Class imbalance handled (weighted loss / augmentation /
      resampling), model improved, results compared to baseline
- [ ] **M4:** Clean writeup connecting results to real fab defect
      triage workflows; repo polished for portfolio

## Why This Project Matters (for recruiters reading this)

This project deliberately mirrors production inspection-to-classification
pipelines used at semicap companies (Applied Materials, KLA, Lam Research)
for defect review and yield diagnostics — connecting hands-on fab
process/inspection experience with an end-to-end applied ML pipeline
(data handling → modeling → evaluation → discussion of real-world
constraints like class imbalance).

## Results

_(To be filled in as milestones are completed.)_
