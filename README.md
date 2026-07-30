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

- [x] **M1:** Environment set up, dataset downloaded, EDA
      complete — class distribution, sample wafer visualizations, image
      size/shape analysis
- [x] **M2:** Baseline CNN trained, confusion matrix + accuracy/F1 logged
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

### Exploratory Data Analysis
- **172,950** labeled wafer maps used (out of 811,457 total; the rest are
  unlabeled and excluded from supervised training)
- **Severe class imbalance**: most common class ("none") outnumbers the
  rarest class by roughly **989:1**
- Wafer map dimensions vary substantially across the dataset (from tiny,
  likely-corrupted outliers like a (15,3) grid up to (153,187)); most
  common shape is around (25,27). All maps are resized to a fixed
  **32×32** target for CNN input, using nearest-neighbor interpolation to
  preserve the categorical (pass/fail/blank) nature of the data
- A small number of degenerate wafer maps (extreme outlier shapes,
  likely data artifacts rather than genuine signal) were identified and
  excluded via a minimum-area filter before training

### Baseline CNN (no class-imbalance handling)

A small custom 2-layer CNN (see `src/model.py`) was trained for 10
epochs on the full labeled dataset (80/20 train/validation split).

| Metric | Score |
|---|---|
| Overall accuracy | 96.0% |
| Weighted-avg F1 | 0.958 |
| **Macro-avg F1** | **0.751** |

The gap between weighted-avg and macro-avg F1 tells the real story:
overall accuracy is dominated by the "none" class (85% of validation
data, F1 = 0.984), which masks poor performance on rarer defect types.

**Per-class breakdown (selected):**

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| none | 29,489 | 0.984 | 0.984 | 0.984 |
| Edge-Ring | 1,941 | 0.965 | 0.955 | 0.960 |
| Center | 891 | 0.875 | 0.909 | 0.892 |
| Edge-Loc | 988 | 0.697 | 0.830 | 0.758 |
| Loc | 717 | 0.594 | 0.640 | 0.616 |
| **Scratch** | **244** | 0.600 | **0.012** | **0.024** |

**Key finding:** the model essentially fails to detect the rarest class
(Scratch, 244 samples) — recall of just 1.2% means it correctly
identifies almost none of the actual Scratch-type defects, despite
strong overall accuracy. This is a textbook class-imbalance failure
mode, and directly mirrors a real risk in fab yield analysis: naive
accuracy metrics can hide poor detection of rare-but-critical defect
signatures.

### Milestone 3, Attempt 1: Weighted Loss (raw inverse frequency)

To address the Scratch-detection failure above, `CrossEntropyLoss` was
given per-class weights inversely proportional to class frequency —
rare classes penalized more heavily when misclassified.

| Metric | Baseline | Weighted (raw inverse freq.) |
|---|---|---|
| Overall accuracy | 96.0% | 81.4% |
| Macro-avg F1 | 0.751 | 0.611 |
| Scratch recall | 0.012 | **0.201** |
| Scratch precision | 0.600 | 0.015 |
| "none" F1 | 0.984 | 0.894 |

**What happened:** Scratch recall improved meaningfully (the model now
actually attempts to detect it), but at a steep cost — precision
collapsed across several classes (Edge-Loc: 0.697→0.312, Loc:
0.594→0.271, Scratch: 0.600→0.015), and macro-avg F1 got *worse*, not
better. Raw inverse-frequency weighting turned out to be too aggressive:
because Scratch is ~989x rarer than "none," its weight became extreme
enough to push the model into over-predicting rare classes broadly,
trading false negatives for a flood of false positives instead of
striking a genuine balance.

**Takeaway:** naively "fixing" class imbalance can just move the problem
rather than solve it — this is a real, common pitfall worth being able
to speak to, not a mistake to hide.

### Milestone 3, Attempt 2: Weighted Loss (inverse square-root frequency)

Softened the weighting scheme — using `1/sqrt(class_count)` instead of
`1/class_count` — so rare classes are still upweighted, but far less
extremely, aiming for a middle ground between the baseline and the
over-corrected first attempt.

_(Results to be added once this run completes.)_

### Next: Milestone 4

Final writeup polishing the comparison across all three models above,
and connecting the class-imbalance findings explicitly back to
real-world fab yield/defect-triage risk.