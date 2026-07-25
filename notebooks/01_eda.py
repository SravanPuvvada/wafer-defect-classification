# %% [markdown]
# # Wafer Map Defect Classification — Exploratory Data Analysis
#
# Goal for this notebook (Milestone 1):
# 1. Load the WM-811K dataset
# 2. Understand class distribution (and confirm the imbalance problem)
# 3. Visualize sample wafer maps for each defect class
# 4. Understand image size variability (wafers are NOT all the same shape)
#
# NOTE: Open this file in Jupyter and it will render as a notebook
# (VS Code / JupyterLab both support the "py:percent" format directly).
# If you prefer a real .ipynb, run: `jupytext --to notebook 01_eda.py`

# %%
import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_labeled_dataset, get_class_distribution, DEFECT_CLASSES

sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Load data
#
# Download LSWMD.pkl from Kaggle first:
# https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
# and place it at `../data/LSWMD.pkl`

# %%
df = load_labeled_dataset("../data/LSWMD.pkl")
df.head()

# %% [markdown]
# ## 2. Class distribution
#
# This is the single most important chart in this notebook. Real fab data
# is heavily imbalanced — most wafers are defect-free or fall into a
# handful of common categories, while some patterns (e.g. Near-full) are
# rare. This is exactly the kind of imbalance a process/yield engineer
# deals with in practice, and it directly shapes the modeling choices
# in Milestone 3 (weighted loss / augmentation / resampling).

# %%
dist = get_class_distribution(df)
print(dist)

fig, ax = plt.subplots(figsize=(9, 5))
dist.plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_title("Wafer Defect Class Distribution (WM-811K, labeled subset)")
ax.set_ylabel("Count")
ax.set_xlabel("Defect Class")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../reports/figures/class_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Sample wafer maps per class
#
# Visualizing a handful of examples per class builds intuition for what
# the CNN will need to learn to distinguish — e.g. "Edge-Ring" is a ring
# near the wafer boundary, "Center" is a cluster near the middle,
# "Scratch" is a linear streak.

# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for i, cls in enumerate(dist.index[:9]):
    sample = df[df["label"] == cls].iloc[0]
    axes[i].imshow(sample["waferMap"], cmap="viridis")
    axes[i].set_title(f"{cls} (n={dist[cls]})")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("../reports/figures/sample_wafers_per_class.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Image size variability
#
# Wafer maps are NOT a fixed resolution — die grid size varies by
# product/lot. This matters directly for model design: we'll need a
# resize/pad step before feeding into a CNN. Worth quantifying up front.

# %%
shapes = df["waferMap"].apply(lambda x: x.shape)
shape_counts = shapes.value_counts().head(15)
print("Most common wafer map shapes:")
print(shape_counts)

heights = shapes.apply(lambda s: s[0])
widths = shapes.apply(lambda s: s[1])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(widths, heights, alpha=0.05, s=5)
ax.set_xlabel("Width (die columns)")
ax.set_ylabel("Height (die rows)")
ax.set_title("Wafer Map Dimension Variability")
plt.tight_layout()
plt.savefig("../reports/figures/dimension_variability.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Takeaways (fill in after running)
#
# - Total labeled wafers: ...
# - Most common class: ... / Rarest class: ...
# - Imbalance ratio (largest class / smallest class): ...
# - Typical wafer map size: ... (decide target resize dimension for CNN, e.g. 64x64)
#
# **Next step (Milestone 2):** build a PyTorch Dataset that resizes/pads
# wafer maps to a fixed size and trains a baseline CNN.
