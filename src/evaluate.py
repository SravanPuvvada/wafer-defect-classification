"""
Evaluation script — loads the trained baseline model and produces:
1. A confusion matrix (visual: which classes get confused with which)
2. Per-class precision, recall, F1-score

Why this file matters more than the accuracy number train.py already
printed: overall accuracy is a misleading metric on an imbalanced
dataset like this one. A model can score ~96% overall while performing
very poorly on rare classes, simply by being excellent at the dominant
class. This script exposes that honestly, class by class.
"""

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_labeled_dataset
from dataset import WaferMapDataset, IDX_TO_LABEL, DEFECT_CLASSES
from model import WaferCNN


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the exact same train/val split used during training.
    # IMPORTANT: random_split's split depends on the random seed/state at
    # the time it's called — for a fully rigorous re-creation of the same
    # validation set, a fixed seed should be set before this split (see
    # note at bottom). For now this reproduces an equivalent-sized split.
    df = load_labeled_dataset("../data/LSWMD.pkl")
    full_dataset = WaferMapDataset(df, target_size=32, min_area=100)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    torch.manual_seed(42)  # match the seed ideally also set in train.py
    _, val_ds = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Load the trained model
    model = WaferCNN(num_classes=9, input_size=32).to(device)
    model.load_state_dict(torch.load("../models/baseline_cnn.pt", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Per-class report: precision, recall, F1 ---
    # Precision: of everything the model called "Scratch", how much
    #            actually was Scratch? (false-alarm rate)
    # Recall: of everything that actually was "Scratch", how much did
    #         the model correctly catch? (miss rate)
    # F1: harmonic mean of precision and recall — a single balanced score
    print("\n=== Per-Class Classification Report ===")
    report = classification_report(
        all_labels, all_preds,
        target_names=DEFECT_CLASSES,
        digits=3,
        zero_division=0
    )
    print(report)

    with open("../reports/classification_report.txt", "w") as f:
        f.write(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=DEFECT_CLASSES, yticklabels=DEFECT_CLASSES
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix — Baseline CNN")
    plt.tight_layout()
    plt.savefig("../reports/figures/confusion_matrix.png", dpi=150)
    plt.show()

    print("\nSaved classification report to reports/classification_report.txt")
    print("Saved confusion matrix to reports/figures/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
