"""
Training loop for the baseline wafer defect classification CNN.

This is intentionally a "baseline" — no class-imbalance handling yet
(that's Milestone 3). The point of this run is to get an honest first
number to improve upon, and to confirm the full pipeline (data -> model
-> training -> saving) works end to end.
"""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_loader import load_labeled_dataset
from dataset import WaferMapDataset
from model import WaferCNN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and split data ---
    df = load_labeled_dataset("../data/LSWMD.pkl")

    # Optional: subsample the dataset if running on a memory-constrained
    # machine. Stratified sampling keeps roughly the same class
    # proportions as the full dataset, so this remains a fair (if
    # smaller) baseline rather than accidentally skewing toward one class.
    MAX_SAMPLES = None  # set to None to use the full dataset
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        frac = MAX_SAMPLES / len(df)
        # NOTE: groupby(...).apply(lambda g: g.sample(...)) silently drops
        # the grouping column ("label") in newer pandas versions — using
        # sample_indices + .loc[] instead avoids that pitfall entirely.
        sample_indices = (
            df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=42).index)
        )
        flat_indices = np.concatenate(sample_indices.values) if hasattr(sample_indices, "values") else sample_indices
        df = df.loc[flat_indices].reset_index(drop=True)
        print(f"Subsampled to {len(df)} wafers (from full labeled set) "
              f"to reduce memory usage.")

    full_dataset = WaferMapDataset(df, target_size=32, min_area=100)

    # 80/20 train/validation split. Validation data is held out and never
    # trained on, so it gives an honest estimate of how the model performs
    # on wafers it hasn't seen — without this, you'd have no way to tell
    # if the model is genuinely learning or just memorizing.
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    # Fixed seed ensures this exact same split can be reproduced later in
    # evaluate.py — without this, re-running random_split would produce a
    # DIFFERENT validation set each time, meaning evaluate.py could
    # accidentally test on data the model was actually trained on.
    torch.manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # --- Model, loss, optimizer ---
    model = WaferCNN(num_classes=9, input_size=32).to(device)

    # CrossEntropyLoss is the standard choice for multi-class
    # classification — it combines softmax + negative log-likelihood in
    # one step, and expects raw model outputs (not pre-softmaxed), which
    # is exactly what WaferCNN.forward() returns.
    criterion = nn.CrossEntropyLoss()

    # Adam is a reliable, commonly-used optimizer that adapts the learning
    # rate per-parameter automatically — a sensible default for a baseline
    # rather than hand-tuning plain SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # one epoch = one full pass through the training data

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()  # enables dropout during training
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()          # clear gradients from last step
            outputs = model(images)        # forward pass
            loss = criterion(outputs, labels)
            loss.backward()                # compute gradients
            optimizer.step()               # update model weights

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation phase ---
        model.eval()  # disables dropout for a stable, honest evaluation
        val_correct, val_total = 0, 0
        with torch.no_grad():  # no need to track gradients when just evaluating
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # Save the trained model weights so it can be reloaded later for
    # evaluation (confusion matrix etc.) without retraining from scratch.
    torch.save(model.state_dict(), "../models/baseline_cnn.pt")
    print("Model saved to models/baseline_cnn.pt")


if __name__ == "__main__":
    train()