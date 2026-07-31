"""
PyTorch Dataset for the WM-811K wafer map defect classification task.

Why this file exists:
PyTorch models don't train directly on a pandas DataFrame. They expect a
`Dataset` object that knows how to (a) report how many samples exist, and
(b) return one (image_tensor, label) pair at a time when asked for index i.
A `DataLoader` then wraps this to serve up shuffled mini-batches during
training. This file is that translation layer between your cleaned
DataFrame and what PyTorch actually needs.
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from data_loader import DEFECT_CLASSES

# Map each string label to an integer index (models work with numbers,
# not strings). e.g. "Center" -> 0, "Donut" -> 1, ..., "none" -> 8
LABEL_TO_IDX = {label: idx for idx, label in enumerate(DEFECT_CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}


class WaferMapDataset(Dataset):
    def __init__(self, dataframe, target_size=32, min_area=100, augment=False):
        """
        dataframe : the labeled (and ideally outlier-filtered) wafer DataFrame
        target_size : every wafer map gets resized to (target_size, target_size)
                       so that all images fed to the CNN are a uniform shape,
                       which PyTorch requires for batching.
        min_area : safety net — drops any degenerate wafer maps (like the
                   (15,3) one you found) that slipped through, based on
                   total die count.
        augment : if True, applies a random 90/180/270-degree rotation to
                  each sample on every access. Used together with
                  oversampling (WeightedRandomSampler) so that rare-class
                  examples aren't just repeated identically many times —
                  each repeat looks slightly different, which helps the
                  model learn the general pattern rather than memorizing
                  exact pixel arrangements. Rotation is a safe augmentation
                  choice here specifically because a defect pattern's
                  identity doesn't change under rotation (a Scratch is
                  still a Scratch at any angle) — unlike, say, digit
                  images, where rotating a 6 can turn it into a 9.
        """
        areas = dataframe["waferMap"].apply(lambda x: x.shape[0] * x.shape[1])
        self.df = dataframe[areas >= min_area].reset_index(drop=True)
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        # Tells PyTorch (and you) how many samples are in this dataset.
        return len(self.df)

    def __getitem__(self, idx):
        # Called by DataLoader once per sample. Must return a
        # (tensor, label) pair for a single index.
        row = self.df.iloc[idx]
        wafer = row["waferMap"].astype(np.float32)

        # Resize every wafer map to a fixed (target_size, target_size) grid.
        # cv2.resize expects (width, height) order, and INTER_NEAREST keeps
        # the values as discrete categories (0/1/2) rather than blending
        # them into meaningless fractional values, which a smoother
        # interpolation method would do.
        resized = cv2.resize(
            wafer, (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST
        )

        if self.augment:
            # Randomly rotate by 0, 90, 180, or 270 degrees. np.rot90 is
            # used (rather than cv2's rotation) because it's an exact,
            # lossless rotation for square arrays — no interpolation
            # artifacts, which matters since these are categorical values.
            k = np.random.randint(0, 4)
            resized = np.rot90(resized, k=k).copy()  # .copy() avoids a
                                                        # negative-stride
                                                        # array, which
                                                        # torch.tensor()
                                                        # cannot accept
                                                        # directly

        # Normalize pixel values from {0,1,2} down to a 0-1 float range.
        # Neural networks train more reliably on small, consistently-scaled
        # inputs rather than raw arbitrary integers.
        resized = resized / 2.0

        # Add a "channel" dimension: CNNs expect input shaped
        # (channels, height, width) — even for a single-channel (grayscale)
        # image like this, PyTorch still wants that leading channel axis.
        tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)

        label_idx = LABEL_TO_IDX[row["label"]]
        return tensor, label_idx