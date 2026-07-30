"""
A small CNN for wafer map defect classification.

Why a custom small CNN, rather than a huge pretrained model (like ResNet)?
Pretrained models are trained on natural photos (cats, cars, faces) — their
learned features (edges, textures, colors) don't transfer meaningfully to
32x32 single-channel wafer die grids, which are a completely different kind
of image. A small custom CNN, built for this specific input size and
problem, is both more appropriate and much easier to explain in an
interview than "I used a pretrained model I don't fully understand."
"""

import torch.nn as nn
import torch.nn.functional as F


class WaferCNN(nn.Module):
    def __init__(self, num_classes=9, input_size=32):
        super().__init__()

        # --- Convolutional feature extraction ---
        # Each conv layer learns to detect increasingly complex spatial
        # patterns: early layers pick up simple edges/blobs, later layers
        # combine those into more complex shapes (e.g. "a ring near the
        # boundary" for Edge-Ring, or "a cluster near the center" for
        # Center-type defects).
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Pooling halves the spatial dimensions after each conv layer —
        # reduces computation and makes the network somewhat tolerant to
        # small shifts in exactly where a defect pattern sits on the wafer.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two pool operations, a 32x32 input becomes 8x8
        # (32 -> 16 -> 8), with 32 channels from conv2.
        flattened_size = 32 * (input_size // 4) * (input_size // 4)

        # --- Fully connected classifier head ---
        # Takes the extracted spatial features and maps them to a final
        # decision across the 9 possible defect classes.
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.3)  # randomly disables neurons during
                                         # training to reduce overfitting —
                                         # useful here since the dataset,
                                         # especially rare classes, is small
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten from (batch, channels, h, w) to (batch, features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # raw scores per class (no softmax here — the loss
                          # function we'll use applies it internally)
        return x