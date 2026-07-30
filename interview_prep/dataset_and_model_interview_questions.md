# Interview Prep — Dataset & Model Implementation (Project 1)

Questions an interviewer could reasonably ask based specifically on
`dataset.py` and `model.py`, organized so you can test yourself. Try
answering from memory first, then check against the explanation.

---

## Part 1: `dataset.py`

### Q1. Why does `WaferMapDataset` inherit from `torch.utils.data.Dataset`, and what methods does that require you to implement?

**Answer:** PyTorch's `DataLoader` (which handles batching, shuffling,
and parallel data loading during training) expects any dataset to follow
a standard interface: `__len__` (how many samples exist) and
`__getitem__` (return one sample given an index). Inheriting from
`Dataset` and implementing these two methods is what makes a custom
class "pluggable" into PyTorch's training machinery, regardless of what
the underlying data source actually is (a DataFrame, files on disk, a
database, etc.).

### Q2. Why resize every wafer map to the same target size? What would happen if you didn't?

**Answer:** PyTorch batches multiple samples together into a single
tensor for efficient parallel computation. This requires every sample in
a batch to have identical dimensions — you cannot stack arrays of
different shapes into one tensor. Without resizing, you'd get a runtime
error the moment `DataLoader` tried to batch two differently-shaped
wafer maps together.

### Q3. Why `cv2.INTER_NEAREST` specifically, instead of a smoother interpolation method like bilinear or bicubic?

**Answer:** The die values (0/1/2) are categorical, not continuous —
they represent discrete states (blank/pass/fail), not a smooth physical
quantity like brightness in a photo. Smooth interpolation methods
blend neighboring pixel values, which could produce a meaningless value
like 1.5 between a "pass" (1) and a "fail" (2). Nearest-neighbor
interpolation instead copies the closest original value, preserving the
categorical integrity of the data.

*(Good follow-up to be ready for: "When WOULD you use bilinear/bicubic
instead?" — Answer: for continuous-valued image data, like natural
photographs, where smooth blending between pixel values is meaningful
and even desirable.)*

### Q4. Why normalize pixel values (dividing by 2.0) before feeding them into the model?

**Answer:** Neural networks train more stably and converge faster when
inputs are on a small, consistent scale (commonly 0-1). Without
normalization, larger raw values can cause unstable gradient updates
during training, and different features/inputs on very different scales
can make optimization harder. Since die values are only ever 0, 1, or 2,
dividing by 2 is a simple min-max style scaling to bring everything into
[0, 1].

### Q5. What does `.unsqueeze(0)` do, and why is it necessary here?

**Answer:** It adds a new dimension of size 1 at the specified position
(index 0, meaning the front). CNNs in PyTorch expect input shaped
`(channels, height, width)` — even single-channel (grayscale-like) data
needs an explicit channel dimension. Without `.unsqueeze(0)`, the tensor
would be shaped `(32, 32)` instead of `(1, 32, 32)`, and the first
convolutional layer (`in_channels=1`) would throw a shape-mismatch error.

### Q6. Why filter out wafers with very small area (`min_area`) before training?

**Answer:** Extremely small/degenerate wafer maps (like the (15,3)
outlier found during EDA) are likely data artifacts rather than genuine
signal — they carry very little real spatial information and can act as
noise during training, potentially confusing the model rather than
teaching it anything meaningful about actual defect patterns. This is a
data-cleaning decision made explicit and documented, not an accidental
oversight.

### Q7. What's the difference between `__len__`/`__getitem__` here and just handing PyTorch a big NumPy array directly?

**Answer:** A `Dataset` class allows lazy, on-demand processing — each
sample is only resized/normalized/converted when actually requested
(e.g., during a training batch), rather than doing all that
preprocessing upfront for the entire dataset at once. This is much more
memory-efficient for large datasets and also naturally supports things
like data augmentation applied differently on each access.

---

## Part 2: `model.py`

### Q8. Walk me through what happens to a single wafer map as it passes through this network, start to finish.

**Answer (talking points, in order):**
1. Input: `(1, 32, 32)` — one channel, 32x32 grid
2. `conv1`: learns 16 different simple pattern detectors → output `(16, 32, 32)`
3. ReLU: zeroes out negative activations, introduces non-linearity
4. Pool: halves spatial size → `(16, 16, 16)`
5. `conv2`: learns 32 more complex patterns from the 16 input channels → `(32, 16, 16)`
6. ReLU + Pool again → `(32, 8, 8)`
7. Flatten → a single vector of length `32*8*8 = 2048`
8. `fc1`: compress to 128 numbers, ReLU applied
9. Dropout: randomly zero 30% of these 128 values (training only)
10. `fc2`: compress to 9 raw scores, one per defect class
11. (Outside the model) softmax + argmax elsewhere turns these into a predicted class

### Q9. Why two convolutional layers instead of one, or five?

**Answer:** This reflects a classic CNN design principle — stacking
multiple conv layers lets the network build a *hierarchy* of features:
early layers detect simple local patterns (edges, small blobs), later
layers combine those into more complex, larger-scale shapes (e.g., a
ring near the wafer boundary). Two layers is a reasonable, deliberately
modest choice for a 32x32 input with a manageable number of classes —
too many layers on such small images would risk overfitting (especially
given how few examples exist for some rare classes) and add unnecessary
computational cost without meaningful benefit at this image resolution.

### Q10. What is `padding=1` doing in the conv layers, and why does it matter?

**Answer:** By default, a convolution shrinks the spatial dimensions
slightly (since the sliding window can't center itself on border
pixels). Padding adds a border of zeros around the input before
convolving, so the output spatial size matches the input size. This
keeps the math predictable, especially useful for tracking exact
dimensions through multiple layers (as done manually for
`flattened_size`).

### Q11. Why MaxPool instead of, say, AveragePool?

**Answer:** Max pooling keeps the strongest activation in each region,
which tends to preserve the most salient/distinctive features (e.g.,
"is there a strong fail-cluster signal here at all") — generally
considered better for classification tasks focused on detecting the
presence of specific patterns. Average pooling smooths things out,
which can dilute sharp, localized signals — sometimes preferred in other
contexts (e.g., certain segmentation tasks) but less common as a default
for classification CNNs like this one.

### Q12. Why is Dropout only applied once, between the two fully connected layers, and not after the convolutional layers?

**Answer:** This is a design choice, not a hard rule — dropout can be
applied after conv layers too (sometimes called spatial dropout). Here
it's placed in the fully-connected section because that's typically
where a small network like this is most at risk of overfitting (dense
layers have many parameters relative to a small dataset), while the
convolutional layers, with fewer parameters and enforced spatial
structure, are comparatively less prone to overfitting in a network this
size.

### Q13. Why does `forward()` NOT apply softmax at the end?

**Answer:** `nn.CrossEntropyLoss` (used in `train.py`) internally
combines a softmax operation with negative log-likelihood loss for
numerical stability reasons. If softmax were applied twice (once in the
model, once inside the loss function), it would distort the loss
calculation and hurt training. The raw, un-normalized outputs (logits)
are the correct thing to return from `forward()`.

### Q14. How would you modify this architecture if you had a much larger dataset with higher-resolution images?

**Answer (good to have a considered opinion here):** Likely add more
convolutional layers to build a deeper feature hierarchy, possibly
increase the number of filters per layer, consider batch normalization
between conv layers to stabilize training at greater depth, and
potentially explore transfer learning from a pretrained model if the
images resembled natural photographs (which wafer maps do not, so a
from-scratch architecture remains reasonable even at larger scale here).

### Q15. This architecture is quite small/simple. Why not just use a pretrained model like ResNet?

**Answer:** Pretrained models like ResNet are trained on natural
photographs (ImageNet) — their learned low-level features (color
gradients, natural textures, object edges) don't transfer meaningfully
to single-channel, categorical wafer die grids, which are a
fundamentally different kind of image data. A small custom CNN
purpose-built for this input is both more appropriate for the data and
far easier to fully explain and justify than "I used a pretrained model"
— which also demonstrates a deeper, more hands-on understanding of CNN
fundamentals rather than relying on a black box.

---

## Related concepts worth knowing solidly (even if not directly in the code)

These are natural follow-up territory in an interview, even though
they're not explicitly implemented here — know at least a one-paragraph
explanation for each:

- **Batch normalization** — what it does, why it stabilizes/speeds up
  training, and why this project doesn't currently use it (small
  network, could be a stated "future improvement")
- **Overfitting vs underfitting** — how to recognize each from
  train/val accuracy curves, and general strategies to address both
- **Data augmentation** — techniques (rotation, flipping) and *why*
  rotation/flipping might be especially reasonable for wafer maps
  specifically (a defect pattern's meaning doesn't change if the wafer
  is rotated — unlike, say, a photo of digit "6" which becomes "9" when
  flipped)
- **Class imbalance handling techniques** — weighted loss functions,
  oversampling minority classes, undersampling majority classes,
  synthetic oversampling (SMOTE-style, though less standard for image
  data) — directly relevant since you'll implement this in Milestone 3
- **Confusion matrix** — how to read one, and why overall accuracy alone
  is a misleading metric on an imbalanced dataset like this
  (precision/recall/F1 per class matter more here)
- **Backpropagation & gradient descent** — the general concept, even
  without deriving the calculus, and what "vanishing/exploding
  gradients" means at a conceptual level
- **Learning rate** — what happens if it's too high vs too low, and what
  a learning rate scheduler does
- **Epoch vs batch vs iteration** — precise definitions, since these are
  often used loosely but interviewers may probe the distinction
  specifically
- **Train/validation/test split** — why three-way splits exist in more
  rigorous setups (test set held out even from validation-driven
  decisions like hyperparameter tuning), even though this project
  currently only does train/val
- **Why CNNs specifically suit spatial/grid data** — parameter sharing
  (the same filter is reused across the whole image) and translation
  invariance, as the core ideas distinguishing CNNs from plain
  fully-connected networks
