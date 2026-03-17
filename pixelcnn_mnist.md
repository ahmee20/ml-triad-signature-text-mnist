# PixelCNN on Binarized MNIST

This section trains a PixelCNN on binarized MNIST and compares three configurations.

**Data preparation**
- Dataset folder: `mnist/` (IDX files).
- Binarization threshold `0.33`.
- Subsets: train `12000`, validation `2000`, test `2000` images.

**Model**
- Masked convolutions (A and B) with batch normalization and dropout.
- Three experiments vary hidden channels, layer count, dropout, and learning rate.

**Result summary**
- Best model: **PixelCNN Experiment 2 ? Deeper** with best validation NLL `0.0885` and test NLL `0.0863`.
- The generated sample grids show digit-like structure without mode collapse.

Curves, sample grids, and the full run log are included in `README.md`.
