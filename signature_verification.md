# Signature Verification

This section builds three pipelines to classify signatures by signer identity and compares classical features to a small CNN.

**Data preparation**
- Images are loaded from `signature-verification-dataset/`, converted to grayscale, resized to `128x128`, inverted when needed, blurred, and Otsu-thresholded.
- A stratified train/test split uses `TEST_SIZE = 0.25`.

**Models**
- HOG features + Logistic Regression (standardized).
- SIFT descriptors ? bag-of-visual-words (MiniBatchKMeans, codebook size `64`) + Logistic Regression.
- Small CNN with 4 convolutional blocks and a 2-layer MLP head (trained for `12` epochs, batch size `32`, learning rate `1e-3`).

**Result summary**
- The CNN is the best-performing model with accuracy `0.9935`, precision `0.9944`, recall `0.9911`, and F1-score `0.9923`.
- The per-signer classification report shows near-perfect precision/recall across signers, consistent with the confusion matrix in the visuals.

Visuals and the full run log are included in `README.md`.
