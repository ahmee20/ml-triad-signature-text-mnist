# Word-Level LSTM Sentence Completion

This section trains a word-level LSTM on the Shakespeare corpus and compares three hyperparameter configurations.

**Data preparation**
- Source file: `shakespeare-plays/versions/4/alllines.txt`.
- Tokenization keeps words and punctuation; tokens are lowercased with `<eol>` markers.
- Sequence length `6`, minimum word frequency `3`, maximum vocab size `8000`, and a token cap of `140000`.
- Train/validation split uses `TRAIN_RATIO = 0.9` with early stopping patience `3`.

**Experiments**
- Experiment 1: Baseline LSTM.
- Experiment 2: Deeper LSTM with larger embedding/hidden sizes.
- Experiment 3: Regularized LSTM with higher dropout.

**Result summary**
- The best validation loss and perplexity come from **Experiment 2 ? Deeper LSTM**.
- This model is selected for qualitative sentence completion comparison.

Curves, comparisons, and the full run log are included in `README.md`.
