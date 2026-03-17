# Signature Text PixelCNN

A compact, single-notebook project that compares classical features and neural models across three tasks: signature verification, word-level LSTM sentence completion, and PixelCNN generation on binarized MNIST. Plots and full run logs are included for reproducibility.

**Project Structure**
- `signature-text-pixelcnn.ipynb`
- `signature_verification.md`
- `lstm_sentence_completion.md`
- `pixelcnn_mnist.md`
- `signature-verification-dataset/`
- `shakespeare-plays/`
- `mnist/`
- `report_images/`

**Datasets**
- `signature-verification-dataset/`: signature images used for signer classification and forged vs genuine analysis.
- `shakespeare-plays/`: text corpus used for word-level language modeling.
- `mnist/`: IDX-format MNIST files used for binarized PixelCNN experiments.

**Notebook**
- `signature-text-pixelcnn.ipynb` contains all code for data loading, training, evaluation, and plotting.
- Outputs are cleared in the notebook; the results and logs are captured below.

**Signature Verification Results**

Key visuals:

![Signature Sample Signatures](report_images/signature_sample_signatures.png)
![Signature Class Distribution](report_images/signature_class_distribution.png)
![Signature HOG Learning Curve](report_images/signature_hog_learning_curve.png)
![Signature SIFT Learning Curve](report_images/signature_sift_learning_curve.png)
![Signature CNN Accuracy Curve](report_images/signature_cnn_accuracy_curve.png)
![Signature CNN Error Curve](report_images/signature_cnn_error_curve.png)
![Signature Metrics Comparison](report_images/signature_metrics_comparison.png)
![Signature Confusion Matrix](report_images/signature_confusion_matrix.png)

<details>
<summary>Run log</summary>

```text
Installing missing packages: torchvision
Dataset directory: C:\Users\Capricon\Desktop\genAI\signature-verification-dataset
Total images: 4298
Unique signers: 64
         count
status        
forged    2020
genuine   2278
Extracting HOG features...
Extracting SIFT descriptors...
Training CNN on cpu
Epoch 01 | train_loss=2.5646 | test_loss=0.9553 | train_acc=0.3894 | test_acc=0.7786
Epoch 02 | train_loss=0.8182 | test_loss=0.5960 | train_acc=0.7809 | test_acc=0.8409
Epoch 03 | train_loss=0.4081 | test_loss=0.2072 | train_acc=0.8995 | test_acc=0.9479
Epoch 04 | train_loss=0.2402 | test_loss=0.1569 | train_acc=0.9345 | test_acc=0.9628
Epoch 05 | train_loss=0.1607 | test_loss=0.1794 | train_acc=0.9572 | test_acc=0.9516
Epoch 06 | train_loss=0.1204 | test_loss=0.0954 | train_acc=0.9659 | test_acc=0.9795
Epoch 07 | train_loss=0.0866 | test_loss=0.0646 | train_acc=0.9792 | test_acc=0.9860
Epoch 08 | train_loss=0.0633 | test_loss=0.0293 | train_acc=0.9842 | test_acc=0.9963
Epoch 09 | train_loss=0.0542 | test_loss=0.0322 | train_acc=0.9870 | test_acc=0.9944
Epoch 10 | train_loss=0.0494 | test_loss=0.0284 | train_acc=0.9863 | test_acc=0.9935
Epoch 11 | train_loss=0.0464 | test_loss=0.0467 | train_acc=0.9876 | test_acc=0.9888
Epoch 12 | train_loss=0.0565 | test_loss=0.0356 | train_acc=0.9842 | test_acc=0.9935
Classification report for best model:
    accuracy                           0.99      1075
   macro avg       0.99      0.99      0.99      1075
weighted avg       0.99      0.99      0.99      1075
Conclusion:
The best-performing approach on this train/test split is CNN with accuracy=0.9935, precision=0.9944, recall=0.9911, and F1-score=0.9923. Compared with the weakest result (SIFT BoVW + Logistic Regression), the best model improves F1-score by 0.1725.
Interpretation: if the CNN is best, learned features captured signer-specific stroke patterns more effectively than hand-crafted descriptors; if HOG or SIFT is best, the dataset size or signature consistency favored manual feature engineering.
```

</details>

**Word-Level LSTM Results**

Key visuals:

![LSTM Experiment 1 Curves](report_images/lstm_experiment1_curves.png)
![LSTM Experiment 2 Curves](report_images/lstm_experiment2_curves.png)
![LSTM Experiment 3 Curves](report_images/lstm_experiment3_curves.png)
![LSTM Hyperparameter Comparison](report_images/lstm_hyperparameter_comparison.png)

<details>
<summary>Run log</summary>

```text
Device: cpu
Text path: C:\Users\Capricon\Desktop\genAI\shakespeare-plays\versions\4\alllines.txt
Total raw lines: 111396
Total tokens used: 140000
Vocabulary size: 3404
Training sequences: 125994
Validation sequences: 14000
Example tokens: ['scene', 'i', '.', 'london', '.', 'the', 'palace', '.', '<eol>', 'enter', 'king', 'henry', ',', 'lord', 'john', 'of', 'lancaster', ',', 'the', 'earl']
Experiment 1 - Baseline | epoch=01 | train_loss=5.3161 | val_loss=5.0649 | train_acc=0.1711 | val_acc=0.1817 | val_ppl=158.37
Experiment 1 - Baseline | epoch=02 | train_loss=4.8421 | val_loss=4.9075 | train_acc=0.2006 | val_acc=0.1926 | val_ppl=135.31
Experiment 1 - Baseline | epoch=03 | train_loss=4.6360 | val_loss=4.8295 | train_acc=0.2107 | val_acc=0.2021 | val_ppl=125.15
Experiment 1 - Baseline | epoch=04 | train_loss=4.4817 | val_loss=4.7990 | train_acc=0.2185 | val_acc=0.2047 | val_ppl=121.39
Experiment 1 - Baseline | epoch=05 | train_loss=4.3550 | val_loss=4.7878 | train_acc=0.2267 | val_acc=0.2044 | val_ppl=120.04
Experiment 1 - Baseline | epoch=06 | train_loss=4.2438 | val_loss=4.7854 | train_acc=0.2323 | val_acc=0.2062 | val_ppl=119.75
Experiment 1 - Baseline | epoch=07 | train_loss=4.1419 | val_loss=4.7993 | train_acc=0.2377 | val_acc=0.2046 | val_ppl=121.43
Experiment 1 - Baseline | epoch=08 | train_loss=4.0511 | val_loss=4.8183 | train_acc=0.2433 | val_acc=0.2092 | val_ppl=123.75
Experiment 1 - Baseline | epoch=09 | train_loss=3.9637 | val_loss=4.8495 | train_acc=0.2482 | val_acc=0.2069 | val_ppl=127.67
Experiment 1 - Baseline | epoch=10 | train_loss=3.8871 | val_loss=4.8802 | train_acc=0.2517 | val_acc=0.2069 | val_ppl=131.65
Experiment 1 - Baseline | epoch=11 | train_loss=3.8119 | val_loss=4.9148 | train_acc=0.2582 | val_acc=0.2054 | val_ppl=136.29
Experiment 1 - Baseline | epoch=12 | train_loss=3.7374 | val_loss=4.9617 | train_acc=0.2619 | val_acc=0.2039 | val_ppl=142.83
Experiment 1 - Baseline | epoch=13 | train_loss=3.6691 | val_loss=5.0043 | train_acc=0.2689 | val_acc=0.2012 | val_ppl=149.06
Experiment 1 - Baseline | epoch=14 | train_loss=3.6052 | val_loss=5.0405 | train_acc=0.2741 | val_acc=0.2026 | val_ppl=154.54
Experiment 1 - Baseline | epoch=15 | train_loss=3.5401 | val_loss=5.0845 | train_acc=0.2809 | val_acc=0.2007 | val_ppl=161.50
Experiment 1 - Baseline | epoch=16 | train_loss=3.4814 | val_loss=5.1163 | train_acc=0.2854 | val_acc=0.1976 | val_ppl=166.71
Experiment 1 - Baseline | epoch=17 | train_loss=3.4258 | val_loss=5.1586 | train_acc=0.2937 | val_acc=0.2000 | val_ppl=173.92
Experiment 1 - Baseline | epoch=18 | train_loss=3.3681 | val_loss=5.2049 | train_acc=0.3000 | val_acc=0.1962 | val_ppl=182.16
Experiment 1 - Baseline | epoch=19 | train_loss=3.3127 | val_loss=5.2532 | train_acc=0.3051 | val_acc=0.1964 | val_ppl=191.18
Experiment 1 - Baseline | epoch=20 | train_loss=3.2574 | val_loss=5.2911 | train_acc=0.3134 | val_acc=0.1931 | val_ppl=198.57
Experiment 1 - Baseline | epoch=21 | train_loss=3.2143 | val_loss=5.3256 | train_acc=0.3189 | val_acc=0.1949 | val_ppl=205.53
Experiment 1 - Baseline | epoch=22 | train_loss=3.1616 | val_loss=5.3732 | train_acc=0.3248 | val_acc=0.1934 | val_ppl=215.55
Experiment 1 - Baseline | epoch=23 | train_loss=3.1166 | val_loss=5.4117 | train_acc=0.3312 | val_acc=0.1906 | val_ppl=224.01
Experiment 1 - Baseline | epoch=24 | train_loss=3.0720 | val_loss=5.4558 | train_acc=0.3377 | val_acc=0.1905 | val_ppl=234.12
Experiment 1 - Baseline | epoch=25 | train_loss=3.0272 | val_loss=5.4917 | train_acc=0.3447 | val_acc=0.1889 | val_ppl=242.66
Experiment 1 - Baseline | epoch=26 | train_loss=2.9851 | val_loss=5.5348 | train_acc=0.3511 | val_acc=0.1896 | val_ppl=253.37
Experiment 1 - Baseline | epoch=27 | train_loss=2.9463 | val_loss=5.5690 | train_acc=0.3565 | val_acc=0.1874 | val_ppl=262.17
Experiment 1 - Baseline | epoch=28 | train_loss=2.9026 | val_loss=5.6081 | train_acc=0.3637 | val_acc=0.1868 | val_ppl=272.63
Experiment 1 - Baseline | epoch=29 | train_loss=2.8715 | val_loss=5.6456 | train_acc=0.3673 | val_acc=0.1877 | val_ppl=283.04
Experiment 1 - Baseline | epoch=30 | train_loss=2.8304 | val_loss=5.6797 | train_acc=0.3749 | val_acc=0.1859 | val_ppl=292.86
Experiment 1 - Baseline | epoch=31 | train_loss=2.7962 | val_loss=5.7223 | train_acc=0.3797 | val_acc=0.1852 | val_ppl=305.62
Experiment 1 - Baseline | epoch=32 | train_loss=2.7621 | val_loss=5.7713 | train_acc=0.3861 | val_acc=0.1802 | val_ppl=320.96
Experiment 1 - Baseline | epoch=33 | train_loss=2.7271 | val_loss=5.7978 | train_acc=0.3915 | val_acc=0.1798 | val_ppl=329.56
Experiment 1 - Baseline | epoch=34 | train_loss=2.6905 | val_loss=5.8244 | train_acc=0.3963 | val_acc=0.1797 | val_ppl=338.45
Experiment 1 - Baseline | epoch=35 | train_loss=2.6678 | val_loss=5.8743 | train_acc=0.4001 | val_acc=0.1787 | val_ppl=355.78
Experiment 1 - Baseline | epoch=36 | train_loss=2.6386 | val_loss=5.9042 | train_acc=0.4049 | val_acc=0.1807 | val_ppl=366.57
Experiment 1 - Baseline | epoch=37 | train_loss=2.6007 | val_loss=5.9333 | train_acc=0.4116 | val_acc=0.1778 | val_ppl=377.41
Experiment 1 - Baseline | epoch=38 | train_loss=2.5756 | val_loss=5.9725 | train_acc=0.4163 | val_acc=0.1801 | val_ppl=392.49
Experiment 1 - Baseline | epoch=39 | train_loss=2.5517 | val_loss=6.0038 | train_acc=0.4203 | val_acc=0.1777 | val_ppl=404.96
Experiment 1 - Baseline | epoch=40 | train_loss=2.5235 | val_loss=6.0408 | train_acc=0.4244 | val_acc=0.1774 | val_ppl=420.22
Experiment 1 - Baseline | epoch=41 | train_loss=2.4965 | val_loss=6.0949 | train_acc=0.4286 | val_acc=0.1770 | val_ppl=443.61
Experiment 1 - Baseline | epoch=42 | train_loss=2.4757 | val_loss=6.1212 | train_acc=0.4333 | val_acc=0.1764 | val_ppl=455.43
Experiment 1 - Baseline | epoch=43 | train_loss=2.4492 | val_loss=6.1486 | train_acc=0.4379 | val_acc=0.1755 | val_ppl=468.05
Experiment 1 - Baseline | epoch=44 | train_loss=2.4269 | val_loss=6.1737 | train_acc=0.4421 | val_acc=0.1749 | val_ppl=479.98
Experiment 1 - Baseline | epoch=45 | train_loss=2.4046 | val_loss=6.2155 | train_acc=0.4437 | val_acc=0.1752 | val_ppl=500.43
Experiment 1 - Baseline | epoch=46 | train_loss=2.3791 | val_loss=6.2303 | train_acc=0.4487 | val_acc=0.1768 | val_ppl=507.89
Experiment 1 - Baseline | epoch=47 | train_loss=2.3616 | val_loss=6.2880 | train_acc=0.4513 | val_acc=0.1719 | val_ppl=538.06
Experiment 1 - Baseline | epoch=48 | train_loss=2.3387 | val_loss=6.3038 | train_acc=0.4554 | val_acc=0.1734 | val_ppl=546.63
Experiment 1 - Baseline | epoch=49 | train_loss=2.3165 | val_loss=6.3269 | train_acc=0.4604 | val_acc=0.1726 | val_ppl=559.39
Experiment 1 - Baseline | epoch=50 | train_loss=2.2982 | val_loss=6.3726 | train_acc=0.4629 | val_acc=0.1710 | val_ppl=585.58
Experiment 2 - Deeper LSTM | epoch=01 | train_loss=5.2431 | val_loss=5.0283 | train_acc=0.1789 | val_acc=0.1884 | val_ppl=152.67
Experiment 2 - Deeper LSTM | epoch=02 | train_loss=4.7904 | val_loss=4.8736 | train_acc=0.2068 | val_acc=0.2024 | val_ppl=130.79
Experiment 2 - Deeper LSTM | epoch=03 | train_loss=4.5623 | val_loss=4.7907 | train_acc=0.2210 | val_acc=0.2083 | val_ppl=120.38
Experiment 2 - Deeper LSTM | epoch=04 | train_loss=4.3830 | val_loss=4.7687 | train_acc=0.2306 | val_acc=0.2113 | val_ppl=117.76
Experiment 2 - Deeper LSTM | epoch=05 | train_loss=4.2294 | val_loss=4.7782 | train_acc=0.2409 | val_acc=0.2112 | val_ppl=118.89
Experiment 2 - Deeper LSTM | epoch=06 | train_loss=4.0860 | val_loss=4.8024 | train_acc=0.2503 | val_acc=0.2099 | val_ppl=121.81
Experiment 2 - Deeper LSTM | epoch=07 | train_loss=3.9464 | val_loss=4.8338 | train_acc=0.2598 | val_acc=0.2080 | val_ppl=125.68
Experiment 2 - Deeper LSTM | epoch=08 | train_loss=3.8167 | val_loss=4.8969 | train_acc=0.2679 | val_acc=0.2043 | val_ppl=133.87
Experiment 2 - Deeper LSTM | epoch=09 | train_loss=3.6903 | val_loss=4.9721 | train_acc=0.2789 | val_acc=0.2071 | val_ppl=144.32
Experiment 2 - Deeper LSTM | epoch=10 | train_loss=3.5701 | val_loss=5.0407 | train_acc=0.2886 | val_acc=0.2024 | val_ppl=154.58
Experiment 3 - Regularized | epoch=01 | train_loss=5.3505 | val_loss=5.1274 | train_acc=0.1706 | val_acc=0.1811 | val_ppl=168.58
Experiment 3 - Regularized | epoch=02 | train_loss=4.9640 | val_loss=4.9735 | train_acc=0.1970 | val_acc=0.1964 | val_ppl=144.53
Experiment 3 - Regularized | epoch=03 | train_loss=4.7774 | val_loss=4.8862 | train_acc=0.2078 | val_acc=0.2006 | val_ppl=132.44
Experiment 3 - Regularized | epoch=04 | train_loss=4.6421 | val_loss=4.8373 | train_acc=0.2150 | val_acc=0.2055 | val_ppl=126.12
Experiment 3 - Regularized | epoch=05 | train_loss=4.5324 | val_loss=4.8013 | train_acc=0.2220 | val_acc=0.2098 | val_ppl=121.67
Experiment 3 - Regularized | epoch=06 | train_loss=4.4375 | val_loss=4.8032 | train_acc=0.2274 | val_acc=0.2098 | val_ppl=121.90
Experiment 3 - Regularized | epoch=07 | train_loss=4.3594 | val_loss=4.8001 | train_acc=0.2327 | val_acc=0.2091 | val_ppl=121.53
Experiment 3 - Regularized | epoch=08 | train_loss=4.2879 | val_loss=4.8162 | train_acc=0.2381 | val_acc=0.2101 | val_ppl=123.49
Experiment 3 - Regularized | epoch=09 | train_loss=4.2205 | val_loss=4.8326 | train_acc=0.2427 | val_acc=0.2096 | val_ppl=125.54
Experiment 3 - Regularized | epoch=10 | train_loss=4.1579 | val_loss=4.8546 | train_acc=0.2464 | val_acc=0.2091 | val_ppl=128.33
Best experiment selected for inference: Experiment 2 - Deeper LSTM
Interpretation:
- Lower validation loss and perplexity indicate more fluent and confident next-word predictions.
- Higher validation accuracy indicates stronger exact next-word prediction performance.
- Compare the generated completions above to judge coherence, repetition, and fluency qualitatively.
```

</details>

**PixelCNN on Binarized MNIST Results**

Key visuals:

![Binarized MNIST Samples](report_images/mnist_binarized_samples.png)
![PixelCNN Experiment 1 NLL Curve](report_images/pixelcnn_experiment1_nll_curve.png)
![PixelCNN Experiment 2 NLL Curve](report_images/pixelcnn_experiment2_nll_curve.png)
![PixelCNN Experiment 3 NLL Curve](report_images/pixelcnn_experiment3_nll_curve.png)
![PixelCNN Generated Samples Exp 1](report_images/pixelcnn_generated_samples_exp1.png)
![PixelCNN Generated Samples Exp 2](report_images/pixelcnn_generated_samples_exp2.png)
![PixelCNN Generated Samples Exp 3](report_images/pixelcnn_generated_samples_exp3.png)
![PixelCNN Hyperparameter Comparison](report_images/pixelcnn_hyperparameter_comparison.png)

<details>
<summary>Run log</summary>

```text
Device: cuda
Train/Val/Test sizes: 12000 2000 2000
Using dataset files:
  mnist/train-images.idx3-ubyte
  mnist/train-labels.idx1-ubyte
  mnist/t10k-images.idx3-ubyte
  mnist/t10k-labels.idx1-ubyte
PixelCNN Experiment 1 - Baseline | epoch=01 | train_nll=0.2311 | val_nll=0.1017
PixelCNN Experiment 1 - Baseline | epoch=02 | train_nll=0.1003 | val_nll=0.0954
PixelCNN Experiment 1 - Baseline | epoch=03 | train_nll=0.0959 | val_nll=0.0936
PixelCNN Experiment 1 - Baseline | epoch=04 | train_nll=0.0942 | val_nll=0.0923
PixelCNN Experiment 1 - Baseline | epoch=05 | train_nll=0.0931 | val_nll=0.0921
PixelCNN Experiment 1 - Baseline | epoch=06 | train_nll=0.0924 | val_nll=0.0912
PixelCNN Experiment 1 - Baseline | epoch=07 | train_nll=0.0920 | val_nll=0.0909
PixelCNN Experiment 1 - Baseline | epoch=08 | train_nll=0.0914 | val_nll=0.0904
PixelCNN Experiment 1 - Baseline | epoch=09 | train_nll=0.0911 | val_nll=0.0907
PixelCNN Experiment 1 - Baseline | epoch=10 | train_nll=0.0906 | val_nll=0.0901
PixelCNN Experiment 1 - Baseline | epoch=11 | train_nll=0.0905 | val_nll=0.0899
PixelCNN Experiment 1 - Baseline | epoch=12 | train_nll=0.0903 | val_nll=0.0899
PixelCNN Experiment 2 - Deeper | epoch=01 | train_nll=0.2220 | val_nll=0.1010
PixelCNN Experiment 2 - Deeper | epoch=02 | train_nll=0.1019 | val_nll=0.0948
PixelCNN Experiment 2 - Deeper | epoch=03 | train_nll=0.0971 | val_nll=0.0927
PixelCNN Experiment 2 - Deeper | epoch=04 | train_nll=0.0950 | val_nll=0.0916
PixelCNN Experiment 2 - Deeper | epoch=05 | train_nll=0.0940 | val_nll=0.0913
PixelCNN Experiment 2 - Deeper | epoch=06 | train_nll=0.0929 | val_nll=0.0906
PixelCNN Experiment 2 - Deeper | epoch=07 | train_nll=0.0922 | val_nll=0.0901
PixelCNN Experiment 2 - Deeper | epoch=08 | train_nll=0.0919 | val_nll=0.0897
PixelCNN Experiment 2 - Deeper | epoch=09 | train_nll=0.0914 | val_nll=0.0899
PixelCNN Experiment 2 - Deeper | epoch=10 | train_nll=0.0910 | val_nll=0.0895
PixelCNN Experiment 2 - Deeper | epoch=11 | train_nll=0.0908 | val_nll=0.0893
PixelCNN Experiment 2 - Deeper | epoch=12 | train_nll=0.0905 | val_nll=0.0889
PixelCNN Experiment 2 - Deeper | epoch=13 | train_nll=0.0902 | val_nll=0.0887
PixelCNN Experiment 2 - Deeper | epoch=14 | train_nll=0.0901 | val_nll=0.0885
PixelCNN Experiment 3 - Regularized | epoch=01 | train_nll=0.2261 | val_nll=0.1016
PixelCNN Experiment 3 - Regularized | epoch=02 | train_nll=0.1036 | val_nll=0.0951
PixelCNN Experiment 3 - Regularized | epoch=03 | train_nll=0.0986 | val_nll=0.0935
PixelCNN Experiment 3 - Regularized | epoch=04 | train_nll=0.0962 | val_nll=0.0922
PixelCNN Experiment 3 - Regularized | epoch=05 | train_nll=0.0951 | val_nll=0.0917
PixelCNN Experiment 3 - Regularized | epoch=06 | train_nll=0.0943 | val_nll=0.0913
PixelCNN Experiment 3 - Regularized | epoch=07 | train_nll=0.0936 | val_nll=0.0908
PixelCNN Experiment 3 - Regularized | epoch=08 | train_nll=0.0931 | val_nll=0.0907
PixelCNN Experiment 3 - Regularized | epoch=09 | train_nll=0.0927 | val_nll=0.0902
PixelCNN Experiment 3 - Regularized | epoch=10 | train_nll=0.0924 | val_nll=0.0903
PixelCNN Experiment 3 - Regularized | epoch=11 | train_nll=0.0921 | val_nll=0.0898
PixelCNN Experiment 3 - Regularized | epoch=12 | train_nll=0.0918 | val_nll=0.0898
PixelCNN Experiment 3 - Regularized | epoch=13 | train_nll=0.0916 | val_nll=0.0895
PixelCNN Experiment 3 - Regularized | epoch=14 | train_nll=0.0913 | val_nll=0.0895
Best experiment: PixelCNN Experiment 2 - Deeper
Evaluation guidance:
- Lower validation/test NLL means the model modeled the pixel distribution more accurately.
- Generated images should look digit-like, structurally coherent, and not collapse to noise or a single repeated pattern.
- Pixel density close to the MNIST training distribution is a useful sanity check for fluency of image generation.
Reference training-set pixel density: 0.14900042116641998
```

</details>
