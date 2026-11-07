# fraud-spike-detector

Unsupervised fraud and fraud-burst detection on 284,807 card transactions. Autoencoders are trained on normal transactions only, and the reconstruction error is the fraud score. The scores are compared with PCA, using a time-based split. PyTorch.

Work in progress.

## Data

Credit Card Fraud Detection (Worldline and ULB): 284,807 transactions over two days, 492 of them fraud. Download `creditcard.csv` from Kaggle (`mlg-ulb/creditcardfraud`) and put it in `data/`. The data is not included in this repository.

## Split

By time, so the model is always tested on transactions that come after the ones it learned from.

| split | time window (seconds) | rows | frauds |
|---|---|---|---|
| train | 0 to 86,400 (day 1) | 144,786 | 281 |
| validation | 86,400 to 129,600 | 47,445 | 96 |
| test | 129,600 to 172,792 | 92,576 | 115 |

The model is fitted on the 144,505 normal train rows only. Amount is replaced by log(1 + Amount), then the 29 features (V1 to V28 and Amount) are scaled with the mean and standard deviation of those rows. The same numbers are used for validation and test.

## Progress

PCA baseline, scored by squared reconstruction error summed over the 29 features, PR-AUC on validation:

| components k | PR-AUC |
|---|---|
| 2 | 0.3474 |
| 4 | 0.3504 |
| 8 | 0.3609 |
| 16 | 0.4836 |
| 20 | 0.5530 |
| 24 | 0.5656 |
| 27 | 0.8100 |
| 28 | 0.6875 |
| 29 | 0.1065 |

A scorer that ranks transactions at random would get about 0.002 (96 frauds in 47,445 rows). Only validation was used to choose k. The test set has not been touched.

Next: linear and deep autoencoders, then window-based spike detection.
