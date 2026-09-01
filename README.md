# Banking Marketing Targets (Deep Neural Network)

A Keras/TensorFlow neural network that predicts whether a bank client will subscribe to a term deposit after a marketing call, trained on the classic Portuguese bank marketing dataset.

## Overview

Banks run phone-based marketing campaigns to sell term deposits, but most calls don't convert — in this dataset only about 12% of clients say yes. The goal here is to flag which clients are actually worth calling, using the data collected during and before the campaign (client profile, contact info, previous campaign outcomes).

The notebook walks through the full pipeline: load the data, explore it, encode/scale it, train a small feed-forward network with class weighting to deal with the imbalance, and evaluate on a held-out test set.

## Features

- EDA: missing value and duplicate checks, target distribution, numeric feature distributions with outlier detection (IQR/boxplots), mixed-type correlation matrix via `dython`
- Label encoding for categorical features, `StandardScaler` for numeric features
- Stratified train/validation split (80/20) on top of the dataset's separate train/test files
- Dense neural network (Keras `Sequential`) with dropout regularization
- Class weighting to handle the ~88/12 class imbalance
- Early stopping (on validation AUC) and learning-rate reduction on plateau
- Evaluation with accuracy, F1, ROC-AUC, confusion matrix and ROC curve

## Tech Stack

| Category      | Tools |
|---------------|-------|
| Language      | Python 3.12 |
| ML/AI         | Keras, TensorFlow |
| Data          | pandas, NumPy, scikit-learn |
| Visualization | matplotlib, seaborn, dython |
| Environment   | Jupyter Notebook |

## Project Structure

```
.
└── banking-dataset-marketing-targets-deep-nn.ipynb
```

Everything (EDA, preprocessing, model, training, evaluation) lives in a single notebook.

## Dataset

**Banking Dataset - Marketing Targets** (Portuguese bank direct marketing campaigns), loaded from:

```
train.csv  -> 45,211 rows
test.csv   ->  4,521 rows
17 columns, semicolon-separated (sep=';')
```

- **Target:** `y` — whether the client subscribed to a term deposit (`no` / `yes`), encoded to 0/1
- **Class balance:** ~88% `no` vs ~12% `yes`
- **Features:** 16 client/campaign attributes — mix of numeric (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) and categorical (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, etc.)

**Preprocessing steps:**
- Label-encode the target and all categorical columns (train/test encoded together to keep category mapping consistent)
- Standard-scale the 7 numeric columns
- Split the encoded train set into train/validation (80/20, stratified, `random_state=42`)

Resulting shapes: `X_train (36168, 16)`, `X_val (9043, 16)`, `X_test_final (4521, 16)`.

## Methodology

```
raw CSVs (train.csv / test.csv)
      │
      ▼
EDA (nulls, duplicates, target balance, distributions, outliers, correlations)
      │
      ▼
Preprocessing (label encoding + standard scaling)
      │
      ▼
Train/val split (stratified)
      │
      ▼
Train Keras model (class-weighted, early stopping, LR reduction)
      │
      ▼
Evaluate on test.csv (accuracy, F1, ROC-AUC, confusion matrix, ROC curve)
```

## Model

A fully connected network built with Keras `Sequential`:

```
Dense(256) -> ReLU -> Dropout(0.45)
Dense(256) -> ReLU -> Dropout(0.45)
Dense(1)   -> Sigmoid
```

- 70,401 trainable parameters
- Optimizer: Adam, loss: binary cross-entropy
- Metrics tracked during training: accuracy, AUC
- `class_weight='balanced'` to compensate for the imbalanced target
- Callbacks: `EarlyStopping` (monitors `val_auc`, patience 10, restores best weights), `ReduceLROnPlateau` (monitors `val_loss`, factor 0.5, patience 5, min_lr 1e-6)
- Trained for up to 100 epochs, batch size 128

## Results

Evaluated on the held-out `test.csv` (4,521 samples, threshold 0.5):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8093 |
| F1-score  | 0.5113 |
| ROC-AUC   | 0.9074 |

```
              precision    recall  f1-score   support

          no       0.98      0.80      0.88      4000
         yes       0.36      0.87      0.51       521

    accuracy                           0.81      4521
   macro avg       0.67      0.83      0.70      4521
weighted avg       0.91      0.81      0.84      4521
```

Class weighting pushes recall on the minority `yes` class up to 0.87, at the cost of precision (0.36) — the model is tuned to catch likely subscribers rather than to avoid false positives, which fits the "who should we call" use case better than raw accuracy would.

## Installation

```bash
git clone <repo-url>
cd banking-dataset-marketing-targets-deep-nn

pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow dython

jupyter notebook banking-dataset-marketing-targets-deep-nn.ipynb
```

The notebook expects `train.csv` and `test.csv` from the Banking Dataset - Marketing Targets dataset, semicolon-separated, placed at the paths referenced in the "Load Data" cell (update the paths if you're not running on Kaggle).

## Usage

Run the notebook top to bottom:

1. Load libraries and data
2. Run the EDA cells to inspect the data
3. Run preprocessing (encoding + scaling)
4. Build and train the model
5. Review the evaluation cells (metrics, confusion matrix, ROC curve) at the end

## Limitations

- Single train/val split, no k-fold cross-validation
- No hyperparameter search — layer sizes, dropout rate and batch size were set manually
- Precision on the `yes` class is low (0.36); usable for ranking/prioritizing clients to call, not as a hard yes/no filter
- Categorical features are label-encoded rather than one-hot encoded or embedded, which imposes an artificial ordinal relationship on nominal categories
- No model export/serialization step — the trained model isn't saved for reuse outside the notebook

## Future Improvements

- Try one-hot encoding or entity embeddings for categorical features instead of label encoding
- Add cross-validation and a proper hyperparameter search
- Compare against simpler baselines (logistic regression, gradient boosting) to see if the NN is actually worth the added complexity
- Save the trained model and scaler/encoders for inference outside the notebook
- Tune the decision threshold instead of using the default 0.5, given the precision/recall trade-off in the results
