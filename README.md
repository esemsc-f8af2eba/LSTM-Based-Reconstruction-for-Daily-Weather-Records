# LSTM-Based-Reconstruction-for-Daily-Weather-Records
Public

## Overview
This project focuses on reconstructing missing daily weather measurements from corrupted multivariate time-series data spanning four decades for a European city. Three decades are provided in both corrupted and uncorrupted form for supervised training, and the fourth decade (`test_set.csv`) contains only corrupted data. The objective is to train a neural network from scratch and generate a completed file `test_set_nogaps.csv` that preserves the original format while filling missing values.

---

## Execution Environment

All experiments were conducted in **Google Colab** using a **GPU runtime (NVIDIA A100)**.  
The project used the default Colab Python environment without creating a custom conda environment.

Primary libraries used:
- PyTorch
- NumPy
- pandas
- matplotlib
- scikit-learn

---

## Dataset Structure

Training data (paired corrupted and clean versions):
- `training_set_0.csv` and `training_set_0_nogaps.csv`
- `training_set_1.csv` and `training_set_1_nogaps.csv`
- `training_set_2.csv` and `training_set_2_nogaps.csv`

Test data:
- `test_set.csv` (corrupted only)

Columns in each file:
- `date`
- `cloud_cover`
- `sunshine`
- `global_radiation`
- `max_temp`
- `mean_temp`
- `min_temp`
- `precipitation`
- `pressure`

---

## Workflow Summary

### Step 1 – Data Loading
- Mounted Google Drive in Colab.
- Loaded all training decades (corrupted and clean pairs) and the corrupted test dataset.
- Verified consistency of columns and data alignment.

### Step 2 – Exploratory Analysis
- Plotted the first 365 days of one training decade using separate line plots for each variable.
- Compared corrupted vs uncorrupted series to visualise missing patterns.
- Generated histograms across all decades to inspect distribution changes caused by corruption.

### Step 3 – Data Preprocessing
- Applied feature-wise scaling based on distribution characteristics:
  - MinMaxScaler for bounded or skewed variables.
  - RobustScaler for variables with outliers.
  - StandardScaler for approximately Gaussian variables.
- Temporarily filled NaNs during model training to prevent instability.

### Step 4 – PyTorch Data Pipeline
- Created supervised training pairs: corrupted inputs + clean targets.
- Created test dataset with corrupted inputs only.
- Constructed `TensorDataset` and `DataLoader` objects.
- Visualised batches to confirm correct pairing and structure.

### Step 5 – Model Design and Training
- Implemented and trained an LSTM-based sequence model from scratch.
- Loss function: Mean Squared Error (MSE).
- Optimiser: Adam.
- Used a validation split to monitor convergence.
- Tracked training and validation loss to ensure stable optimisation.

### Step 6 – Test Set Imputation
- Applied the trained model to the corrupted test dataset.
- Replaced only missing entries while preserving observed values.
- Visualised reconstructed segments using different colours to distinguish imputed values.

### Step 7 – Output Generation
- Saved the final dataset as `test_set_nogaps.csv`.
- Ensured:
  - Same row count and column count as original.
  - Same column names and ordering.
  - Only missing values were replaced.

---

## AI Usage
ChatGPT (OpenAI) was used for:
- Clarifying assessment requirements.
- Drafting structured write-ups.
- Debugging model training issues and PyTorch pipeline setup.

All neural networks were implemented and trained from scratch. No pre-trained models were used.

---

## Final Checks Before Submission
- `test_set_nogaps.csv` contains no NaN values in weather feature columns.
- Output schema exactly matches `test_set.csv`.
- All notebook cells have been executed and saved.