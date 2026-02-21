# Deep Learning for Multivariate Weather Time-Series Imputation

## Overview
This repository contains the implementation for Assessment 2. The objective of this project is to reconstruct missing daily weather measurements from corrupted multivariate time-series data using a neural network model. The dataset consists of four decades of daily weather observations for a European city. Three decades are provided in both corrupted and uncorrupted form (for supervised training), while the fourth decade contains only corrupted data and is used for testing. The final output of this project is an imputed version of the test dataset (`test_set_nogaps.csv`) with missing values reconstructed.

## Dataset Description
Each dataset contains the following columns: `date`, `cloud_cover`, `sunshine`, `global_radiation`, `max_temp`, `mean_temp`, `min_temp`, `precipitation`, and `pressure`. The data represents daily sequential observations.

The training data is located in the `training_set/` directory and includes:
- `training_set_0.csv` and `training_set_0_nogaps.csv`
- `training_set_1.csv` and `training_set_1_nogaps.csv`
- `training_set_2.csv` and `training_set_2_nogaps.csv`

Each decade includes a corrupted version (with missing values) and a corresponding complete version (ground truth). The test dataset (`test_set.csv`) contains corrupted data only. The task is to reconstruct its missing values.

## Methodology
The workflow consists of data loading and exploratory visualisation (line plots for temporal inspection and histograms for distribution analysis), followed by feature-wise preprocessing using StandardScaler, RobustScaler, and MinMaxScaler depending on variable distribution characteristics. A PyTorch data pipeline was constructed using `TensorDataset` and `DataLoader`, pairing corrupted inputs with clean targets for supervised learning.

An LSTM-based sequence model was designed to learn temporal dependencies and cross-variable relationships. The model was trained using Mean Squared Error (MSE) loss and the Adam optimiser, with a validation split to monitor generalisation performance. No pre-trained networks were used.

After training, the model was applied to the corrupted test dataset to predict missing values. Only NaN entries were replaced, while existing observed values were preserved. The completed dataset was exported as `test_set_nogaps.csv` with the same format, column ordering, and row structure as the original test file.

## Model Configuration
- Architecture: LSTM-based sequence model  
- Input dimension: 8 weather variables  
- Hidden size: 128  
- Number of layers: 1  
- Loss function: MSE  
- Optimiser: Adam  

## Dependencies
Python 3.x, PyTorch, NumPy, Pandas, Matplotlib, and Scikit-learn.

## Reproducibility
To reproduce the results, ensure the datasets are placed in the correct directory and execute all cells in `Assessment.ipynb` sequentially. The imputed test dataset will be generated automatically.

## Sources
Implementation references include PyTorch and Scikit-learn official documentation, general resources on LSTM-based time-series modelling, and ChatGPT (used for drafting structure and debugging support).
