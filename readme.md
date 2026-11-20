# Drone Propeller Fault Classification (1D CNN & Dual TF-CNN)

Jupyter notebooks for classifying drone propeller states from `QDrone_data` `.mat` files in the **DronePropA Motion Trajectories Dataset** using PyTorch.

## Dataset

- **Dataset:** [DronePropA: Motion Trajectories Dataset for Commercial Drones with Defective Propellers](https://data.mendeley.com/datasets/ftdyxrr3c5/1)

## Notebooks & Approaches

1. **1D CNN (`1dcnn.ipynb`)**
   - Uses `CausalConv1d`, `ChannelLayerNorm1d`, `ReLU`, pooling layers, and a linear classification head (`num_classes = 4`).
   - Segments fixed-length windows (`window_len = 1000`) with per-window mean removal and supports time-domain as well as FFT magnitude (`np.abs(np.fft.fft(...)) / window_len`) representations.
   - Includes channel correlation inspection (`plot_named_channels_correlation`) across mapped telemetry channels (`IMUs`, gyroscopes, accelerometers, battery level, height range, and motor/ESC commands).

2. **Dual TF-CNN (Time-Frequency CNN)**
   - Combines time-domain sensor windows and frequency-domain representations to capture both temporal dynamics and spectral fault signatures across propeller states.
   - Evaluates model performance using stratified train/test splits, learning curves, confusion matrices, and per-class classification metrics.

## Fault Classes

- `F0 (Healthy)`
- `F1 (Edge Cut)`
- `F2 (Crack)`
- `F3 (Surface Cut)`

## Dataset Setup

1. Download or place the DronePropA `.mat` files containing `QDrone_data` in a local directory (such as `./data`).
2. Update `drone_dir` in the notebook configuration cell to match your local dataset path:

```python
drone_dir = r"./data"
```

## Requirements

- Python 3.x
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `torch`
- `scikit-learn`

## Evaluation Metrics & Plots Guide

- **Correlation Plot (Pearson):** Measures linear relationships between channels from `-1` to `+1` (`+1` for strong positive co-movement, `-1` for strong negative co-movement, and `0` for little or no linear relationship).
- **Confusion Matrix:** Compares true class labels (rows) against predicted class labels (columns). Diagonal entries show correct classifications, while off-diagonal entries reveal which fault types are confused with one another.
- **Precision:** The fraction of predicted instances for a class that are actually correct (helps assess false positives).
- **Recall:** The fraction of actual instances of a class that the model correctly identifies (helps assess false negatives).
- **F1-Score:** The harmonic mean of precision and recall for each class.
- **Support:** The actual number of true test samples belonging to each class in the evaluation split.
- **Accuracy, Macro Avg, and Weighted Avg:** Overall accuracy across all test samples, the unweighted mean across classes (`macro avg`), and the sample-weighted mean across classes (`weighted avg`).

## Results

- **DronePropA Results:** All the outputs and plots are present in the DronePropA CNN Results pdf file