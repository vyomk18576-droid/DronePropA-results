# Drone Propeller Fault Classification (Nested 5×5 CV: 1D CNN & Dual TF-CNN)

PyTorch implementations for multi-class UAV propeller fault diagnosis from `QDrone_data` `.mat` flight logs in the **DronePropA Motion Trajectories Dataset**, evaluated under strict flight-file-level **Nested 5×5 Stratified Cross-Validation**.

## Dataset

- **Dataset:** [DronePropA: Motion Trajectories Dataset for Commercial Drones with Defective Propellers](https://data.mendeley.com/datasets/ftdyxrr3c5/1)
- **Scope:** 130 flight recordings sampled at 1 kHz, segmented into 9,992 non-overlapping 1-second windows (`window_len = 1000`).
- **Fault Classes:**
  - `F0 (Healthy)` — 40 flights
  - `F1 (Edge Cut)` — 30 flights
  - `F2 (Crack)` — 30 flights
  - `F3 (Surface Cut)` — 30 flights

## Architectures Included

1. **Single-Channel Causal 1D CNN Baseline (Channel 31 — Accel Z)**
   - Uses `CausalConv1d` (`bias=True`), `ChannelLayerNorm1d` (`elementwise_affine=False`), `ReLU`, `AvgPool1d`, `AdaptiveAvgPool1d`, and a linear classification head (`num_classes = 4`).
   - Applies per-window DC mean removal (`window_len = 1000`) and trains with Adam (`lr = 0.01`, `weight_decay = 1e-3`, `batch_size = 512`, `50` epochs).
   - Supports three filter-width variants: Model 1 `(16, 32)`, Model 2 `(32, 64)`, and Model 3 `(64, 128)`.

2. **13-Channel Dual-Branch Time-Frequency CNN (Dual TF-CNN)**
   - Uses 13 correlation-pruned telemetry channels: Gyroscope #1 (Roll, Pitch, Yaw), Accelerometer #1 (X, Y, Z), Accelerometer #2 (X, Y, Z), and 4 Motor Commands (`[29, 30, 31, 26, 27, 28, 35, 36, 37, 46, 48, 50, 52]`).
   - Applies global per-flight DC mean removal prior to windowing and pairs each temporal window with its length-normalized FFT magnitude spectrum (`np.abs(np.fft.fft(...)) / window_len`).
   - Processes temporal dynamics via a causal 1D-CNN branch and spectral harmonics via a parallel symmetric 1D-CNN branch with channel-wise `LayerNorm` and `bias=True`, followed by Global Average Pooling and feature fusion (`lr = 0.001`).

## Evaluation Protocol (Nested 5×5 Stratified CV)

To prevent temporal window leakage across train, validation, and test splits, all partitions are performed strictly at the **`.mat` flight-file level** (25 total runs per configuration):

- **Outer 5-Fold Split:** 104 development flights (80%) / 26 completely unseen test flights (20%) per fold, ensuring every flight is tested out-of-fold once.
- **Inner 5-Fold Split:** ~83 training flights / ~21 validation flights inside each 104-flight development set. The lowest validation-loss checkpoint is saved per inner fold, and all 5 inner checkpoints are evaluated individually and combined via softmax probability averaging (5-model ensemble) on the 26 held-out test flights.

## Nested 5×5 CV Results Summary

| Architecture Family | Model | Channels | Params | Single-Model Acc. (%) | 5-Model Ens. Acc. (%) | Macro Prec. | Macro Rec. | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Single-Channel)** | Model 1 | 1 | 2,852 | 62.74 ± 5.46 | 65.51 ± 6.10 | 0.6642 | 0.6465 | 0.6513 |
| **1D CNN (Single-Channel)** | Model 2 | 1 | 10,820 | 64.90 ± 5.84 | 67.85 ± 6.03 | 0.6867 | 0.6626 | 0.6690 |
| **1D CNN (Single-Channel)** | Model 3 | 1 | 42,116 | 63.85 ± 5.31 | 66.89 ± 6.25 | 0.6790 | 0.6597 | 0.6648 |
| **Dual-Branch TF-CNN** | Proposed | 13 | 77,700 | **80.49 ± 2.97** | **83.58 ± 3.17** | **0.8306** | **0.8315** | **0.8309** |

## Generated Outputs & Plots

Each Nested 5×5 CV run automatically generates and saves 300 DPI evaluation figures:
- **Learning Curves:** Mean ± 1 Std training and validation loss and accuracy across all 25 runs.
- **Confusion Matrix:** Aggregated out-of-fold predictions across all 130 unseen flights (9,992 windows).
- **Classification Report Heatmap:** Per-class precision, recall, F1-score, macro avg, and weighted avg.
- **t-SNE Projection:** 2D latent feature clusters extracted from the lowest-validation-loss checkpoint evaluated on its unseen 26-flight outer test fold.
- **Benchmarking against existing research paper:** Refer to the uploaded "benchmark comparisons.png"
- **Research paper link:** https://ieeexplore.ieee.org/document/11576037

- **Published Univariate Benchmarks vs. Strict Nested CV:** Martinez-Ríos et al. reported $63.01\%$ (WST-SVM) and $68.20\%$ / $69.75\%$ / $68.10\%$ (2-layer causal 1D-CNN Models 1–3) accuracy on single-axis z-axis acceleration under window-level splits. Under strict flight-level Nested $5 \times 5$ CV (zero shared flights across train and test), single-channel 1D-CNN Models 1–3 average $62.74\% \pm 5.46\%$, $64.90\% \pm 5.84\%$, and $63.85\% \pm 5.31\%$ single-model accuracy ($65.51\% \pm 6.10\%$, $67.85\% \pm 6.03\%$, and $66.89\% \pm 6.25\%$ with 5-model ensembling).  

- **Effect of Window-Level Split Leakage:** Random or window-level $80/20$ splitting ($67.38\%$ accuracy for Model 1, $70.94\%$ accuracy for Model 2, and $73.19\%$ accuracy for Model 3 on single-channel input) allows adjacent $1\text{ s}$ windows from the same flight to appear across train and test sets, inflating accuracy relative to unseen-flight Nested CV.

- **13-Channel Dual TF-CNN Gain on Unseen Flights:** Coupling multi-axis gyroscope, accelerometer and motor-command telemetry with time-frequency branches reaches $80.49\% \pm 2.97\%$ single-model and $83.58\% \pm 3.17\%$ 5-model ensemble accuracy ($0.8309$ macro F1) across all $130$ unseen flights.

- **13-Channel Dual TF-CNN vs. Single-Channel Baselines:** Lifting macro avg F1 from 0.6513 (Model 1), 0.6690 (Model 2), and 0.6648 (Model 3) on single-channel Channel 31 to 0.8309 (0.8306 macro precision, 0.8315 macro recall) on the 13-channel Dual TF-CNN demonstrates that multi-axis gyroscope. accelerometer and motor-command channels paired with time-frequency modeling  improve balanced fault detection across unseen flights. 

- **Balanced Precision and Recall:** Almost equal macro avg precision (0.8306) and recall (0.8315) on the 13-channel Dual TF-CNN indicate consistent fault discrimination across classes without heavy false-positive or false-negative skew on unseen flight recordings.  

- **Class Separability Across Health States:** High F1-scores on F0 (Healthy) (0.8932) and F1 (Edge Cut) (0.8828) confirm clear separability for healthy flight and blade-edge defects, whereas lower F1-scores on F2 (Crack) (0.7935) and F3 (Surface Cut) (0.7540) highlight overlap between subtle structural damage signatures. 

- **Unweighted Four-Class Benchmark:** Because DronePropA has more healthy recordings (40 flights, 3,063 windows) than each fault class (30 flights each, 2,302–2,318 windows), macro avg metrics confirm that gains hold across all four classes rather than being driven solely by healthy windows.   

## Dataset Setup

1. Download the DronePropA `.mat` files containing `QDrone_data` and place them in a local directory (e.g., `./data` or `/content/drone_data` on Google Colab).
2. Update `drone_dir` in the configuration cell to match your dataset path:

```python
drone_dir = r"./data"