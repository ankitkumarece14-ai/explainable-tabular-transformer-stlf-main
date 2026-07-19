# ⚡  Residential Load Forecasting Using Explainable Tabular Transformers

> **Companion Code Repository** — This repository contains the complete source code, datasets, experimental results, and reproducibility artifacts for the journal paper:
>
> **" Residential Load Forecasting Using Explainable Tabular Transformers"**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Methodology](#methodology)
  - [Feature Engineering](#feature-engineering)
  - [Models Implemented](#models-implemented)
  - [Training Pipeline](#training-pipeline)
- [Results Summary](#results-summary)
  - [UCI Dataset Results](#uci-dataset-results)
  - [Custom IoT Dataset Results](#custom-iot-dataset-results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Citation](#citation)
- [License](#license)

---


**On the custom IoT dataset, FT-Transformer achieved R² = 0.9676, RMSE = 1.0873, MAE = 0.8656, and MAPE = 1.47%.** **On the UCI benchmark, TabPFN achieved R² = 0.9955 and RMSE = 0.0209.**

---

## Repository Structure

```
.
├── README.md                              # This file
│
├── UCI_full.PY                            # Full pipeline for UCI dataset
├── Custom_full.py                         # Full pipeline for Custom IoT dataset
├── mycsv.py                               # Feature CSV generation utility
│
├── household_power_consumption.txt        # UCI dataset (raw)
├── hostel_Data.csv                        # Custom hostel IoT dataset (raw)
│
├── UCI_result/                            # UCI experiment outputs
│   ├── comparison.csv                     #   Model comparison table
│   ├── hyperparams/                       #   All model hyperparameters (JSON)
│   │   ├── all_hyperparams.json
│   │   ├── FT-T_*_hyperparams.json
│   │   ├── TabNet_*_hyperparams.json
│   │   ├── LSTM_hyperparams.json
│   │   ├── BiLSTM_hyperparams.json
│   │   ├── TabPFN_hyperparams.json
│   │   └── hyperparams_summary.md
│   ├── predictions/                       #   Per-model prediction CSVs
│   │   ├── all_model_predictions.csv
│   │   ├── detailed_comparison_with_hyperparams.csv
│   │   └── <ModelName>_predictions.csv    #   (16 individual model files)
│   ├── plots/                             #   Visualization outputs
│   │   ├── all_scatter.png                #     Actual vs Predicted scatter
│   │   ├── metrics_comparison.png         #     Bar chart of R², RMSE, MAE
│   │   ├── model_ranking.png              #     Overall model ranking
│   │   ├── time_series_top6.png           #     Time-series overlay (top 6)
│   │   ├── ft_transformer_comparison.png  #     FT-Transformer variants
│   │   └── training_curves.png            #     Loss convergence curves
│   └── shap/                              #   SHAP explainability outputs
│       ├── shap_summary_LightGBM.png
│       ├── shap_summary_RandomForest.png
│       ├── shap_bar_LightGBM.png
│       ├── shap_bar_RandomForest.png
│       └── shap_values_*.csv
│
└── Custom_Results/                        # Custom IoT dataset experiment outputs
    ├── comparison.csv                     #   Model comparison table
    ├── hyperparams/                       #   All model hyperparameters (JSON)
    ├── predictions/                       #   Per-model prediction CSVs
    └── plots/                             #   Visualization outputs
        ├── all_scatter.png
        ├── metrics_comparison.png
        ├── model_ranking.png
        ├── time_series_top6.png
        ├── ft_transformer_comparison.png
        ├── residuals_distribution.png
        └── training_curves.png
```

---

## Datasets

### 1. UCI Household Power Consumption (Public Benchmark)

| Property | Value |
|---|---|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) |
| **Records** | ~2,075,259 (minute-level) |
| **Duration** | Dec 2006 – Nov 2010 (~47 months) |
| **Target** | `Global_active_power` (kW) |
| **Resampled** | Hourly (`h`) for experiments |
| **Features Used** | 19 engineered features (temporal, lag-24, rolling-48, power factor, voltage, reactive power) |

### 2. Custom Hostel IoT Dataset (Real-World Edge Deployment)

| Property | Value |
|---|---|
| **Source** | Custom collection via Raspberry Pi + Hall-effect current sensors + SX1262 LoRa |
| **Environment** | Residential hostel under realistic deployment conditions |
| **Raw Columns** | `IST_TIME`, `S1_IRMS`, `S1_PEAK`, `S1_EST_APPARENT_POWER`, `S2_IRMS`, `S2_PEAK`, `S2_EST_APPARENT_POWER` |
| **Target** | `TOTAL_POWER = S1_EST_APPARENT_POWER + S2_EST_APPARENT_POWER` (VA) |
| **Features Used** | 12 DL-optimized engineered features |
| **Challenges** | Sensor noise, packet loss, clock drift, temporal irregularities |

---

## Methodology

### Feature Engineering

#### Custom IoT Dataset — 12 Engineered Features

| # | Feature | Formula | Category |
|---|---------|---------|----------|
| 1 | `log_s1_irms` | log(1 + S1\_IRMS) | Log-scaled current |
| 2 | `log_s2_irms` | log(1 + S2\_IRMS) | Log-scaled current |
| 3 | `s1_irms_diff` | S1\_IRMS(t) − S1\_IRMS(t−1) | Differential |
| 4 | `s2_irms_diff` | S2\_IRMS(t) − S2\_IRMS(t−1) | Differential |
| 5 | `power_imbalance` | (S1 − S2) / (S1 + S2 + ε) | Power imbalance |
| 6 | `s1_peak_to_irms` | S1\_PEAK / S1\_IRMS | Crest factor |
| 7 | `s2_peak_to_irms` | S2\_PEAK / S2\_IRMS | Crest factor |
| 8 | `irms_ratio` | S1\_IRMS / S2\_IRMS | Current ratio |
| 9 | `s1_volatility` | std(diff(S1\_IRMS), window=5) | Volatility |
| 10 | `total_volatility` | s1\_vol + s2\_vol | Volatility |
| 11 | `sin_time` | sin(2π · hour/24) | Cyclical temporal |
| 12 | `cos_time` | cos(2π · hour/24) | Cyclical temporal |

#### UCI Dataset — 19 Engineered Features

Includes temporal features (hour, day\_of\_week, month, quarter, is\_weekend), cyclical encodings (sin/cos for hour, day, month), lag features (lag\_24), rolling statistics (mean, std, min, max at window=48), power factor, voltage, and global reactive power.

> ⚠️ **No data leakage**: All features use only present or past information. Chronological train/validation/test splitting is enforced throughout. No target-derived features or future look-ahead.

### Models Implemented

| Category | Model | Key Characteristics |
|----------|-------|---------------------|
| **Traditional ML** | Linear Regression | Baseline linear model |
| | Ridge / Lasso | L2 / L1 regularized linear models |
| | Decision Tree | Non-linear, interpretable |
| | Random Forest | Bagging ensemble of trees |
| | Gradient Boosting | Sequential boosting (sklearn) |
| | XGBoost | Optimized gradient boosting |
| | LightGBM | Histogram-based gradient boosting + Optuna tuning |
| | SVR | Support vector regression (RBF kernel) |
| | KNN | Distance-weighted k-nearest neighbors |
| | MLP (sklearn) | Feedforward neural network |
| **Deep Learning** | MLP (PyTorch) | Custom MLP with BatchNorm + Dropout |
| | LSTM | Long Short-Term Memory with self-attention |
| | BiLSTM | Bidirectional LSTM |
| | GRU | Gated Recurrent Unit |
| **Tabular DL** | FT-Transformer | Feature Tokenizer + Transformer (Small, SmallDeep, Medium, MediumWide, Best) |
| | TabNet | Attentive tabular network with learned sparsity (Small, Large) |
| | TabPFN | Pre-trained tabular foundation model (zero-shot) |

### Training Pipeline

```
Raw Sensor Data → Preprocessing → Temporally Causal Feature Engineering → StandardScaler
                                                     ↓
                                    70/15/15 Chronological Split
                                                     ↓
                                    ┌─────────────────────────────────────────┐
                                    │  Model Training                        │
                                    │  • Optuna HPO (LightGBM, TabNet, MLP)  │
                                    │  • Early Stopping + LR Scheduling      │
                                    │  • Mixed Precision (AMP) for FT-T      │
                                    │  • GPU Memory Management               │
                                    └─────────────────────────────────────────┘
                                                     ↓
                               Evaluation (R², RMSE, MAE, MAPE, MASE)
                                                     ↓
                            ┌──────────────┬──────────────┬──────────────┐
                            │  SHAP (XAI)  │  Ablation    │  Deployment  │
                            │  Analysis    │  Studies     │  Efficiency  │
                            └──────────────┴──────────────┴──────────────┘
```

- **Data Split**: Chronological 70/15/15 (train / validation / test) — no random shuffling to respect temporal ordering
- **Scaling**: `StandardScaler` applied to both features (X) and target (y)
- **Evaluation Metrics**: R², RMSE, MAE, MAPE, MASE
- **Reproducibility**: `random_state=42`, `torch.manual_seed(42)`, `np.random.seed(42)`

---

## Results Summary

### UCI Dataset Results

Hourly Global Active Power Forecasting on test set:

| Rank | Model | R² | RMSE | MAE | MAPE (%) | Training Time (s) |
|------|-------|----|------|-----|----------|-------------------|
| 🥇 | **TabPFN** | **0.9955** | **0.0209** | **0.0042** | **0.60** | 0.33 |
| 🥈 | FT-T\_SmallDeep | 0.9939 | 0.0243 | 0.0167 | 2.23 | 2.07 |
| 🥉 | FT-T\_Medium | 0.9840 | 0.0395 | 0.0216 | 2.90 | 1.90 |
| 4 | FT-T\_Small | 0.9824 | 0.0414 | 0.0235 | 3.30 | 0.99 |
| 5 | FT-T\_MediumWide | 0.9823 | 0.0415 | 0.0289 | 3.75 | 0.92 |
| 6 | FT-T\_Best | 0.9578 | 0.0641 | 0.0236 | 3.52 | 3.50 |
| 7 | MLP\_sklearn | 0.9349 | 0.0797 | 0.0575 | 6.75 | 2.15 |
| 8 | TabNet\_Large | 0.8926 | 0.1024 | 0.0670 | 10.14 | 22.34 |
| 9 | MLP\_PyTorch | 0.8680 | 0.1135 | 0.0808 | 9.96 | 1.02 |
| 10 | LightGBM | 0.8449 | 0.1230 | 0.0483 | 6.11 | 0.16 |
| 11 | TabNet\_Small | 0.8206 | 0.1323 | 0.0818 | 13.02 | 5.40 |
| 12 | RandomForest | 0.7958 | 0.1412 | 0.1003 | 13.79 | 0.13 |
| 13 | DecisionTree | 0.7199 | 0.1653 | 0.0911 | 11.83 | 0.005 |
| 14 | LinearRegression | 0.5007 | 0.2207 | 0.1693 | 21.30 | 0.001 |
| 15 | BiLSTM | 0.4445 | 0.2376 | 0.1837 | 27.42 | 1.16 |
| 16 | LSTM | 0.3710 | 0.2528 | 0.1940 | 28.37 | 0.73 |

### Custom IoT Dataset Results

Total Power (S1 + S2) Forecasting on test set:

| Rank | Model | R² | RMSE | MAE | MAPE (%) | Training Time (s) |
|------|-------|----|------|-----|----------|-------------------|
| 🥇 | **FT-T\_Best** | **0.9747** | **0.960** | **0.775** | **1.30** | 77.77 |
| 🥈 | MLP\_PyTorch | 0.9482 | 1.375 | 0.949 | 1.58 | 7.89 |
| 🥉 | MLP\_sklearn | 0.9406 | 1.471 | 1.124 | 1.84 | 1.82 |
| 4 | FT-T\_Small | 0.9384 | 1.499 | 1.201 | 1.97 | 2.32 |
| 5 | FT-T\_Medium | 0.9201 | 1.707 | 1.292 | 2.13 | 7.91 |
| 6 | RandomForest | 0.8297 | 2.492 | 1.641 | 2.75 | 0.30 |
| 7 | TabNet\_Small | 0.8055 | 2.663 | 1.939 | 3.30 | 92.39 |
| 8 | DecisionTree | 0.7684 | 2.906 | 1.988 | 3.27 | 0.03 |
| 9 | LinearRegression | 0.6450 | 3.597 | 2.338 | 3.69 | 0.002 |
| 10 | BiLSTM | −0.032 | 6.142 | 4.210 | 6.74 | 9.27 |
| 11 | LSTM | −0.202 | 6.628 | 4.612 | 7.30 | 12.18 |
| 12 | LightGBM | −1.438 | 9.427 | 3.050 | 4.71 | 0.12 |

### Key Findings

- **FT-Transformer** consistently achieves top-tier performance on both datasets, demonstrating the effectiveness of tabular transformer architectures for residential energy forecasting
- **TabPFN** achieves the highest R² (0.9955) on the UCI benchmark — a pre-trained foundation model requiring minimal training time (~0.33 s)
- **Tabular DL models** (FT-Transformer, TabPFN) outperform traditional gradient boosting and tree-based models on both datasets under temporally causal feature engineering
- **LSTM/BiLSTM** underperform compared to tabular models, suggesting that feature-tokenization via multi-head self-attention captures non-linear electrical relationships more effectively than sequential recurrence for this task
- **Domain-specific feature engineering** (crest factors, volatility, power imbalance, cyclical encodings) is critical to enabling deep learning superiority over simpler models

---

## Installation & Setup

### Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU recommended (CPU supported)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm
pip install pytorch-tabnet
pip install rtdl-revisiting-models      # FT-Transformer
pip install tabpfn                       # TabPFN
pip install optuna                       # Hyperparameter optimization
pip install shap                         # Explainability
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Run UCI dataset experiment
python UCI_full.PY --data household_power_consumption.txt --freqs h

# Run Custom IoT dataset experiment
python Custom_full.py

# Generate feature CSV only
python mycsv.py
```

---

## Usage

### UCI Dataset Pipeline

```bash
# Full run with Optuna tuning, ablation study, and SHAP
python UCI_full.PY --data household_power_consumption.txt

# Without Optuna tuning (faster)
python UCI_full.PY --data household_power_consumption.txt --no-optuna

# Without ablation study
python UCI_full.PY --data household_power_consumption.txt --no-ablation

# Without SHAP XAI analysis
python UCI_full.PY --data household_power_consumption.txt --no-xai

# Generate CSV templates only
python UCI_full.PY --generate-templates
```

### Custom IoT Dataset Pipeline

```bash
# Full run (all models: FT-Transformer × 3 variants, TabNet, LSTM, BiLSTM, MLP, Traditional ML)
python Custom_full.py
```

### Feature Dataset Generation

```bash
# Generate 12-feature dataset CSV with documentation
python mycsv.py
```

This produces:
- `DATASET_12FEATURES_AND_TARGET.csv` — All 12 features + target + train/val/test split labels
- `FEATURE_DOCUMENTATION.txt` — Detailed feature descriptions, formulas, and raw-column mappings

---

## Visualizations

All generated plots are saved in the `plots/` subdirectory under each experiment result:

| Plot | Description |
|------|-------------|
| `all_scatter.png` | Actual vs Predicted scatter plots for all models |
| `metrics_comparison.png` | Side-by-side bar charts of R², RMSE, MAE, MAPE |
| `model_ranking.png` | Overall model ranking by R² score |
| `time_series_top6.png` | Time-series overlay of top 6 model predictions |
| `ft_transformer_comparison.png` | Comparison across FT-Transformer variants |
| `training_curves.png` | Training/validation loss convergence curves |
| `residuals_distribution.png` | Residual error distributions per model |

### SHAP Explainability (UCI dataset)

| Plot | Description |
|------|-------------|
| `shap_summary_*.png` | SHAP beeswarm summary plots showing feature impact direction |
| `shap_bar_*.png` | SHAP global mean feature importance bar charts |
| `shap_values_*.csv` | Raw SHAP values for further analysis |

---


--

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) for the Household Power Consumption dataset
- [rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models) for the FT-Transformer implementation
- [TabPFN](https://github.com/automl/TabPFN) for the pre-trained tabular foundation model
- [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet) for the TabNet implementation
- [SHAP](https://github.com/shap/shap) for model interpretability
- [Optuna](https://github.com/optuna/optuna) for Bayesian hyperparameter optimization
- Indian Institute of Technology Patna and NIELIT Patna

---

<p align="center">
  <i>For questions or issues, please open a GitHub Issue or contact the corresponding author at <b>ankit.kumar@iitp.ac.in</b></i>
</p>

