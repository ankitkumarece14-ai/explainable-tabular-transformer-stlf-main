"""
================================================================================
COMPLETE ENSEMBLE: S1+S2 Power Prediction
Generates: 12 features + target CSV (no model predictions)
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")


# ============================================================
# PART 1: DATA LOADING
# ============================================================

def load_and_prepare_data(data_path):
    """Load data."""
    print(f"\nLoading data from {data_path}...")
    
    df = pd.read_csv(data_path, sep=',', engine='python')
    df.columns = df.columns.str.strip()
    
    datetime_col = 'IST_TIME' if 'IST_TIME' in df.columns else df.columns[0]
    
    try:
        df['datetime'] = pd.to_datetime(df[datetime_col], format='%d-%m-%Y %H:%M')
    except:
        df['datetime'] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
    
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df.drop([datetime_col], axis=1, errors='ignore')
    
    numeric_cols = ['S1_IRMS', 'S1_PEAK', 'S1_EST_APPARENT_POWER',
                    'S2_IRMS', 'S2_PEAK', 'S2_EST_APPARENT_POWER']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.interpolate(method='time', limit=10).ffill().bfill()
    
    df['TOTAL_POWER'] = df['S1_EST_APPARENT_POWER'] + df['S2_EST_APPARENT_POWER']
    
    print(f"Loaded {len(df):,} rows")
    return df


# ============================================================
# PART 2: FEATURE ENGINEERING (12 FEATURES)
# ============================================================

def engineer_features(df, target_col='TOTAL_POWER'):
    """Create 12 features."""
    df = df.copy()
    
    # Raw values
    s1_irms = df['S1_IRMS']
    s1_peak = df['S1_PEAK']
    s2_irms = df['S2_IRMS']
    s2_peak = df['S2_PEAK']
    
    # 1-2: Log currents
    df['log_s1_irms'] = np.log1p(s1_irms)
    df['log_s2_irms'] = np.log1p(s2_irms)
    
    # 3-4: Differences
    df['s1_irms_diff'] = s1_irms.diff().fillna(0)
    df['s2_irms_diff'] = s2_irms.diff().fillna(0)
    
    # 5: Power imbalance
    df['power_imbalance'] = (s1_irms - s2_irms) / (s1_irms + s2_irms + 1e-8)
    
    # 6-7: Crest factors
    df['s1_peak_to_irms'] = s1_peak / (s1_irms + 1e-8)
    df['s2_peak_to_irms'] = s2_peak / (s2_irms + 1e-8)
    
    # 8: Current ratio
    df['irms_ratio'] = s1_irms / (s2_irms + 1e-8)
    
    # 9-10: Volatility
    s1_shifted = s1_irms.shift(1)
    df['s1_volatility'] = s1_shifted.diff().rolling(window=5, min_periods=1).std().fillna(0)
    s2_shifted = s2_irms.shift(1)
    s2_vol = s2_shifted.diff().rolling(window=5, min_periods=1).std().fillna(0)
    df['total_volatility'] = df['s1_volatility'] + s2_vol
    
    # 11-12: Temporal
    time_decimal = df.index.hour + df.index.minute / 60.0
    df['sin_time'] = np.sin(2 * np.pi * time_decimal / 24)
    df['cos_time'] = np.cos(2 * np.pi * time_decimal / 24)
    
    # Drop originals
    drop_cols = ['S1_IRMS', 'S1_PEAK', 'S1_EST_APPARENT_POWER',
                 'S2_IRMS', 'S2_PEAK', 'S2_EST_APPARENT_POWER']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df = df.dropna()
    return df


# 12 features list
FEATURES_12 = [
    'log_s1_irms', 'log_s2_irms',
    's1_irms_diff', 's2_irms_diff',
    'power_imbalance',
    's1_peak_to_irms', 's2_peak_to_irms',
    'irms_ratio',
    's1_volatility', 'total_volatility',
    'sin_time', 'cos_time'
]


# ============================================================
# PART 3: PREPARE DATA
# ============================================================

def prepare_data(df):
    """Prepare data."""
    print(f"\n12 FEATURES:")
    for i, f in enumerate(FEATURES_12, 1):
        print(f"  {i:2d}. {f}")
    
    X = df[FEATURES_12].values
    y = df['TOTAL_POWER'].values
    
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    timestamps = df.index[n_train + n_val:]
    
    print(f"\nSplit: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return {
        'X_train': X_train_s, 'y_train': y_train_s, 'y_train_orig': y_train,
        'X_val': X_val_s, 'y_val': y_val_s, 'y_val_orig': y_val,
        'X_test': X_test_s, 'y_test': y_test_s, 'y_test_orig': y_test,
        'scaler_y': scaler_y, 'timestamps': timestamps,
        'X_train_raw': X_train, 'X_val_raw': X_val, 'X_test_raw': X_test,
        'df_full': df  # Keep full dataframe for CSV generation
    }


# ============================================================
# PART 4: SIMPLE CSV GENERATION (12 FEATURES + TARGET)
# ============================================================

def generate_final_csv(data_dict, output_dir):
    """
    Generate CSV with 12 features + target for ALL data (train+val+test)
    """
    print(f"\n{'='*70}")
    print("GENERATING FINAL CSV: 12 FEATURES + TARGET")
    print(f"{'='*70}")
    
    df_full = data_dict['df_full']
    
    # Create output dataframe
    out_df = pd.DataFrame()
    out_df['Timestamp'] = df_full.index
    
    # Add 12 features with clear names
    for i, feat in enumerate(FEATURES_12, 1):
        out_df[f'F{i:02d}_{feat}'] = df_full[feat].values
    
    # Add target
    out_df['Target_Total_Power'] = df_full['TOTAL_POWER'].values
    
    # Add split indicator
    n = len(df_full)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    split_col = ['Train'] * n_train + ['Validation'] * n_val + ['Test'] * (n - n_train - n_val)
    out_df['Split'] = split_col
    
    # Save
    csv_path = os.path.join(output_dir, 'DATASET_12FEATURES_AND_TARGET.csv')
    out_df.to_csv(csv_path, index=False)
    
    print(f"[SAVE] {csv_path}")
    print(f"       Rows: {len(out_df)}, Columns: {len(out_df.columns)}")
    print(f"       Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")
    
    return out_df


def generate_documentation(output_dir):
    """Generate feature documentation."""
    doc = """12 FEATURES FOR S1+S2 POWER PREDICTION
==========================================

TARGET VARIABLE:
Target_Total_Power = S1_EST_APPARENT_POWER + S2_EST_APPARENT_POWER
- Combined apparent power from both sources (VA)

FEATURES (12):

F01_log_s1_irms:
  Formula: log(1 + S1_IRMS)
  Raw: S1_IRMS (RMS current source 1)
  Meaning: Log-scaled current from source 1

F02_log_s2_irms:
  Formula: log(1 + S2_IRMS)
  Raw: S2_IRMS (RMS current source 2)
  Meaning: Log-scaled current from source 2

F03_s1_irms_diff:
  Formula: S1_IRMS(t) - S1_IRMS(t-1)
  Raw: Consecutive S1_IRMS values
  Meaning: Current change rate for source 1

F04_s2_irms_diff:
  Formula: S2_IRMS(t) - S2_IRMS(t-1)
  Raw: Consecutive S2_IRMS values
  Meaning: Current change rate for source 2

F05_power_imbalance:
  Formula: (S1_IRMS - S2_IRMS) / (S1_IRMS + S2_IRMS + epsilon)
  Raw: S1_IRMS, S2_IRMS
  Meaning: Load balance between sources [-1 to 1]
  Note: 0=balanced, +1=all S1, -1=all S2

F06_s1_peak_to_irms:
  Formula: S1_PEAK / S1_IRMS
  Raw: S1_PEAK (peak current), S1_IRMS
  Meaning: Crest factor (waveform shape) for source 1
  Note: ~1.414 for sine wave, higher for distorted

F07_s2_peak_to_irms:
  Formula: S2_PEAK / S2_IRMS
  Raw: S2_PEAK, S2_IRMS
  Meaning: Crest factor for source 2

F08_irms_ratio:
  Formula: S1_IRMS / (S2_IRMS + epsilon)
  Raw: S1_IRMS, S2_IRMS
  Meaning: Current ratio between sources

F09_s1_volatility:
  Formula: std(diff(S1_IRMS), window=5)
  Raw: S1_IRMS over 5 timesteps
  Meaning: Short-term variability of source 1

F10_total_volatility:
  Formula: s1_volatility + s2_volatility
  Raw: Calculated volatilities
  Meaning: Combined system variability

F11_sin_time:
  Formula: sin(2 * pi * (hour + minute/60) / 24)
  Raw: Timestamp
  Meaning: Time of day (cyclical)

F12_cos_time:
  Formula: cos(2 * pi * (hour + minute/60) / 24)
  Raw: Timestamp
  Meaning: Time of day (cyclical, orthogonal to F11)

RAW INPUT COLUMNS (original data):
- IST_TIME: Timestamp
- S1_IRMS: RMS current source 1 (Amps)
- S1_PEAK: Peak current source 1 (Amps)
- S1_EST_APPARENT_POWER: Apparent power source 1 (VA)
- S2_IRMS: RMS current source 2 (Amps)
- S2_PEAK: Peak current source 2 (Amps)
- S2_EST_APPARENT_POWER: Apparent power source 2 (VA)

DATASET SPLIT:
- Train: 70% (first portion)
- Validation: 15% (middle portion)
- Test: 15% (last portion)

NO DATA LEAKAGE:
- All features use only present or past information
- No target values used as features
- No future information
"""
    
    doc_path = os.path.join(output_dir, 'FEATURE_DOCUMENTATION.txt')
    with open(doc_path, 'w') as f:
        f.write(doc)
    
    print(f"[SAVE] Documentation: {doc_path}")
    return doc


# ============================================================
# PART 5: MAIN EXECUTION
# ============================================================

def main(data_path="hostel_Data.csv"):
    """Main function."""
    print("="*70)
    print("S1+S2 POWER PREDICTION - 12 FEATURES DATASET GENERATION")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dataset_12features_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput: {output_dir}")
    
    # Load and process data
    df = load_and_prepare_data(data_path)
    df = engineer_features(df)
    data = prepare_data(df)
    
    # Generate CSV and documentation
    generate_final_csv(data, output_dir)
    generate_documentation(output_dir)
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"Files created:")
    print(f"  1. DATASET_12FEATURES_AND_TARGET.csv")
    print(f"  2. FEATURE_DOCUMENTATION.txt")
    print(f"\nLocation: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    data_file = "hostel_Data.csv"  # Change to your file
    main(data_file)