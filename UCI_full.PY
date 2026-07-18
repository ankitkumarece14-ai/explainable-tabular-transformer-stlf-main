"""
================================================================================
COMPLETE ENSEMBLE: Traditional ML + Deep Learning + Advanced Features
Features: lag_6, lag_12, lag_24 + rolling_12, rolling_24, rolling_48
INCLUDES: power_factor, NO Global_intensity, NO lag_1, NO lag_3

ADDITIONAL FEATURES:
- Optuna Hyperparameter Tuning for all models
- LightGBM integration
- TabPFN integration
- SHAP for XAI (Explainable AI)
- Energy vs Accuracy Tradeoff Analysis
- Ablation Studies
- CSV generation for hour/minute/day
- Comprehensive visualization (Actual vs Predicted, Heatmaps, etc.)
- MEMORY-EFFICIENT FT-TRANSFORMER with Automatic GPU Cleanup
- Mixed Precision Training (AMP)
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import time
import os
import json
import gc
from datetime import datetime
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Traditional ML
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Optuna for Hyperparameter Tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

# SHAP for XAI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

# FT-Transformer
try:
    from rtdl_revisiting_models import FTTransformer
    FTTRANSFORMER_AVAILABLE = True
except ImportError:
    try:
        from rtdl import FTTransformer
        FTTRANSFORMER_AVAILABLE = True
    except ImportError:
        FTTRANSFORMER_AVAILABLE = False

# TabPFN
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

# ============================================================
# PART 1: DATA LOADING & PREPROCESSING
# ============================================================

def load_data(data_path="household_power_consumption.txt"):
    """Load UCI Household Power Consumption dataset."""
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(
        data_path,
        sep=';',
        na_values=['?', 'nan', 'NaN', ''],
        low_memory=False
    )
    
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df.drop(['Date', 'Time'], axis=1, errors='ignore')
    
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded {len(df):,} rows")
    
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.interpolate(method='time', limit=24)
        df = df.ffill(limit=24).bfill(limit=24)
        df = df.fillna(df.mean())
        print(f"Cleaned {missing:,} missing values")
    
    return df


def resample_data(df, freq):
    """Resample to specified frequency."""
    freq_map = {
        'H': 'h', 'D': 'd', 'MIN': 'min', 'Min': 'min',
        'min': 'min', 'h': 'h', 'd': 'd'
    }
    freq = 'h'
    agg_dict = {
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    df_resampled = df.resample(freq).agg(agg_dict).dropna()
    print(f"Resampled to {freq}: {len(df):,} -> {len(df_resampled):,} rows")
    return df_resampled


# ============================================================
# PART 2: FEATURE ENGINEERING
# ============================================================

def engineer_features_final(df, target_col='Global_active_power'):
    """
    FINAL FEATURE SET with requested lags and rolling windows:
    - Lags: 6, 12, 24 (medium to long term)
    - Rolling: 12, 24, 48 (multiple time scales)
    - NO lag_1, NO lag_3 (too recent - leakage)
    - NO Global_intensity (leakage)
    - INCLUDES power_factor
    """
    df = df.copy()
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df.index.quarter
    
    # Cyclical encoding
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # LAG FEATURES: 6, 12, 24
    for lag in [6, 12, 24]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # ROLLING STATISTICS: 12, 24, 48
    shifted = df[target_col].shift(1)
    for window in [12, 24, 48]:
        df[f'rolling_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = shifted.rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = shifted.rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = shifted.rolling(window=window, min_periods=1).max()
    
    # POWER FACTOR (includes Global_active_power and Global_reactive_power)
    df['power_factor'] = df[target_col] / np.sqrt(df[target_col]**2 + df['Global_reactive_power']**2 + 1e-8)
    
    df = df.dropna()
    return df


def get_final_features(df, target_col='Global_active_power'):
    """Get final feature list with lag_6, lag_12, lag_24, rolling_12,24,48, and power_factor."""
    features = [
        # Time features
        'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'quarter',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month',
        
        # Lag features: 6, 12, 24
        #'Global_active_power_lag_6',
        #'Global_active_power_lag_12',
        'Global_active_power_lag_24',
        
        # Rolling statistics: 12, 24, 48
        #'rolling_mean_12', 'rolling_std_12', 'rolling_min_12', 'rolling_max_12',
        #'rolling_mean_24', 'rolling_std_24', 'rolling_min_24', 'rolling_max_24',
        'rolling_mean_48', 'rolling_std_48', 'rolling_min_48', 'rolling_max_48',
        
        # Power factor
        'power_factor',
        
        # Independent electrical measurements
        'Voltage',
        'Global_reactive_power',
    ]
    
    feature_cols = [c for c in features if c in df.columns]
    
    # Leakage detection
    forbidden_exact = ['Global_intensity', 'total_sub_metering', 'TM']
    forbidden_contains = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    forbidden_endswith = ['_lag_1', '_lag_2', '_lag_3', '_lag_4', '_lag_5', 
                          '_lag_7', '_lag_8', '_lag_9', '_lag_10', '_lag_11',
                          '_lag_13', '_lag_14', '_lag_15', '_lag_16', '_lag_17',
                          '_lag_18', '_lag_19', '_lag_20', '_lag_21', '_lag_22',
                          '_lag_23', '_lag_25']
    
    for col in feature_cols:
        if col in forbidden_exact:
            raise ValueError(f"LEAKAGE (exact): {col}")
        for bad in forbidden_contains:
            if bad in col:
                raise ValueError(f"LEAKAGE (contains): {col}")
        for bad in forbidden_endswith:
            if col.endswith(bad):
                raise ValueError(f"LEAKAGE (lag): {col}")
    
    print(f"\nFinal features ({len(feature_cols)}): {feature_cols}")
    return feature_cols


def prepare_data(df, target_col='Global_active_power', train_ratio=0.7, val_ratio=0.15):
    """Prepare train/val/test splits."""
    feature_cols = get_final_features(df, target_col)
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Chronological split
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    # Scale
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return (X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s,
            y_train, y_val, y_test, scaler_y, feature_cols, X_train, X_val, X_test)


# ============================================================
# PART 3: METRICS CALCULATION (FROM SCRATCH)
# ============================================================

def calculate_metrics_from_scratch(y_true, y_pred):
    """Calculate all metrics manually without sklearn shortcuts."""
    n = len(y_true)
    
    # R² (Coefficient of Determination)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # MASE (Mean Absolute Scaled Error)
    # Scale error by naive forecast (previous value)
    naive_errors = np.abs(y_true[1:] - y_true[:-1])
    mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1e-8
    mase = mae / mae_naive if mae_naive != 0 else np.inf
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MASE': mase
    }


# ============================================================
# PART 4: TRADITIONAL ML WITH OPTUNA TUNING
# ============================================================

class TraditionalML:
    def __init__(self, n_jobs=-1, random_state=42, use_optuna=True):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.models = {}
        self.results = {}
        self.training_times = {}
        self.energy_scores = {}
        
    def init_models(self):
        self.models = {
            'LinearRegression': LinearRegression(n_jobs=self.n_jobs),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.001, random_state=self.random_state, max_iter=10000),
            'DecisionTree': DecisionTreeRegressor(max_depth=10, min_samples_split=10, 
                                                 min_samples_leaf=5, random_state=self.random_state),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                                   min_samples_split=5, n_jobs=self.n_jobs,
                                                   random_state=self.random_state, max_samples=0.8),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                         max_depth=5, random_state=self.random_state),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
            'KNN': KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=self.n_jobs),
            'MLP_sklearn': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                                        alpha=0.001, batch_size=256, max_iter=1000,
                                        early_stopping=True, random_state=self.random_state),
        }
        return self.models
    
    def optuna_tune_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optuna hyperparameter tuning for LightGBM."""
        if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
            return None
            
        print(f"\nOptuna tuning for LightGBM ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'verbose': -1,
                'random_state': self.random_state
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best LightGBM params: {study.best_params}")
        return study.best_params
    
    def fit_and_evaluate(self, X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
                        y_train, y_val, y_test, scaler_y, X_train_raw=None, X_val_raw=None, X_test_raw=None):
        self.init_models()
        
        # Add LightGBM with Optuna tuning
        if LIGHTGBM_AVAILABLE:
            if self.use_optuna:
                best_params = self.optuna_tune_lightgbm(X_train, y_train_s, X_val, y_val_s)
                if best_params:
                    self.models['LightGBM'] = lgb.LGBMRegressor(**best_params, verbose=-1, random_state=self.random_state)
            else:
                self.models['LightGBM'] = lgb.LGBMRegressor(
                    learning_rate=0.05, num_leaves=31, n_estimators=500,
                    verbose=-1, random_state=self.random_state
                )
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train_s)
            train_time = time.time() - start_time
            
            y_pred_train = scaler_y.inverse_transform(model.predict(X_train).reshape(-1, 1)).flatten()
            y_pred_val = scaler_y.inverse_transform(model.predict(X_val).reshape(-1, 1)).flatten()
            y_pred_test = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
            
            self.results[name] = {
                'train': calculate_metrics_from_scratch(y_train, y_pred_train),
                'val': calculate_metrics_from_scratch(y_val, y_pred_val),
                'test': calculate_metrics_from_scratch(y_test, y_pred_test)
            }
            self.training_times[name] = train_time
            
            # Energy efficiency score (inverse of training time normalized)
            self.energy_scores[name] = 1.0 / (train_time + 1e-6)
            
            print(f"  Test R²: {self.results[name]['test']['R2']:.4f}, Time: {train_time:.2f}s")
        
        return self.results, self.training_times, self.energy_scores


# ============================================================
# PART 5: DEEP LEARNING WITH OPTUNA TUNING
# ============================================================

class DeepLearningModels:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42, use_optuna=True):
        self.device = device
        self.random_state = random_state
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.results = {}
        self.training_times = {}
        self.energy_scores = {}
        print(f"Using device: {self.device}")
    
    class MLPModel(nn.Module):
        def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x).squeeze()
    
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            _, (h_n, _) = self.lstm(x)
            return self.fc(self.dropout(h_n[-1])).squeeze()
    
    class BiLSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, 1)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            _, (h_n, _) = self.lstm(x)
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
            return self.fc(self.dropout(h)).squeeze()
    
    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            _, h_n = self.gru(x)
            return self.fc(self.dropout(h_n[-1])).squeeze()
    
    def optuna_tune_mlp(self, X_train, y_train, X_val, y_val, input_dim, n_trials=30):
        """Optuna tuning for MLP hyperparameters."""
        if not OPTUNA_AVAILABLE:
            return [128, 64, 32], 0.2, 0.001
            
        print(f"\nOptuna tuning for MLP ({n_trials} trials)...")
        
        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_dims = []
            for i in range(n_layers):
                hidden_dims.append(trial.suggest_int(f'n_units_l{i}', 32, 256, log=True))
            
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            
            model = self.MLPModel(input_dim, hidden_dims, dropout).to(self.device)
            train_loader, val_loader, _ = self.create_loaders(X_train, y_train, X_val, y_val, X_val, y_val, 256)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
            # Quick training for evaluation
            for epoch in range(20):
                model.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(Xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    val_losses.append(criterion(model(Xb), yb).item())
            
            return np.mean(val_losses)
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best = study.best_params
        hidden_dims = [best[f'n_units_l{i}'] for i in range(best['n_layers'])]
        print(f"Best MLP config: hidden={hidden_dims}, dropout={best['dropout']:.3f}, lr={best['lr']:.5f}")
        return hidden_dims, best['dropout'], best['lr']
    
    def create_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=256):
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size),
                DataLoader(test_ds, batch_size=batch_size))
    
    def train_model(self, model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    val_losses.append(criterion(model(Xb), yb).item())
            
            avg_val = np.mean(val_losses)
            scheduler.step(avg_val)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: val_loss={avg_val:.6f}")
            
            if avg_val < best_loss:
                best_loss = avg_val
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        return model
    
    def evaluate(self, model, test_loader, y_test_orig, scaler_y):
        model.eval()
        preds = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(self.device)
                preds.extend(model(Xb).cpu().numpy())
        
        y_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return calculate_metrics_from_scratch(y_test_orig, y_pred)
    
    def fit_and_evaluate(self, X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
                        y_train, y_val, y_test, scaler_y, input_dim, epochs=100):
        train_loader, val_loader, test_loader = self.create_loaders(
            X_train, y_train_s, X_val, y_val_s, X_test, y_test_s)
        
        # Optuna tune MLP
        if self.use_optuna:
            hidden_dims, dropout, lr = self.optuna_tune_mlp(X_train, y_train_s, X_val, y_val_s, input_dim)
        else:
            hidden_dims, dropout, lr = [128, 64, 32], 0.2, 0.001
        
        models_config = [
            ('MLP_PyTorch', self.MLPModel(input_dim, hidden_dims, dropout)),
            ('LSTM', self.LSTMModel(input_dim, 64, 2, 0.2)),
            ('BiLSTM', self.BiLSTMModel(input_dim, 64, 2, 0.2)),
            ('GRU', self.GRUModel(input_dim, 64, 2, 0.2)),
        ]
        
        for name, model in models_config:
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            trained = self.train_model(model, train_loader, val_loader, epochs, lr, 15)
            train_time = time.time() - start_time
            
            self.results[name] = self.evaluate(trained, test_loader, y_test, scaler_y)
            self.training_times[name] = train_time
            self.energy_scores[name] = 1.0 / (train_time + 1e-6)
            
            print(f"  Test R²: {self.results[name]['R2']:.4f}, Time: {train_time:.2f}s")
        
        return self.results, self.training_times, self.energy_scores


# ============================================================
# PART 6: TABNET WITH OPTUNA TUNING
# ============================================================

class TabNetTuned:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_optuna=True):
        self.device = device
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.results = {}
        self.training_times = {}
        self.energy_scores = {}
        self.best_params = None
        
    def optuna_tune_tabnet(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Optuna hyperparameter tuning for TabNet."""
        if not OPTUNA_AVAILABLE:
            return {'n_d_a': 64, 'n_steps': 3, 'gamma': 1.5, 'lambda_sparse': 1e-4, 'lr': 2e-2}
            
        print(f"\nOptuna tuning for TabNet ({n_trials} trials)...")
        
        y_train_t = y_train.reshape(-1, 1)
        y_val_t = y_val.reshape(-1, 1)
        
        def objective(trial):
            params = {
                'n_d': trial.suggest_int('n_d', 8, 64),
                'n_a': trial.suggest_int('n_a', 8, 64),
                'n_steps': trial.suggest_int('n_steps', 3, 10),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True),
                'lr': trial.suggest_float('lr', 1e-3, 5e-2, log=True),
                'n_shared': trial.suggest_int('n_shared', 1, 3),
                'n_independent': trial.suggest_int('n_independent', 1, 3)
            }
            
            model = TabNetRegressor(
                n_d=params['n_d'], n_a=params['n_a'], n_steps=params['n_steps'],
                gamma=params['gamma'], n_independent=params['n_independent'], 
                n_shared=params['n_shared'],
                lambda_sparse=params['lambda_sparse'], optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=params['lr']), mask_type='entmax',
                verbose=0, device_name=self.device
            )
            
            model.fit(X_train, y_train_t, eval_set=[(X_val, y_val_t)],
                     max_epochs=30, patience=10, batch_size=256, virtual_batch_size=128)
            
            y_pred = model.predict(X_val).flatten()
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best = study.best_params
        self.best_params = {
            'n_d_a': best['n_d'], 'n_steps': best['n_steps'], 'gamma': best['gamma'],
            'lambda_sparse': best['lambda_sparse'], 'lr': best['lr'],
            'n_shared': best['n_shared'], 'n_independent': best['n_independent']
        }
        self.best_params['n_a'] = best['n_a']
        
        print(f"Best TabNet params: {self.best_params}")
        return self.best_params
    
    def fit_and_evaluate(self, X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
                         y_train, y_val, y_test, scaler_y):
        if not TABNET_AVAILABLE:
            print("TabNet not available, skipping...")
            return {}, {}, {}
        
        if self.use_optuna:
            self.optuna_tune_tabnet(X_train, y_train_s, X_val, y_val_s)
        else:
            self.best_params = {'n_d_a': 64, 'n_steps': 3, 'gamma': 1.5, 
                              'lambda_sparse': 1e-4, 'lr': 2e-2, 'n_shared': 2, 'n_independent': 2}
        
        print("\nTraining final TabNet with best params...")
        y_train_t = y_train_s.reshape(-1, 1)
        y_val_t = y_val_s.reshape(-1, 1)
        
        start_time = time.time()
        
        model = TabNetRegressor(
            n_d=self.best_params['n_d_a'], n_a=self.best_params.get('n_a', self.best_params['n_d_a']),
            n_steps=self.best_params['n_steps'], gamma=self.best_params['gamma'],
            n_independent=self.best_params.get('n_independent', 2), 
            n_shared=self.best_params.get('n_shared', 2),
            lambda_sparse=self.best_params['lambda_sparse'],
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=self.best_params['lr']),
            mask_type='entmax', verbose=10, device_name=self.device
        )
        
        model.fit(X_train, y_train_t, eval_set=[(X_val, y_val_t)],
                 max_epochs=200, patience=30, batch_size=256, virtual_batch_size=128)
        
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test).flatten()
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        self.results['TabNet_Tuned'] = calculate_metrics_from_scratch(y_test, y_pred_inv)
        self.training_times['TabNet_Tuned'] = train_time
        self.energy_scores['TabNet_Tuned'] = 1.0 / (train_time + 1e-6)
        
        print(f"  Test R²: {self.results['TabNet_Tuned']['R2']:.4f}, Time: {train_time:.2f}s")
        
        # Store model for SHAP
        self.model = model
        
        return self.results, self.training_times, self.energy_scores
# ============================================================
# PART 7: MEMORY-EFFICIENT FT-TRANSFORMER WITH OPTUNA TUNING
# ============================================================

class FTTransformerTuned:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_optuna=True):
        self.device = device
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.results = {}
        self.training_times = {}
        self.energy_scores = {}
        self.best_config = None
        self.model = None
        self.use_amp = True  # Automatic Mixed Precision
        
    def clear_memory(self):
        """Clear GPU memory before/after training."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
    def optuna_tune_fttransformer(self, X_train, y_train, X_val, y_val, input_dim, n_trials=30):
        """Optuna hyperparameter tuning for FT-Transformer."""
        if not OPTUNA_AVAILABLE:
            return {'n_blocks': 3, 'd_token': 64, 'n_heads': 4, 'lr': 1e-3, 'dropout': 0.1}
            
        print(f"\nOptuna tuning for FT-Transformer ({n_trials} trials)...")
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        def objective(trial):
            cfg = {
                'n_blocks': trial.suggest_int('n_blocks', 2, 6),
                'd_token': trial.suggest_int('d_token', 64, 256, log=True),
                'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
                'lr': trial.suggest_float('lr', 5e-5, 5e-3, log=True)
            }
            
            # Clear memory before building model
            self.clear_memory()
            
            model = FTTransformer(
                n_cont_features=input_dim, cat_cardinalities=[], d_out=1,
                n_blocks=cfg['n_blocks'], d_block=cfg['d_token'],
                attention_n_heads=cfg['n_heads'], attention_dropout=cfg['dropout'],
                ffn_d_hidden=None, ffn_d_hidden_multiplier=4/3,
                ffn_dropout=cfg['dropout'], residual_dropout=0.0,
            ).to(self.device)
            
            optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=cfg['lr'], weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            # Use AMP for memory efficiency during tuning
            scaler = torch.cuda.amp.GradScaler() if (self.use_amp and self.device == 'cuda') else None
            
            # Quick training
            for epoch in range(20):
                model.train()
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = model(None, X_train_t).squeeze()
                        loss = criterion(pred, y_train_t)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(None, X_train_t).squeeze()
                    loss = criterion(pred, y_train_t)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                if scaler:
                    with torch.cuda.amp.autocast():
                        val_loss = criterion(model(None, X_val_t).squeeze(), y_val_t).item()
                else:
                    val_loss = criterion(model(None, X_val_t).squeeze(), y_val_t).item()
            
            # Clean up
            del model
            self.clear_memory()
            
            return val_loss
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best = study.best_params
        self.best_config = {
            'n_blocks': best['n_blocks'], 'd_token': best['d_token'],
            'n_heads': best['n_heads'], 'lr': best['lr'], 'dropout': best['dropout']
        }
        
        print(f"Best FT-Transformer config: {self.best_config}")
        return self.best_config
    
    def fit_and_evaluate(self, X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
                     y_train, y_val, y_test, scaler_y, input_dim):
        if not FTTRANSFORMER_AVAILABLE:
            print("FT-Transformer not available, skipping...")
            return {}, {}, {}
        
        if self.use_optuna:
            self.optuna_tune_fttransformer(X_train, y_train_s, X_val, y_val_s, input_dim)
        else:
            self.best_config = {'n_blocks': 3, 'd_token': 128, 'n_heads': 4, 'lr': 1e-3, 'dropout': 0.1}
        
        print("\nTraining final FT-Transformer with best config...")
        print(f"  Config: n_blocks={self.best_config['n_blocks']}, d_token={self.best_config['d_token']}, n_heads={self.best_config['n_heads']}")
        
        # Clear memory before training
        self.clear_memory()
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train_s).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val_s).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        
        start_time = time.time()
        
        self.model = FTTransformer(
            n_cont_features=input_dim, cat_cardinalities=[], d_out=1,
            n_blocks=self.best_config['n_blocks'], d_block=self.best_config['d_token'],
            attention_n_heads=self.best_config['n_heads'], attention_dropout=self.best_config['dropout'],
            ffn_d_hidden=None, ffn_d_hidden_multiplier=4/3,
            ffn_dropout=self.best_config['dropout'], residual_dropout=0.0,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.make_parameter_groups(), lr=self.best_config['lr'], weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Automatic Mixed Precision
        scaler = torch.cuda.amp.GradScaler() if (self.use_amp and self.device == 'cuda') else None
        
        best_val = float('inf')
        patience = 30
        patience_counter = 0
        best_state = None
        
        for epoch in range(200):
            # Training with AMP
            self.model.train()
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    pred = self.model(None, X_train_t).squeeze()
                    loss = criterion(pred, y_train_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = self.model(None, X_train_t).squeeze()
                loss = criterion(pred, y_train_t)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                if scaler:
                    with torch.cuda.amp.autocast():
                        val_pred = self.model(None, X_val_t).squeeze()
                        val_loss = criterion(val_pred, y_val_t).item()
                else:
                    val_pred = self.model(None, X_val_t).squeeze()
                    val_loss = criterion(val_pred, y_val_t).item()
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                mem_info = ""
                if self.device == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1024**3
                    mem_info = f" | GPU: {mem:.1f}GB"
                print(f"  Epoch {epoch+1}: val_loss={val_loss:.6f}{mem_info}")
            
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break
            
            # Periodic memory cleanup
            if self.device == 'cuda' and (epoch + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        train_time = time.time() - start_time
        
        # Batch prediction to avoid OOM
        self.model.eval()
        test_preds = []
        batch_size = 1024
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = X_test_t[i:i+batch_size]
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = self.model(None, batch).cpu().numpy().flatten()
                else:
                    pred = self.model(None, batch).cpu().numpy().flatten()
                test_preds.extend(pred)
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        y_pred_inv = scaler_y.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
        
        self.results['FT-Transformer_Tuned'] = calculate_metrics_from_scratch(y_test_orig, y_pred_inv)
        self.training_times['FT-Transformer_Tuned'] = train_time
        self.energy_scores['FT-Transformer_Tuned'] = 1.0 / (train_time + 1e-6)
        
        print(f"  Test R²: {self.results['FT-Transformer_Tuned']['R2']:.4f}, Time: {train_time:.2f}s")
        
        # Clear memory after training
        self.clear_memory()
        
        return self.results, self.training_times, self.energy_scores
    
    def predict(self, X):
        """Predict with batch processing to save memory."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        batch_size = 1024
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            X_t = torch.FloatTensor(batch).to(self.device)
            with torch.no_grad():
                pred = self.model(None, X_t).cpu().numpy().flatten()
            predictions.extend(pred)
            
            del X_t
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return np.array(predictions)


# ============================================================
# PART 8: TABPFN MODEL
# ============================================================

class TabPFNModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.training_times = {}
        self.energy_scores = {}
        
    def fit_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, y_test_orig, scaler_y):
        if not TABPFN_AVAILABLE:
            print("TabPFN not available, skipping...")
            return {}, {}, {}
        
        print("\nTraining TabPFN...")
        start_time = time.time()
        
        # Combine train and val for TabPFN (it uses internal validation)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        model = TabPFNRegressor(device=self.device, N_ensemble_configurations=4)
        model.fit(X_combined, y_combined)
        
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        self.results['TabPFN'] = calculate_metrics_from_scratch(y_test_orig, y_pred_inv)
        self.training_times['TabPFN'] = train_time
        self.energy_scores['TabPFN'] = 1.0 / (train_time + 1e-6)
        
        print(f"  Test R²: {self.results['TabPFN']['R2']:.4f}, Time: {train_time:.2f}s")
        
        return self.results, self.training_times, self.energy_scores


# ============================================================
# PART 9: SHAP XAI ANALYSIS
# ============================================================

class XAIAnalyzer:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_model(self, model, X_train, X_test, feature_names, model_name, y_test=None, y_pred=None):
        """Generate SHAP analysis for a model."""
        if not SHAP_AVAILABLE:
            print("SHAP not available, skipping XAI analysis...")
            return
        
        print(f"\nGenerating SHAP analysis for {model_name}...")
        
        try:
            # Tree-based models
            if hasattr(model, 'feature_importances_') or 'Tree' in str(type(model)) or 'Forest' in str(type(model)) or 'Boost' in str(type(model)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            
            # Linear models
            elif hasattr(model, 'coef_'):
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test)
            
            # Deep learning / Other models - use KernelExplainer with sample
            else:
                # Use k-means summary for background
                background = shap.kmeans(X_train, 100)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_test[:100])  # Sample for speed
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
            plt.title(f'SHAP Feature Importance (Bar) - {model_name}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/shap_bar_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Dependence plots for top 3 features
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_indices = np.argsort(mean_abs_shap)[-3:][::-1]
                
                for idx in top_indices:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names, show=False)
                    plt.title(f'SHAP Dependence - {feature_names[idx]}')
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/shap_dependence_{model_name}_{feature_names[idx]}.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Feature importance CSV
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Mean_SHAP': np.abs(shap_values).mean(axis=0),
                    'Std_SHAP': np.abs(shap_values).std(axis=0)
                }).sort_values('Mean_SHAP', ascending=False)
                
                importance_df.to_csv(f'{self.output_dir}/shap_importance_{model_name}.csv', index=False)
            
            print(f"  SHAP analysis saved for {model_name}")
            
        except Exception as e:
            print(f"  SHAP analysis failed for {model_name}: {str(e)}")


# ============================================================
# PART 10: VISUALIZATION
# ============================================================

class ResultVisualizer:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_actual_vs_predicted(self, y_true, y_pred_dict, freq, n_samples=1000):
        """Plot actual vs predicted for all models."""
        n_models = len(y_pred_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        # Sample for visibility
        indices = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
        indices = np.sort(indices)
        
        for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            ax = axes[idx]
            
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
            
            ax.scatter(y_true_sample, y_pred_sample, alpha=0.5, s=10, c='blue', edgecolors='none')
            
            # Perfect prediction line
            min_val = min(y_true_sample.min(), y_pred_sample.min())
            max_val = max(y_true_sample.max(), y_pred_sample.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Power Consumption', fontsize=10)
            ax.set_ylabel('Predicted Power Consumption', fontsize=10)
            ax.set_title(f'{model_name} - Actual vs Predicted', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add R² annotation
            r2 = r2_score(y_true, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide empty subplots
        for idx in range(len(y_pred_dict), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/actual_vs_predicted_{freq}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_time_series_comparison(self, y_true, y_pred_dict, freq, n_points=500):
        """Plot time series comparison."""
        n_models = len(y_pred_dict)
        
        fig, axes = plt.subplots(n_models + 1, 1, figsize=(15, 3 * (n_models + 1)), sharex=True)
        
        # Plot actual
        axes[0].plot(y_true[:n_points], label='Actual', color='black', linewidth=2)
        axes[0].set_title('Actual Power Consumption', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Power')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot each model
        for idx, (model_name, y_pred) in enumerate(y_pred_dict.items(), 1):
            ax = axes[idx]
            ax.plot(y_true[:n_points], label='Actual', color='black', alpha=0.7, linewidth=1.5)
            ax.plot(y_pred[:n_points], label='Predicted', color='red', alpha=0.7, linewidth=1.5)
            ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Power')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.xlabel('Time Steps')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_series_comparison_{freq}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics_heatmap(self, results_dict, freq):
        """Plot metrics heatmap."""
        models = list(results_dict.keys())
        metrics = ['R2', 'RMSE', 'MAE', 'MAPE']
        
        # Create matrix
        data = []
        for model in models:
            row = [results_dict[model][m] for m in metrics]
            data.append(row)
        
        # Normalize for visualization (R2 should be max, others min)
        data = np.array(data)
        data_norm = data.copy()
        
        # Invert RMSE, MAE, MAPE for consistent coloring (higher = better)
        for i in range(1, 4):
            data_norm[:, i] = 1 / (data[:, i] + 1e-6)
            data_norm[:, i] = (data_norm[:, i] - data_norm[:, i].min()) / (data_norm[:, i].max() - data_norm[:, i].min() + 1e-6)
        
        # R2 is already 0-1
        data_norm[:, 0] = np.clip(data[:, 0], 0, 1)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))
        
        im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.4f}',
                              ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        ax.set_title(f'Model Performance Heatmap - {freq} Frequency', fontsize=14, fontweight='bold', pad=20)
        fig.colorbar(im, ax=ax, label='Normalized Score (Higher = Better)')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/metrics_heatmap_{freq}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_energy_accuracy_tradeoff(self, results_dict, energy_scores, freq):
        """Plot energy vs accuracy tradeoff."""
        models = list(results_dict.keys())
        r2_scores = [results_dict[m]['R2'] for m in models]
        energies = [energy_scores.get(m, 0) for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        scatter = ax.scatter(energies, r2_scores, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')
        
        # Add model labels
        for i, model in enumerate(models):
            ax.annotate(model, (energies[i], r2_scores[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Energy Efficiency Score (1/Training Time)', fontsize=12)
        ax.set_ylabel('R² Score (Accuracy)', fontsize=12)
        ax.set_title(f'Energy vs Accuracy Tradeoff - {freq} Frequency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        median_energy = np.median(energies)
        median_r2 = np.median(r2_scores)
        ax.axvline(median_energy, color='red', linestyle='--', alpha=0.5)
        ax.axhline(median_r2, color='red', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(max(energies) * 0.95, max(r2_scores) * 0.98, 'High Accuracy\nHigh Efficiency', 
               ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(min(energies) * 1.05, max(r2_scores) * 0.98, 'High Accuracy\nLow Efficiency', 
               ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/energy_accuracy_tradeoff_{freq}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_residuals(self, y_true, y_pred_dict, freq):
        """Plot residual analysis."""
        n_models = len(y_pred_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            ax = axes[idx]
            residuals = y_true - y_pred
            
            ax.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Residuals', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model_name} - Residual Distribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            ax.text(0.95, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for idx in range(len(y_pred_dict), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/residuals_{freq}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_comparison_bar(self, results_dict, freq):
        """Plot bar chart comparison of all models."""
        models = list(results_dict.keys())
        metrics = ['R2', 'RMSE', 'MAE']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [results_dict[m][metric] for m in models]
            
            # Color by rank
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
            if metric != 'R2':  # Lower is better for RMSE, MAE
                sorted_indices = np.argsort(values)
                colors = [colors[i] for i in np.argsort(sorted_indices)]
            
            bars = ax.barh(models, values, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xlabel(metric, fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.suptitle(f'Model Performance Comparison - {freq} Frequency', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison_bar_{freq}.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# PART 11: ABLATION STUDY
# ============================================================

class AblationStudy:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_ablation(self, df, target_col='Global_active_power', train_ratio=0.7, val_ratio=0.15):
        """Run ablation study by removing feature groups."""
        print("\n" + "="*80)
        print("ABLATION STUDY")
        print("="*80)
        
        feature_groups = {
            'All Features': None,  # Keep all
            'No Lag Features': ['Global_active_power_lag_6', 'Global_active_power_lag_12', 'Global_active_power_lag_24'],
            'No Rolling Features': [c for c in df.columns if 'rolling_' in c],
            'No Time Features': ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'quarter',
                                'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month'],
            'No Cyclical Features': ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month'],
            'No Power Factor': ['power_factor'],
            'Only Lag Features': 'inverse',  # Keep only lags
            'Only Rolling Features': 'inverse',  # Keep only rolling
        }
        
        results = {}
        base_features = get_final_features(df, target_col)
        
        for group_name, features_to_remove in feature_groups.items():
            print(f"\n--- Testing: {group_name} ---")
            
            if features_to_remove == 'inverse':
                # Keep only specific group
                if 'Only Lag' in group_name:
                    current_features = [c for c in base_features if 'lag_' in c]
                else:
                    current_features = [c for c in base_features if 'rolling_' in c]
                # Add target and basic features
                current_features = ['hour', 'day_of_week'] + current_features
            elif features_to_remove is None:
                current_features = base_features
            else:
                current_features = [c for c in base_features if c not in features_to_remove]
            
            if len(current_features) == 0:
                continue
                
            print(f"Features used: {len(current_features)}")
            
            # Prepare data
            X = df[current_features].values
            y = df[target_col].values
            
            n = len(X)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            X_train, y_train = X[:n_train], y[:n_train]
            X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
            X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
            
            scaler_X = StandardScaler()
            X_train_s = scaler_X.fit_transform(X_train)
            X_val_s = scaler_X.transform(X_val)
            X_test_s = scaler_X.transform(X_test)
            
            scaler_y = StandardScaler()
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # Train LightGBM for quick evaluation
            if LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42)
                model.fit(X_train_s, y_train_s)
                
                y_pred = scaler_y.inverse_transform(model.predict(X_test_s).reshape(-1, 1)).flatten()
                metrics = calculate_metrics_from_scratch(y_test, y_pred)
                results[group_name] = metrics
                
                print(f"  R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
        
        # Save ablation results
        ablation_df = pd.DataFrame(results).T
        ablation_df.to_csv(f'{self.output_dir}/ablation_study_results.csv')
        
        # Plot ablation results
        self.plot_ablation_results(results)
        
        return results
    
    def plot_ablation_results(self, results):
        """Plot ablation study results."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        groups = list(results.keys())
        r2_scores = [results[g]['R2'] for g in groups]
        
        colors = ['green' if r2 == max(r2_scores) else 'red' if r2 < max(r2_scores) * 0.95 else 'orange' 
                  for r2 in r2_scores]
        
        bars = ax.barh(groups, r2_scores, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_title('Ablation Study: Feature Group Impact', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, r2_scores):
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# PART 12: CROSS-VALIDATION
# ============================================================

class CrossValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def time_series_cv(self, X, y, model_class, model_params, scaler_y):
        """Time series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            y_pred = scaler_y.inverse_transform(model.predict(X_val).reshape(-1, 1)).flatten()
            y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
            
            r2 = r2_score(y_val_orig, y_pred)
            scores.append(r2)
            print(f"  Fold {fold+1}: R² = {r2:.4f}")
        
        return {
            'mean_r2': np.mean(scores),
            'std_r2': np.std(scores),
            'folds': scores
        }


# ============================================================
# PART 13: MAIN EXECUTION
# ============================================================

def run_complete_analysis(data_path="household_power_consumption.txt", frequencies=['min', 'h', 'd'], 
                         use_optuna=True, run_ablation=True, run_xai=True):
    """Run complete analysis for all frequencies."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f'ensemble_results_{timestamp}'
    os.makedirs(base_output_dir, exist_ok=True)
    
    all_results_summary = {}
    
    for freq in frequencies:
        print(f"\n{'#'*100}")
        print(f"# PROCESSING FREQUENCY: {freq}")
        print(f"{'#'*100}")
        
        output_dir = f'{base_output_dir}/{freq}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess
        df = load_data(data_path)
        df = resample_data(df, freq)
        df = engineer_features_final(df)
        
        # Prepare data
        (X_train, X_val, X_test, y_train_s, y_val_s, y_test_s,
         y_train, y_val, y_test, scaler_y, feature_cols, 
         X_train_raw, X_val_raw, X_test_raw) = prepare_data(df)
        
        input_dim = X_train.shape[1]
        
        print(f"\n{'='*80}")
        print(f"DATA SUMMARY - {freq} FREQUENCY")
        print(f"Features: {input_dim} | Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
        print(f"{'='*80}")
        
        all_results = {}
        all_times = {}
        all_energy = {}
        all_predictions = {}
        
        # Store models for XAI
        trained_models = {}
        
        # 1. Traditional ML + LightGBM
        print("\n" + "#"*70)
        print("# TRADITIONAL ML MODELS + LIGHTGBM")
        print("#"*70)
        trad = TraditionalML(n_jobs=-1, random_state=42, use_optuna=use_optuna)
        trad_results, trad_times, trad_energy = trad.fit_and_evaluate(
            X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
            y_train, y_val, y_test, scaler_y, X_train_raw, X_val_raw, X_test_raw
        )
        all_results.update({k: v['test'] for k, v in trad_results.items()})
        all_times.update(trad_times)
        all_energy.update(trad_energy)
        
        # Store predictions
        for name, model in trad.models.items():
            y_pred = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
            all_predictions[name] = y_pred
            trained_models[name] = model
        
        # 2. Deep Learning
        print("\n" + "#"*70)
        print("# DEEP LEARNING MODELS (PYTORCH)")
        print("#"*70)
        dl = DeepLearningModels(device='cuda' if torch.cuda.is_available() else 'cpu', use_optuna=use_optuna)
        dl_results, dl_times, dl_energy = dl.fit_and_evaluate(
            X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
            y_train, y_val, y_test, scaler_y, input_dim, epochs=100
        )
        all_results.update(dl_results)
        all_times.update(dl_times)
        all_energy.update(dl_energy)
        
        # 3. TabNet
        print("\n" + "#"*70)
        print("# TABNET WITH OPTUNA HYPERPARAMETER TUNING")
        print("#"*70)
        tabnet = TabNetTuned(device='cuda' if torch.cuda.is_available() else 'cpu', use_optuna=use_optuna)
        tabnet_results, tabnet_times, tabnet_energy = tabnet.fit_and_evaluate(
            X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
            y_train, y_val, y_test, scaler_y
        )
        all_results.update(tabnet_results)
        all_times.update(tabnet_times)
        all_energy.update(tabnet_energy)
        
        # Store TabNet predictions
        if TABNET_AVAILABLE and hasattr(tabnet, 'model'):
            y_pred = scaler_y.inverse_transform(tabnet.model.predict(X_test).flatten().reshape(-1, 1)).flatten()
            all_predictions['TabNet_Tuned'] = y_pred
        
        # 4. FT-Transformer (Memory-Efficient)
        print("\n" + "#"*70)
        print("# FT-TRANSFORMER WITH OPTUNA HYPERPARAMETER TUNING (MEMORY-EFFICIENT)")
        print("#"*70)
        ft = FTTransformerTuned(device='cuda' if torch.cuda.is_available() else 'cpu', use_optuna=use_optuna)
        ft_results, ft_times, ft_energy = ft.fit_and_evaluate(
            X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
            y_train, y_val, y_test, scaler_y, input_dim
        )
        all_results.update(ft_results)
        all_times.update(ft_times)
        all_energy.update(ft_energy)
        
        # Store FT-Transformer predictions
        if FTTRANSFORMER_AVAILABLE and ft.model is not None:
            y_pred_s = ft.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
            all_predictions['FT-Transformer_Tuned'] = y_pred
            trained_models['FT-Transformer_Tuned'] = ft
        
        # 5. TabPFN
        print("\n" + "#"*70)
        print("# TABPFN")
        print("#"*70)
        tabpfn = TabPFNModel(device='cuda' if torch.cuda.is_available() else 'cpu')
        tabpfn_results, tabpfn_times, tabpfn_energy = tabpfn.fit_and_evaluate(
            X_train, y_train_s, X_val, y_val_s, X_test, y_test_s,
            y_train, y_val, y_test, scaler_y
        )
        all_results.update(tabpfn_results)
        all_times.update(tabpfn_times)
        all_energy.update(tabpfn_energy)
        
        # ==================== VISUALIZATION ====================
        print("\n" + "#"*70)
        print("# GENERATING VISUALIZATIONS")
        print("#"*70)
        
        viz = ResultVisualizer(output_dir=output_dir)
        
        # 1. Actual vs Predicted plots
        print("Generating Actual vs Predicted plots...")
        viz.plot_actual_vs_predicted(y_test, all_predictions, freq)
        
        # 2. Time series comparison
        print("Generating Time Series comparison...")
        viz.plot_time_series_comparison(y_test, all_predictions, freq)
        
        # 3. Metrics heatmap
        print("Generating Metrics Heatmap...")
        viz.plot_metrics_heatmap(all_results, freq)
        
        # 4. Energy vs Accuracy tradeoff
        print("Generating Energy-Accuracy Tradeoff...")
        viz.plot_energy_accuracy_tradeoff(all_results, all_energy, freq)
        
        # 5. Residual analysis
        print("Generating Residual Analysis...")
        viz.plot_residuals(y_test, all_predictions, freq)
        
        # 6. Model comparison bar chart
        print("Generating Model Comparison Bar Chart...")
        viz.plot_model_comparison_bar(all_results, freq)
        
        # ==================== XAI / SHAP ANALYSIS ====================
        if run_xai and SHAP_AVAILABLE:
            print("\n" + "#"*70)
            print("# SHAP XAI ANALYSIS")
            print("#"*70)
            
            xai = XAIAnalyzer(output_dir=output_dir)
            
            # Analyze key models
            key_models = ['LightGBM', 'RandomForest', 'GradientBoosting', 'TabNet_Tuned']
            for model_name in key_models:
                if model_name in trained_models:
                    xai.analyze_model(
                        trained_models[model_name], 
                        X_train_raw if 'TabNet' not in model_name else X_train,
                        X_test_raw if 'TabNet' not in model_name else X_test,
                        feature_cols, 
                        model_name,
                        y_test, 
                        all_predictions.get(model_name)
                    )
        
        # ==================== SAVE RESULTS TO CSV ====================
        print("\n" + "#"*70)
        print("# SAVING RESULTS TO CSV")
        print("#"*70)
        
        # Main results CSV
        results_df = pd.DataFrame({
            'Model': list(all_results.keys()),
            'R2': [v['R2'] for v in all_results.values()],
            'RMSE': [v['RMSE'] for v in all_results.values()],
            'MAE': [v['MAE'] for v in all_results.values()],
            'MAPE': [v['MAPE'] for v in all_results.values()],
            'MASE': [v.get('MASE', np.nan) for v in all_results.values()],
            'Training_Time': [all_times.get(k, np.nan) for k in all_results.keys()],
            'Energy_Efficiency': [all_energy.get(k, np.nan) for k in all_results.keys()]
        })
        
        # Sort by R2
        results_df = results_df.sort_values('R2', ascending=False)
        results_df.to_csv(f'{output_dir}/model_comparison_results.csv', index=False)
        print(f"Saved: {output_dir}/model_comparison_results.csv")
        
        # Detailed metrics CSV
        detailed_df = pd.DataFrame(all_results).T
        detailed_df.to_csv(f'{output_dir}/detailed_metrics.csv')
        print(f"Saved: {output_dir}/detailed_metrics.csv")
        
        # Predictions CSV (sample)
        predictions_df = pd.DataFrame({
            'Actual': y_test[:1000]
        })
        for model_name, preds in all_predictions.items():
            predictions_df[f'{model_name}_Predicted'] = preds[:1000]
        predictions_df.to_csv(f'{output_dir}/predictions_sample.csv', index=False)
        print(f"Saved: {output_dir}/predictions_sample.csv")
        
        # Training times CSV
        times_df = pd.DataFrame({
            'Model': list(all_times.keys()),
            'Training_Time_Seconds': list(all_times.values()),
            'Energy_Efficiency_Score': [all_energy.get(k, np.nan) for k in all_times.keys()]
        })
        times_df.to_csv(f'{output_dir}/training_efficiency.csv', index=False)
        print(f"Saved: {output_dir}/training_efficiency.csv")
        
        # Feature list CSV
        feature_df = pd.DataFrame({
            'Feature_Index': range(len(feature_cols)),
            'Feature_Name': feature_cols,
            'Category': ['Time' if 'hour' in f or 'day' in f or 'month' in f or 'quarter' in f or 'weekend' in f or 'year' in f
                        else 'Cyclical' if 'sin' in f or 'cos' in f
                        else 'Lag' if 'lag' in f
                        else 'Rolling' if 'rolling' in f
                        else 'PowerFactor' if 'power_factor' in f
                        else 'Electrical' for f in feature_cols]
        })
        feature_df.to_csv(f'{output_dir}/feature_list.csv', index=False)
        print(f"Saved: {output_dir}/feature_list.csv")
        
        # Print summary
        print("\n" + "="*80)
        print(f"FINAL RESULTS: {freq} FREQUENCY - TEST SET PERFORMANCE (RANKED BY R²)")
        print("="*80)
        print(f"{'Model':<25} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'Time(s)':>10}")
        print("-"*80)
        
        for _, row in results_df.iterrows():
            print(f"{row['Model']:<25} {row['R2']:>10.4f} {row['RMSE']:>10.4f} "
                  f"{row['MAE']:>10.4f} {row['MAPE']:>9.2f}% {row['Training_Time']:>10.2f}")
        
        print("="*80)
        
        # Store for overall summary
        all_results_summary[freq] = {
            'results': all_results,
            'times': all_times,
            'energy': all_energy,
            'best_model': results_df.iloc[0]['Model'] if len(results_df) > 0 else None,
            'best_r2': results_df.iloc[0]['R2'] if len(results_df) > 0 else None
        }
        
        # ==================== ABLATION STUDY ====================
        if run_ablation:
            print("\n" + "#"*70)
            print("# RUNNING ABLATION STUDY")
            print("#"*70)
            
            ablation = AblationStudy(output_dir=output_dir)
            ablation.run_ablation(df, target_col='Global_active_power')
    
    # ==================== FINAL SUMMARY ACROSS ALL FREQUENCIES ====================
    print("\n" + "#"*100)
    print("# FINAL SUMMARY ACROSS ALL FREQUENCIES")
    print("#"*100)
    
    summary_rows = []
    for freq, data in all_results_summary.items():
        for model, metrics in data['results'].items():
            summary_rows.append({
                'Frequency': freq,
                'Model': model,
                'R2': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'Training_Time': data['times'].get(model, np.nan),
                'Energy_Efficiency': data['energy'].get(model, np.nan)
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{base_output_dir}/complete_summary_all_frequencies.csv', index=False)
    print(f"\nSaved complete summary: {base_output_dir}/complete_summary_all_frequencies.csv")
    
    # Best model per frequency
    best_per_freq = summary_df.loc[summary_df.groupby('Frequency')['R2'].idxmax()]
    best_per_freq.to_csv(f'{base_output_dir}/best_model_per_frequency.csv', index=False)
    print(f"Saved best per frequency: {base_output_dir}/best_model_per_frequency.csv")
    
    # Pivot table for comparison
    pivot_r2 = summary_df.pivot(index='Model', columns='Frequency', values='R2')
    pivot_r2.to_csv(f'{base_output_dir}/r2_comparison_matrix.csv')
    print(f"Saved R² matrix: {base_output_dir}/r2_comparison_matrix.csv")
    
    # Create final comparison heatmap across frequencies
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_r2) * 0.5)))
    
    im = ax.imshow(pivot_r2.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(pivot_r2.columns)))
    ax.set_yticks(np.arange(len(pivot_r2.index)))
    ax.set_xticklabels(pivot_r2.columns)
    ax.set_yticklabels(pivot_r2.index)
    
    # Add text annotations
    for i in range(len(pivot_r2.index)):
        for j in range(len(pivot_r2.columns)):
            text = ax.text(j, i, f'{pivot_r2.iloc[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('R² Score Comparison Across Frequencies', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    fig.colorbar(im, ax=ax, label='R² Score')
    
    plt.tight_layout()
    plt.savefig(f'{base_output_dir}/frequency_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"All results saved in: {base_output_dir}/")
    print(f"{'='*80}")
    
    return all_results_summary, base_output_dir


def generate_model_parameter_template():
    """Generate the model parameter template CSV."""
    params_data = [
        # LightGBM
        {'Model': 'LightGBM', 'Parameter': 'learning_rate', 'Values': '0.01, 0.05, 0.1', 'Type': 'float', 'Tuning_Range': '0.001-0.3'},
        {'Model': 'LightGBM', 'Parameter': 'num_leaves', 'Values': '31, 50, 100', 'Type': 'int', 'Tuning_Range': '20-150'},
        {'Model': 'LightGBM', 'Parameter': 'max_depth', 'Values': '-1, 5, 10', 'Type': 'int', 'Tuning_Range': '3-12'},
        {'Model': 'LightGBM', 'Parameter': 'n_estimators', 'Values': '100, 500, 1000', 'Type': 'int', 'Tuning_Range': '100-2000'},
        {'Model': 'LightGBM', 'Parameter': 'subsample', 'Values': '0.8, 1.0', 'Type': 'float', 'Tuning_Range': '0.6-1.0'},
        
        # TabNet
        {'Model': 'TabNet', 'Parameter': 'n_d', 'Values': '8, 16, 32', 'Type': 'int', 'Tuning_Range': '8-64'},
        {'Model': 'TabNet', 'Parameter': 'n_a', 'Values': '8, 16, 32', 'Type': 'int', 'Tuning_Range': '8-64'},
        {'Model': 'TabNet', 'Parameter': 'n_steps', 'Values': '3, 5, 7', 'Type': 'int', 'Tuning_Range': '3-10'},
        {'Model': 'TabNet', 'Parameter': 'gamma', 'Values': '1.0, 1.5, 2.0', 'Type': 'float', 'Tuning_Range': '1.0-2.0'},
        {'Model': 'TabNet', 'Parameter': 'lambda_sparse', 'Values': '1e-4, 1e-3', 'Type': 'float', 'Tuning_Range': '1e-5-1e-2'},
        
        # MLP
        {'Model': 'MLP', 'Parameter': 'hidden_layers', 'Values': '2, 3, 4', 'Type': 'int', 'Tuning_Range': '2-5'},
        {'Model': 'MLP', 'Parameter': 'hidden_units', 'Values': '64, 128, 256', 'Type': 'int', 'Tuning_Range': '32-512'},
        {'Model': 'MLP', 'Parameter': 'dropout', 'Values': '0.2, 0.3', 'Type': 'float', 'Tuning_Range': '0.1-0.5'},
        {'Model': 'MLP', 'Parameter': 'learning_rate', 'Values': '0.001, 0.01', 'Type': 'float', 'Tuning_Range': '1e-4-1e-2'},
        
        # FT-Transformer
        {'Model': 'FT-Transformer', 'Parameter': 'n_blocks', 'Values': '2, 3, 4', 'Type': 'int', 'Tuning_Range': '2-6'},
        {'Model': 'FT-Transformer', 'Parameter': 'd_token', 'Values': '64, 128, 192', 'Type': 'int', 'Tuning_Range': '64-256'},
        {'Model': 'FT-Transformer', 'Parameter': 'n_heads', 'Values': '2, 4, 8', 'Type': 'int', 'Tuning_Range': '2,4,8'},
        {'Model': 'FT-Transformer', 'Parameter': 'dropout', 'Values': '0.1, 0.2', 'Type': 'float', 'Tuning_Range': '0.0-0.3'},
        {'Model': 'FT-Transformer', 'Parameter': 'learning_rate', 'Values': '1e-3, 5e-4', 'Type': 'float', 'Tuning_Range': '5e-5-5e-3'},
        
        # TabPFN
        {'Model': 'TabPFN', 'Parameter': 'N_ensemble_configurations', 'Values': '4, 8', 'Type': 'int', 'Tuning_Range': '1-32'},
    ]
    
    return pd.DataFrame(params_data)


def generate_empty_results_template():
    """Generate empty results template for manual filling."""
    models = ['TabNet', 'LightGBM', 'MLP', 'FT-Transformer', 'TabPFN', 'LinearRegression', 
              'Ridge', 'Lasso', 'RandomForest', 'GradientBoosting', 'SVR', 'KNN', 'MLP_sklearn',
              'MLP_PyTorch', 'LSTM', 'BiLSTM', 'GRU']
    
    template = pd.DataFrame({
        'Model': models,
        'RMSE': [''] * len(models),
        'MAE': [''] * len(models),
        'R2': [''] * len(models),
        'MASE': [''] * len(models),
        'MAPE': [''] * len(models),
        'Training_Time': [''] * len(models),
        'Energy_Efficiency': [''] * len(models),
        'Frequency': [''] * len(models)
    })
    
    return template


def generate_ablation_template():
    """Generate ablation study template."""
    experiments = [
        'All Features',
        'No Lag Features',
        'No Rolling Features',
        'No Time Features',
        'No Cyclical Features',
        'No Power Factor',
        'Only Lag Features',
        'Only Rolling Features',
        'Only Time Features',
        'Remove Voltage',
        'Remove Global_reactive_power'
    ]
    
    template = pd.DataFrame({
        'Experiment': experiments,
        'Removed_Feature': ['None'] + ['lag_6, lag_12, lag_24'] + ['rolling_12, rolling_24, rolling_48'] +
                          ['hour, day_of_week, month, etc.'] + ['sin_*, cos_*'] + ['power_factor'] +
                          ['All except lags'] + ['All except rolling'] + ['All except time'] +
                          ['Voltage'] + ['Global_reactive_power'],
        'R2_Impact': [''] * len(experiments),
        'RMSE_Impact': [''] * len(experiments),
        'Performance_Change': [''] * len(experiments)
    })
    
    return template


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Ensemble Power Consumption Forecasting')
    parser.add_argument('--data', type=str, default='household_power_consumption.txt', 
                       help='Path to data file')
    parser.add_argument('--freqs', nargs='+', default=['min', 'h', 'd'],
                       help='Frequencies to process: min, h, d')
    parser.add_argument('--no-optuna', action='store_true', 
                       help='Disable Optuna hyperparameter tuning')
    parser.add_argument('--no-ablation', action='store_true',
                       help='Disable ablation study')
    parser.add_argument('--no-xai', action='store_true',
                       help='Disable SHAP XAI analysis')
    parser.add_argument('--generate-templates', action='store_true',
                       help='Generate empty CSV templates and exit')
    
    args = parser.parse_args()
    
    # Generate templates if requested
    if args.generate_templates:
        print("Generating template files...")
        
        # Model parameters template
        param_template = generate_model_parameter_template()
        param_template.to_csv('model_parameters_template.csv', index=False)
        print("Saved: model_parameters_template.csv")
        
        # Empty results template
        results_template = generate_empty_results_template()
        results_template.to_csv('results_template.csv', index=False)
        print("Saved: results_template.csv")
        
        # Ablation study template
        ablation_template = generate_ablation_template()
        ablation_template.to_csv('ablation_study_template.csv', index=False)
        print("Saved: ablation_study_template.csv")
        
        print("\nTemplates generated successfully!")
        exit(0)
    
    # Run complete analysis
    print("""
    ================================================================================
    COMPLETE ENSEMBLE: Power Consumption Forecasting
    ================================================================================
    Features: lag_6, lag_12, lag_24 + rolling_12, rolling_24, rolling_48 + power_factor
    Models: TabNet, LightGBM, MLP, FT-Transformer, TabPFN + Traditional ML
    Capabilities:
      - Optuna Hyperparameter Tuning
      - SHAP XAI Analysis
      - Energy-Accuracy Tradeoff Analysis
      - Ablation Studies
      - Cross-Validation
      - Comprehensive Visualizations
      - MEMORY-EFFICIENT FT-TRANSFORMER with Automatic GPU Cleanup
      - Mixed Precision Training (AMP) for FT-Transformer
    ================================================================================
    """)
    
    # Check available libraries
    print("Library Status:")
    print(f"  Optuna: {'✓' if OPTUNA_AVAILABLE else '✗'}")
    print(f"  LightGBM: {'✓' if LIGHTGBM_AVAILABLE else '✗'}")
    print(f"  SHAP: {'✓' if SHAP_AVAILABLE else '✗'}")
    print(f"  TabNet: {'✓' if TABNET_AVAILABLE else '✗'}")
    print(f"  FT-Transformer: {'✓' if FTTRANSFORMER_AVAILABLE else '✗'}")
    print(f"  TabPFN: {'✓' if TABPFN_AVAILABLE else '✗'}")
    print(f"  PyTorch: {'✓' if torch.cuda.is_available() else '✓ (CPU)'}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print()
    
    # Run analysis
    results, output_dir = run_complete_analysis(
        data_path=args.data,
        frequencies=args.freqs,
        use_optuna=not args.no_optuna,
        run_ablation=not args.no_ablation,
        run_xai=not args.no_xai
    )
    
    print(f"\n{'='*80}")
    print("EXECUTION COMPLETE")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*80}")