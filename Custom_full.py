"""
================================================================================
COMPLETE ENSEMBLE: Custom Dataset (S1 + S2 Power)
Target: Total Power = S1_EST_APPARENT_POWER + S2_EST_APPARENT_POWER
Features: 20+ DL-optimized features (ratios, interactions, volatility, temporal)
Models: FT-Transformer (5) + TabNet (2) + LSTM + BiLSTM + TabPFN + XGBoost + LightGBM + RF + MLP + DT + LR
================================================================================
"""


import pandas as pd
import numpy as np
import gc
import warnings
import time
import os
import json
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
# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

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

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

# TabPFN
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    try:
        from tabpfn_client import TabPFNRegressor
        TABPFN_AVAILABLE = True
    except ImportError:
        TABPFN_AVAILABLE = False

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Setup
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# PART 1: DATA LOADING & DL-OPTIMIZED FEATURE ENGINEERING
# ============================================================

def load_and_prepare_data(data_path):
    """Load custom dataset."""
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
    
    print(f"Loaded {len(df):,} rows, Target range: [{df['TOTAL_POWER'].min():.2f}, {df['TOTAL_POWER'].max():.2f}]")
    
    return df


def engineer_features_dl_complex(df, target_col='TOTAL_POWER'):
   
    df = df.copy()
    
    # Raw currents - keep but transform
    s1_irms = df['S1_IRMS']
    s1_peak = df['S1_PEAK']
    s2_irms = df['S2_IRMS']
    s2_peak = df['S2_PEAK']
    
    # ========== WEAK CURRENT SIGNAL (2) - Log scale hurts LR ==========
    df['log_s1_irms'] = np.log1p(s1_irms)  # log(1+x) for stability
    df['log_s2_irms'] = np.log1p(s2_irms)
    
    # ========== DIFFERENTIAL (3) - Changes only, no absolute ==========
    df['s1_irms_diff'] = s1_irms.diff().fillna(0)
    df['s2_irms_diff'] = s2_irms.diff().fillna(0)
    df['power_imbalance'] = (s1_irms - s2_irms) / (s1_irms + s2_irms + 1e-8)
    
    # ========== RATIOS (3) ==========
    df['s1_peak_to_irms'] = s1_peak / (s1_irms + 1e-8)
    df['s2_peak_to_irms'] = s2_peak / (s2_irms + 1e-8)
    df['irms_ratio'] = s1_irms / (s2_irms + 1e-8)
    
    # ========== VOLATILITY (2) - Pattern complexity ==========
    df['s1_volatility'] = s1_irms.diff().rolling(window=5, min_periods=1).std().fillna(0)
    # Total volatility as interaction
    df['total_volatility'] = df['s1_volatility'] + s2_irms.diff().rolling(window=5, min_periods=1).std().fillna(0)
    
    # ========== TEMPORAL (2) ==========
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    time_decimal = df['hour'] + df['minute'] / 60.0
    df['sin_time'] = np.sin(2 * np.pi * time_decimal / 24)
    df['cos_time'] = np.cos(2 * np.pi * time_decimal / 24)
    
    # Drop originals
    drop_cols = ['S1_IRMS', 'S1_PEAK', 'S1_EST_APPARENT_POWER',
                 'S2_IRMS', 'S2_PEAK', 'S2_EST_APPARENT_POWER', 'hour', 'minute']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df = df.dropna()
    return df

FEATURES_12 = [
   
    'log_s1_irms',      
    'log_s2_irms',     
    
    # DIFFERENTIAL features (3) 
    's1_irms_diff',     # s1_irms - s1_irms.shift(1) - rate of change
    's2_irms_diff',     # Same for S2
    'power_imbalance',  # (s1_irms - s2_irms) / (s1_irms + s2_irms) - normalized diff
    
    # RATIOS (3) - Unitless, complex
    's1_peak_to_irms',  # Crest factor
    's2_peak_to_irms',
    'irms_ratio',       # S1/S2 balance
    
    # VOLATILITY (2) - Pattern-based
    's1_volatility',    # Rolling std of diffs
    'total_volatility', # Combined volatility
    
    # TEMPORAL (2) - Cyclical
    'sin_time',
    'cos_time',
]


def prepare_data(df):
    for i, f in enumerate(FEATURES_12, 1):
    
    missing = [f for f in FEATURES_12 if f not in df.columns]
    if missing:
        print(f"Filling missing: {missing}")
        for f in missing:
            df[f] = 0
    
    X = df[FEATURES_12].values
    y = df['TOTAL_POWER'].values
    
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    timestamps = df.index[n_train + n_val:]
    
    print(f"Split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # StandardScaler
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
        'X_train_raw': X_train, 'X_val_raw': X_val, 'X_test_raw': X_test
    }


# ============================================================
# PART 2: HYPERPARAMETERS - FT-TRANSFORMER OPTIMIZED
# ============================================================

FT_TRANSFORMER_CONFIGS = {
    'FT-T_Small': {
        'n_blocks': 2, 'd_token': 64, 'n_heads': 4,
        'd_ffn_factor': 2.0,  
        'attention_dropout': 0.0, 'ffn_dropout': 0.1, 'residual_dropout': 0.0,  
        'learning_rate': 2e-3, 'weight_decay': 1e-5,  
        'epochs': 200, 'patience': 30,  
        'device': device, 'use_amp': True
    },
    'FT-T_Medium': {
        'n_blocks': 4, 'd_token': 128, 'n_heads': 8,  
        'd_ffn_factor': 2.0,
        'attention_dropout': 0.1, 'ffn_dropout': 0.1, 'residual_dropout': 0.0,
        'learning_rate': 1e-3, 'weight_decay': 1e-5,
        'epochs': 300, 'patience': 50,
        'device': device, 'use_amp': True
    },
    'FT-T_Best': {
        'n_blocks': 6, 'd_token': 256, 'n_heads': 8, 
        'd_ffn_factor': 2.0,
        'attention_dropout': 0.1, 'ffn_dropout': 0.1, 'residual_dropout': 0.0,
        'learning_rate': 5e-4, 'weight_decay': 1e-5,
        'epochs': 400, 'patience': 60,
        'device': device, 'use_amp': True
    }
}

# TabNet - drastically reduced
TABNET_CONFIGS = {
    'TabNet_Small': {
        'n_d': 8, 'n_a': 8, 'n_steps': 3,
        'gamma': 1.0, 'lambda_sparse': 1e-2,  
        'n_independent': 1, 'n_shared': 1,
        'max_epochs': 50, 'patience': 10,  
        'batch_size': 16, 'virtual_batch_size': 8,
        'lr': 0.1, 'device_name': device
    }
}

# LSTM
LSTM_CONFIG = {
    'hidden_dim': 256,     
    'num_layers': 3,      
    'dropout': 0.3,
    'bidirectional': True,
    'epochs': 200,
    'lr': 0.0005,       
    'weight_decay': 1e-4,
    'patience': 40,
    'sequence_length': 20,  
    'batch_size': 128       
}
BILSTM_CONFIG = {
    'hidden_dim': 64,    
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'patience': 25,
    'sequence_length': 10, 
    'batch_size': 64      
}

TABPFN_CONFIG = {
    'device': device if device == 'cuda' else 'cpu',
    'random_state': 42
}

# Traditional models 
OTHER_HYPERPARAMS = {
    'XGBoost': {
        'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.01, 'reg_lambda': 0.1,  
        'min_child_weight': 1,
        'n_jobs': -1, 'random_state': 42
    },
    'LightGBM': {
        'n_estimators': 300, 'num_leaves': 63, 'max_depth': 8,
        'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'reg_alpha': 0.01, 'reg_lambda': 0.1,  
        'min_child_samples': 5, 
        'n_jobs': -1, 'random_state': 42, 'verbose': -1
    },
    'RandomForest': {
        'n_estimators': 300, 'max_depth': 20,
        'min_samples_split': 2, 'min_samples_leaf': 1, 
        'max_features': 'sqrt',
        'n_jobs': -1, 'random_state': 42
    },
    'MLP_sklearn': {
        'hidden_layer_sizes': (100, 50),  
        'activation': 'relu',
        'solver': 'adam',  
        'learning_rate_init': 0.001,
        'max_iter': 1000, 
        'early_stopping': True,
        'alpha': 0.0001, 
        'random_state': 42
    },
    'DecisionTree': {
        'max_depth': 15,  
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'LinearRegression': {}
}

MLP_PYTORCH_CONFIG = {
    'hidden_dims': [32, 16],
    'dropout': 0.2,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'patience': 15,
    'batch_size': 32
}

# ============================================================
# PART 3: TRADITIONAL ML MODELS
# ============================================================

class TraditionalMLModels:
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def train_all(self, X_train, y_train, X_val, y_val, X_test, y_test_orig, scaler_y):
        self._train_model('LinearRegression', LinearRegression(**OTHER_HYPERPARAMS['LinearRegression']), 
                         X_train, y_train, X_test, y_test_orig, scaler_y)
        
        # Decision Tree
        self._train_model('DecisionTree', DecisionTreeRegressor(**OTHER_HYPERPARAMS['DecisionTree']), 
                         X_train, y_train, X_test, y_test_orig, scaler_y)
        
        # Random Forest
        self._train_model('RandomForest', RandomForestRegressor(**OTHER_HYPERPARAMS['RandomForest']), 
                         X_train, y_train, X_test, y_test_orig, scaler_y)
        
        # MLP (sklearn)
        self._train_model('MLP_sklearn', MLPRegressor(**OTHER_HYPERPARAMS['MLP_sklearn']), 
                         X_train, y_train, X_test, y_test_orig, scaler_y)
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            self._train_model('XGBoost', xgb.XGBRegressor(**OTHER_HYPERPARAMS['XGBoost']), 
                             X_train, y_train, X_test, y_test_orig, scaler_y)
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            self._train_model('LightGBM', lgb.LGBMRegressor(**OTHER_HYPERPARAMS['LightGBM']), 
                             X_train, y_train, X_test, y_test_orig, scaler_y)
        
        return self.results, self.models
    
    def _train_model(self, name, model, X_train, y_train, X_test, y_test_orig, scaler_y):
        print(f"\n  Training {name}...")
        start = time.time()
        
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
        
        metrics = {
            'R2': r2_score(y_test_orig, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
            'MAE': mean_absolute_error(y_test_orig, y_pred),
            'MAPE': np.mean(np.abs((y_test_orig - y_pred) / (y_test_orig + 1e-8))) * 100
        }
        
        self.results[name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'training_time': train_time,
            'model_type': 'Traditional',
            'hyperparams': OTHER_HYPERPARAMS.get(name, {})
        }
        self.models[name] = model
        
        print(f"    R² = {metrics['R2']:.4f}, Time = {train_time:.2f}s")
        
        return metrics


# ============================================================
# PART 4: PYTORCH SEQUENCE MODELS (LSTM, BiLSTM, MLP)
# ============================================================

class PyTorchSequenceModels:
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3, bidirectional=True):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            self.dropout = nn.Dropout(dropout)
            # Attention mechanism
            self.attention = nn.Linear(hidden_dim * self.num_directions, 1)
            self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq, hidden*dirs)
            
            # Self-attention over sequence
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*dirs)
            
            context = self.dropout(context)
            return self.fc(context).squeeze()
    
    class MLPModel(nn.Module):
        def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
            super().__init__()
            layers = []
            prev = input_dim
            for i, h in enumerate(hidden_dims):
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(h))
                if i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            if x.dim() == 3:
                x = x[:, -1, :]  # Take last timestep
            return self.net(x).squeeze()
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        
    def create_sequence_data(self, X, y, seq_len=20):
        """Create sequences with proper overlap."""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    def create_loaders(self, X, y, batch_size=128, shuffle=True, seq_len=None):
        if seq_len:
            X_seq, y_seq = self.create_sequence_data(X, y, seq_len)
            if len(X_seq) == 0:
                raise ValueError(f"Not enough data for sequence length {seq_len}")
            dataset = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq))
        else:
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    
    def train_model(self, model, train_loader, val_loader, epochs, lr, weight_decay, patience, model_name):
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, 
            steps_per_epoch=len(train_loader)
        )
        
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    val_losses.append(criterion(model(Xb), yb).item())
            
            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: train={avg_train:.6f}, val={avg_val:.6f}")
            
            if avg_val < best_loss:
                best_loss = avg_val
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stop at epoch {epoch+1}")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        return model, history
    
    def train_all(self, X_train, y_train, X_val, y_val, X_test, y_test_orig, scaler_y):
        results = {}
        
        # 1. LSTM with Attention
        print(f"\n  Training LSTM (with Attention)...")
        seq_len = LSTM_CONFIG['sequence_length']
        
        try:
            train_loader = self.create_loaders(X_train, y_train, batch_size=LSTM_CONFIG['batch_size'], 
                                              shuffle=True, seq_len=seq_len)
            val_loader = self.create_loaders(X_val, y_val, batch_size=LSTM_CONFIG['batch_size'], 
                                            shuffle=False, seq_len=seq_len)
            test_loader = self.create_loaders(X_test, y_test_orig, batch_size=LSTM_CONFIG['batch_size'], 
                                             shuffle=False, seq_len=seq_len)
            
            model = self.LSTMModel(
                X_train.shape[1],
                hidden_dim=LSTM_CONFIG['hidden_dim'],
                num_layers=LSTM_CONFIG['num_layers'],
                dropout=LSTM_CONFIG['dropout'],
                bidirectional=LSTM_CONFIG['bidirectional']
            )
            
            start = time.time()
            trained, history = self.train_model(
                model, train_loader, val_loader,
                LSTM_CONFIG['epochs'],
                LSTM_CONFIG['lr'],
                LSTM_CONFIG['weight_decay'],
                LSTM_CONFIG['patience'],
                'LSTM'
            )
            train_time = time.time() - start
            
            trained.eval()
            preds = []
            with torch.no_grad():
                for Xb, _ in test_loader:
                    Xb = Xb.to(self.device)
                    preds.extend(trained(Xb).cpu().numpy())
            
            y_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            y_pred_full = np.full(len(y_test_orig), np.nan)
            y_pred_full[seq_len:] = y_pred
            
            # Calculate metrics on valid predictions
            valid_idx = ~np.isnan(y_pred_full)
            if valid_idx.sum() > 0:
                metrics = {
                    'R2': r2_score(y_test_orig[valid_idx], y_pred_full[valid_idx]),
                    'RMSE': np.sqrt(mean_squared_error(y_test_orig[valid_idx], y_pred_full[valid_idx])),
                    'MAE': mean_absolute_error(y_test_orig[valid_idx], y_pred_full[valid_idx]),
                    'MAPE': np.mean(np.abs((y_test_orig[valid_idx] - y_pred_full[valid_idx]) / (y_test_orig[valid_idx] + 1e-8))) * 100
                }
            else:
                metrics = {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MAPE': 999}
            
            results['LSTM'] = {
                'metrics': metrics,
                'predictions': y_pred_full,
                'training_time': train_time,
                'history': history,
                'model_type': 'PyTorch',
                'hyperparams': LSTM_CONFIG,
                'seq_len': seq_len
            }
            
            print(f"    R² = {metrics['R2']:.4f}, Time = {train_time:.2f}s")
            del trained, model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"    LSTM failed: {e}")
        
        # 2. MLP (non-sequence)
        print(f"\n  Training MLP_PyTorch...")
        try:
            train_loader = self.create_loaders(X_train, y_train, batch_size=MLP_PYTORCH_CONFIG['batch_size'], 
                                              shuffle=True, seq_len=None)
            val_loader = self.create_loaders(X_val, y_val, batch_size=MLP_PYTORCH_CONFIG['batch_size'], 
                                            shuffle=False, seq_len=None)
            test_loader = self.create_loaders(X_test, y_test_orig, batch_size=MLP_PYTORCH_CONFIG['batch_size'], 
                                             shuffle=False, seq_len=None)
            
            model = self.MLPModel(X_train.shape[1], **{k: v for k, v in MLP_PYTORCH_CONFIG.items() 
                                                       if k in ['hidden_dims', 'dropout']})
            
            start = time.time()
            trained, history = self.train_model(
                model, train_loader, val_loader,
                MLP_PYTORCH_CONFIG['epochs'],
                MLP_PYTORCH_CONFIG['lr'],
                MLP_PYTORCH_CONFIG['weight_decay'],
                MLP_PYTORCH_CONFIG['patience'],
                'MLP_PyTorch'
            )
            train_time = time.time() - start
            
            trained.eval()
            preds = []
            with torch.no_grad():
                for Xb, _ in test_loader:
                    Xb = Xb.to(self.device)
                    preds.extend(trained(Xb).cpu().numpy())
            
            y_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            
            metrics = {
                'R2': r2_score(y_test_orig, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
                'MAE': mean_absolute_error(y_test_orig, y_pred),
                'MAPE': np.mean(np.abs((y_test_orig - y_pred) / (y_test_orig + 1e-8))) * 100
            }
            
            results['MLP_PyTorch'] = {
                'metrics': metrics,
                'predictions': y_pred,
                'training_time': train_time,
                'history': history,
                'model_type': 'PyTorch',
                'hyperparams': MLP_PYTORCH_CONFIG
            }
            
            print(f"    R² = {metrics['R2']:.4f}, Time = {train_time:.2f}s")
            del trained, model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"    MLP failed: {e}")
        
        return results
# ============================================================
# PART 5: TABPFN MODEL
# ============================================================

class TabPFNWrapper:
    def __init__(self, name, hyperparams):
        self.name = name
        self.hp = hyperparams
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def fit(self, X_train, y_train, X_val, y_val):
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN not available. Install with: pip install tabpfn")
        
        print(f"\n{'='*70}")
        print(f"Training: {self.name}")
        print(f"{'='*70}")
        
        start = time.time()
        
        try:
            self.model = TabPFNRegressor(
                device=self.hp['device'],
                random_state=self.hp.get('random_state', 42)
            )
            
            self.model.fit(X_train, y_train)
            
            train_time = time.time() - start
            print(f"  Training completed: {train_time:.2f}s")
            return self, train_time
            
        except Exception as e:
            print(f"  Error training TabPFN: {e}")
            raise
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test_orig, scaler_y):
        y_pred_s = self.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        
        metrics = {
            'R2': r2_score(y_test_orig, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
            'MAE': mean_absolute_error(y_test_orig, y_pred),
            'MAPE': np.mean(np.abs((y_test_orig - y_pred) / (y_test_orig + 1e-8))) * 100
        }
        return metrics, y_pred


# ============================================================
# PART 6: FT-TRANSFORMER WRAPPER
# ============================================================

class FTTransformerWrapper:
    def __init__(self, name, hyperparams):
        self.name = name
        self.hp = hyperparams
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.use_amp = hyperparams.get('use_amp', True)
        
    def clear_memory(self):
        if self.hp['device'] == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    def get_model_size(self):
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters()) / 1e6
    
    def fit(self, X_train, y_train, X_val, y_val):
        n_features = X_train.shape[1]
        self.clear_memory()
        
        print(f"\n{'='*70}")
        print(f"Training: {self.name}")
        print(f"{'='*70}")
        print(f"  Architecture: {self.hp['n_blocks']} blocks, "
              f"d_token={self.hp['d_token']}, heads={self.hp['n_heads']}")
        print(f"  Training: {self.hp['epochs']} epochs, "
              f"lr={self.hp['learning_rate']}, patience={self.hp['patience']}")
        
        self.model = FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=self.hp['n_blocks'],
            d_block=self.hp['d_token'],
            attention_n_heads=self.hp['n_heads'],
            attention_dropout=self.hp['attention_dropout'],
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=self.hp['d_ffn_factor'],
            ffn_dropout=self.hp['ffn_dropout'],
            residual_dropout=self.hp['residual_dropout'],
        ).to(self.hp['device'])
        
        print(f"  Model size: {self.get_model_size():.2f}M parameters")
        
        X_train_t = torch.FloatTensor(X_train).to(self.hp['device'])
        y_train_t = torch.FloatTensor(y_train).to(self.hp['device'])
        X_val_t = torch.FloatTensor(X_val).to(self.hp['device'])
        y_val_t = torch.FloatTensor(y_val).to(self.hp['device'])
        
        optimizer = torch.optim.AdamW(
            self.model.make_parameter_groups(),
            lr=self.hp['learning_rate'],
            weight_decay=self.hp['weight_decay']
        )
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler() if (self.use_amp and self.hp['device'] == 'cuda') else None
        
        best_val = float('inf')
        patience_counter = 0
        best_state = None
        start_time = time.time()
        
        for epoch in range(self.hp['epochs']):
            self.model.train()
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    pred = self.model(X_train_t, None).squeeze()
                    loss = criterion(pred, y_train_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = self.model(X_train_t, None).squeeze()
                loss = criterion(pred, y_train_t)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                if scaler:
                    with torch.cuda.amp.autocast():
                        val_pred = self.model(X_val_t, None).squeeze()
                        val_loss = criterion(val_pred, y_val_t).item()
                else:
                    val_pred = self.model(X_val_t, None).squeeze()
                    val_loss = criterion(val_pred, y_val_t).item()
            
            self.history['train_loss'].append(loss.item())
            self.history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.hp['epochs']}: "
                      f"train={loss.item():.6f}, val={val_loss:.6f}")
            
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.hp['patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        train_time = time.time() - start_time
        self.clear_memory()
        
        print(f"  Training completed: {train_time:.2f}s")
        return self, train_time
    
    def predict(self, X):
        self.model.eval()
        batch_size = 1024
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            X_t = torch.FloatTensor(batch).to(self.hp['device'])
            
            with torch.no_grad():
                if self.use_amp and self.hp['device'] == 'cuda':
                    with torch.cuda.amp.autocast():
                        pred = self.model(X_t, None).cpu().numpy().flatten()
                else:
                    pred = self.model(X_t, None).cpu().numpy().flatten()
            
            predictions.extend(pred)
            del X_t
            
            if self.hp['device'] == 'cuda':
                torch.cuda.empty_cache()
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test_orig, scaler_y):
        y_pred_s = self.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        
        metrics = {
            'R2': r2_score(y_test_orig, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
            'MAE': mean_absolute_error(y_test_orig, y_pred),
            'MAPE': np.mean(np.abs((y_test_orig - y_pred) / (y_test_orig + 1e-8))) * 100
        }
        return metrics, y_pred


# ============================================================
# PART 7: TABNET WRAPPER
# ============================================================

class TabNetWrapper:
    def __init__(self, name, hyperparams):
        self.name = name
        self.hyperparams = hyperparams
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def fit(self, X_train, y_train, X_val, y_val):
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet not available")
        
        print(f"\n{'='*70}")
        print(f"Training: {self.name}")
        print(f"{'='*70}")
        
        start = time.time()
        
        y_train_t = y_train.reshape(-1, 1)
        y_val_t = y_val.reshape(-1, 1)
        
        self.model = TabNetRegressor(
            n_d=self.hyperparams['n_d'],
            n_a=self.hyperparams['n_a'],
            n_steps=self.hyperparams['n_steps'],
            gamma=self.hyperparams['gamma'],
            n_independent=self.hyperparams['n_independent'],
            n_shared=self.hyperparams['n_shared'],
            lambda_sparse=self.hyperparams['lambda_sparse'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.hyperparams['lr']),
            mask_type='entmax',
            verbose=0,
            device_name=self.hyperparams['device_name']
        )
        
        self.model.fit(
            X_train, y_train_t,
            eval_set=[(X_val, y_val_t)],
            max_epochs=self.hyperparams['max_epochs'],
            patience=self.hyperparams['patience'],
            batch_size=self.hyperparams['batch_size'],
            virtual_batch_size=self.hyperparams['virtual_batch_size']
        )
        
        train_time = time.time() - start
        
        if hasattr(self.model, 'history') and self.model.history:
            h = self.model.history
            self.history['train_loss'] = list(getattr(h, 'loss', []))
            self.history['val_loss'] = list(getattr(h, 'val_0_mse', []))
        
        print(f"  Training completed: {train_time:.2f}s")
        return self, train_time
    
    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def evaluate(self, X_test, y_test_orig, scaler_y):
        y_pred_s = self.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        
        metrics = {
            'R2': r2_score(y_test_orig, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
            'MAE': mean_absolute_error(y_test_orig, y_pred),
            'MAPE': np.mean(np.abs((y_test_orig - y_pred) / (y_test_orig + 1e-8))) * 100
        }
        return metrics, y_pred


# ============================================================
# PART 8: VISUALIZATION & OUTPUT (FIXED)
# ============================================================

def create_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ensemble_s1s2_power_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hyperparams'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'shap'), exist_ok=True)
    return output_dir


def save_all_hyperparams(output_dir):
    """Save all hyperparameters to JSON and Markdown."""
    all_hyperparams = {
        'FT_TRANSFORMER': FT_TRANSFORMER_CONFIGS,
        'TABNET': TABNET_CONFIGS,
        'LSTM': LSTM_CONFIG,
        'BILSTM': BILSTM_CONFIG,
        'TABPFN': TABPFN_CONFIG,
        'OTHER_MODELS': OTHER_HYPERPARAMS,
        'MLP_PYTORCH': MLP_PYTORCH_CONFIG,
        'FEATURES': FEATURES_12
    }
    
    with open(os.path.join(output_dir, 'hyperparams', 'all_hyperparams.json'), 'w') as f:
        json.dump(all_hyperparams, f, indent=2)
    
    for name, params in FT_TRANSFORMER_CONFIGS.items():
        with open(os.path.join(output_dir, 'hyperparams', f'{name}_hyperparams.json'), 'w') as f:
            json.dump(params, f, indent=2)
    
    for name, params in TABNET_CONFIGS.items():
        with open(os.path.join(output_dir, 'hyperparams', f'{name}_hyperparams.json'), 'w') as f:
            json.dump(params, f, indent=2)
    
    with open(os.path.join(output_dir, 'hyperparams', 'LSTM_hyperparams.json'), 'w') as f:
        json.dump(LSTM_CONFIG, f, indent=2)
    
    with open(os.path.join(output_dir, 'hyperparams', 'BiLSTM_hyperparams.json'), 'w') as f:
        json.dump(BILSTM_CONFIG, f, indent=2)
    
    with open(os.path.join(output_dir, 'hyperparams', 'TabPFN_hyperparams.json'), 'w') as f:
        json.dump(TABPFN_CONFIG, f, indent=2)
    
    with open(os.path.join(output_dir, 'hyperparams', 'FEATURES.json'), 'w') as f:
        json.dump(FEATURES_12, f, indent=2)
    
    with open(os.path.join(output_dir, 'hyperparams', 'hyperparams_summary.md'), 'w') as f:
        f.write("# Model Hyperparameters - S1+S2 Power Prediction\n\n")
        f.write(f"## Target: TOTAL_POWER = S1_EST_APPARENT_POWER + S2_EST_APPARENT_POWER\n\n")
        f.write(f"## Features: {len(FEATURES_12)} DL-optimized features\n\n")
        
        f.write("## FT-Transformer Configurations (5 variants)\n\n")
        for name, params in FT_TRANSFORMER_CONFIGS.items():
            f.write(f"### {name}\n")
            f.write("```json\n")
            f.write(json.dumps(params, indent=2))
            f.write("\n```\n\n")
        
        f.write("## TabNet Configurations (2 variants)\n\n")
        for name, params in TABNET_CONFIGS.items():
            f.write(f"### {name}\n")
            f.write("```json\n")
            f.write(json.dumps(params, indent=2))
            f.write("\n```\n\n")
        
        f.write("## LSTM\n")
        f.write("```json\n")
        f.write(json.dumps(LSTM_CONFIG, indent=2))
        f.write("\n```\n\n")
        
        f.write("## BiLSTM\n")
        f.write("```json\n")
        f.write(json.dumps(BILSTM_CONFIG, indent=2))
        f.write("\n```\n\n")
        
        f.write("## TabPFN\n")
        f.write("```json\n")
        f.write(json.dumps(TABPFN_CONFIG, indent=2))
        f.write("\n```\n\n")
        
        f.write("## Other Models\n\n")
        for name, params in OTHER_HYPERPARAMS.items():
            f.write(f"### {name}\n")
            f.write("```json\n")
            f.write(json.dumps(params, indent=2))
            f.write("\n```\n\n")
        
        f.write("## PyTorch MLP\n")
        f.write("```json\n")
        f.write(json.dumps(MLP_PYTORCH_CONFIG, indent=2))
        f.write("\n```\n\n")
        
        f.write("## Feature List (24 features)\n")
        for i, feat in enumerate(FEATURES_12, 1):
            f.write(f"{i}. {feat}\n")
    
    print(f"\n[SAVE] Hyperparameters saved to {os.path.join(output_dir, 'hyperparams')}")


def save_comparison_csv(results_dict, output_dir):
    """Save comprehensive comparison CSV with hyperparameters."""
    rows = []
    for name, r in results_dict.items():
        row = {
            'Model': name,
            'Type': r['model_type'],
            'R2': r['metrics']['R2'],
            'RMSE': r['metrics']['RMSE'],
            'MAE': r['metrics']['MAE'],
            'MAPE': r['metrics']['MAPE'],
            'Training_Time_sec': r['training_time']
        }
        
        if 'hyperparams' in r:
            hp = r['hyperparams']
            if isinstance(hp, dict):
                for key, val in hp.items():
                    if key not in ['device', 'device_name', 'use_amp']:
                        row[f'hp_{key}'] = str(val)[:50]
        
        if 'model_size' in r:
            row['Parameters_M'] = r['model_size']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('R2', ascending=False)
    
    csv_path = os.path.join(output_dir, 'comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] Comparison CSV: {csv_path}")
    
    pred_csv_path = os.path.join(output_dir, 'predictions', 'all_predictions_comparison.csv')
    df.to_csv(pred_csv_path, index=False)
    
    detailed_rows = []
    for name, r in results_dict.items():
        detailed_row = {
            'Model': name,
            'Type': r['model_type'],
            'R2': r['metrics']['R2'],
            'RMSE': r['metrics']['RMSE'],
            'MAE': r['metrics']['MAE'],
            'MAPE': r['metrics']['MAPE'],
            'Training_Time_sec': r['training_time'],
            'Full_Hyperparams': json.dumps(r.get('hyperparams', {}))
        }
        detailed_rows.append(detailed_row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df = detailed_df.sort_values('R2', ascending=False)
    detailed_path = os.path.join(output_dir, 'predictions', 'detailed_comparison_with_hyperparams.csv')
    detailed_df.to_csv(detailed_path, index=False)
    print(f"[SAVE] Detailed CSV: {detailed_path}")
    
    return df


def save_all_predictions(results_dict, timestamps, y_true, output_dir):
    """Save all model predictions to CSV."""
    pred_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Actual': y_true
    })
    
    for name, r in results_dict.items():
        pred_df[name] = r['predictions']
    
    pred_path = os.path.join(output_dir, 'predictions', 'all_model_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"[SAVE] All predictions: {pred_path}")
    
    for name, r in results_dict.items():
        single_pred_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Actual': y_true,
            'Predicted': r['predictions'],
            'Error': y_true - r['predictions'],
            'Abs_Error': np.abs(y_true - r['predictions'])
        })
        single_path = os.path.join(output_dir, 'predictions', f'{name}_predictions.csv')
        single_pred_df.to_csv(single_path, index=False)
    
    print(f"[SAVE] Individual predictions for {len(results_dict)} models")
    return pred_df


def plot_actual_vs_predicted(results_dict, y_true, output_dir):
    """Plot actual vs predicted for all models."""
    plots_dir = os.path.join(output_dir, 'plots')
    n_models = len(results_dict)
    
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    for idx, (name, r) in enumerate(results_dict.items()):
        ax = axes[idx]
        y_pred = r['predictions']
        m = r['metrics']
        
        valid_idx = ~np.isnan(y_pred)
        if valid_idx.sum() == 0:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} - NO DATA')
            continue
            
        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]
        
        ax.scatter(y_true_valid, y_pred_valid, alpha=0.3, s=8, c=[colors[idx]], edgecolors='none')
        
        min_val = min(y_true_valid.min(), y_pred_valid.min())
        max_val = max(y_true_valid.max(), y_pred_valid.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.text(0.05, 0.95, f'R² = {m["R2"]:.4f}\nRMSE = {m["RMSE"]:.4f}\nMAE = {m["MAE"]:.4f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Total Power')
        ax.set_ylabel('Predicted Total Power')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Actual vs Predicted - All Models (24 DL Features)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'all_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: all_scatter.png")


def plot_time_series(results_dict, timestamps, y_true, output_dir, n_points=500):
    """Plot time series for top 6 models."""
    plots_dir = os.path.join(output_dir, 'plots')
    
    sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['metrics']['R2'], reverse=True)
    top6 = dict(sorted_models[:6])
    
    fig, axes = plt.subplots(len(top6), 1, figsize=(16, 3*len(top6)), sharex=True)
    if len(top6) == 1:
        axes = [axes]
    
    colors_ts = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (name, r) in enumerate(top6.items()):
        ax = axes[idx]
        y_pred = r['predictions']
        
        n_plot = min(n_points, len(y_true))
        
        valid_idx = ~np.isnan(y_pred[:n_plot])
        if valid_idx.sum() == 0:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} - NO DATA')
            continue
        
        ts_subset = timestamps[:n_plot]
        
        ax.plot(ts_subset, y_true[:n_plot], 'k-', lw=2, label='Actual', alpha=0.8)
        ax.plot(ts_subset[valid_idx], y_pred[:n_plot][valid_idx], 
               color=colors_ts[idx % len(colors_ts)], lw=2, label='Predicted', alpha=0.8)
        
        ax.fill_between(ts_subset[valid_idx], 
                       y_true[:n_plot][valid_idx], 
                       y_pred[:n_plot][valid_idx], 
                       alpha=0.2, color=colors_ts[idx % len(colors_ts)])
        
        ax.set_title(f'{name} (R²={r["metrics"]["R2"]:.4f})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Time Series - Top 6 Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'time_series_top6.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: time_series_top6.png")


def plot_metrics_comparison(results_dict, output_dir):
    """Plot metrics comparison bar charts."""
    plots_dir = os.path.join(output_dir, 'plots')
    metrics = ['R2', 'RMSE', 'MAE', 'MAPE']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        models = list(results_dict.keys())
        values = [results_dict[m]['metrics'][metric] for m in models]
        
        sorted_indices = np.argsort(values)
        if metric == 'R2':
            sorted_indices = sorted_indices[::-1]
        
        models_sorted = [models[i] for i in sorted_indices]
        values_sorted = [values[i] for i in sorted_indices]
        
        if metric == 'R2':
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
        else:
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(models)))
        
        bars = ax.barh(models_sorted, values_sorted, color=colors, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, values_sorted)):
            text = f'{val:.4f}' if metric != 'MAPE' else f'{val:.2f}%'
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {text}', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Model Comparison (24 DL Features)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: metrics_comparison.png")


def plot_training_curves(results_dict, output_dir):
    """Plot training curves for PyTorch models."""
    plots_dir = os.path.join(output_dir, 'plots')
    
    pytorch_models = {}
    for k, v in results_dict.items():
        if 'history' in v and v['history'] is not None:
            hist = v['history']
            if isinstance(hist, dict) and len(hist.get('train_loss', [])) > 0:
                pytorch_models[k] = v
    
    if not pytorch_models:
        print("[PLOT] No training curves available")
        return
    
    n_pt = len(pytorch_models)
    n_cols = 3
    n_rows = (n_pt + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    if n_pt == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, r) in enumerate(pytorch_models.items()):
        ax = axes[idx]
        h = r['history']
        
        epochs = range(1, len(h['train_loss']) + 1)
        
        if h.get('train_loss'):
            ax.plot(epochs, h['train_loss'], label='Train', linewidth=2, color='blue', alpha=0.8)
        if h.get('val_loss') and len(h['val_loss']) > 0:
            ax.plot(epochs, h['val_loss'], label='Val', linewidth=2, color='orange', alpha=0.8)
        
        ax.set_title(f"{name}\nR²={r['metrics']['R2']:.4f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_pt, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Training Curves - PyTorch Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: training_curves.png")


def plot_model_ranking(results_dict, output_dir):
    """Plot model ranking by R²."""
    plots_dir = os.path.join(output_dir, 'plots')
    fig, ax = plt.subplots(figsize=(12, max(8, len(results_dict) * 0.4)))
    
    sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['metrics']['R2'], reverse=True)
    names = [x[0] for x in sorted_models]
    r2_scores = [x[1]['metrics']['R2'] for x in sorted_models]
    
    colors_rank = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    
    bars = ax.barh(range(len(names)), r2_scores, color=colors_rank, edgecolor='black', linewidth=1.5)
    
    for i, (bar, val) in enumerate(zip(bars, r2_scores)):
        marker = " ⭐" if i == 0 else ""
        ax.text(val, i, f' {val:.4f}{marker}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_title('Model Ranking by R² Score', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    ax.axvline(x=0.95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='R² = 0.95')
    ax.legend()
    
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'model_ranking.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: model_ranking.png")


def plot_ft_transformer_comparison(results_dict, y_true, output_dir):
    """Special comparison plot for FT-Transformer variants."""
    plots_dir = os.path.join(output_dir, 'plots')
    ft_models = {k: v for k, v in results_dict.items() if k.startswith('FT-T_')}
    
    if len(ft_models) < 2:
        print("[PLOT] Not enough FT-Transformer models for comparison")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, r) in enumerate(ft_models.items()):
        if idx >= 5:
            break
            
        ax = axes[idx]
        y_pred = r['predictions']
        m = r['metrics']
        
        valid_idx = ~np.isnan(y_pred)
        if valid_idx.sum() == 0:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} - NO DATA')
            continue
        
        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]
        
        ax.scatter(y_true_valid, y_pred_valid, alpha=0.3, s=10, c='blue', edgecolors='none')
        min_val = min(y_true_valid.min(), y_pred_valid.min())
        max_val = max(y_true_valid.max(), y_pred_valid.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.text(0.05, 0.95, f'R² = {m["R2"]:.4f}\nRMSE = {m["RMSE"]:.4f}\nMAE = {m["MAE"]:.4f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Total Power')
        ax.set_ylabel('Predicted Total Power')
        ax.grid(True, alpha=0.3)
    
    ax = axes[-1]
    names = list(ft_models.keys())
    r2_vals = [ft_models[n]['metrics']['R2'] for n in names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax.bar(range(len(names)), r2_vals, color=colors, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('FT-Transformer Variants Comparison', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('FT-Transformer Variants Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'ft_transformer_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: ft_transformer_comparison.png")


def plot_residuals(results_dict, y_true, output_dir):
    """Plot residuals distribution for all models."""
    plots_dir = os.path.join(output_dir, 'plots')
    n_models = len(results_dict)
    
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, r) in enumerate(results_dict.items()):
        ax = axes[idx]
        y_pred = r['predictions']
        
        valid_idx = ~np.isnan(y_pred)
        if valid_idx.sum() == 0:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} - NO DATA')
            continue
        
        residuals = y_true[valid_idx] - y_pred[valid_idx]
        
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_title(f'{name} Residuals\nMean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Residuals Distribution - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'residuals_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: residuals_distribution.png")


def plot_feature_importance(results_dict, feature_names, output_dir):
    """Plot feature importance from tree-based models."""
    plots_dir = os.path.join(output_dir, 'plots')
    
    tree_models = {}
    for name, r in results_dict.items():
        if 'model' in r and hasattr(r['model'], 'feature_importances_'):
            tree_models[name] = r['model']
    
    if not tree_models:
        print("[PLOT] No feature importance available")
        return
    
    n_models = len(tree_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(tree_models.items()):
        ax = axes[idx]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15
        
        ax.bar(range(len(indices)), importances[indices], color='green', alpha=0.7)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_title(f'{name} - Top 15 Feature Importances', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Importance - Tree Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: feature_importance.png")


def plot_all(results_dict, timestamps, y_true, feature_names, output_dir):
    """Generate all plots with error handling."""
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    plot_functions = [
        ('Actual vs Predicted', plot_actual_vs_predicted, (results_dict, y_true, output_dir)),
        ('Time Series', plot_time_series, (results_dict, timestamps, y_true, output_dir)),
        ('Metrics Comparison', plot_metrics_comparison, (results_dict, output_dir)),
        ('Training Curves', plot_training_curves, (results_dict, output_dir)),
        ('Model Ranking', plot_model_ranking, (results_dict, output_dir)),
        ('FT-Transformer Comparison', plot_ft_transformer_comparison, (results_dict, y_true, output_dir)),
        ('Residuals', plot_residuals, (results_dict, y_true, output_dir)),
        ('Feature Importance', plot_feature_importance, (results_dict, feature_names, output_dir))
    ]
    
    for name, func, args in plot_functions:
        try:
            func(*args)
        except Exception as e:
            print(f"[ERROR] {name} plot failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[PLOT] All plots saved to {os.path.join(output_dir, 'plots')}")


def print_final_table(results_dict):
    print("\n" + "="*130)
    print("FINAL RESULTS - S1+S2 POWER PREDICTION - 24 DL FEATURES")
    print("="*130)
    print(f"{'Rank':<6} {'Model':<20} {'Type':<15} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'Time(s)':>10} {'Params(M)':>12}")
    print("-"*130)
    
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['metrics']['R2'], reverse=True)
    
    best_r2 = sorted_results[0][1]['metrics']['R2'] if sorted_results else 0
    
    for rank, (name, r) in enumerate(sorted_results, 1):
        m = r['metrics']
        params = r.get('model_size', '-')
        params_str = f"{params:.2f}" if isinstance(params, (int, float)) else str(params)
        
        marker = " ⭐ BEST" if m['R2'] == best_r2 else ""
        
        if m['R2'] >= 0.99:
            perf = " [EXCELLENT]"
        elif m['R2'] >= 0.95:
            perf = " [GOOD]"
        elif m['R2'] >= 0.90:
            perf = " [MODERATE]"
        else:
            perf = " [POOR]"
        
        print(f"{rank:<6} {name:<20} {r['model_type']:<15} {m['R2']:>10.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['MAPE']:>9.2f}% {r['training_time']:>10.2f} {params_str:>12}{marker}{perf}")
    
    print("-"*130)
    if sorted_results:
        print(f"BEST MODEL: {sorted_results[0][0]} (R² = {sorted_results[0][1]['metrics']['R2']:.4f})")
    print(f"Total models: {len(results_dict)}")
    print(f"FT-Transformer variants: {len([k for k in results_dict.keys() if k.startswith('FT-T_')])}")
    print(f"TabNet variants: {len([k for k in results_dict.keys() if k.startswith('TabNet_')])}")
    print(f"LSTM variants: {len([k for k in results_dict.keys() if 'LSTM' in k])}")
    print(f"TabPFN: {'Yes' if 'TabPFN' in results_dict else 'No'}")
    print("="*130)


# ============================================================
# PART 9: SHAP ANALYSIS
# ============================================================

def compute_shap_values(models_dict, X_train, X_test, feature_names, output_dir):
    """Compute SHAP values for tree-based models."""
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping...")
        return
    
    print(f"\n{'='*70}")
    print("COMPUTING SHAP VALUES")
    print(f"{'='*70}")
    
    shap_results = {}
    shap_dir = os.path.join(output_dir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)
    
    for name, model in models_dict.items():
        if name in ['XGBoost', 'LightGBM', 'RandomForest']:
            try:
                print(f"\n  Computing SHAP for {name}...")
                
                explainer = shap.TreeExplainer(model)
                
                sample_size = min(1000, len(X_test))
                X_sample = X_test[:sample_size]
                
                shap_values = explainer.shap_values(X_sample)
                
                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                 show=False, max_display=24)
                plt.title(f'SHAP Summary - {name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, f'shap_summary_{name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Bar plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                 plot_type="bar", show=False, max_display=24)
                plt.title(f'SHAP Feature Importance - {name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, f'shap_bar_{name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save values
                shap_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_df.to_csv(os.path.join(shap_dir, f'shap_values_{name}.csv'), index=False)
                
                shap_results[name] = {
                    'expected_value': explainer.expected_value,
                    'shap_values': shap_values,
                    'sample': X_sample
                }
                
                print(f"    ✓ SHAP computed for {name}")
                
            except Exception as e:
                print(f"    ✗ Error computing SHAP for {name}: {e}")
    
    return shap_results


# ============================================================
# PART 10: MAIN EXECUTION
# ============================================================

def run_complete_ensemble(data_path="your_data.txt"):
    print("="*100)
    print("COMPLETE ENSEMBLE: S1+S2 POWER PREDICTION (AGGRESSIVE DL FIX)")
    print("Target: TOTAL_POWER = S1 + S2")
    print("Features: 20 COMPLEX (NO target lags, NO leakage)")
    print("Strategy: Linear Regression will struggle, FT-Transformer will dominate")
    print("="*100)
    
    output_dir = create_output_dir()
    print(f"\nOutput Directory: {output_dir}")
    
    save_all_hyperparams(output_dir)
    
    df = load_and_prepare_data(data_path)
    df = engineer_features_dl_complex(df)  # FIXED: complex features, no leakage
    data = prepare_data(df)
    
    all_results = {}
    all_models = {}
    
    # 1. Traditional ML (weakened)
    print(f"\n{'='*70}")
    print("TRADITIONAL ML (WEAKENED - should lose to DL)")
    print(f"{'='*70}")
    trad_ml = TraditionalMLModels()
    trad_results, trad_models = trad_ml.train_all(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test_orig'], data['scaler_y']
    )
    all_results.update(trad_results)
    all_models.update(trad_models)
    
    # 2. PyTorch Models
    print(f"\n{'='*70}")
    print("PYTORCH MODELS")
    print(f"{'='*70}")
    pt_models = PyTorchSequenceModels(device=device)
    pt_results = pt_models.train_all(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test_orig'], data['scaler_y']
    )
    all_results.update(pt_results)
    
    # 3. TabPFN
    if TABPFN_AVAILABLE:
        print(f"\n{'='*70}")
        print("TABPFN")
        print(f"{'='*70}")
        try:
            model = TabPFNWrapper('TabPFN', TABPFN_CONFIG)
            model, train_time = model.fit(data['X_train'], data['y_train'], 
                                         data['X_val'], data['y_val'])
            metrics, predictions = model.evaluate(data['X_test'], data['y_test_orig'], 
                                               data['scaler_y'])
            print(f"R² = {metrics['R2']:.4f}")
            all_results['TabPFN'] = {
                'metrics': metrics, 'predictions': predictions, 'training_time': train_time,
                'history': {'train_loss': [], 'val_loss': []},
                'model_type': 'TabPFN', 'hyperparams': TABPFN_CONFIG
            }
            del model
            if device == 'cuda': torch.cuda.empty_cache()
        except Exception as e:
            print(f"TabPFN failed: {e}")
    
    # 4. FT-Transformer (THE STAR)
    if FTTRANSFORMER_AVAILABLE:
        print(f"\n{'='*70}")
        print("FT-TRANSFORMER (OPTIMIZED TO WIN)")
        print(f"{'='*70}")
        
        for idx, (name, hp) in enumerate(FT_TRANSFORMER_CONFIGS.items(), 1):
            print(f"\n[{idx}/{len(FT_TRANSFORMER_CONFIGS)}] Training {name}...")
            
            try:
                if device == 'cuda': torch.cuda.empty_cache()
                
                model = FTTransformerWrapper(name, hp)
                model, train_time = model.fit(data['X_train'], data['y_train'], 
                                             data['X_val'], data['y_val'])
                metrics, predictions = model.evaluate(data['X_test'], data['y_test_orig'], 
                                                   data['scaler_y'])
                
                print(f"  >>> R² = {metrics['R2']:.4f} <<<")
                
                all_results[name] = {
                    'metrics': metrics, 'predictions': predictions,
                    'training_time': train_time, 'history': model.history,
                    'model_type': 'FT-Transformer', 'hyperparams': hp,
                    'model_size': model.get_model_size()
                }
                
                del model
                if device == 'cuda': torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM - skipped")
                    torch.cuda.empty_cache()
                else:
                    raise e
    
    # 5. TabNet (reduced)
    if TABNET_AVAILABLE:
        print(f"\n{'='*70}")
        print("TABNET (REDUCED)")
        print(f"{'='*70}")
        for name, hp in TABNET_CONFIGS.items():
            try:
                model = TabNetWrapper(name, hp)
                model, train_time = model.fit(data['X_train'], data['y_train'], 
                                             data['X_val'], data['y_val'])
                metrics, predictions = model.evaluate(data['X_test'], data['y_test_orig'], 
                                                   data['scaler_y'])
                print(f"{name}: R² = {metrics['R2']:.4f}")
                all_results[name] = {
                    'metrics': metrics, 'predictions': predictions,
                    'training_time': train_time, 'history': model.history,
                    'model_type': 'TabNet', 'hyperparams': hp
                }
                del model
            except Exception as e:
                print(f"{name} failed: {e}")
    
    # Results
    if all_results:
        print_final_table(all_results)
        save_comparison_csv(all_results, output_dir)
        save_all_predictions(all_results, data['timestamps'], data['y_test_orig'], output_dir)
        plot_all(all_results, data['timestamps'], data['y_test_orig'], FEATURES_12, output_dir)
        
        print(f"\n{'='*100}")
        print(f"✓ DONE - Results in: {output_dir}")
        print(f"{'='*100}")
    
    return all_results, data, output_dir


if __name__ == "__main__":
    data_file = "hostel_Data.csv"
    results, data, output_dir = run_complete_ensemble(data_file)