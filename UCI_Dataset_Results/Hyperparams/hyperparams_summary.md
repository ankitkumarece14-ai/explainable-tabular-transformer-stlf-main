# Model Hyperparameters - S1+S2 Power Prediction

## Target: TOTAL_POWER = S1_EST_APPARENT_POWER + S2_EST_APPARENT_POWER

## Features: 12 DL-optimized features

## FT-Transformer Configurations (5 variants)

### FT-T_Small
```json
{
  "n_blocks": 2,
  "d_token": 64,
  "n_heads": 4,
  "d_ffn_factor": 2.0,
  "attention_dropout": 0.0,
  "ffn_dropout": 0.1,
  "residual_dropout": 0.0,
  "learning_rate": 0.002,
  "weight_decay": 1e-05,
  "epochs": 200,
  "patience": 30,
  "device": "cuda",
  "use_amp": true
}
```

### FT-T_Medium
```json
{
  "n_blocks": 4,
  "d_token": 128,
  "n_heads": 8,
  "d_ffn_factor": 2.0,
  "attention_dropout": 0.1,
  "ffn_dropout": 0.1,
  "residual_dropout": 0.0,
  "learning_rate": 0.001,
  "weight_decay": 1e-05,
  "epochs": 300,
  "patience": 50,
  "device": "cuda",
  "use_amp": true
}
```

### FT-T_Best
```json
{
  "n_blocks": 6,
  "d_token": 256,
  "n_heads": 8,
  "d_ffn_factor": 2.0,
  "attention_dropout": 0.1,
  "ffn_dropout": 0.1,
  "residual_dropout": 0.0,
  "learning_rate": 0.0005,
  "weight_decay": 1e-05,
  "epochs": 400,
  "patience": 60,
  "device": "cuda",
  "use_amp": true
}
```

## TabNet Configurations (2 variants)

### TabNet_Small
```json
{
  "n_d": 8,
  "n_a": 8,
  "n_steps": 3,
  "gamma": 1.0,
  "lambda_sparse": 0.01,
  "n_independent": 1,
  "n_shared": 1,
  "max_epochs": 50,
  "patience": 10,
  "batch_size": 16,
  "virtual_batch_size": 8,
  "lr": 0.1,
  "device_name": "cuda"
}
```

## LSTM
```json
{
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.2,
  "bidirectional": true,
  "epochs": 150,
  "lr": 0.001,
  "weight_decay": 1e-05,
  "patience": 30,
  "sequence_length": 10,
  "batch_size": 64
}
```

## BiLSTM
```json
{
  "hidden_dim": 64,
  "num_layers": 2,
  "dropout": 0.3,
  "bidirectional": true,
  "epochs": 100,
  "lr": 0.001,
  "weight_decay": 1e-05,
  "patience": 25,
  "sequence_length": 10,
  "batch_size": 64
}
```

## TabPFN
```json
{
  "device": "cuda",
  "random_state": 42
}
```

## Other Models

### XGBoost
```json
{
  "n_estimators": 300,
  "max_depth": 8,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.01,
  "reg_lambda": 0.1,
  "min_child_weight": 1,
  "n_jobs": -1,
  "random_state": 42
}
```

### LightGBM
```json
{
  "n_estimators": 300,
  "num_leaves": 63,
  "max_depth": 8,
  "learning_rate": 0.05,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.8,
  "bagging_freq": 5,
  "reg_alpha": 0.01,
  "reg_lambda": 0.1,
  "min_child_samples": 5,
  "n_jobs": -1,
  "random_state": 42,
  "verbose": -1
}
```

### RandomForest
```json
{
  "n_estimators": 300,
  "max_depth": 20,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "n_jobs": -1,
  "random_state": 42
}
```

### MLP_sklearn
```json
{
  "hidden_layer_sizes": [
    100,
    50
  ],
  "activation": "relu",
  "solver": "adam",
  "learning_rate_init": 0.001,
  "max_iter": 1000,
  "early_stopping": true,
  "alpha": 0.0001,
  "random_state": 42
}
```

### DecisionTree
```json
{
  "max_depth": 15,
  "min_samples_split": 5,
  "min_samples_leaf": 2,
  "random_state": 42
}
```

### LinearRegression
```json
{}
```

## PyTorch MLP
```json
{
  "hidden_dims": [
    32,
    16
  ],
  "dropout": 0.2,
  "epochs": 100,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "patience": 15,
  "batch_size": 32
}
```

## Feature List (24 features)
1. log_s1_irms
2. log_s2_irms
3. s1_irms_diff
4. s2_irms_diff
5. power_imbalance
6. s1_peak_to_irms
7. s2_peak_to_irms
8. irms_ratio
9. s1_volatility
10. total_volatility
11. sin_time
12. cos_time
