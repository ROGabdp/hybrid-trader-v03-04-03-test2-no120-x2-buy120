# -*- coding: utf-8 -*-
"""
TWII 20 日預測模型參數優化腳本 (Hyperparameter Optimizer)
針對 2000-01-01 ~ 2022-12-31 長週期數據尋找最佳參數

測試參數組合 (Grid Search):
1. LSTM Units: [128, 256]
2. Dropout Rate: [0.2, 0.3, 0.5]
3. Batch Size: [16, 32, 64]

使用方式：
  python twii_model_optimizer_20d.py
"""

import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
import twii_model_registry_20d as model_registry

# =============================================================================
# 設定
# =============================================================================
TRAIN_START = "2000-01-01"
TRAIN_END = "2022-12-31"

# Grid Search 候選參數
PARAM_GRID = {
    'lstm_units': [128, 256],
    'dropout_rate': [0.2, 0.3, 0.5],
    'batch_size': [16, 32, 64]
}

def optimize():
    print("\n" + "=" * 70)
    print("  🚀 TWII T+20 Model Hyperparameter Optimizer")
    print("=" * 70)
    print(f"  Training Range: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  Parameter Grid: {PARAM_GRID}")
    print("=" * 70)

    # 1. 準備數據 (只做一次預處理)
    df = model_registry.download_data_by_date_range(TRAIN_START, TRAIN_END)
    
    # 這裡 split_ratio 設為 0.8 來進行驗證，不使用 0.9 或 0.99
    # 因為我們需要一個足夠大的 Validation Set 來評估參數好壞
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler, _, _, n_features = \
        model_registry.preprocess_for_training(df, train_ratio=0.8)

    print(f"\n[Data] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 產生所有組合
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    best_r2 = -float('inf')
    best_config = None

    print(f"\n[Optimizer] Starting Grid Search on {len(combinations)} combinations...\n")

    for i, config in enumerate(combinations):
        print("-" * 60)
        print(f"🔄 Trial {i+1}/{len(combinations)}: {config}")
        
        # 設定隨機種子重現性
        np.random.seed(42)
        tf.random.set_seed(42)

        # 建立模型
        model = model_registry.build_lstm_ssam_model(
            time_steps=model_registry.LOOKBACK,
            n_features=n_features,
            lstm_units=config['lstm_units'],
            dropout_rate=config['dropout_rate']
        )

        # 訓練
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # 優化時稍微嚴格一點，節省時間
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=30,  # 優化時 Epochs 稍微減少，避免太久
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0  # 靜默模式
        )

        # 評估
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_predicted = target_scaler.inverse_transform(y_pred_scaled).flatten()

        rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
        r2 = r2_score(y_actual, y_predicted)

        print(f"   👉 Result: R² = {r2:.4f} | RMSE = {rmse:.2f}")

        results.append({
            **config,
            'r2': r2,
            'rmse': rmse
        })

        if r2 > best_r2:
            best_r2 = r2
            best_config = config
            print(f"   🏆 New Best Found!")

    # 輸出總結
    print("\n" + "=" * 70)
    print("🏅 Optimization Results Summary")
    print("=" * 70)
    
    results_df = pd.DataFrame(results).sort_values(by='r2', ascending=False)
    print(results_df)

    print("\n" + "=" * 70)
    print(f"🏆 Best Configuration: {best_config}")
    print(f"   Best R²: {best_r2:.4f}")
    print("=" * 70)
    
    # 建議指令
    if best_config:
        print("\nTo update `twii_model_registry_20d.py`, use these values:")
        for k, v in best_config.items():
            print(f"  {k.upper()} = {v}")

if __name__ == "__main__":
    optimize()
