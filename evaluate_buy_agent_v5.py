#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Buy Agent V5 評估腳本
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

V5_MODELS_PATH = "models_hybrid_v5"
START_DATE = '2017-10-16'
END_DATE = '2025-12-26'

def main():
    print("=" * 70)
    print("Buy Agent V5 評估腳本 - 對稱獎勵版本")
    print(f"測試期間：{START_DATE} ~ {END_DATE}")
    print("=" * 70)
    
    import ptrl_hybrid_system as hybrid
    from stable_baselines3 import PPO
    
    print("\n[System] 載入 LSTM 模型...")
    hybrid.load_best_lstm_models()
    
    print("\n[Data] 載入 ^TWII 資料...")
    twii_raw = hybrid._load_local_twii_data(start_date="2000-01-01")
    twii_df = hybrid.calculate_features(twii_raw, twii_raw, ticker="^TWII", use_cache=False)
    
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)
    test_df = twii_df[(twii_df.index >= start_ts) & (twii_df.index <= end_ts)].copy()
    
    print(f"[Data] 測試資料：{len(test_df)} 筆")
    
    print("\n[Model] 載入 Buy Agent V5...")
    buy_model = PPO.load(os.path.join(V5_MODELS_PATH, "ppo_buy_twii_final.zip"))
    print(f"  ✅ 載入成功")
    
    feature_cols = hybrid.FEATURE_COLS
    
    print("\n[Eval] 開始評估...")
    results = []
    
    for i, (date, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="評估進度")):
        obs = row[feature_cols].values.astype(np.float32)
        action, _ = buy_model.predict(obs.reshape(1, -1), deterministic=True)
        
        # 取得信心度
        obs_tensor = buy_model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
        action_dist = buy_model.policy.get_distribution(obs_tensor)
        action_probs = action_dist.distribution.probs.detach().cpu().numpy()[0]
        conf_buy = action_probs[1]
        conf_hold = action_probs[0]
        
        # 計算未來120天最高報酬
        future_idx = twii_df.index.get_loc(date)
        future_start = future_idx + 1
        future_end = min(future_idx + 121, len(twii_df))
        
        if future_end > future_start:
            max_high = twii_df.iloc[future_start:future_end]['High'].max()
            max_ret = (max_high / row['Close']) - 1
        else:
            max_ret = np.nan
        
        results.append({
            'Date': date,
            'Action': int(action),
            'Confidence_Buy': conf_buy,
            'Confidence_Hold': conf_hold,
            'Max_Return_120d': max_ret,
            'Success': max_ret >= 0.10 if not np.isnan(max_ret) else np.nan
        })
    
    df = pd.DataFrame(results)
    df.to_csv('buy_agent_v5_evaluation.csv', index=False)
    
    # 統計分析
    valid_df = df[~df['Success'].isna()]
    buy_df = valid_df[valid_df['Action'] == 1]
    hold_df = valid_df[valid_df['Action'] == 0]
    
    print("\n" + "=" * 70)
    print("📊 V5 Buy Agent 分析結果")
    print("=" * 70)
    
    print(f"\n📅 總測試天數：{len(valid_df)}")
    print(f"   - 買入決策：{len(buy_df)} 天 ({len(buy_df)/len(valid_df)*100:.1f}%)")
    print(f"   - 不買決策：{len(hold_df)} 天 ({len(hold_df)/len(valid_df)*100:.1f}%)")
    
    if len(buy_df) > 0:
        correct_buy = len(buy_df[buy_df['Success'] == True])
        print(f"\n🎯 買入成功率：{correct_buy/len(buy_df)*100:.1f}% ({correct_buy}/{len(buy_df)})")
        print(f"   買入平均信心度：{buy_df['Confidence_Buy'].mean():.3f}")
    
    if len(hold_df) > 0:
        correct_hold = len(hold_df[hold_df['Success'] == False])
        print(f"\n🛡️ 不買正確率：{correct_hold/len(hold_df)*100:.1f}% ({correct_hold}/{len(hold_df)})")
        print(f"   不買平均信心度：{hold_df['Confidence_Hold'].mean():.3f}")
    
    # 整體準確率
    total_correct = len(buy_df[buy_df['Success']==True]) + len(hold_df[hold_df['Success']==False])
    print(f"\n📈 整體準確率：{total_correct/len(valid_df)*100:.1f}% ({total_correct}/{len(valid_df)})")
    
    # 信心度分層
    print("\n" + "=" * 70)
    print("📊 信心度分層分析（買入決策）")
    print("=" * 70)
    
    if len(buy_df) > 0:
        bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['0-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        buy_df = buy_df.copy()
        buy_df['Conf_Bin'] = pd.cut(buy_df['Confidence_Buy'], bins=bins, labels=labels)
        
        for label in labels:
            bin_df = buy_df[buy_df['Conf_Bin'] == label]
            if len(bin_df) > 0:
                success = len(bin_df[bin_df['Success'] == True])
                rate = success / len(bin_df) * 100
                print(f"   信心度 {label}：{rate:.1f}% 成功率 ({success}/{len(bin_df)})")
    
    print("\n" + "=" * 70)
    print("✅ 評估完成！結果已儲存至 buy_agent_v5_evaluation.csv")
    print("=" * 70)

if __name__ == "__main__":
    main()
