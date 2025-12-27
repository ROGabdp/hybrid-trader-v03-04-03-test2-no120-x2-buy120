#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Buy Agent 評估腳本 - 120天目標版本
================================================================================
測試期間：2017-10-16 ~ 2025-12-26
每天讓 Buy Agent 判斷是否買入，記錄信心度與成功率

成功定義：買入後 120 天內最高漲幅 >= 10%
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# 設定
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
V4_MODELS_PATH = os.path.join(PROJECT_PATH, "models_hybrid_v4")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "buy_agent_evaluation_120d.csv")

START_DATE = '2017-10-16'
END_DATE = '2025-12-26'


def main():
    print("=" * 70)
    print("Buy Agent 評估腳本 - 120天目標版本")
    print(f"測試期間：{START_DATE} ~ {END_DATE}")
    print("=" * 70)
    
    # 載入模組
    import ptrl_hybrid_system as hybrid
    from stable_baselines3 import PPO
    
    # 載入 LSTM 模型
    print("\n[System] 載入 LSTM 模型...")
    hybrid.load_best_lstm_models()
    
    # 載入資料
    print("\n[Data] 載入 ^TWII 資料...")
    twii_raw = hybrid._load_local_twii_data(start_date="2000-01-01")
    twii_df = hybrid.calculate_features(twii_raw, twii_raw, ticker="^TWII", use_cache=False)
    
    # 篩選測試期間
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)
    test_df = twii_df[(twii_df.index >= start_ts) & (twii_df.index <= end_ts)].copy()
    
    print(f"[Data] 測試資料：{len(test_df)} 筆 ({test_df.index[0].strftime('%Y-%m-%d')} ~ {test_df.index[-1].strftime('%Y-%m-%d')})")
    
    # 載入 Buy Agent
    print("\n[Model] 載入 Buy Agent...")
    buy_model_path = os.path.join(V4_MODELS_PATH, "ppo_buy_twii_final.zip")
    if not os.path.exists(buy_model_path):
        print(f"[Error] 找不到模型：{buy_model_path}")
        return
    buy_model = PPO.load(buy_model_path)
    print(f"  ✅ 載入成功：{buy_model_path}")
    
    # 準備特徵欄位
    feature_cols = hybrid.FEATURE_COLS
    
    # 評估每一天
    print("\n[Eval] 開始評估...")
    results = []
    
    for i, (date, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="評估進度")):
        # 準備觀察值
        obs = row[feature_cols].values.astype(np.float32)
        
        # 取得 Buy Agent 預測
        action, _ = buy_model.predict(obs.reshape(1, -1), deterministic=True)
        
        # 取得動作機率（信心度）
        obs_tensor = buy_model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
        action_dist = buy_model.policy.get_distribution(obs_tensor)
        action_probs = action_dist.distribution.probs.detach().cpu().numpy()[0]
        
        confidence_hold = action_probs[0]  # 不買的信心度
        confidence_buy = action_probs[1]   # 買入的信心度
        
        # 計算實際結果（120天內最高漲幅）
        close_price = row['Close']
        
        # 取得未來 120 天的最高價（T+1 到 T+120，不包含當天）
        future_idx = twii_df.index.get_loc(date)
        future_start_idx = future_idx + 1  # 從明天開始
        future_end_idx = min(future_idx + 121, len(twii_df))  # T+1 到 T+120
        
        if future_end_idx > future_start_idx:
            future_highs = twii_df.iloc[future_start_idx:future_end_idx]['High'].values
            max_high = np.max(future_highs)
            actual_max_return = (max_high / close_price) - 1
        else:
            actual_max_return = np.nan
        
        # 判斷是否成功（120天內最高漲幅 >= 10%）
        is_success = actual_max_return >= 0.10 if not np.isnan(actual_max_return) else np.nan
        
        # 判斷決策是否正確
        if action == 1:  # 買入
            action_label = "買入"
            if is_success == True:
                result_type = "正確買入"
                reward = 2.0
            elif is_success == False:
                result_type = "錯誤買入"
                reward = -0.5
            else:
                result_type = "未知"
                reward = np.nan
        else:  # 不買
            action_label = "不買"
            if is_success == True:
                result_type = "錯過機會"
                reward = -1.0
            elif is_success == False:
                result_type = "正確迴避"
                reward = 0.5
            else:
                result_type = "未知"
                reward = np.nan
        
        results.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Close': close_price,
            'Next_120d_Max_Return': actual_max_return,
            'Target (>10%)': is_success,
            'Action': action,
            'Action_Label': action_label,
            'Confidence_Buy': confidence_buy,
            'Confidence_Hold': confidence_hold,
            'Reward': reward,
            'Result_Type': result_type
        })
    
    # 轉換為 DataFrame
    results_df = pd.DataFrame(results)
    
    # 儲存結果
    results_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[Output] 結果已儲存：{OUTPUT_PATH}")
    
    # ==========================================================================
    # 統計分析
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📊 統計分析")
    print("=" * 70)
    
    # 過濾掉未知結果
    valid_df = results_df[results_df['Result_Type'] != '未知'].copy()
    
    # 基本統計
    total_days = len(valid_df)
    buy_days = len(valid_df[valid_df['Action'] == 1])
    hold_days = len(valid_df[valid_df['Action'] == 0])
    
    print(f"\n📅 測試天數：{total_days} 天")
    print(f"   - 買入決策：{buy_days} 天 ({buy_days/total_days*100:.1f}%)")
    print(f"   - 不買決策：{hold_days} 天 ({hold_days/total_days*100:.1f}%)")
    
    # 成功率分析
    correct_buy = len(valid_df[valid_df['Result_Type'] == '正確買入'])
    wrong_buy = len(valid_df[valid_df['Result_Type'] == '錯誤買入'])
    missed = len(valid_df[valid_df['Result_Type'] == '錯過機會'])
    correct_hold = len(valid_df[valid_df['Result_Type'] == '正確迴避'])
    
    print(f"\n✅ 決策結果分布：")
    print(f"   - 正確買入：{correct_buy} 次")
    print(f"   - 錯誤買入：{wrong_buy} 次")
    print(f"   - 錯過機會：{missed} 次")
    print(f"   - 正確迴避：{correct_hold} 次")
    
    # 買入成功率
    if buy_days > 0:
        buy_success_rate = correct_buy / buy_days * 100
        print(f"\n🎯 買入成功率：{buy_success_rate:.1f}% ({correct_buy}/{buy_days})")
    
    # 不買正確率
    if hold_days > 0:
        hold_success_rate = correct_hold / hold_days * 100
        print(f"🛡️ 不買正確率：{hold_success_rate:.1f}% ({correct_hold}/{hold_days})")
    
    # 整體準確率
    total_correct = correct_buy + correct_hold
    overall_accuracy = total_correct / total_days * 100
    print(f"\n📈 整體準確率：{overall_accuracy:.1f}% ({total_correct}/{total_days})")
    
    # 平均信心度
    print(f"\n💡 平均信心度：")
    print(f"   - 買入決策平均信心：{valid_df[valid_df['Action']==1]['Confidence_Buy'].mean():.3f}")
    print(f"   - 不買決策平均信心：{valid_df[valid_df['Action']==0]['Confidence_Hold'].mean():.3f}")
    
    # 平均獎勵
    avg_reward = valid_df['Reward'].mean()
    print(f"\n🏆 平均獎勵：{avg_reward:.3f}")
    
    # 信心度與成功率的關係
    print("\n" + "=" * 70)
    print("📊 信心度分層分析（買入決策）")
    print("=" * 70)
    
    buy_df = valid_df[valid_df['Action'] == 1].copy()
    if len(buy_df) > 0:
        buy_df['Confidence_Bin'] = pd.cut(buy_df['Confidence_Buy'], 
                                           bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
                                           labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'])
        
        for bin_name in ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']:
            bin_df = buy_df[buy_df['Confidence_Bin'] == bin_name]
            if len(bin_df) > 0:
                success = len(bin_df[bin_df['Result_Type'] == '正確買入'])
                rate = success / len(bin_df) * 100
                print(f"   信心度 {bin_name}：{rate:.1f}% 成功率 ({success}/{len(bin_df)})")
    
    print("\n" + "=" * 70)
    print("✅ 評估完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
