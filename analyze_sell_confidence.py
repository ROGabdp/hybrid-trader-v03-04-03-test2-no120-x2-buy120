import glob
import pandas as pd
import os
import numpy as np

def analyze_confidence():
    base_dir = r"d:\000-github-repositories\hybrid-trader-v03-04-03-test2-no120-x2-buy120\results_backtest_v5_dca_hybrid_no_filter_fixed_lstm"
    pattern = os.path.join(base_dir, "daily_action_strat2_*.csv")
    
    files = glob.glob(pattern)
    files.sort()
    
    print(f"\n{'='*30}\n Sell Agent Confidence Analysis \n{'='*30}")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        date_range = filename.replace("daily_action_strat2_", "").replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
        except:
            continue
            
        if df.empty or 'ai_sell_conf' not in df.columns:
            continue
            
        # Filter for days where we had positions (so sell decision was relevant)
        # Assuming if 'ai_sell_conf' is not NaN, the agent evaluated selling
        active_days = df.dropna(subset=['ai_sell_conf']).copy()
        
        if active_days.empty:
            continue
            
        avg_conf = active_days['ai_sell_conf'].mean()
        median_conf = active_days['ai_sell_conf'].median()
        
        # Detect "High Confidence Holds" -> High sell confidence but Action was HOLD
        high_conf_holds = active_days[
            (active_days['ai_sell_conf'] > 0.8) & 
            (active_days['ai_action'] == 'HOLD')
        ]
        
        # Detect "Low Confidence Sells" -> Low sell confidence but Action was SELL
        low_conf_sells = active_days[
            (active_days['ai_sell_conf'] < 0.6) & 
            (active_days['ai_action'] == 'SELL')
        ]
        
        # Calculate correlation between Sell Confidence and Price Change (Next Day)
        # This checks if high sell confidence correctly predicts drops
        active_days['next_return'] = active_days['price'].pct_change().shift(-1)
        correlation = active_days['ai_sell_conf'].corr(active_days['next_return'])
        
        print(f"\nPeriod: {date_range}")
        print(f"  - Avg Sell Confidence: {avg_conf:.4f}")
        print(f"  - Median Sell Confidence: {median_conf:.4f}")
        print(f"  - High Conf (>0.8) but HOLD: {len(high_conf_holds)} days ({len(high_conf_holds)/len(active_days)*100:.1f}%)")
        print(f"  - Low Conf (<0.6) but SELL: {len(low_conf_sells)} trades")
        print(f"  - Correlation (Sell Conf vs Next Day Return): {correlation:.4f} (Negative is good for Sell)")

        # Deep dive into one period if meaningful
        if len(high_conf_holds) > 0:
             print(f"    * Note: Agent often wants to sell (Avg Conf {avg_conf:.2f}) but holds. Possible 'Greed' or 'Argmax' blocking.")

if __name__ == "__main__":
    analyze_confidence()
