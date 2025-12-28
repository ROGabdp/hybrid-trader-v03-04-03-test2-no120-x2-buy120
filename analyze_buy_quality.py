import glob
import pandas as pd
import numpy as np
import os

def analyze_buy_quality():
    base_dir = r"d:\000-github-repositories\hybrid-trader-v03-04-03-test2-no120-x2-buy120\results_backtest_v5_dca_hybrid_no_filter_fixed_lstm"
    # Using strat2 files as they represent the AI's pure decisions better (though strat1 uses same buy logic usually)
    # Actually, strat2 is "Shared" budget, but the buy signals in daily_action are what we care about.
    # daily_action_strat2_*.csv contains 'ai_action' column.
    
    pattern = os.path.join(base_dir, "daily_action_strat2_*.csv")
    files = glob.glob(pattern)
    files.sort()
    
    all_buys = []
    
    print(f"{'Period':<20} | {'Buys':<5} | {'10d Ret':<8} | {'20d Ret':<8} | {'60d Ret':<8} | {'60d MDD':<8} | {'WinRate(60d)':<10}")
    print("-" * 100)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        date_range = filename.replace("daily_action_strat2_", "").replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
        except:
            continue
            
        if df.empty or 'ai_action' not in df.columns:
            continue
            
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Identify BUY rows
        buy_indices = df[df['ai_action'] == 'BUY'].index
        
        if len(buy_indices) == 0:
            continue
            
        period_buys = []
        
        for idx in buy_indices:
            buy_price = df.loc[idx, 'price']
            buy_date = df.loc[idx, 'date']
            
            # Calculate forward metrics
            # We need future prices. 
            # Slice from idx+1 to idx+60
            future_window = df.iloc[idx+1 : idx+61]
            
            if len(future_window) < 1:
                continue
                
            # 10d Return
            if len(future_window) >= 10:
                price_10d = future_window.iloc[9]['price']
                ret_10d = (price_10d - buy_price) / buy_price
            else:
                ret_10d = (future_window.iloc[-1]['price'] - buy_price) / buy_price
                
            # 20d Return
            if len(future_window) >= 20:
                price_20d = future_window.iloc[19]['price']
                ret_20d = (price_20d - buy_price) / buy_price
            else:
                ret_20d = (future_window.iloc[-1]['price'] - buy_price) / buy_price
            
            # 60d Return
            if len(future_window) >= 60:
                price_60d = future_window.iloc[59]['price']
                ret_60d = (price_60d - buy_price) / buy_price
            else:
                ret_60d = (future_window.iloc[-1]['price'] - buy_price) / buy_price
                
            # 60d Max Drawdown (lowest close relative to buy price)
            # MDD here is defined as: min(price in window) / buy_price - 1
            # This measures "how much did it dip below my entry price?"
            min_price = future_window['price'].min()
            mdd_60d = (min_price - buy_price) / buy_price
            
            period_buys.append({
                'ret_10d': ret_10d,
                'ret_20d': ret_20d,
                'ret_60d': ret_60d,
                'mdd_60d': mdd_60d
            })
            
        if not period_buys:
            continue
            
        df_period = pd.DataFrame(period_buys)
        
        avg_10d = df_period['ret_10d'].mean() * 100
        avg_20d = df_period['ret_20d'].mean() * 100
        avg_60d = df_period['ret_60d'].mean() * 100
        avg_mdd = df_period['mdd_60d'].mean() * 100
        
        # Win rate: % of buys that are positive after 60 days
        win_rate = (len(df_period[df_period['ret_60d'] > 0]) / len(df_period)) * 100
        
        output_line = f"{date_range:<20} | {len(period_buys):<5} | {avg_10d:>7.2f}% | {avg_20d:>7.2f}% | {avg_60d:>7.2f}% | {avg_mdd:>7.2f}% | {win_rate:>9.1f}%"
        print(output_line)
        all_buys.append(output_line)

    with open("buy_quality_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Period':<20} | {'Buys':<5} | {'10d Ret':<8} | {'20d Ret':<8} | {'60d Ret':<8} | {'60d MDD':<8} | {'WinRate(60d)':<10}\n")
        f.write("-" * 100 + "\n")
        for line in all_buys:
            f.write(line + "\n")

if __name__ == "__main__":
    analyze_buy_quality()
