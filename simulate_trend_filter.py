import pandas as pd
import numpy as np

def simulate_trend_filter():
    file_path = r"d:\000-github-repositories\hybrid-trader-v03-04-03-test2-no120-x2-buy120\results_backtest_v5_dca_hybrid_no_filter_fixed_lstm\daily_action_strat2_20220207_20230428.csv"
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate MAs
    df['MA20'] = df['price'].rolling(window=20).mean()
    df['MA60'] = df['price'].rolling(window=60).mean()
    df['MA120'] = df['price'].rolling(window=120).mean()
    
    # Determine Logic
    # We want to see when the filter would have switched to "BEAR" mode.
    # Criteria 1: Price < MA60 (Faster)
    # Criteria 2: Price < MA120 (Slower)
    # Criteria 3: MA60 < MA120 (Death Cross)
    
    print(f"\nAnalyzing 2022 Bear Market (Start Price: {df['price'].iloc[0]:.2f})")
    
    peak_price = df['price'].max()
    peak_date = df.loc[df['price'].idxmax(), 'date'].date()
    print(f"Peak Price: {peak_price:.2f} on {peak_date}")
    
    # 1. Price < MA60
    bear_ma60 = df[df['price'] < df['MA60']]
    if not bear_ma60.empty:
        first_signal = bear_ma60.iloc[0]
        dd_at_signal = (first_signal['price'] - peak_price) / peak_price * 100
        print(f"\n[Trigger: Price < MA60]")
        print(f"  Date: {first_signal['date'].date()}")
        print(f"  Price: {first_signal['price']:.2f}")
        print(f"  Drawdown from Peak at Trigger: {dd_at_signal:.2f}% (Lag Cost)")
    
    # 2. Price < MA120
    bear_ma120 = df[df['price'] < df['MA120']]
    if not bear_ma120.empty:
        first_signal = bear_ma120.iloc[0]
        dd_at_signal = (first_signal['price'] - peak_price) / peak_price * 100
        print(f"\n[Trigger: Price < MA120]")
        print(f"  Date: {first_signal['date'].date()}")
        print(f"  Price: {first_signal['price']:.2f}")
        print(f"  Drawdown from Peak at Trigger: {dd_at_signal:.2f}% (Lag Cost)")

    # 3. MA60 < MA120 (Death Cross)
    # Need to find the first day where MA60 crosses below MA120
    # Since we don't have enough history for MA120 at the very start of this specific CSV, 
    # this might be nan for the first 120 rows. 
    # Let's see if we have valid data.
    
    valid_cross = df.dropna(subset=['MA120'])
    death_cross = valid_cross[valid_cross['MA60'] < valid_cross['MA120']]
    
    if not death_cross.empty:
        first_signal = death_cross.iloc[0]
        dd_at_signal = (first_signal['price'] - peak_price) / peak_price * 100
        print(f"\n[Trigger: MA60 < MA120 (Death Cross)]")
        print(f"  Date: {first_signal['date'].date()}")
        print(f"  Price: {first_signal['price']:.2f}")
        print(f"  Drawdown from Peak at Trigger: {dd_at_signal:.2f}% (Lag Cost)")
    else:
        print("\n[Trigger: MA60 < MA120] No Death Cross detected in this file (or insufficient pre-data).")

if __name__ == "__main__":
    simulate_trend_filter()
