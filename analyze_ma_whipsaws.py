import pandas as pd
import numpy as np

def analyze_ma_noise():
    # Load a long period of data (using TWII data or a long backtest result)
    # We can use the 'twii_data_from_2000_01_01.csv' directly if available, 
    # or one of the daily_action files that covers a long range.
    # The file 'results_backtest_v5_dca_hybrid_no_filter_fixed_lstm/daily_action_strat2_20171016_20231013.csv' is good.
    
    file_path = r"d:\000-github-repositories\hybrid-trader-v03-04-03-test2-no120-x2-buy120\results_backtest_v5_dca_hybrid_no_filter_fixed_lstm\daily_action_strat2_20171016_20231013.csv"
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate MAs
    df['MA60'] = df['price'].rolling(window=60).mean()
    df['MA120'] = df['price'].rolling(window=120).mean()
    
    print(f"\n{'='*60}")
    print(f"{'MA Regime Switch Analysis (2017-2023)':^60}")
    print(f"{'='*60}")
    
    for length in [60, 120]:
        ma_col = f'MA{length}'
        
        # 1. Count Regime Switches
        # Bear Mode: Price < MA
        # Bull Mode: Price >= MA
        
        # We look for crosses
        df['regime'] = np.where(df['price'] < df[ma_col], 'BEAR', 'BULL')
        df['switch'] = df['regime'] != df['regime'].shift(1)
        
        # Filter out NaN (first N days)
        valid_df = df.dropna(subset=[ma_col])
        if valid_df.empty: continue
        
        switches = valid_df[valid_df['switch']]
        num_switches = len(switches)
        
        # 2. Whipsaw Analysis (Short-lived Bear Signals)
        # Identify "Bear Starts" -> Regime becomes BEAR
        bear_starts = switches[switches['regime'] == 'BEAR']
        
        short_lived_bear = 0
        whipsaw_days = 20 # If regime flips back to BULL within 20 days, it's a whipsaw
        
        for idx in bear_starts.index:
            # Look ahead N days in valid_df (need to map back to position)
            # Find the next switch to BULL
            future_switches = switches[(switches.index > idx) & (switches['regime'] == 'BULL')]
            
            if not future_switches.empty:
                next_bull_idx = future_switches.index[0]
                duration_days = (valid_df.loc[next_bull_idx, 'date'] - valid_df.loc[idx, 'date']).days
                
                if duration_days <= whipsaw_days:
                    short_lived_bear += 1
        
        # 3. Time in Bear Mode
        bear_days = len(valid_df[valid_df['regime'] == 'BEAR'])
        total_days = len(valid_df)
        bear_ratio = bear_days / total_days * 100
        
        print(f"\n[MA{length}] Filter Logic: Price < MA{length}")
        print(f"  - Total Switches: {num_switches} (Avg {num_switches/6:.1f} per year)")
        print(f"  - Bear Signals: {len(bear_starts)}")
        print(f"  - False Alarms (<{whipsaw_days} days): {short_lived_bear} ({short_lived_bear/len(bear_starts)*100:.1f}%)")
        print(f"  - Time in Bear Mode: {bear_ratio:.1f}%")
        
        # Specific check for 2022
        # Did it stay in Bear Mode consistently?
        mask_2022 = (valid_df['date'] >= '2022-01-01') & (valid_df['date'] <= '2022-12-31')
        df_2022 = valid_df[mask_2022]
        if not df_2022.empty:
            bear_2022 = len(df_2022[df_2022['regime'] == 'BEAR'])
            print(f"  - 2022 Bear Coverage: {bear_2022/len(df_2022)*100:.1f}% of trading days")

if __name__ == "__main__":
    analyze_ma_noise()
