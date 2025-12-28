import glob
import pandas as pd
import os

def analyze_trades():
    # Define the directory and pattern
    base_dir = r"d:\000-github-repositories\hybrid-trader-v03-04-03-test2-no120-x2-buy120\results_backtest_v5_dca_hybrid_no_filter_fixed_lstm"
    pattern = os.path.join(base_dir, "trades_strat2_*.csv")
    
    files = glob.glob(pattern)
    files.sort()
    
    results = []
    
    print(f"Found {len(files)} trade log files.")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract date range from filename: trades_strat2_20171016_20231013.csv
        date_range = filename.replace("trades_strat2_", "").replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            # print(f"Skipping empty file: {filename}")
            continue
            
        if df.empty:
            continue
            
        total_trades = len(df)
        wins = df[df['profit'] > 0]
        losses = df[df['profit'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = df['profit'].sum()
        gross_profit = wins['profit'].sum()
        gross_loss = abs(losses['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Stop Loss Analysis (Assuming ~ -8% threshold)
        STOP_THRESHOLD = -0.079 
        
        stop_loss_trades = df[df['return'] <= STOP_THRESHOLD]
        voluntary_trades = df[df['return'] > STOP_THRESHOLD]
        
        stop_loss_count = len(stop_loss_trades)
        voluntary_count = len(voluntary_trades)
        
        voluntary_wins = voluntary_trades[voluntary_trades['profit'] > 0]
        voluntary_win_rate = (len(voluntary_wins) / voluntary_count * 100) if voluntary_count > 0 else 0
        voluntary_avg_return = voluntary_trades['return'].mean() * 100 if voluntary_count > 0 else 0
        
        stop_avg_return = stop_loss_trades['return'].mean() * 100 if stop_loss_count > 0 else 0
        
        avg_hold_days = df['hold_days'].mean()
        
        results.append({
            "Period": date_range,
            "Trades": total_trades,
            "Win Rate": f"{win_rate:.2f}%",
            "Net Profit": f"${total_profit:,.0f}",
            "PF": f"{profit_factor:.2f}",
            "SL %": f"{(stop_loss_count/total_trades*100):.1f}%",
            "Vol %": f"{(voluntary_count/total_trades*100):.1f}%",
            "Vol Win": f"{voluntary_win_rate:.2f}%",
            "Vol Ret": f"{voluntary_avg_return:.2f}%",
            "Hold Days": f"{avg_hold_days:.1f}"
        })

    # Create DataFrame for display
    results_df = pd.DataFrame(results)
    
    # Print formatted table manually to avoid tabulate dependency
    print(f"{'Period':<20} | {'Trades':<6} | {'Win Rate':<8} | {'Net Profit':<12} | {'PF':<5} | {'SL %':<6} | {'Vol %':<6} | {'Vol Win':<8} | {'Vol Ret':<8} | {'Hold Days':<9}")
    print("-" * 115)
    for index, row in results_df.iterrows():
        print(f"{row['Period']:<20} | {row['Trades']:<6} | {row['Win Rate']:<8} | {row['Net Profit']:<12} | {row['PF']:<5} | {row['SL %']:<6} | {row['Vol %']:<6} | {row['Vol Win']:<8} | {row['Vol Ret']:<8} | {row['Hold Days']:<9}")

if __name__ == "__main__":
    analyze_trades()
