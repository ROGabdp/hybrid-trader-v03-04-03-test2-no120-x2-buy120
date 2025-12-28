import glob
import pandas as pd
import numpy as np
import os

def analyze_v4_filter_impact():
    # V4 Result Directory (With Filter)
    v4_dir = r"d:\000-github-repositories\hybrid-trader-v03-04-03-test2-no120-x2-buy120\results_backtest_v4_dca_hybrid_with_filter_fixed_lstm"
    
    # We look for metrics_comparison CSVs to get high level stats
    # and trades_strat2 CSVs to get trade details
    pattern = os.path.join(v4_dir, "metrics_comparison_*.csv")
    files = glob.glob(pattern)
    files.sort()
    
    print(f"\n{'='*80}")
    print(f"{'V4 Filter Impact Analysis (Bear Market Focus)':^80}")
    print(f"{'='*80}")
    print(f"{'Period':<20} | {'Trades':<6} | {'Win Rate':<8} | {'Tot Return':<10} | {'Max DD':<8} | {'Sharpe':<6}")
    print("-" * 80)
    
    bear_years = ['2018', '2022', '20200102'] # Keywords to identify bear/crisis periods
    
    output_lines = []
    
    for file_path in files:
        filename = os.path.basename(file_path)
        date_range = filename.replace("metrics_comparison_", "").replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
        except:
            continue
            
        # Transpose V4 metrics because "Strategy" is in columns (Strat1, Strat2...)
        # Current format: Metric, Strat1, Strat2, ...
        # We want: Strategy, Metric1, Metric2...
        df = df.set_index('Metric').transpose()
        df['Strategy'] = df.index
        
        # Extract Strategy 2 (Shared) metrics as it's the one we track
        # Columns usually: Strategy, Total Return, Annualized Return, Sharpe Ratio, Max Drawdown, Total Trades, Win Rate
        strat2 = df[df['Strategy'] == 'Strat2_Shared']
        
        if strat2.empty:
            continue
            
        trades = int(float(strat2['AI_Trades'].values[0])) if 'AI_Trades' in strat2.columns else 0
        win_rate = strat2['AI_Win_Rate_Pct'].values[0] if 'AI_Win_Rate_Pct' in strat2.columns else "N/A"
        tot_ret = strat2['Total_Return_Pct'].values[0]
        max_dd = strat2['Max_Drawdown_Pct'].values[0]
        sharpe = strat2['Sharpe_Ratio'].values[0]
        
        # Highlight bear markets
        is_bear = any(y in date_range for y in bear_years)
        prefix = "[BEAR] " if is_bear else "       "
        
        # Format string values if they are strings
        def fmt_pct(val):
            if isinstance(val, str) and '%' in val: return val
            return f"{float(val)*100:.2f}%" if isinstance(val, (int, float)) else str(val)
        
        line = f"{prefix}{date_range:<17} | {trades:<6} | {win_rate:<8} | {fmt_pct(tot_ret):<10} | {fmt_pct(max_dd):<8} | {sharpe:<6}"
        print(line)
        output_lines.append(line)
        
    output_lines.append(f"{'='*80}\n")
    
    # Deep dive into 2022 Trades if available
    analyze_2022_trades(v4_dir, output_lines)

    with open("v4_filter_report.txt", "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

def analyze_2022_trades(base_dir, output_lines):
    # Find 2022 trade log
    pattern = os.path.join(base_dir, "trades_strat2_2022*.csv")
    files = glob.glob(pattern)
    
    if not files:
        output_lines.append("No dedicated 2022 trade log found.")
        return

    output_lines.append("Details for 2022 Bear Market (With Filter):")
    for f in files:
        df = pd.read_csv(f)
        output_lines.append(f"File: {os.path.basename(f)}")
        if df.empty:
            output_lines.append("  No AI trades executed.")
        else:
            output_lines.append(f"  Total Trades: {len(df)}")
            output_lines.append(f"  Avg Return: {df['return'].mean()*100:.2f}%")
            output_lines.append(f"  Win Rate: {len(df[df['profit']>0])/len(df)*100:.1f}%")
            output_lines.append(f"  Max Loss: {df['return'].min()*100:.2f}%")

if __name__ == "__main__":
    analyze_v4_filter_impact()
