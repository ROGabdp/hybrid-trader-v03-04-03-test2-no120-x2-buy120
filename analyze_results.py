import pandas as pd
import numpy as np
import glob
import os

RESULTS_DIR = 'results_backtest_v5_dca_hybrid_dynamic_filter_fixed_lstm'

def analyze_equity(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # MDD
    roll_max = df['value'].cummax()
    drawdown = (df['value'] - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # Return
    initial = df['value'].iloc[0]
    final = df['value'].iloc[-1]
    total_ret = (final - initial) / initial
    
    # Annualized
    days = (df.index[-1] - df.index[0]).days
    annualized = (1 + total_ret) ** (365/days) - 1 if days > 0 else 0
    
    return total_ret, max_dd, annualized

def analyze_trades(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) == 0:
            return 0, 0, 0
        
        wins = len(df[df['profit'] > 0])
        total = len(df)
        win_rate = wins / total
        avg_profit = df['profit'].mean()
        return total, win_rate, avg_profit
    except:
        return 0, 0, 0

lines = []
lines.append(f"{'Period':<25} {'Total Ret':<10} {'MDD':<10} {'Ann. Ret':<10} {'Trades':<8} {'Win Rate':<10}")
lines.append("-" * 80)

files = glob.glob(os.path.join(RESULTS_DIR, 'equity_curve_*.csv'))
for eq_file in files:
    try:
        basename = os.path.basename(eq_file)
        period = basename.replace('equity_curve_', '').replace('.csv', '')
        
        ret, mdd, ann = analyze_equity(eq_file)
        
        trade_file = os.path.join(RESULTS_DIR, f'trades_{period}.csv')
        trades, win_rate, avg_prof = analyze_trades(trade_file)
        
        lines.append(f"{period:<25} {ret*100:6.2f}%    {mdd*100:6.2f}%    {ann*100:6.2f}%    {trades:<8} {win_rate*100:6.2f}%")
    except Exception as e:
        lines.append(f"Error processing {basename}: {e}")

with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("Analysis saved to analysis_summary.txt")
