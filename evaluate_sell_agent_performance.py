import pandas as pd
import numpy as np
import os

# Configuration
TRADES_CSV = r'results_backtest_v5_dca_hybrid_no_filter_fixed_lstm/trades_strat2_20230103_20251226.csv'
STOP_LOSS_THRESHOLD = -0.08

def evaluate_performance():
    if not os.path.exists(TRADES_CSV):
        print(f"Error: File not found: {TRADES_CSV}")
        return

    print(f"Reading trades from: {TRADES_CSV}")
    df = pd.read_csv(TRADES_CSV)
    
    if len(df) == 0:
        print("No trades found in the file.")
        return

    # Basic Metrics
    total_trades = len(df)
    winners = df[df['profit'] > 0]
    losers = df[df['profit'] <= 0]
    
    win_rate = len(winners) / total_trades * 100
    
    avg_return_all = df['return'].mean() * 100
    avg_return_winners = winners['return'].mean() * 100 if len(winners) > 0 else 0
    avg_return_losers = losers['return'].mean() * 100 if len(losers) > 0 else 0
    
    total_profit = df['profit'].sum()
    gross_profit = winners['profit'].sum()
    gross_loss = abs(losers['profit'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Hold Time Analysis
    avg_hold_days = df['hold_days'].mean()
    max_hold_days = df['hold_days'].max()
    min_hold_days = df['hold_days'].min()
    
    # Stop Loss Analysis
    # Assuming any return <= -8% is a stop loss (approximate, slippage could make it worse)
    stop_loss_hits = df[df['return'] <= STOP_LOSS_THRESHOLD]
    stop_loss_rate = len(stop_loss_hits) / total_trades * 100
    
    # AI Performance (excluding stop losses to see pure judgment vs forced exit)
    # This is tricky because some "bad decisions" might not hit stop loss but still lose.
    # But checking "Voluntary Sells" (return > -8%) might give insight into "taking profit" vs "cutting loss early".
    voluntary_sells = df[df['return'] > STOP_LOSS_THRESHOLD]
    voluntary_win_rate = len(voluntary_sells[voluntary_sells['profit'] > 0]) / len(voluntary_sells) * 100 if len(voluntary_sells) > 0 else 0
    
    print("\n" + "="*40)
    print("🤖 Sell Agent Performance Evaluation")
    print("="*40)
    print(f"Total Trades:       {total_trades}")
    print(f"Win Rate:           {win_rate:.2f}% ({len(winners)} W / {len(losers)} L)")
    print(f"Profit Factor:      {profit_factor:.2f}")
    print(f"Total Profit:       ${total_profit:,.2f}")
    print("-" * 30)
    print(f"Avg Return (All):   {avg_return_all:.2f}%")
    print(f"Avg Return (Wins):  {avg_return_winners:.2f}%")
    print(f"Avg Return (Loss):  {avg_return_losers:.2f}%")
    print("-" * 30)
    print(f"Avg Hold Days:      {avg_hold_days:.1f} days")
    print(f"Max Hold Days:      {max_hold_days} days")
    print("-" * 30)
    print("🛡️ Stop Loss Analysis")
    print(f"Stop Loss Hits:     {len(stop_loss_hits)} ({stop_loss_rate:.2f}%)")
    print(f"Avg Loss on Stop:   {stop_loss_hits['return'].mean()*100:.2f}% (Target: -8%)")
    print("-" * 30)
    print("🧠 AI Voluntary Decision (Return > -8%)")
    print(f"Trades Count:       {len(voluntary_sells)}")
    print(f"Win Rate:           {voluntary_win_rate:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_performance()
