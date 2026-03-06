#!/usr/bin/env python3
"""
Manual backfill: Pull closed trades from MT5 history and record them 
into the adaptive learner (bypassing the race condition bug).

Usage:
    python -m scripts.manual_backfill
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.broker.mt5_connector import MT5Connector
from src.ai.adaptive_learner import AdaptiveLearner
from config.strategy_config import PAIRS

def main():
    print("🔄 Manual backfill: Loading trade history from MT5...")
    
    # Initialize broker connection
    broker = MT5Connector()
    
    # Check if connected (either relay or direct MT5)
    if not broker.connected and not broker.relay_mode:
        print("❌ Not connected to MT5")
        return
    
    # Get trade history (last 7 days)
    history = broker.get_trade_history(hours=168)
    if not history:
        print("❌ No trade history found")
        return
    
    print(f"📊 Found {len(history)} deals in history")
    
    # Load adaptive learner
    learner = AdaptiveLearner()
    
    # Build set of already-known position IDs
    existing = set()
    for t in learner.trade_history:
        pid = t.get('position_id') or t.get('ticket', 0)
        if pid:
            existing.add(pid)
    
    print(f"📋 Already recorded: {len(existing)} trades")
    
    imported = 0
    skipped_pair = 0
    skipped_dup = 0
    
    for deal in history:
        pos_id = deal.get('position_id', deal.get('ticket', 0))
        if pos_id in existing:
            skipped_dup += 1
            continue
        
        profit = deal.get('profit', 0) + deal.get('swap', 0) + deal.get('commission', 0)
        
        # Normalise pair format: EURUSD → EUR/USD
        raw_pair = deal.get('pair', '')
        if len(raw_pair) == 6 and '/' not in raw_pair:
            pair = f"{raw_pair[:3]}/{raw_pair[3:]}"
        else:
            pair = raw_pair
        
        # Skip trades for pairs we're not currently trading
        if pair not in PAIRS:
            skipped_pair += 1
            continue
        
        # The closing deal's type is the *exit* direction — flip for the original signal
        exit_dir = deal.get('type', '')
        trade_type = 'SELL' if exit_dir == 'BUY' else 'BUY'
        
        # Record the trade
        learner.record_trade({
            'pair': pair,
            'signal': trade_type,
            'profit_loss': profit,
            'entry_price': deal.get('open_price', 0),
            'exit_price': deal.get('price', 0),
            'exit_type': 'TAKE_PROFIT' if profit > 0 else 'STOP_LOSS',
            'model_signals': {},   # Unknown for historical trades
            'position_id': pos_id,
            'timestamp': deal.get('time', ''),
        })
        
        is_win = profit > 0
        print(f"  {'✅' if is_win else '❌'} {pair} {trade_type} | P/L: ${profit:+.2f} | ticket={pos_id}")
        imported += 1
    
    # Save
    learner.save()
    
    print()
    print(f"📊 Backfill complete:")
    print(f"   Imported:      {imported}")
    print(f"   Skipped (dup): {skipped_dup}")
    print(f"   Skipped (pair):{skipped_pair}")
    print(f"   Total in learner: {len(learner.trade_history)}")
    
    if learner.trade_history:
        wins = sum(1 for t in learner.trade_history if t.get('profit_loss', 0) > 0)
        losses = len(learner.trade_history) - wins
        total_pnl = sum(t.get('profit_loss', 0) for t in learner.trade_history)
        print(f"   Win/Loss: {wins}/{losses} ({100*wins/len(learner.trade_history):.1f}%)")
        print(f"   Total P/L: ${total_pnl:+.2f}")

if __name__ == '__main__':
    main()
