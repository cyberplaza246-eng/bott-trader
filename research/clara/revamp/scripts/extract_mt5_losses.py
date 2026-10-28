#!/usr/bin/env python3
"""
Extract and analyze MT5 trade history for a date window.

Usage:
    python scripts/extract_mt5_losses.py --start 2026-03-05 --end 2026-03-06

Outputs:
    - data/trade_analysis_{start}_{end}.csv  (raw trades)
    - data/trade_analysis_{start}_{end}.json (analysis summary)
    - Prints recommendations for TP/SL adjustments
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests


def fetch_mt5_history(relay_url: str, hours: int = 168) -> list:
    """Fetch trade history from MT5 relay server."""
    try:
        resp = requests.get(f"{relay_url}/history", params={"hours": hours}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("deals", [])
        print(f"❌ Relay returned status {resp.status_code}")
        return []
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to MT5 relay at {relay_url}")
        print("   Make sure relay_server.py is running on your Windows machine")
        return []
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []


def parse_trades_from_logs(log_path: str, start_date: datetime, end_date: datetime) -> list:
    """Fallback: Parse trades from trades.log file."""
    trades = []
    if not os.path.exists(log_path):
        return trades
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                msg = entry.get('message', '')
                
                # Parse ORDER_PLACED entries
                if 'ORDER_PLACED' in msg and 'RELAY' in msg:
                    parts = msg.split('|')
                    trade = {}
                    for part in parts:
                        part = part.strip()
                        if 'Pair:' in part:
                            trade['pair'] = part.split(':')[1].strip()
                        elif 'Type:' in part:
                            trade['type'] = part.split(':')[1].strip()
                        elif 'Lot:' in part:
                            trade['lot'] = float(part.split(':')[1].strip())
                        elif 'SL:' in part:
                            trade['sl'] = float(part.split(':')[1].strip())
                        elif 'TP:' in part:
                            trade['tp'] = float(part.split(':')[1].strip())
                        elif 'Ticket:' in part:
                            trade['ticket'] = int(part.split(':')[1].strip())
                    
                    if trade.get('pair'):
                        trades.append(trade)
                        
            except (json.JSONDecodeError, ValueError):
                continue
    
    return trades


def calculate_pip_distance(price1: float, price2: float, pair: str) -> float:
    """Calculate pip distance between two prices."""
    if 'JPY' in pair:
        return abs(price1 - price2) / 0.01
    return abs(price1 - price2) / 0.0001


def analyze_trades(trades: list, start_date: datetime, end_date: datetime) -> dict:
    """Analyze trades and generate recommendations."""
    
    # Filter by date
    filtered = []
    for t in trades:
        trade_time = t.get('time', '')
        if trade_time:
            try:
                if isinstance(trade_time, str):
                    trade_dt = datetime.fromisoformat(trade_time.replace('Z', '+00:00').replace('+00:00', ''))
                else:
                    trade_dt = trade_time
                    
                if start_date <= trade_dt <= end_date:
                    filtered.append(t)
            except:
                continue
    
    if not filtered:
        return {'error': 'No trades found in date range'}
    
    analysis = {
        'date_range': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        },
        'total_trades': len(filtered),
        'by_pair': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []}),
        'by_direction': {'BUY': {'wins': 0, 'losses': 0}, 'SELL': {'wins': 0, 'losses': 0}},
        'exit_analysis': {'sl_hits': 0, 'tp_hits': 0, 'manual': 0},
        'sl_distances': [],
        'tp_distances': [],
        'win_sl_distances': [],
        'loss_sl_distances': [],
        'trades': filtered
    }
    
    total_pnl = 0
    wins = 0
    losses = 0
    
    for t in filtered:
        pair = t.get('pair', t.get('symbol', 'UNKNOWN'))
        profit = t.get('profit', 0) + t.get('commission', 0) + t.get('swap', 0)
        direction = t.get('type', 'UNKNOWN')
        
        total_pnl += profit
        is_win = profit > 0
        
        if is_win:
            wins += 1
            analysis['by_direction'].get(direction, {})['wins'] = analysis['by_direction'].get(direction, {'wins': 0, 'losses': 0})['wins'] + 1
        else:
            losses += 1
            analysis['by_direction'].get(direction, {})['losses'] = analysis['by_direction'].get(direction, {'wins': 0, 'losses': 0})['losses'] + 1
        
        # Track by pair
        pair_stats = analysis['by_pair'][pair]
        if is_win:
            pair_stats['wins'] += 1
        else:
            pair_stats['losses'] += 1
        pair_stats['total_pnl'] += profit
        pair_stats['trades'].append({
            'profit': profit,
            'direction': direction,
            'time': t.get('time', ''),
            'price': t.get('price', 0)
        })
    
    analysis['summary'] = {
        'total_pnl': round(total_pnl, 2),
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / len(filtered) * 100, 1) if filtered else 0,
        'avg_win': round(sum(t.get('profit', 0) for t in filtered if t.get('profit', 0) > 0) / max(wins, 1), 2),
        'avg_loss': round(sum(t.get('profit', 0) for t in filtered if t.get('profit', 0) < 0) / max(losses, 1), 2)
    }
    
    # Convert defaultdict to regular dict for JSON serialization
    analysis['by_pair'] = {k: dict(v) for k, v in analysis['by_pair'].items()}
    
    return analysis


def generate_recommendations(analysis: dict) -> dict:
    """Generate TP/SL adjustment recommendations based on analysis."""
    
    if 'error' in analysis:
        return {'error': analysis['error']}
    
    summary = analysis.get('summary', {})
    win_rate = summary.get('win_rate', 0)
    avg_win = abs(summary.get('avg_win', 0))
    avg_loss = abs(summary.get('avg_loss', 0))
    
    recommendations = {
        'diagnosis': [],
        'tp_adjustments': {},
        'sl_adjustments': {},
        'entry_adjustments': {}
    }
    
    # Win rate analysis
    if win_rate < 40:
        recommendations['diagnosis'].append(
            f"⚠️ Win rate critically low ({win_rate}%) — SL likely too tight or entries too late"
        )
        recommendations['sl_adjustments']['multiplier_change'] = '+0.2×ATR (wider SL)'
        recommendations['sl_adjustments']['reason'] = 'Low win rate suggests stops getting hit by noise'
    elif win_rate < 50:
        recommendations['diagnosis'].append(
            f"⚠️ Win rate below breakeven ({win_rate}%) — consider widening SL"
        )
        recommendations['sl_adjustments']['multiplier_change'] = '+0.1×ATR'
    else:
        recommendations['diagnosis'].append(f"✅ Win rate acceptable ({win_rate}%)")
    
    # R:R analysis
    if avg_loss > 0:
        actual_rr = avg_win / avg_loss
        recommendations['actual_rr'] = round(actual_rr, 2)
        
        if actual_rr < 1.0:
            recommendations['diagnosis'].append(
                f"⚠️ Negative R:R ({actual_rr:.2f}) — avg win ${avg_win:.2f} < avg loss ${avg_loss:.2f}"
            )
            recommendations['tp_adjustments']['ratio_change'] = '+0.2R (wider TP)'
            recommendations['tp_adjustments']['reason'] = 'Let winners run longer'
        elif actual_rr < 1.3:
            recommendations['diagnosis'].append(
                f"⚠️ Marginal R:R ({actual_rr:.2f}) — consider widening TP"
            )
            recommendations['tp_adjustments']['ratio_change'] = '+0.1R'
    
    # Direction bias
    by_dir = analysis.get('by_direction', {})
    buy_wr = by_dir.get('BUY', {}).get('wins', 0) / max(by_dir.get('BUY', {}).get('wins', 0) + by_dir.get('BUY', {}).get('losses', 0), 1) * 100
    sell_wr = by_dir.get('SELL', {}).get('wins', 0) / max(by_dir.get('SELL', {}).get('wins', 0) + by_dir.get('SELL', {}).get('losses', 0), 1) * 100
    
    if abs(buy_wr - sell_wr) > 20:
        weaker = 'BUY' if buy_wr < sell_wr else 'SELL'
        recommendations['diagnosis'].append(
            f"⚠️ Direction imbalance: {weaker} signals underperforming ({buy_wr:.0f}% vs {sell_wr:.0f}%)"
        )
        recommendations['entry_adjustments']['direction_filter'] = f"Increase confidence threshold for {weaker} by +10%"
    
    # Pair analysis
    for pair, stats in analysis.get('by_pair', {}).items():
        pair_wr = stats['wins'] / max(stats['wins'] + stats['losses'], 1) * 100
        if pair_wr < 35 and stats['wins'] + stats['losses'] >= 3:
            recommendations['diagnosis'].append(
                f"⚠️ {pair} severely underperforming ({pair_wr:.0f}% WR) — consider pausing"
            )
    
    # Concrete parameter recommendations
    if win_rate < 45:
        recommendations['new_parameters'] = {
            'sl_atr_mult': 1.4,  # Wider than current 1.2
            'sl_min_atr_mult': 1.1,  # Wider floor
            'tp_base_ratio': 1.2,  # Slightly tighter TP for faster wins
            'entry_threshold': 0.50,  # Stricter entries
            'drift_tolerance_atr': 3.0,  # Allow more drift (was 2.5)
        }
    elif win_rate < 55:
        recommendations['new_parameters'] = {
            'sl_atr_mult': 1.3,
            'sl_min_atr_mult': 1.0,
            'tp_base_ratio': 1.15,
            'entry_threshold': 0.47,
            'drift_tolerance_atr': 2.8,
        }
    else:
        recommendations['new_parameters'] = {
            'sl_atr_mult': 1.2,
            'sl_min_atr_mult': 1.0,
            'tp_base_ratio': 1.3,
            'entry_threshold': 0.45,
            'drift_tolerance_atr': 2.5,
        }
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Analyze MT5 trade history')
    parser.add_argument('--start', type=str, default='2026-03-05', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2026-03-06', help='End date (YYYY-MM-DD)')
    parser.add_argument('--hours', type=int, default=168, help='Hours of history to fetch (default: 168 = 7 days)')
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d') + timedelta(days=1)  # Include end date
    
    print(f"📊 Analyzing trades from {args.start} to {args.end}")
    print("=" * 60)
    
    # Try MT5 relay first
    relay_url = os.getenv('MT5_RELAY_URL', 'http://127.0.0.1:5555')
    print(f"🔌 Connecting to MT5 relay at {relay_url}...")
    
    trades = fetch_mt5_history(relay_url, args.hours)
    
    if not trades:
        print("⚠️ No trades from MT5 relay, trying log file fallback...")
        log_path = Path(__file__).parent.parent / 'logs' / 'trades.log'
        trades = parse_trades_from_logs(str(log_path), start_date, end_date)
    
    if not trades:
        print("❌ No trades found. Make sure:")
        print("   1. MT5 relay is running (python mt5_relay/relay_server.py)")
        print("   2. You have trades in the date range")
        print("   3. Or check logs/trades.log for recorded trades")
        return
    
    print(f"✅ Found {len(trades)} total trades")
    
    # Analyze
    analysis = analyze_trades(trades, start_date, end_date)
    
    if 'error' in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    # Generate recommendations
    recs = generate_recommendations(analysis)
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 ANALYSIS SUMMARY")
    print("=" * 60)
    
    summary = analysis.get('summary', {})
    print(f"Total Trades: {analysis['total_trades']}")
    print(f"Win Rate: {summary.get('win_rate', 0)}%")
    print(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
    print(f"Avg Win: ${summary.get('avg_win', 0):.2f}")
    print(f"Avg Loss: ${summary.get('avg_loss', 0):.2f}")
    
    if 'actual_rr' in recs:
        print(f"Actual R:R: {recs['actual_rr']:.2f}")
    
    print("\n📊 By Pair:")
    for pair, stats in analysis.get('by_pair', {}).items():
        wr = stats['wins'] / max(stats['wins'] + stats['losses'], 1) * 100
        print(f"  {pair}: {stats['wins']}W/{stats['losses']}L ({wr:.0f}%) | PnL: ${stats['total_pnl']:.2f}")
    
    print("\n🔍 DIAGNOSIS")
    print("-" * 40)
    for diag in recs.get('diagnosis', []):
        print(f"  {diag}")
    
    print("\n🎯 RECOMMENDED PARAMETERS")
    print("-" * 40)
    params = recs.get('new_parameters', {})
    for key, val in params.items():
        print(f"  {key}: {val}")
    
    # Save outputs
    output_dir = Path(__file__).parent.parent / 'data'
    output_prefix = f"trade_analysis_{args.start}_{args.end}"
    
    # Save CSV
    csv_path = output_dir / f"{output_prefix}.csv"
    with open(csv_path, 'w') as f:
        f.write("time,pair,type,profit,price\n")
        for t in analysis.get('trades', []):
            f.write(f"{t.get('time','')},{t.get('pair','')},{t.get('type','')},{t.get('profit',0)},{t.get('price',0)}\n")
    print(f"\n💾 Saved: {csv_path}")
    
    # Save JSON
    json_path = output_dir / f"{output_prefix}.json"
    
    # Clean up for JSON serialization
    output_data = {
        'summary': summary,
        'by_pair': analysis.get('by_pair', {}),
        'by_direction': analysis.get('by_direction', {}),
        'recommendations': recs
    }
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"💾 Saved: {json_path}")
    
    return recs


if __name__ == '__main__':
    main()
