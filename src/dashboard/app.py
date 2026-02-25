"""
Web Dashboard for the AI Trading Bot

Provides a real-time monitoring interface:
  - Account status, balance, equity
  - Open positions and P/L
  - Signal history
  - Model performance metrics
  - Adaptive learning stats

Run standalone:  python -m scripts.run_dashboard
"""
import os
import json
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global references (set by start_dashboard)
_bot_ref = None
_learner_ref = None

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Forex Bot - Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --border: #30363d;
    --text: #c9d1d9; --text2: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }
  .header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 1.3em; color: var(--accent); }
  .header .status { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }
  .status.live { background: rgba(63, 185, 80, 0.2); color: var(--green); }
  .status.paper { background: rgba(210, 153, 34, 0.2); color: var(--yellow); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; padding: 24px; }
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
  .card h2 { font-size: 0.95em; color: var(--text2); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .metric:last-child { border-bottom: none; }
  .metric .label { color: var(--text2); }
  .metric .value { font-weight: 600; }
  .value.positive { color: var(--green); }
  .value.negative { color: var(--red); }
  .signal-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.9em; }
  .signal-badge { padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; }
  .signal-badge.BUY { background: rgba(63,185,80,.2); color: var(--green); }
  .signal-badge.SELL { background: rgba(248,81,73,.2); color: var(--red); }
  .signal-badge.SKIP, .signal-badge.HOLD { background: rgba(139,148,158,.2); color: var(--text2); }
  .bar-chart { display: flex; align-items: end; gap: 4px; height: 60px; margin-top: 8px; }
  .bar { flex: 1; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 2px; transition: height 0.3s; }
  .bar.loss { background: var(--red); }
  .weight-bar { height: 8px; background: var(--border); border-radius: 4px; margin: 4px 0; overflow: hidden; }
  .weight-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.5s; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th, td { padding: 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--text2); font-weight: 500; }
  .refresh-btn { background: var(--accent); color: #fff; border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.85em; }
  .refresh-btn:hover { opacity: 0.85; }
  @media(max-width:700px) { .grid { grid-template-columns: 1fr; padding: 12px; } }
</style>
</head>
<body>
<div class="header">
  <h1>🤖 AI Forex Trading Bot</h1>
  <div>
    <span class="status" id="mode-badge">Loading...</span>
    <button class="refresh-btn" onclick="refresh()">↻ Refresh</button>
  </div>
</div>
<div class="grid" id="dashboard">
  <div class="card" id="account-card"><h2>Account</h2><p>Loading...</p></div>
  <div class="card" id="tier-card"><h2>Account Tier</h2><p>Loading...</p></div>
  <div class="card" id="risk-card"><h2>Risk Status</h2><p>Loading...</p></div>
  <div class="card" id="positions-card"><h2>Open Positions</h2><p>Loading...</p></div>
  <div class="card" id="signals-card"><h2>Recent Signals</h2><p>Loading...</p></div>
  <div class="card" id="weights-card"><h2>Model Weights</h2><p>Loading...</p></div>
  <div class="card" id="performance-card"><h2>Performance</h2><p>Loading...</p></div>
</div>
<script>
function $(id){return document.getElementById(id)}
function cls(v,pos=true){return pos?v>=0?'positive':'negative':'';}

async function refresh(){
  try{
    const r=await fetch('/api/status');
    const d=await r.json();
    renderAccount(d.account);
    renderTier(d.tier);
    renderRisk(d.risk);
    renderPositions(d.positions);
    renderSignals(d.signals);
    renderWeights(d.weights);
    renderPerformance(d.performance);
    const badge=$('mode-badge');
    badge.textContent=d.mode.toUpperCase();
    badge.className='status '+ d.mode;
  }catch(e){console.error(e);}
}

function renderAccount(a){
  if(!a)return;
  $('account-card').innerHTML=`<h2>Account</h2>
    <div class="metric"><span class="label">Balance</span><span class="value">$${a.balance.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Equity</span><span class="value">$${a.equity.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Profit</span><span class="value ${cls(a.profit)}">$${a.profit>=0?'+':''}${a.profit.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Leverage</span><span class="value">1:${a.leverage}</span></div>`;
}
function renderTier(t){
  if(!t){return;}
  const growthCls=t.account_growth>=0?'positive':'negative';
  const pct=t.next_tier_at?Math.min(100,((t.next_tier_at-t.balance_to_next)/t.next_tier_at)*100).toFixed(0):100;
  $('tier-card').innerHTML=`<h2>Account Tier</h2>
    <div class="metric"><span class="label">Current Tier</span><span class="value" style="color:var(--accent)">${t.tier_description}</span></div>
    <div class="metric"><span class="label">Growth</span><span class="value ${growthCls}">${t.account_growth>=0?'+':''}${t.account_growth.toFixed(1)}%</span></div>
    <div class="metric"><span class="label">Max Lot Size</span><span class="value">${t.max_lot_size}</span></div>
    <div class="metric"><span class="label">Max Trades</span><span class="value">${t.max_concurrent_trades}</span></div>
    <div class="metric"><span class="label">Risk/Trade</span><span class="value">${t.risk_percent}%</span></div>
    <div class="metric"><span class="label">Next Tier</span><span class="value">${t.next_tier} ($${t.balance_to_next.toFixed(0)} away)</span></div>
    <div class="weight-bar"><div class="weight-fill" style="width:${pct}%"></div></div>`;
}
function renderRisk(r){
  if(!r)return;
  $('risk-card').innerHTML=`<h2>Risk Status</h2>
    <div class="metric"><span class="label">Daily Loss</span><span class="value ${r.daily_loss>0?'negative':''}">$${r.daily_loss.toFixed(2)} (${r.daily_loss_percent.toFixed(1)}%)</span></div>
    <div class="metric"><span class="label">Open Trades</span><span class="value">${r.open_trades}</span></div>
    <div class="metric"><span class="label">Can Trade</span><span class="value ${r.can_trade?'positive':'negative'}">${r.can_trade?'YES':'NO'}</span></div>
    <div class="metric"><span class="label">Threshold</span><span class="value">${(r.confidence_threshold*100).toFixed(0)}%</span></div>`;
}

function renderPositions(positions){
  if(!positions||!positions.length){$('positions-card').innerHTML='<h2>Open Positions</h2><p style="color:var(--text2)">No open positions</p>';return;}
  let rows=positions.map(p=>`<tr><td>${p.pair}</td><td><span class="signal-badge ${p.type}">${p.type}</span></td><td>${p.open_price.toFixed(5)}</td><td class="${cls(p.profit)}">${p.profit>=0?'+':''}${p.profit.toFixed(2)}</td></tr>`).join('');
  $('positions-card').innerHTML=`<h2>Open Positions</h2><table><tr><th>Pair</th><th>Type</th><th>Entry</th><th>P/L</th></tr>${rows}</table>`;
}

function renderSignals(signals){
  if(!signals||!signals.length){$('signals-card').innerHTML='<h2>Recent Signals</h2><p style="color:var(--text2)">No signals yet</p>';return;}
  let rows=signals.slice(0,10).map(s=>`<div class="signal-row"><span>${s.pair}</span><span class="signal-badge ${s.signal}">${s.signal}</span><span>${(s.confidence*100).toFixed(0)}%</span><span style="color:var(--text2)">${s.time||''}</span></div>`).join('');
  $('signals-card').innerHTML=`<h2>Recent Signals</h2>${rows}`;
}

function renderWeights(w){
  if(!w){return;}
  let bars=Object.entries(w).map(([k,v])=>`<div style="margin-bottom:8px"><div class="metric"><span class="label">${k.toUpperCase()}</span><span class="value">${(v*100).toFixed(1)}%</span></div><div class="weight-bar"><div class="weight-fill" style="width:${v*100}%"></div></div></div>`).join('');
  $('weights-card').innerHTML=`<h2>Model Weights (Adaptive)</h2>${bars}`;
}

function renderPerformance(p){
  if(!p){return;}
  $('performance-card').innerHTML=`<h2>Performance</h2>
    <div class="metric"><span class="label">Total Trades</span><span class="value">${p.total_trades}</span></div>
    <div class="metric"><span class="label">Win Rate</span><span class="value ${p.win_rate>=0.5?'positive':'negative'}">${(p.win_rate*100).toFixed(1)}%</span></div>
    <div class="metric"><span class="label">Total P/L</span><span class="value ${cls(p.total_pnl)}">$${p.total_pnl>=0?'+':''}${p.total_pnl.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Consec. Losses</span><span class="value">${p.consecutive_losses}</span></div>
    <div class="metric"><span class="label">Max Consec. Losses</span><span class="value">${p.max_consecutive_losses}</span></div>`;
}

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/status')
def api_status():
    """Full status endpoint for the dashboard."""
    status = {
        'mode': 'paper',
        'account': {
            'balance': 50.0, 'equity': 50.0, 'profit': 0.0, 'leverage': 100
        },
        'risk': {
            'daily_loss': 0.0, 'daily_loss_percent': 0.0,
            'open_trades': 0, 'can_trade': True, 'confidence_threshold': 0.75
        },
        'positions': [],
        'signals': [],
        'weights': {'lstm': 0.35, 'sentiment': 0.30, 'technical': 0.20, 'volume': 0.15},
        'performance': {
            'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0,
            'consecutive_losses': 0, 'max_consecutive_losses': 0
        },
        'tier': {
            'tier_description': 'Micro ($0-$200)', 'account_growth': 0.0,
            'max_lot_size': 0.05, 'max_concurrent_trades': 2,
            'risk_percent': 1.0, 'next_tier': 'Mini ($200-$1K)',
            'balance_to_next': 150.0, 'next_tier_at': 200.0,
        },
    }

    if _bot_ref:
        try:
            status['mode'] = _bot_ref.mode

            # Account info
            if _bot_ref.broker:
                acct = _bot_ref.broker.get_account_info()
                if acct:
                    status['account'] = {
                        'balance': acct.get('balance', 50),
                        'equity': acct.get('equity', 50),
                        'profit': acct.get('profit', 0),
                        'leverage': acct.get('leverage', 100),
                    }

            # Risk status (includes tier info)
            risk = _bot_ref.risk_manager.get_daily_status()
            status['risk'] = {
                'daily_loss': risk['daily_loss'],
                'daily_loss_percent': risk['daily_loss_percent'],
                'open_trades': risk['open_trades'],
                'max_concurrent_trades': risk.get('max_concurrent_trades', 3),
                'can_trade': risk['can_trade'],
                'confidence_threshold': getattr(
                    _learner_ref, 'confidence_threshold', 0.75
                ) if _learner_ref else 0.75,
            }

            # Tier info
            status['tier'] = {
                'tier_description': risk.get('tier_description', 'Micro'),
                'account_growth': risk.get('account_growth', 0),
                'max_lot_size': risk.get('max_lot_size', 0.05),
                'max_concurrent_trades': risk.get('max_concurrent_trades', 2),
                'risk_percent': risk.get('risk_percent', 1.0),
                'next_tier': risk.get('next_tier', 'N/A'),
                'balance_to_next': risk.get('balance_to_next', 0),
                'next_tier_at': risk.get('balance_to_next', 0) + risk.get('current_balance', 50),
            }

            # Open positions
            if _bot_ref.broker:
                positions = _bot_ref.broker.get_open_positions()
                status['positions'] = positions or []

            # Signals from history
            if hasattr(_bot_ref, 'signal_history'):
                status['signals'] = _bot_ref.signal_history[-20:]

        except Exception as e:
            status['error'] = str(e)

    # Adaptive learner
    if _learner_ref:
        try:
            status['weights'] = _learner_ref.get_adjusted_weights()
            perf = _learner_ref.get_performance_summary()
            status['performance'] = {
                'total_trades': perf['total_trades'],
                'win_rate': perf['win_rate'],
                'total_pnl': perf['total_pnl'],
                'consecutive_losses': perf['consecutive_losses'],
                'max_consecutive_losses': perf['max_consecutive_losses'],
            }
        except Exception:
            pass

    return jsonify(status)


@app.route('/api/trades')
def api_trades():
    """Return trade history."""
    if _learner_ref:
        return jsonify(_learner_ref.trade_history[-100:])
    return jsonify([])


@app.route('/api/model-accuracy')
def api_model_accuracy():
    """Return per-model accuracy stats."""
    if _learner_ref:
        return jsonify(dict(_learner_ref.model_accuracy))
    return jsonify({})


def start_dashboard(bot=None, learner=None, port=5000):
    """
    Start the dashboard server.

    Args:
        bot:     TradingBot instance (optional)
        learner: AdaptiveLearner instance (optional)
        port:    HTTP port
    """
    global _bot_ref, _learner_ref
    _bot_ref = bot
    _learner_ref = learner

    print(f"\n🌐 Dashboard running at http://localhost:{port}")
    print(f"   Open in browser to monitor the bot\n")

    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
