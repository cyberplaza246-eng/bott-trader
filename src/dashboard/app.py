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
    --orange: #db6d28;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }
  .header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
  .header h1 { font-size: 1.3em; color: var(--accent); }
  .header .status { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }
  .status.live { background: rgba(63, 185, 80, 0.2); color: var(--green); }
  .status.paper { background: rgba(210, 153, 34, 0.2); color: var(--yellow); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; padding: 24px; }
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
  .card.full-width { grid-column: 1 / -1; }
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
  .signal-badge.WIN { background: rgba(63,185,80,.15); color: var(--green); }
  .signal-badge.LOSS { background: rgba(248,81,73,.15); color: var(--red); }
  .signal-badge.TP { background: rgba(63,185,80,.15); color: var(--green); }
  .signal-badge.SL { background: rgba(248,81,73,.15); color: var(--red); }
  .weight-bar { height: 8px; background: var(--border); border-radius: 4px; margin: 4px 0; overflow: hidden; }
  .weight-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.5s; }
  .weight-fill.green { background: var(--green); }
  .weight-fill.red { background: var(--red); }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th, td { padding: 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--text2); font-weight: 500; }
  .refresh-btn { background: var(--accent); color: #fff; border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.85em; }
  .refresh-btn:hover { opacity: 0.85; }
  .last-update { font-size: 0.75em; color: var(--text2); margin-right: 10px; }
  .pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--green); margin-right: 6px; animation: pulse-anim 2s infinite; }
  @keyframes pulse-anim { 0%,100%{ box-shadow: 0 0 0 0 rgba(63,185,80,0.6); } 50%{ box-shadow: 0 0 0 6px rgba(63,185,80,0); } }
  .flash { animation: flash-anim 0.4s; }
  @keyframes flash-anim { 0%{ opacity:0.5; } 100%{ opacity:1; } }
  /* Stats row – 3 mini-cards */
  .stats-row { display: flex; gap: 12px; margin-bottom: 12px; }
  .stat-box { flex: 1; text-align: center; padding: 12px 8px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
  .stat-box .big { font-size: 1.6em; font-weight: 700; line-height: 1.2; }
  .stat-box .sub { font-size: 0.75em; color: var(--text2); margin-top: 2px; }
  /* Win/loss mini bar */
  .wl-bar { display: flex; height: 6px; border-radius: 3px; overflow: hidden; margin: 8px 0 4px; background: var(--border); }
  .wl-bar .win-part { background: var(--green); }
  .wl-bar .loss-part { background: var(--red); }
  /* Pair perf grid */
  .pair-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.9em; }
  .pair-row:last-child { border-bottom: none; }
  .pair-row .pair-name { font-weight: 600; min-width: 70px; }
  .pair-row .pair-wr { min-width: 50px; text-align: right; }
  .pair-row .pair-pnl { min-width: 70px; text-align: right; font-weight: 600; }
  /* Session badges */
  .session-grid { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .session-badge { padding: 6px 12px; border-radius: 6px; background: var(--bg); border: 1px solid var(--border); font-size: 0.8em; text-align: center; }
  .session-badge .sname { font-weight: 600; color: var(--accent); text-transform: capitalize; }
  .session-badge .swl { margin-top: 2px; color: var(--text2); }
  @media(max-width:700px) { .grid { grid-template-columns: 1fr; padding: 12px; } .stats-row { flex-direction: column; } }
</style>
</head>
<body>
<div class="header">
  <h1><span class="pulse"></span>🤖 AI Forex Trading Bot</h1>
  <div>
    <span class="last-update" id="last-update"></span>
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
  <div class="card" id="pair-perf-card"><h2>Pair Performance</h2><p>Loading...</p></div>
  <div class="card full-width" id="trade-history-card"><h2>Trade History</h2><p>Loading...</p></div>
</div>
<script>
function $(id){return document.getElementById(id)}
function cls(v){return v>=0?'positive':'negative';}

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
    renderPairPerf(d.performance);
    renderTradeHistory(d.trade_history||[]);
    const badge=$('mode-badge');
    badge.textContent=d.mode.toUpperCase();
    badge.className='status '+ d.mode;
    const now=new Date();
    $('last-update').textContent='Updated '+now.toLocaleTimeString();
    document.querySelectorAll('.card').forEach(c=>{c.classList.remove('flash');void c.offsetWidth;c.classList.add('flash');});
  }catch(e){
    $('last-update').textContent='⚠ Connection lost';
    console.error(e);
  }
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
  if(!t)return;
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
  let totalPnl=positions.reduce((s,p)=>s+(p.profit||0),0);
  let rows=positions.map(p=>`<tr><td>${p.pair}</td><td><span class="signal-badge ${p.type}">${p.type}</span></td><td>${Number(p.open_price).toFixed(5)}</td><td class="${cls(p.profit)}" style="font-weight:600">${p.profit>=0?'+':''}${p.profit.toFixed(2)}</td></tr>`).join('');
  $('positions-card').innerHTML=`<h2>Open Positions <span style="float:right;font-size:0.85em" class="${cls(totalPnl)}">Net: $${totalPnl>=0?'+':''}${totalPnl.toFixed(2)}</span></h2><table><tr><th>Pair</th><th>Type</th><th>Entry</th><th>P/L</th></tr>${rows}</table>`;
}

function renderSignals(signals){
  if(!signals||!signals.length){$('signals-card').innerHTML='<h2>Recent Signals</h2><p style="color:var(--text2)">No signals yet</p>';return;}
  let rows=signals.slice(0,10).map(s=>`<div class="signal-row"><span>${s.pair}</span><span class="signal-badge ${s.signal}">${s.signal}</span><span>${(s.confidence*100).toFixed(0)}%</span><span style="color:var(--text2)">${s.time||''}</span></div>`).join('');
  $('signals-card').innerHTML=`<h2>Recent Signals</h2>${rows}`;
}

function renderWeights(w){
  if(!w)return;
  let bars=Object.entries(w).map(([k,v])=>`<div style="margin-bottom:8px"><div class="metric"><span class="label">${k.toUpperCase()}</span><span class="value">${(v*100).toFixed(1)}%</span></div><div class="weight-bar"><div class="weight-fill" style="width:${v*100}%"></div></div></div>`).join('');
  $('weights-card').innerHTML=`<h2>Model Weights (Adaptive)</h2>${bars}`;
}

function renderPerformance(p){
  if(!p)return;
  const wins=p.total_wins||0;
  const losses=p.total_losses||0;
  const total=p.total_trades||0;
  const wrPct=total>0?(wins/total*100).toFixed(1):'0.0';
  const winBarW=total>0?(wins/total*100):50;
  const avgWin=p.avg_win||0;
  const avgLoss=p.avg_loss||0;
  const bestTrade=p.best_trade||0;
  const worstTrade=p.worst_trade||0;
  const profitFactor=p.profit_factor||0;

  $('performance-card').innerHTML=`<h2>Performance</h2>
    <div class="stats-row">
      <div class="stat-box"><div class="big" style="color:var(--accent)">${total}</div><div class="sub">Total Trades</div></div>
      <div class="stat-box"><div class="big positive">${wins}</div><div class="sub">Wins</div></div>
      <div class="stat-box"><div class="big negative">${losses}</div><div class="sub">Losses</div></div>
    </div>
    <div class="metric"><span class="label">Win Rate</span><span class="value ${parseFloat(wrPct)>=50?'positive':'negative'}">${wrPct}%</span></div>
    <div class="wl-bar"><div class="win-part" style="width:${winBarW}%"></div><div class="loss-part" style="width:${100-winBarW}%"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:0.75em;color:var(--text2);margin-bottom:8px"><span>${wins}W</span><span>${losses}L</span></div>
    <div class="metric"><span class="label">Total P/L</span><span class="value ${cls(p.total_pnl)}">$${p.total_pnl>=0?'+':''}${p.total_pnl.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Avg Win</span><span class="value positive">$${avgWin>=0?'+':''}${avgWin.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Avg Loss</span><span class="value negative">$${avgLoss.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Best Trade</span><span class="value positive">$${bestTrade>=0?'+':''}${bestTrade.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Worst Trade</span><span class="value negative">$${worstTrade.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Profit Factor</span><span class="value ${profitFactor>=1?'positive':'negative'}">${profitFactor.toFixed(2)}</span></div>
    <div class="metric"><span class="label">Consec. Losses</span><span class="value">${p.consecutive_losses}</span></div>
    <div class="metric"><span class="label">Max Consec. Losses</span><span class="value">${p.max_consecutive_losses}</span></div>
    ${p.session_stats?renderSessionBadges(p.session_stats):''}`;
}

function renderSessionBadges(ss){
  if(!ss||!Object.keys(ss).length)return '';
  let badges=Object.entries(ss).map(([name,s])=>{
    const t=s.wins+s.losses;
    const wr=t>0?(s.wins/t*100).toFixed(0):'--';
    return `<div class="session-badge"><div class="sname">${name}</div><div class="swl">${s.wins}W / ${s.losses}L (${wr}%)</div></div>`;
  }).join('');
  return `<div style="margin-top:12px"><span style="color:var(--text2);font-size:0.8em;text-transform:uppercase">Session Performance</span><div class="session-grid">${badges}</div></div>`;
}

function renderPairPerf(p){
  if(!p||!p.pair_stats||!Object.keys(p.pair_stats).length){
    $('pair-perf-card').innerHTML='<h2>Pair Performance</h2><p style="color:var(--text2)">No pair data yet</p>';return;
  }
  let rows=Object.entries(p.pair_stats).map(([pair,s])=>{
    const t=s.wins+s.losses;
    const wr=t>0?(s.wins/t*100).toFixed(0):'--';
    const wrCls=t>0?(s.wins/t>=0.5?'positive':'negative'):'';
    const pnlCls=s.total_pnl>=0?'positive':'negative';
    const barW=t>0?(s.wins/t*100):0;
    return `<div class="pair-row">
      <span class="pair-name">${pair}</span>
      <span style="color:var(--text2);font-size:0.8em">${s.wins}W / ${s.losses}L</span>
      <span class="pair-wr ${wrCls}">${wr}%</span>
      <span class="pair-pnl ${pnlCls}">$${s.total_pnl>=0?'+':''}${s.total_pnl.toFixed(2)}</span>
    </div>
    <div class="wl-bar" style="margin:0 0 6px"><div class="win-part" style="width:${barW}%"></div><div class="loss-part" style="width:${100-barW}%"></div></div>`;
  }).join('');
  $('pair-perf-card').innerHTML=`<h2>Pair Performance</h2>${rows}`;
}

function renderTradeHistory(trades){
  if(!trades||!trades.length){$('trade-history-card').innerHTML='<h2>Trade History</h2><p style="color:var(--text2)">No closed trades yet</p>';return;}
  // Show most recent first, limit to 20
  let recent=trades.slice(-20).reverse();
  let rows=recent.map(t=>{
    const pnl=t.profit_loss||0;
    const win=t.is_win;
    const exitType=(t.exit_type||'').replace('TAKE_PROFIT','TP').replace('STOP_LOSS','SL');
    const exitBadge=exitType==='TP'?'TP':'SL';
    const ts=t.timestamp?t.timestamp.substring(5,16).replace('T',' '):'';
    return `<tr>
      <td style="color:var(--text2)">${ts}</td>
      <td>${t.pair||''}</td>
      <td><span class="signal-badge ${t.signal||''}">${t.signal||''}</span></td>
      <td>${t.entry_price?Number(t.entry_price).toFixed(5):''}</td>
      <td>${t.exit_price?Number(t.exit_price).toFixed(5):''}</td>
      <td><span class="signal-badge ${exitBadge}">${exitType}</span></td>
      <td class="${cls(pnl)}" style="font-weight:600">$${pnl>=0?'+':''}${pnl.toFixed(2)}</td>
      <td><span class="signal-badge ${win?'WIN':'LOSS'}">${win?'WIN':'LOSS'}</span></td>
    </tr>`;
  }).join('');
  $('trade-history-card').innerHTML=`<h2>Trade History <span style="float:right;font-size:0.8em;color:var(--text2)">Last ${recent.length} trades</span></h2>
    <div style="overflow-x:auto"><table><tr><th>Time</th><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th><th>Type</th><th>P/L</th><th>Result</th></tr>${rows}</table></div>`;
}

refresh();
setInterval(refresh, 3000);
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

            # Compute extra stats from trade history
            trades = _learner_ref.trade_history or []
            wins_pnl = [t.get('profit_loss', 0) for t in trades if t.get('is_win')]
            losses_pnl = [t.get('profit_loss', 0) for t in trades if not t.get('is_win')]
            all_pnl = [t.get('profit_loss', 0) for t in trades]

            avg_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0
            avg_loss = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0
            best_trade = max(all_pnl) if all_pnl else 0
            worst_trade = min(all_pnl) if all_pnl else 0
            gross_wins = sum(wins_pnl) if wins_pnl else 0
            gross_losses = abs(sum(losses_pnl)) if losses_pnl else 0
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else (
                999.99 if gross_wins > 0 else 0
            )

            status['performance'] = {
                'total_trades': perf['total_trades'],
                'total_wins': perf.get('total_wins', 0),
                'total_losses': perf.get('total_losses', 0),
                'win_rate': perf['win_rate'],
                'total_pnl': perf['total_pnl'],
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'profit_factor': profit_factor,
                'consecutive_losses': perf['consecutive_losses'],
                'max_consecutive_losses': perf['max_consecutive_losses'],
                'pair_stats': perf.get('pair_stats', {}),
                'session_stats': perf.get('session_stats', {}),
            }

            # Send last 50 closed trades for the history table
            status['trade_history'] = trades[-50:]
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
