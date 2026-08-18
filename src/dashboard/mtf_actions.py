"""Local website for MNQ live actions + Gemini chat."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from flask import Flask, jsonify, make_response, render_template_string, request
from flask_cors import CORS

from src.ai.action_log import (
    OVERNIGHT_FLATTEN_ET,
    OVERNIGHT_TP_TEXT,
    ActionLog,
    build_today_activity,
    format_live_entry_brackets,
    format_stop_line,
    is_overnight_rec,
    read_journal,
    read_status,
)
from src.ai.llm_advisor import (
    LLMTradeAdvisor,
    apply_overnight_live_quote,
    build_advisory_market_packet,
    compute_recipe_levels,
    overlay_overnight_levels,
    extract_direction_callout,
    invalidate_1m_cache,
    load_desk_quote_meta,
    load_locked_backtest_facts,
)
from src.ai.trade_review import JOURNAL_LIVE, JOURNAL_PAPER, read_suggestions, suggest_min_trades
from src.ai.overnight_journal import read_overnight_suggestions, read_zones
from src.ai.window_edge import (
    closed_trade_view,
    live_expectancy,
    read_snapshot,
)
from src.strategy.mnq_15m_ema_eod import ENTRY_WINDOWS, ET

app = Flask(__name__)
CORS(app)
_llm: Optional[LLMTradeAdvisor] = None
_last_ask_ok: Optional[bool] = None

STATUS_STALE_SEC = 30
GEMINI_HISTORY_N = 6
_HTML_PATH = Path(__file__).with_name("mtf_trend_desk.html")
_PACKET_CACHE: Dict[str, Any] = {"t": 0.0, "packet": None}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MNQ desk</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --bd:#30363d; --tx:#c9d1d9; --mut:#8b949e;
          --acc:#58a6ff; --ok:#3fb950; --bad:#f85149; --warn:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--tx); }
  header { display:flex; justify-content:space-between; align-items:center; padding:16px 22px;
           border-bottom:1px solid var(--bd); background:var(--card); gap:12px; flex-wrap:wrap; }
  h1 { font-size:1.15rem; margin:0; color:var(--acc); }
  .sub { color:var(--mut); font-size:.85rem; }
  .pill { padding:4px 10px; border-radius:999px; font-size:.8rem; font-weight:600; }
  .live { background:rgba(63,185,80,.2); color:var(--ok); }
  .paper { background:rgba(210,153,34,.2); color:var(--warn); }
  .overnight { background:rgba(224,180,78,.18); color:var(--warn); }
  .idle { background:rgba(139,148,158,.22); color:var(--mut); }
  .grid { display:grid; grid-template-columns:1.15fr .85fr; gap:16px; padding:18px; }
  @media (max-width:900px){ .grid { grid-template-columns:1fr; } }
  .wide { padding:0 18px 18px; }
  .split { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:900px){ .split { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--bd); border-radius:10px; padding:16px; }
  h2 { margin:0 0 10px; font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; color:var(--mut); }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  th,td { padding:7px 6px; border-bottom:1px solid var(--bd); text-align:left; white-space:nowrap; }
  th { color:var(--mut); font-weight:500; }
  .scroll { overflow-x:auto; }
  .tiny { font-size:.78rem; }
  .k { color:var(--mut); } .ok { color:var(--ok); } .bad { color:var(--bad); }
  .row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--bd); gap:10px; }
  .plain { font-size:1rem; line-height:1.55; }
  .plain .item { margin:0 0 12px; }
  .plain .item:last-child { margin-bottom:0; }
  .plain .lbl { display:block; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase;
                color:var(--mut); margin-bottom:4px; }
  textarea, input { width:100%; background:#0d1117; color:var(--tx); border:1px solid var(--bd);
                    border-radius:8px; padding:10px; font:inherit; }
  button { background:var(--acc); color:#fff; border:none; border-radius:8px; padding:8px 14px; cursor:pointer; }
  #gemini-out { white-space:pre-wrap; color:var(--tx); font-size:.9rem; margin-top:10px; min-height:3em; }
  details.block { background:var(--card); border:1px solid var(--bd); border-radius:10px; padding:12px 16px; }
  details.block > summary { cursor:pointer; color:var(--mut); font-size:.8rem; letter-spacing:.08em;
                            text-transform:uppercase; font-weight:600; }
  details.block[open] > summary { margin-bottom:12px; }
  .help { font-size:.78rem; color:var(--mut); line-height:1.45; margin:0 0 12px; }
  .qhist { font-size:.8rem; color:var(--mut); margin-top:10px; }
  .qhist .row span:first-child { overflow:hidden; text-overflow:ellipsis; }
</style>
</head>
<body>
<header>
  <div>
    <h1>MNQ desk</h1>
    <div class="sub">Read this top card first. Tables live under Details.</div>
  </div>
  <div>
    <span id="mode" class="pill idle">…</span>
    <span id="updated" class="k"></span>
  </div>
</header>
<div class="wide" style="padding-top:18px">
  <div class="card plain" id="plain-card">
    <h2>In plain English</h2>
    <div id="plain"></div>
  </div>
</div>
<div class="grid">
  <div class="card">
    <h2>Ask Gemini</h2>
    <p class="help" style="margin-top:0">Answers questions. Does not place trades or change stops.</p>
    <textarea id="q" rows="3" placeholder="e.g. Should I skip tomorrow if CPI is at 8:30?" onkeydown="onAskKey(event)"></textarea>
    <div style="margin-top:8px"><button onclick="ask()">Ask</button></div>
    <div id="gemini-out"></div>
    <div id="gemini-hist" class="qhist"></div>
  </div>
  <div>
    <div class="card" style="margin-bottom:16px">
      <h2>Status</h2>
      <div id="status"></div>
    </div>
    <div class="card">
      <h2>Suggestions</h2>
      <p id="suggest-meta" class="k" style="margin:0 0 8px;font-size:.8rem"></p>
      <div id="suggestions"></div>
    </div>
  </div>
</div>
<div class="wide">
  <details class="block" open>
    <summary>Details — numbers, closed trades, activity</summary>
    <p class="help" id="glossary"></p>
    <div class="card" style="margin-bottom:16px">
      <h2>Closed trades (from a running bot)</h2>
      <p id="trades-empty" class="k" style="margin:0 0 8px;font-size:.8rem"></p>
      <div class="scroll">
        <table class="tiny"><thead><tr>
          <th>Entry</th><th>Exit</th><th>Dir</th><th>In</th><th>Out</th>
          <th title="Exit Stop">Exit Stop</th>
          <th title="Worst pullback while in the trade">Worst pullback</th>
          <th title="Best unrealized run while in the trade">Best run</th>
          <th>How it exited</th>
          <th>15m</th><th>60m</th><th>Daily</th><th>ATR</th><th>P&amp;L</th>
        </tr></thead>
        <tbody id="trades"></tbody></table>
      </div>
    </div>
    <div class="split" style="margin-bottom:16px">
      <div class="card">
        <h2>Live journal (real closes)</h2>
        <p id="live-adv" class="k" style="margin:0 0 8px;font-size:.8rem"></p>
        <div id="live-exp"></div>
      </div>
      <div class="card">
        <h2>Locked 3-window backtest (not the 6-window Gemini cites)</h2>
        <p id="bt-adv" class="k" style="margin:0 0 8px;font-size:.8rem"></p>
        <div id="bt-exp"></div>
      </div>
    </div>
    <div class="card">
      <h2>Activity</h2>
      <p class="help">Bot entries/exits. Repeat Gemini questions are collapsed under Ask Gemini.</p>
      <table><thead><tr><th>Time</th><th>Kind</th><th>Detail</th></tr></thead>
      <tbody id="actions"></tbody></table>
    </div>
  </details>
</div>
<script>
const WIN_LABEL = {
  '09:35':'Morning 9:35',
  '11:00':'Late morning 11:00',
  '13:30':'Afternoon 13:30',
  'aligned':'daily + 15m + 60m all agree',
  'not aligned':'not all three agree',
};
function fmtTs(v){
  if(!v) return '—';
  const s=String(v).replace('T',' ');
  return s.slice(5,16);
}
function fmtN(v, d){
  if(v==null || v==='') return '—';
  const n=Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : '—';
}
function pfCell(s){
  if(!s || s.n===0) return '<span class="k">no trades</span>';
  const pf=s.pf==null?'n/a':Number(s.pf).toFixed(2);
  const exp=s.expectancy_usd==null?'—':'$'+Number(s.expectancy_usd).toFixed(0);
  const pnl=s.total_pnl_usd==null?'—':'$'+Number(s.total_pnl_usd).toFixed(0);
  const cls=(s.total_pnl_usd||0)>=0?'ok':'bad';
  return `${s.n} trades · profit factor ${pf} · ${exp} per trade · <span class="${cls}">${pnl} total</span>`;
}
function expTables(exp){
  if(!exp) return '<div class="k">No snapshot yet</div>';
  const rows=(title, map, keys)=> {
    const m=map||{};
    const use=keys || Object.keys(m);
    return `<div class="tiny" style="margin-bottom:10px"><div class="k">${title}</div>` +
      use.map(k=>{
        const s=m[k]||{n:0};
        return `<div class="row"><span>${WIN_LABEL[k]||k}</span><span>${pfCell(s)}</span></div>`;
      }).join('') + '</div>';
  };
  const canon=['09:35','11:00','13:30'];
  const extra=Object.keys(exp.by_window||{}).filter(k=>!canon.includes(k));
  const dist=exp.mfe_mae||{};
  const buckets=dist.winner_mfe_buckets||{};
  const bkeys=['0-15','15-20','20-40','40-80','80-120','120-200','200+'];
  const mfe = `<div class="tiny"><div class="k">How far winners ran (best unrealized run, points)</div>` +
    bkeys.map(k=>`<div class="row"><span>${k} pts</span><span>${buckets[k]||0}</span></div>`).join('') +
    `<div class="k" style="margin-top:6px">typical winner run ${dist.median_mfe_winners??'—'} pts · ` +
    `tiny (≤20) ${dist.pct_winners_mfe_le_20??'—'}% · big (≥80) ${dist.pct_winners_mfe_ge_80??'—'}% · ` +
    `typical loser pullback ${dist.median_mae_losers??'—'} pts · ` +
    `${dist.behavior==='trend-follow'?'trend hold, not a tiny scalp':(dist.behavior||'')}</div></div>`;
  const notes=(exp.advisory||[]).filter(x=>!/no live ema15 closes/i.test(String(x))).map(x=>`<div class="row"><span>${x}</span></div>`).join('');
  return rows('When it entered', exp.by_window, canon.concat(extra)) +
    rows('Long vs short', exp.by_side, ['long','short']) +
    rows('Did daily + 15m + 60m agree?', exp.by_alignment, ['aligned','not aligned']) +
    mfe + (notes? `<div class="tiny" style="margin-top:8px">${notes}</div>`:'');
}
function renderPlain(p){
  const items=[
    ['Live', p.live],
    ['Backtest (3 windows)', p.backtest],
    ['Suggestions', p.suggestions],
    ['Gemini', p.gemini],
    ['Two recipes', p.recipe_note],
  ].filter(x=>x[1]);
  document.getElementById('plain').innerHTML = items.map(([k,v])=>
    `<div class="item"><span class="lbl">${k}</span><span>${v}</span></div>`
  ).join('');
}
async function load(){
  const r = await fetch('/api/desk');
  const d = await r.json();
  const run = d.run || {};
  document.getElementById('mode').textContent = run.pill || 'DESK ONLY';
  document.getElementById('mode').className = 'pill ' + (run.pill_class || 'idle');
  document.getElementById('updated').textContent = run.updated_label || '';
  renderPlain(d.plain || {});
  document.getElementById('glossary').textContent =
    'Profit factor = gross wins ÷ gross losses (>1 means wins paid more than losses). ' +
    '$ per trade = average dollars per closed trade. ' +
    'Worst pullback = farthest the trade went against you while open. ' +
    'Best unrealized run = farthest it went in your favor while open. ' +
    'How it exited: Exit Stop, EOD flatten (3:50pm ET), or TP cap.';
  const st = document.getElementById('status');
  st.innerHTML = (d.status_rows||[]).map(([k,v])=>
    `<div class="row"><span class="k">${k}</span><span>${v}</span></div>`
  ).join('');
  document.getElementById('actions').innerHTML = (d.actions||[]).slice().reverse().map(a=>{
    const t=(a.ts||'').replace('T',' ').slice(11,19);
    const det=[a.symbol,a.direction,a.reason,a.pnl!=null?('$'+Number(a.pnl).toFixed(0)):''].filter(Boolean).join(' · ');
    return `<tr><td>${t}</td><td>${a.kind||''}</td><td>${det}</td></tr>`;
  }).join('') || '<tr><td colspan="3" class="k">No bot activity yet (desk only).</td></tr>';
  const hist=d.gemini_questions||[];
  document.getElementById('gemini-hist').innerHTML = hist.length
    ? '<div class="k" style="margin-bottom:6px">Recent questions</div>' + hist.slice().reverse().map(q=>{
        const n=q.repeat>1?` ×${q.repeat}`:'';
        return `<div class="row"><span>${q.question||''}${n}</span></div>`;
      }).join('')
    : '';
  const trades=d.trades||[];
  const liveEmpty=!!(d.expectancy&&d.expectancy.empty);
  document.getElementById('trades-empty').textContent = trades.length
    ? trades.length + ' closes from the journal. Worst pullback / best run fill after a trade has a 1-minute path.'
    : (liveEmpty ? 'This table fills after paper or live closes. Historical results are in the 3-window backtest panel.' : '');
  document.getElementById('trades').innerHTML = trades.slice().reverse().map(t=>{
    const pnl=Number(t.pnl_usd||t.pnl||0);
    const cls=pnl>=0?'ok':'bad';
    return `<tr>
      <td>${fmtTs(t.entry_ts_et||t.entry_time)}</td>
      <td>${fmtTs(t.exit_ts_et||t.exit_time)}</td>
      <td>${(t.direction||t.side||'').toUpperCase()}</td>
      <td>${fmtN(t.entry_price,2)}</td>
      <td>${fmtN(t.exit_price,2)}</td>
      <td>${fmtN(t.stop_distance,1)}</td>
      <td>${fmtN(t.mae_pts,1)}</td>
      <td>${fmtN(t.mfe_pts,1)}</td>
      <td>${t.desk_exit_reason||t.exit_reason||'—'}</td>
      <td>${t.trend_15m_txt||t.trend_15m||'—'}</td>
      <td>${t.trend_60m_txt||t.trend_60m||'—'}</td>
      <td>${t.daily_trend_txt||t.daily_trend||'—'}</td>
      <td>${fmtN(t.atr_at_entry,1)}</td>
      <td class="${cls}">$${pnl.toFixed(2)}</td>
    </tr>`;
  }).join('');
  const live=d.expectancy||{};
  if(live.empty){
    document.getElementById('live-adv').textContent = 'Empty until a paper or live bot records a close.';
    document.getElementById('live-exp').innerHTML = '<div class="k">No live numbers to show yet. Use the 3-window backtest next door.</div>';
  }else{
    document.getElementById('live-adv').textContent = ((live.overall&&live.overall.n||0)+' live closes · advisory only — the bot will not change itself');
    document.getElementById('live-exp').innerHTML = expTables(live);
  }
  const bt=d.backtest_edge||{};
  const btAll=bt.all||{};
  document.getElementById('bt-adv').textContent = bt.n_all
    ? `Locked 3-window recipe on historical 1-minute MNQ (morning 9:35, late-morning 11:00, afternoon 13:30). ${bt.n_all} trades · later-period subset ${bt.n_oos||0}. Advisory only.`
    : 'Run python scripts/analyze_window_edge.py to fill this panel.';
  document.getElementById('bt-exp').innerHTML = expTables(btAll);
  if(bt.oos && bt.n_oos){
    document.getElementById('bt-exp').innerHTML += '<div class="k tiny" style="margin:12px 0 6px">Later period only (after Jun 2026)</div>' + expTables(bt.oos);
  }
  const sg=d.suggestions||{};
  const meta=document.getElementById('suggest-meta');
  if(!sg.ready){
    meta.textContent = sg.note || 'None until 30 real closes. The bot will not change itself.';
    document.getElementById('suggestions').innerHTML='';
  }else{
    meta.textContent = (sg.generated_at_et||sg.generated_at||'') + ' · profit factor ' +
      (sg.overall_pf!=null?sg.overall_pf:'—') + ' · ' + (sg.n_closes||0) +
      ' closes · advisory only (never auto-applied)';
    document.getElementById('suggestions').innerHTML = (sg.suggestions||[]).map(x=>{
      const pf = x.pf!=null?x.pf:'n/a';
      const ov = x.overall_pf!=null?x.overall_pf:'n/a';
      return `<div class="row"><span>${x.tweak||''}</span><span class="k">${x.n} trades · profit factor ${pf} vs ${ov}</span></div>`;
    }).join('') || '<div class="k">No clusters yet</div>';
  }
  if(!document.getElementById('gemini-out').dataset.locked){
    const last=(d.gemini_questions||[]).slice(-1)[0];
    if(last && last.answer && !document.getElementById('gemini-out').textContent){
      document.getElementById('gemini-out').textContent = last.answer;
    }
  }
}
function onAskKey(e){
  if(e.key !== 'Enter') return;
  const isTextarea = (e.target.tagName || '').toLowerCase() === 'textarea';
  if(e.shiftKey && isTextarea) return;
  e.preventDefault();
  ask();
}
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q) return;
  const out=document.getElementById('gemini-out');
  out.dataset.locked='1';
  out.textContent='Thinking…';
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  out.textContent=d.answer||d.error||'';
  load();
}
load(); setInterval(load, 4000);
</script>
</body></html>
"""


def get_llm_advisor() -> LLMTradeAdvisor:
    global _llm
    if _llm is None:
        _llm = LLMTradeAdvisor()
    return _llm


def ask_looks_successful(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith("gemini/llm is off"):
        return False
    if low.startswith("gemini unavailable"):
        return False
    return True


def bot_run_state(
    status: Optional[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    stale_sec: int = STATUS_STALE_SEC,
) -> Dict[str, Any]:
    """desk_only vs paper vs live from the bot status file."""
    now = now or datetime.now(timezone.utc)
    status = status or {}
    updated = status.get("updated_at")
    age_sec: Optional[float] = None
    fresh = False
    if updated:
        try:
            t = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            age_sec = (now - t.astimezone(timezone.utc)).total_seconds()
            fresh = age_sec <= stale_sec
        except (TypeError, ValueError):
            fresh = False
    if not status or not fresh or status.get("running") is False:
        return {
            "state": "desk_only",
            "label": "not trading yet",
            "detail": "desk only / no live bot",
            "pill": "not running",
            "pill_class": "idle",
            "bot_running": False,
            "updated_at": updated,
            "updated_label": "no running bot" if not updated else "status stale",
            "age_sec": age_sec,
        }
    overnight = (
        bool(status.get("overnight_research"))
        or str(status.get("strategy") or "").lower() == "overnight_research"
        or str(status.get("session") or "").lower() == "overnight_research"
    )
    paper = bool(status.get("paper_mode")) or str(status.get("mode") or "").lower() == "paper"
    if overnight:
        return {
            "state": "overnight_research",
            "label": "paper overnight research",
            "detail": "PAPER OVERNIGHT RESEARCH",
            "pill": "PAPER OVERNIGHT RESEARCH",
            "pill_class": "overnight",
            "bot_running": True,
            "updated_at": updated,
            "updated_label": "overnight paper running",
            "age_sec": age_sec,
        }
    if paper:
        return {
            "state": "paper",
            "label": "paper",
            "detail": "bot running (paper)",
            "pill": "PAPER BOT",
            "pill_class": "paper",
            "bot_running": True,
            "updated_at": updated,
            "updated_label": "paper bot running",
            "age_sec": age_sec,
        }
    return {
        "state": "live",
        "label": "live",
        "detail": "bot running (live)",
        "pill": "LIVE BOT",
        "pill_class": "live",
        "bot_running": True,
        "updated_at": updated,
            "updated_label": "live bot running",
        "age_sec": age_sec,
    }


def gemini_desk_status(
    *,
    enabled: bool,
    mode: str = "advisory",
    last_ask_ok: Optional[bool] = None,
) -> Dict[str, Any]:
    mode = (mode or "advisory").strip() or "advisory"
    if not enabled:
        return {"label": "off", "detail": "off", "on": False, "mode": mode}
    if last_ask_ok is True:
        return {"label": f"{mode} / on", "detail": f"{mode} / on", "on": True, "mode": mode}
    if last_ask_ok is False:
        return {"label": f"{mode} / last ask failed", "detail": f"{mode} / last ask failed", "on": False, "mode": mode}
    return {"label": f"{mode} / ready", "detail": f"{mode} / ready", "on": False, "mode": mode}


def collapse_gemini_questions(
    actions: Sequence[Dict[str, Any]],
    n: int = GEMINI_HISTORY_N,
) -> List[Dict[str, Any]]:
    """Keep last N Gemini questions; merge consecutive identical asks."""
    collapsed: List[Dict[str, Any]] = []
    for row in actions:
        if str(row.get("kind") or "") != "gemini_chat":
            continue
        q = str(row.get("reason") or "").strip()
        if collapsed and collapsed[-1]["question"].casefold() == q.casefold():
            collapsed[-1]["repeat"] = int(collapsed[-1].get("repeat") or 1) + 1
            collapsed[-1]["answer"] = row.get("answer")
            collapsed[-1]["ts"] = row.get("ts")
        else:
            collapsed.append({
                "question": q,
                "answer": row.get("answer"),
                "ts": row.get("ts"),
                "repeat": 1,
            })
    return collapsed[-max(1, int(n)):]


def last_ask_ok_from_history(
    questions: Sequence[Dict[str, Any]],
    in_memory: Optional[bool],
) -> Optional[bool]:
    if in_memory is not None:
        return in_memory
    if not questions:
        return None
    return ask_looks_successful(str(questions[-1].get("answer") or ""))


def _money(v: Any) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if n >= 0 else "−"
    return f"{sign}${abs(n):,.0f}"


def _open_label(status: Dict[str, Any], running: bool) -> str:
    opens = status.get("open_positions") or []
    if not isinstance(opens, list):
        opens = []
    names = [f"{p.get('symbol','')} {p.get('direction','')}".strip() for p in opens if isinstance(p, dict)]
    names = [x for x in names if x]
    if names:
        return ", ".join(names)
    if running:
        return "flat (bot running, no trade on)"
    return "flat (desk only / no live bot)"


def build_plain_english(
    *,
    run: Dict[str, Any],
    status: Dict[str, Any],
    snapshot: Dict[str, Any],
    live_empty: bool,
    suggestions: Dict[str, Any],
    gemini: Dict[str, Any],
    facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    facts = facts or load_locked_backtest_facts()
    daily = status.get("daily_pnl")
    if run.get("bot_running"):
        pnl = "—" if daily is None else f"${float(daily):.2f}"
        live = (
            f"{run['label'].capitalize()} — {run['detail']}. "
            f"Open: {_open_label(status, True)}. Daily P&L: {pnl}."
        )
        if run.get("state") == "overnight_research":
            live = (
                "PAPER OVERNIGHT RESEARCH — Globex paper fills only, no Rithmic orders. "
                f"Open: {_open_label(status, True)}. Tonight's paper fills: "
                f"{todays_live_trade_count(status, None)}. Daily P&L: {pnl}."
            )
        if live_empty and run.get("state") != "overnight_research":
            live += " No closed trades recorded yet."
    else:
        live = (
            "Not trading yet — desk only / no live bot. "
            "Open is flat. Daily P&L appears when a paper or live bot is running."
        )
    backtest = (
        "Profitable on historical 1-minute MNQ. Morning 9:35 and late-morning 11:00 "
        "make the money. Afternoon 13:30 loses. Winners usually run far and get flattened "
        "at 3:50pm ET — this is a trend hold, not a tiny scalp."
    )
    min_n = suggest_min_trades()
    if suggestions.get("ready"):
        sug = "Ready in Details. Advisory only — the bot will not change itself."
    else:
        sug = f"None until {min_n} real closes. The bot will not change itself."
    gem_line = gemini.get("label") or "off"
    if gemini.get("on"):
        gem_line += " — last Ask worked. It does not place trades."
    elif gemini.get("label") == "off":
        gem_line = "off (desk chat needs LLM_ENABLED=true)."
    overall = ((snapshot.get("all") or {}).get("overall") or {})
    n_all = snapshot.get("n_all") or overall.get("n")
    pf3 = overall.get("pf")
    try:
        is_pf = float(facts.get("is_pf"))
        oos_pf = float(facts.get("oos_pf"))
        is_pnl = float(facts.get("is_pnl_usd"))
        oos_pnl = float(facts.get("oos_pnl_usd"))
    except (TypeError, ValueError):
        is_pf = oos_pf = is_pnl = oos_pnl = None
    three = ""
    if n_all and pf3 is not None:
        three = f" ({int(n_all)} trades, profit factor {float(pf3):.2f})"
    six = ""
    if is_pf is not None and oos_pf is not None:
        six = (
            f" (in-sample profit factor {is_pf:.2f} / {_money(is_pnl)}; "
            f"later period {oos_pf:.2f} / {_money(oos_pnl)})"
        )
    recipe = (
        f"Gemini cites the official 6-window backtest{six}. "
        f"The Details table is the locked 3-window set{three}. "
        "Same idea, different window count — not a contradiction. "
        "Afternoon 13:30 stays in the locked recipe (not auto-dropped)."
    )
    return {
        "live": live,
        "backtest": backtest,
        "suggestions": sug,
        "gemini": gem_line,
        "recipe_note": recipe,
    }


def format_et_clock(minutes: int) -> str:
    h = int(minutes) // 60
    m = int(minutes) % 60
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


def entry_windows_display() -> List[str]:
    out: List[str] = []
    for start, end in ENTRY_WINDOWS:
        out.append(f"{format_et_clock(start)} – {format_et_clock(end)} ET")
    return out


def _pf2(v: Any) -> Optional[float]:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def desk_market_packet(
    status: Optional[Dict[str, Any]] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    now = time.time()
    cached = _PACKET_CACHE.get("packet")
    cache_key = (
        (status or {}).get("updated_at"),
        (status or {}).get("last_price"),
        (status or {}).get("open"),
    )
    if (
        cached
        and _PACKET_CACHE.get("key") == cache_key
        and now - float(_PACKET_CACHE.get("t") or 0) < 8
    ):
        return cached
    try:
        pkt = build_advisory_market_packet(status=status, recent_trades=trades)
    except Exception:
        pkt = compute_recipe_levels({"price_stale": True, "session": "closed"})
        pkt = {"price_stale": True, "levels": pkt}
    if is_overnight_rec(status or {}):
        pkt = apply_overnight_live_quote(pkt, status)
    _PACKET_CACHE["t"] = now
    _PACKET_CACHE["key"] = cache_key
    _PACKET_CACHE["packet"] = pkt
    return pkt


def build_profit_card(snapshot: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
    overall = ((snapshot or {}).get("all") or {}).get("overall") or {}
    oos = ((snapshot or {}).get("oos") or {}).get("overall") or {}
    n = overall.get("n") or snapshot.get("n_all")
    pf = _pf2(overall.get("pf"))
    oos_pf = _pf2(oos.get("pf"))
    pnl = overall.get("total_pnl_usd")
    yes = pf is not None and pf > 1
    footnote = None
    try:
        footnote = (
            f"A 6-window variant also passed: IS PF {float(facts['is_pf']):.2f} / "
            f"OOS {float(facts['oos_pf']):.2f}"
        )
    except (TypeError, ValueError, KeyError):
        footnote = None
    return {
        "yes": yes,
        "headline": "Yes (Historical Backtest)" if yes else "Not proven on this snapshot",
        "trades": n,
        "pf": pf,
        "total_profit": pnl,
        "oos_pf": oos_pf,
        "strategy_type": "Trend Following",
        "translation": "Winners were larger than losers.",
        "footnote": footnote,
    }


def build_works_card(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    by_win = ((snapshot or {}).get("all") or {}).get("by_window") or {}
    align = ((snapshot or {}).get("all") or {}).get("by_alignment") or {}
    labels = (("09:35", "9:35 AM"), ("11:00", "11:00 AM"), ("13:30", "1:30 PM"))
    windows = []
    for key, name in labels:
        pf = _pf2((by_win.get(key) or {}).get("pf"))
        windows.append({"name": name, "key": key, "pf": pf if pf is not None else 0, "flag": pf is not None and pf < 1})
    return {
        "windows": windows,
        "aligned_pf": _pf2((align.get("aligned") or {}).get("pf")),
        "disagree_pf": _pf2((align.get("not aligned") or {}).get("pf")),
        "translation": "Trade only when all three timeframes point the same way.",
    }


def build_kind_card(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    mfe = ((snapshot or {}).get("all") or {}).get("mfe_mae") or {}
    oos_mfe = ((snapshot or {}).get("oos") or {}).get("mfe_mae") or {}
    win_lo = mfe.get("median_mfe_winners")
    win_hi = oos_mfe.get("median_mfe_winners") or win_lo
    lose = mfe.get("median_mae_losers")
    pct80 = mfe.get("pct_winners_mfe_ge_80")
    body_bits = ["Not a scalp."]
    if win_lo is not None and win_hi is not None:
        body_bits.append(f"Typical winner +{float(win_lo):.0f} to +{float(win_hi):.0f}.")
    if lose is not None:
        body_bits.append(f"Typical loser ~{float(lose):.0f} adverse.")
    if pct80 is not None:
        body_bits.append("Most winners run 80+.")
    return {
        "headline": "Not a scalp.",
        "body": " ".join(body_bits[1:]) if len(body_bits) > 1 else "This strategy catches trends and holds them.",
        "translation": "This strategy catches trends and holds them.",
    }


def _parse_ts_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def is_overnight_research_close(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if str(row.get("session") or "").lower() != "overnight_research":
        return False
    return str(row.get("event") or "close") == "close"


def is_bot_ema15_close(row: Dict[str, Any]) -> bool:
    """True for a real ema15 close — not a backtest dump that only has source=."""
    if not isinstance(row, dict):
        return False
    if str(row.get("session") or "").lower() == "overnight_research":
        return False
    if str(row.get("event") or "close") != "close":
        return False
    return bool(row.get("entry_snapshot") or row.get("window_name") or row.get("atr_stop_pts"))


def todays_live_trade_count(
    status: Optional[Dict[str, Any]],
    closes: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    now: Optional[datetime] = None,
) -> int:
    """Today's paper/live ema15 fills. Prefer the bot heartbeat; never lifetime journal n."""
    status = status or {}
    if status.get("live_trades_today") is not None:
        try:
            return max(0, int(status["live_trades_today"]))
        except (TypeError, ValueError):
            pass
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    today = now.astimezone(ET).date()
    n = 0
    for row in closes or []:
        if not is_bot_ema15_close(row):
            continue
        ts = _parse_ts_utc(row.get("exit_time") or row.get("ts") or row.get("exit_ts_et"))
        if ts is None:
            continue
        if ts.astimezone(ET).date() == today:
            n += 1
    return n


def _quote_age_label(levels: Dict[str, Any], refresh: Optional[Dict[str, Any]] = None) -> str:
    refresh = refresh or {}
    if refresh.get("ok") is False or levels.get("refresh_ok") is False:
        err = refresh.get("error") or levels.get("refresh_error") or "Databento download failed"
        return f"download failed — last file print is not current ({err})"
    age = levels.get("quote_age_min")
    if age is None and levels.get("quote_age_sec") is not None:
        age = round(float(levels["quote_age_sec"]) / 60.0, 1)
    if age is None and refresh.get("quote_age_sec") is not None:
        age = round(float(refresh["quote_age_sec"]) / 60.0, 1)
    if age is None:
        return "unknown"
    mins = float(age)
    if mins < 1:
        return "<1 min"
    whole = int(round(mins))
    if mins > 15:
        return f"{whole} min (stale)"
    return f"{whole} min"


def build_status_card(
    run: Dict[str, Any],
    status: Dict[str, Any],
    live: Dict[str, Any],
    gemini: Dict[str, Any],
    *,
    levels: Optional[Dict[str, Any]] = None,
    refresh: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    overnight = run.get("state") == "overnight_research"
    if overnight:
        bot_s = "PAPER OVERNIGHT RESEARCH"
    elif run.get("state") == "paper":
        bot_s = "paper bot running"
    elif run.get("state") == "live":
        bot_s = "live bot running"
    else:
        bot_s = "not running"
    opens = status.get("open_positions") or []
    names = []
    if isinstance(opens, list):
        names = [f"{p.get('symbol','')} {p.get('direction','')}".strip() for p in opens if isinstance(p, dict)]
        names = [x for x in names if x]
    open_s = ", ".join(names) if names else "Flat"
    daily = status.get("daily_pnl")
    if not run.get("bot_running") or daily is None:
        daily_s = "N/A"
    else:
        daily_s = f"${float(daily):.2f}"
    live_n = todays_live_trade_count(status, None)
    trade_label = "Live trades tonight" if overnight else "Live trades"
    quote_levels = dict(levels or {})
    if overnight and status.get("quote_age_seconds") is not None:
        try:
            quote_levels["quote_age_sec"] = float(status["quote_age_seconds"])
            quote_levels["quote_age_min"] = round(float(status["quote_age_seconds"]) / 60.0, 1)
            quote_levels["refresh_ok"] = True
        except (TypeError, ValueError):
            pass
    rows = [
        ["Bot", bot_s],
        ["Open", open_s],
        ["Daily P&L", daily_s],
        [trade_label, str(live_n)],
        ["Quote age", _quote_age_label(quote_levels, refresh if not overnight else {"ok": True})],
    ]
    if overnight:
        px = status.get("last_price")
        if px not in (None, ""):
            try:
                rows.append(["Ticker last", f"{float(px):.2f}"])
            except (TypeError, ValueError):
                rows.append(["Ticker last", str(px)])
        src = str(status.get("quote_source") or "")
        if src:
            rows.append(["Quote source", src])
        opens = status.get("open_positions") or []
        pos = opens[0] if isinstance(opens, list) and opens and isinstance(opens[0], dict) else {}
        stop_line = format_stop_line({
            **pos,
            "session": "overnight_research",
            "entry": pos.get("entry") or pos.get("entry_price"),
            "stop": pos.get("sl") or pos.get("stop"),
            "atr_stop_pts": pos.get("atr_stop_pts"),
        }) if pos else ""
        rows.append(["Stop", stop_line if stop_line else "—"])
        rows.append(["Target", OVERNIGHT_TP_TEXT])
        fill = status.get("last_paper_fill") or {}
        if fill:
            side = str(fill.get("side") or "")
            px = fill.get("exit_price") or fill.get("entry_price")
            cue = fill.get("cue") or ""
            rows.append(["Last paper fill", f"{side} {cue} @ {px}".strip()])
    elif opens:
        pos = opens[0] if isinstance(opens, list) and opens and isinstance(opens[0], dict) else {}
        if pos:
            stop_line = format_stop_line({
                **pos,
                "entry": pos.get("entry") or pos.get("entry_price"),
                "stop": pos.get("sl") or pos.get("stop"),
                "atr_stop_pts": pos.get("atr_stop_pts"),
            })
            live = format_live_entry_brackets({
                **pos,
                "stop": pos.get("sl") or pos.get("stop"),
                "tp": pos.get("tp") or pos.get("tp_cap"),
                "tp_cap": pos.get("tp_cap") or pos.get("tp"),
            })
            rows.append(["Stop", stop_line if stop_line else "—"])
            rows.append(["Target", live or (f"TP cap {pos.get('tp')}" if pos.get("tp") else "Flatten 15:50 ET")])
    note = str(status.get("last_note") or status.get("rithmic_probe") or "").strip()
    if note:
        rows.append(["Note", note])
    return {
        "headline": bot_s,
        "rows": rows,
    }


def build_signal_card(
    packet: Dict[str, Any],
    questions: Sequence[Dict[str, Any]],
    status: Optional[Dict[str, Any]] = None,
    run: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = status or {}
    overnight = is_overnight_rec(status) or (run or {}).get("state") == "overnight_research"
    levels = (packet or {}).get("levels") or compute_recipe_levels(packet or {})
    if overnight:
        levels = overlay_overnight_levels(
            levels,
            status,
            price=status.get("last_price") or levels.get("last_price"),
            source=str(status.get("quote_source") or levels.get("source") or "rithmic"),
        )
    gem_call = None
    if not overnight:
        for q in reversed(list(questions or [])):
            gem_call = extract_direction_callout(str(q.get("answer") or ""))
            if gem_call:
                break
    waiting = bool(levels.get("waiting", True))
    refresh_ok = levels.get("refresh_ok")
    last_price = levels.get("last_price")
    if refresh_ok is False and not overnight:
        last_price = None
        waiting = True
    rth = (packet or {}).get("rth_levels") or {}
    return {
        "waiting": waiting,
        "overnight": overnight,
        "callout": "WAIT" if refresh_ok is False and not overnight else (levels.get("callout") or "WAIT"),
        "headline": levels.get("first_line") or "Waiting for market data.",
        "last_price": last_price,
        "stop": None if refresh_ok is False and not overnight else levels.get("stop"),
        "stop_pts": None if refresh_ok is False and not overnight else levels.get("stop_pts"),
        "take_profit": OVERNIGHT_TP_TEXT if overnight else levels.get("take_profit"),
        "flatten_et": OVERNIGHT_FLATTEN_ET if overnight else levels.get("flatten_et"),
        "next_window": "flatten 09:25" if overnight else levels.get("next_window"),
        "last_bar_ts_et": levels.get("last_bar_ts_et") or status.get("last_quote_ts_et"),
        "quote_age_min": levels.get("quote_age_min"),
        "quote_age_sec": levels.get("quote_age_sec") or status.get("quote_age_seconds"),
        "refresh_ok": True if overnight or refresh_ok is None else bool(refresh_ok),
        "refresh_error": None if overnight else levels.get("refresh_error"),
        "source": levels.get("source") or status.get("quote_source"),
        "entry": None if refresh_ok is False and not overnight else levels.get("entry"),
        "gemini_callout": gem_call,
        "rules": (
            [
                "Overnight paper: break settled ONH/ONL only. ATR×1.5 stop, 1R TP, flatten 09:25 ET.",
                "Lucid TEST/sim orders (not funded live). Locked RTH ema15 is off overnight.",
            ]
            if overnight
            else [
                "Day recipe: enter only when daily+15m+60m agree.",
                "BUY if 15m and 60m EMA8>EMA21 and prior RTH close > daily EMA20",
                "SELL if 15m and 60m EMA8<EMA21 and prior RTH close < daily EMA20",
            ]
        ),
        "windows": entry_windows_display(),
        "rth_recipe_note": "Day recipe: enter only when daily+15m+60m agree.",
        "rth_callout": rth.get("callout") or rth.get("first_line"),
    }


def _desk_journal_path() -> str:
    status = read_status()
    if status.get("journal_path"):
        return str(status["journal_path"])
    if status.get("paper_mode"):
        return JOURNAL_PAPER
    if os.path.isfile(JOURNAL_LIVE):
        return JOURNAL_LIVE
    return JOURNAL_PAPER


def _desk_trades(n: int = 80):
    status = read_status()
    rows = read_journal(_desk_journal_path(), n=4000)
    if (
        bool(status.get("overnight_research"))
        or str(status.get("strategy") or "") == "overnight_research"
    ):
        closes = [r for r in rows if is_overnight_research_close(r)]
        out = []
        for r in closes[-n:]:
            out.append({
                "entry_ts_et": r.get("entry_ts_et") or r.get("entry_time"),
                "exit_ts_et": r.get("exit_ts_et") or r.get("exit_time"),
                "direction": r.get("direction") or r.get("side"),
                "entry_price": r.get("entry_price"),
                "exit_price": r.get("exit_price"),
                "stop_distance": r.get("atr_stop_pts"),
                "mae_pts": r.get("mae_pts"),
                "mfe_pts": r.get("mfe_pts"),
                "desk_exit_reason": r.get("exit_reason"),
                "trend_15m_txt": r.get("cue") or "—",
                "trend_60m_txt": r.get("zone_name") or "—",
                "daily_trend_txt": "overnight",
                "atr_at_entry": r.get("atr_stop_pts"),
                "pnl_usd": r.get("pnl_usd"),
                "session": "overnight_research",
            })
        return out
    closes = [r for r in rows if is_bot_ema15_close(r)]
    return [closed_trade_view(r) for r in closes[-n:]]


def _llm_enabled() -> bool:
    try:
        return bool(get_llm_advisor().enabled)
    except Exception:
        return os.getenv("LLM_ENABLED", "false").lower() == "true"


@app.get("/")
def home():
    if _HTML_PATH.is_file():
        body = _HTML_PATH.read_text(encoding="utf-8")
    else:
        body = render_template_string(HTML)
    resp = make_response(body)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/desk")
def desk():
    global _last_ask_ok
    log = ActionLog()
    status = read_status()
    actions_all = log.recent(250)
    questions = collapse_gemini_questions(actions_all, GEMINI_HISTORY_N)
    run = bot_run_state(status)
    last_ok = last_ask_ok_from_history(questions, _last_ask_ok)
    gemini = gemini_desk_status(
        enabled=_llm_enabled(),
        mode=os.getenv("LLM_MODE", "advisory"),
        last_ask_ok=last_ok,
    )
    trades = _desk_trades(80)
    raw = read_journal(_desk_journal_path(), n=4000)
    closes = [r for r in raw if is_bot_ema15_close(r)]
    live = live_expectancy(closes)
    snapshot = read_snapshot()
    overnight_active = run.get("state") == "overnight_research"
    zones_payload = read_zones() if overnight_active else {}
    overnight_sugg = read_overnight_suggestions() if overnight_active else {}
    suggestions = overnight_sugg if overnight_active else read_suggestions()
    facts = load_locked_backtest_facts()
    packet = desk_market_packet(status, trades[-5:] if trades else [])
    levels = packet.get("levels") or {}
    refresh = load_desk_quote_meta()
    plain = build_plain_english(
        run=run,
        status=status,
        snapshot=snapshot,
        live_empty=bool(live.get("empty")),
        suggestions=suggestions,
        gemini=gemini,
        facts=facts,
    )
    cards = {
        "status": build_status_card(run, status, live, gemini, levels=levels, refresh=refresh),
        "profit": build_profit_card(snapshot, facts),
        "works": build_works_card(snapshot),
        "kind": build_kind_card(snapshot),
        "signal": build_signal_card(packet, questions, status=status, run=run),
    }
    status_rows = cards["status"]["rows"]
    actions = [a for a in actions_all if str(a.get("kind") or "") != "gemini_chat"]
    feed = build_today_activity(actions_all, status)
    return jsonify({
        "status": status,
        "run": run,
        "gemini": gemini,
        "plain": plain,
        "cards": cards,
        "market": {"levels": levels, "price_1m": packet.get("price_1m"), "price_stale": packet.get("price_stale")},
        "status_rows": status_rows,
        "actions": actions,
        "gemini_questions": questions,
        "trades": trades,
        "expectancy": live,
        "backtest_edge": snapshot,
        "suggestions": suggestions,
        "official_6w": {
            "is_pf": facts.get("is_pf"),
            "is_pnl_usd": facts.get("is_pnl_usd"),
            "oos_pf": facts.get("oos_pf"),
            "oos_pnl_usd": facts.get("oos_pnl_usd"),
            "label": facts.get("label"),
        },
        "refresh": refresh,
        "overnight": {
            "active": overnight_active,
            "zones": status.get("zones") or zones_payload.get("zones") or [],
            "last_paper_fill": status.get("last_paper_fill"),
            "source": status.get("quote_source"),
            "suggestions": overnight_sugg if overnight_active else {},
        },
        "activity": feed.get("events") or [],
        "activity_card": feed.get("card"),
        "developer": feed.get("developer") or [],
    })


@app.post("/api/ask")
def ask():
    global _last_ask_ok
    body = request.get_json(silent=True) or {}
    q = str(body.get("question") or "").strip()
    if not q:
        return jsonify({"error": "empty question"}), 400
    log = ActionLog()
    status = read_status()
    trades = _desk_trades(15)
    packet = desk_market_packet(status, trades)
    ctx: Dict[str, Any] = {
        "status": status,
        "recent_actions": log.recent(20),
        "recent_trades": trades,
        "suggestions": read_suggestions(),
        "market": packet,
    }
    answer = get_llm_advisor().ask(q, ctx)
    _last_ask_ok = ask_looks_successful(answer)
    try:
        log.record("gemini_chat", reason=q[:240], answer=(answer or "")[:500])
    except Exception:
        pass
    levels = (packet.get("levels") or {})
    return jsonify({"answer": answer, "levels": levels})


def _refresh_mnq_quote() -> Dict[str, Any]:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "download_databento_latest.py"
    spec = importlib.util.spec_from_file_location("download_databento_latest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Databento download script")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.refresh_latest_1m(lookback_days=3)


def start_mtf_dashboard(port: int = 5055) -> None:
    import socket

    probe = socket.socket()
    probe.settimeout(0.4)
    try:
        probe.connect(("127.0.0.1", int(port)))
        in_use = True
    except OSError:
        in_use = False
    finally:
        probe.close()
    if in_use:
        print(f"Desk already running at http://127.0.0.1:{port} — not starting a second Flask")
        return
    try:
        _refresh_mnq_quote()
    except Exception as exc:
        meta_path = Path(__file__).resolve().parents[2] / "data" / "mnq_desk_quote.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        msg = str(exc).split("\n")[0][:240]
        meta_path.write_text(
            json.dumps({
                "ok": False,
                "error": f"Databento download failed: {msg}",
                "last_price": None,
                "refreshed_at_et": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )
    invalidate_1m_cache()
    _PACKET_CACHE["t"] = 0.0
    _PACKET_CACHE["packet"] = None
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    try:
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    except OSError as exc:
        err = str(exc).lower()
        if "already in use" in err or getattr(exc, "winerror", None) == 10048:
            print(f"Desk already running at http://127.0.0.1:{port}")
            return
        raise
