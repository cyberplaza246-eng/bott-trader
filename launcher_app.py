#!/usr/bin/env python
"""
BottTrader — Desktop Launcher for Rithmic Futures Trading Bot

A standalone GUI application that wraps LiveRithmicTrader with:
  - Instrument selection (MES, MNQ, NQ)
  - Paper / Live toggle
  - Settings panel (confidence, SL/TP, risk, max positions)
  - Scrolling log output
  - System tray support (minimize to tray)
  - Trade notifications (Windows toast)
  - Auto-update check from GitHub Releases
  - Dashboard launcher (opens Flask dashboard in browser)

Usage:
    python launcher_app.py          # Launch GUI
    pyinstaller --onefile ...       # See build_exe.bat
"""

__version__ = "1.0.0"

import os
import sys
import json
import threading
import queue
import io
import time
import webbrowser
from datetime import datetime
from pathlib import Path

# Resolve base directory (works both as script and as PyInstaller exe)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Add project root to path so imports work
sys.path.insert(0, str(BASE_DIR))

# Load .env from next to the exe / script
_env_path = BASE_DIR / '.env'
try:
    from dotenv import load_dotenv
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    # python-dotenv not installed — load .env manually
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

import customtkinter as ctk

# ─── Settings Persistence ──────────────────────────────────────
SETTINGS_FILE = BASE_DIR / 'settings.json'

DEFAULT_SETTINGS = {
    'ensemble_confidence_threshold': 0.55,
    'stop_loss_multiplier': 2.5,
    'take_profit_ratio': 2.5,
    'risk_per_trade_percent': 1.0,
    'max_concurrent_trades': 3,
    'daily_loss_limit_percent': 3.0,
    'max_contracts_mes': 3,
    'max_contracts_mnq': 3,
    'max_contracts_nq': 1,
}


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception:
            pass
    return settings


def save_settings(settings: dict):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)


def apply_settings_to_env(settings: dict):
    """Push GUI settings into environment variables so the bot picks them up."""
    os.environ['ENSEMBLE_CONFIDENCE_THRESHOLD'] = str(settings['ensemble_confidence_threshold'])
    os.environ['STOP_LOSS_MULTIPLIER'] = str(settings['stop_loss_multiplier'])
    os.environ['TAKE_PROFIT_RATIO'] = str(settings['take_profit_ratio'])
    os.environ['RISK_PER_TRADE_PERCENT'] = str(settings['risk_per_trade_percent'])
    os.environ['MAX_CONCURRENT_TRADES'] = str(int(settings['max_concurrent_trades']))
    os.environ['DAILY_LOSS_LIMIT_PERCENT'] = str(settings['daily_loss_limit_percent'])


# ─── Update Checker ─────────────────────────────────────────────
GITHUB_REPO = "cyberplaza246-eng/bott-trader"


def check_for_update() -> dict | None:
    """Check GitHub for a newer release. Returns {tag, url} or None."""
    try:
        import requests
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
            timeout=5, headers={'Accept': 'application/vnd.github+json'}
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        tag = data.get('tag_name', '').lstrip('v')
        if tag and tag != __version__:
            # Find .exe asset
            dl_url = data.get('html_url', '')
            for asset in data.get('assets', []):
                if asset['name'].lower().endswith('.exe'):
                    dl_url = asset['browser_download_url']
                    break
            return {'tag': tag, 'url': dl_url}
    except Exception:
        pass
    return None


# ─── Credentials Dialog ─────────────────────────────────────────
class CredentialsDialog(ctk.CTkToplevel):
    """Modal dialog for first-time Rithmic credentials setup."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Rithmic Credentials")
        self.geometry("420x320")
        self.resizable(False, False)
        self.grab_set()
        self.result = False

        ctk.CTkLabel(self, text="Enter your Rithmic credentials",
                      font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(18, 10))

        frame = ctk.CTkFrame(self)
        frame.pack(padx=20, fill='x')

        ctk.CTkLabel(frame, text="User ID:").grid(row=0, column=0, sticky='w', padx=8, pady=6)
        self.user_entry = ctk.CTkEntry(frame, width=240)
        self.user_entry.grid(row=0, column=1, padx=8, pady=6)
        self.user_entry.insert(0, os.getenv('RITHMIC_USER_ID', ''))

        ctk.CTkLabel(frame, text="Password:").grid(row=1, column=0, sticky='w', padx=8, pady=6)
        self.pass_entry = ctk.CTkEntry(frame, width=240, show='*')
        self.pass_entry.grid(row=1, column=1, padx=8, pady=6)
        self.pass_entry.insert(0, os.getenv('RITHMIC_PASSWORD', ''))

        ctk.CTkLabel(frame, text="System:").grid(row=2, column=0, sticky='w', padx=8, pady=6)
        self.system_var = ctk.StringVar(value=os.getenv('RITHMIC_SYSTEM', 'Rithmic Paper Trading'))
        self.system_menu = ctk.CTkOptionMenu(frame, variable=self.system_var, width=240,
            values=['Rithmic Paper Trading', 'LucidTrading', 'tradesea', 'Rithmic 01'])
        self.system_menu.grid(row=2, column=1, padx=8, pady=6)

        ctk.CTkLabel(frame, text="Gateway:").grid(row=3, column=0, sticky='w', padx=8, pady=6)
        self.gw_entry = ctk.CTkEntry(frame, width=240, placeholder_text="(optional)")
        self.gw_entry.grid(row=3, column=1, padx=8, pady=6)
        self.gw_entry.insert(0, os.getenv('RITHMIC_GATEWAY', ''))

        btn_frame = ctk.CTkFrame(self, fg_color='transparent')
        btn_frame.pack(pady=16)
        ctk.CTkButton(btn_frame, text="Save", width=120, command=self._save).pack(side='left', padx=8)
        ctk.CTkButton(btn_frame, text="Cancel", width=120, fg_color='gray',
                       command=self.destroy).pack(side='left', padx=8)

    def _save(self):
        uid = self.user_entry.get().strip()
        pw = self.pass_entry.get().strip()
        system = self.system_var.get()
        gw = self.gw_entry.get().strip()

        if not uid or not pw:
            ctk.CTkLabel(self, text="User ID and Password are required",
                          text_color='#f85149').pack()
            return

        # Write to .env
        env_lines = []
        if _env_path.exists():
            env_lines = _env_path.read_text().splitlines()

        # Update or append each key
        key_map = {
            'RITHMIC_USER_ID': uid,
            'RITHMIC_PASSWORD': pw,
            'RITHMIC_SYSTEM': system,
            'RITHMIC_GATEWAY': gw,
        }
        for key, val in key_map.items():
            found = False
            for i, line in enumerate(env_lines):
                if line.startswith(f'{key}=') or line.startswith(f'{key} ='):
                    env_lines[i] = f'{key}={val}'
                    found = True
                    break
            if not found:
                env_lines.append(f'{key}={val}')
            os.environ[key] = val

        _env_path.write_text('\n'.join(env_lines) + '\n')
        self.result = True
        self.destroy()


# ─── Log Redirector ──────────────────────────────────────────────
class QueueWriter(io.TextIOBase):
    """Captures stdout/stderr writes and puts them in a queue for the GUI."""

    def __init__(self, log_queue: queue.Queue, original):
        self._queue = log_queue
        self._original = original

    def write(self, text):
        if text and text.strip():
            self._queue.put(text)
        # Also write to original so console still works during development
        if self._original:
            try:
                self._original.write(text)
                self._original.flush()
            except Exception:
                pass
        return len(text) if text else 0

    def flush(self):
        if self._original:
            try:
                self._original.flush()
            except Exception:
                pass


# ─── Trade Notification Helper ───────────────────────────────────
def _notify(title: str, message: str):
    """Send a Windows toast notification (best-effort)."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name='BottTrader',
            timeout=5,
        )
    except Exception:
        pass


# Pattern matching for trade log lines
_TRADE_PATTERNS = [
    ('BUY', '🟢'),
    ('SELL', '🔴'),
    ('TAKE_PROFIT', '💰'),
    ('STOP_LOSS', '⛔'),
    ('trail', '🔒'),
    ('Daily loss limit', '⚠️'),
]


def _check_trade_notification(text: str):
    """Fire a notification if the log line indicates a trade event."""
    for keyword, icon in _TRADE_PATTERNS:
        if keyword in text:
            # First 120 chars as message
            _notify(f"{icon} BottTrader", text[:120])
            return


# ─── Main Application ───────────────────────────────────────────
class BottTraderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(f"BottTrader v{__version__}")
        self.geometry("780x680")
        self.minsize(650, 550)

        self._trader = None
        self._trader_thread = None
        self._stop_event = threading.Event()
        self._log_queue: queue.Queue = queue.Queue()
        self._dashboard_thread = None
        self._tray_icon = None
        self._running = False
        self._settings = load_settings()

        self._build_ui()
        self._poll_log()
        self._check_update_async()

        # Override window close to minimize to tray
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ──────────────────────────────────────────
    def _build_ui(self):
        # Update banner (hidden by default)
        self._update_frame = ctk.CTkFrame(self, fg_color='#1a3a1a', corner_radius=6)
        self._update_label = ctk.CTkLabel(self._update_frame, text="", text_color='#3fb950')
        self._update_label.pack(side='left', padx=12)
        self._update_btn = ctk.CTkButton(self._update_frame, text="Download", width=90,
                                          fg_color='#238636', hover_color='#2ea043',
                                          command=self._open_update_url)
        self._update_btn.pack(side='right', padx=12, pady=6)
        self._update_url = ''
        # Banner hidden by default — shown if update found

        # Tabview
        self._tabs = ctk.CTkTabview(self, anchor='nw')
        self._tabs.pack(fill='both', expand=True, padx=12, pady=(8, 4))
        tab_trade = self._tabs.add("Trading")
        tab_settings = self._tabs.add("Settings")

        # ── Trading Tab ──────────────────────────────────────────
        top_frame = ctk.CTkFrame(tab_trade, fg_color='transparent')
        top_frame.pack(fill='x', padx=8, pady=(8, 4))

        # Instruments
        instr_frame = ctk.CTkFrame(top_frame)
        instr_frame.pack(side='left', padx=(0, 16))
        ctk.CTkLabel(instr_frame, text="Instruments", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=8, pady=(6, 2))
        self._sym_vars = {}
        for sym in ['MES', 'MNQ', 'NQ']:
            var = ctk.BooleanVar(value=(sym in ('MES', 'MNQ')))
            cb = ctk.CTkCheckBox(instr_frame, text=sym, variable=var)
            cb.pack(anchor='w', padx=12, pady=2)
            self._sym_vars[sym] = var

        # Mode
        mode_frame = ctk.CTkFrame(top_frame)
        mode_frame.pack(side='left', padx=(0, 16))
        ctk.CTkLabel(mode_frame, text="Mode", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=8, pady=(6, 2))
        self._mode_var = ctk.StringVar(value='paper')
        ctk.CTkRadioButton(mode_frame, text="Paper", variable=self._mode_var, value='paper').pack(anchor='w', padx=12, pady=2)
        ctk.CTkRadioButton(mode_frame, text="Live ⚠️", variable=self._mode_var, value='live').pack(anchor='w', padx=12, pady=2)

        # Status
        status_frame = ctk.CTkFrame(top_frame)
        status_frame.pack(side='left', fill='y', padx=(0, 16))
        ctk.CTkLabel(status_frame, text="Status", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=8, pady=(6, 2))
        self._status_label = ctk.CTkLabel(status_frame, text="⏹ Stopped", text_color='#8b949e',
                                           font=ctk.CTkFont(size=13))
        self._status_label.pack(anchor='w', padx=12, pady=2)

        # Buttons
        btn_frame = ctk.CTkFrame(top_frame, fg_color='transparent')
        btn_frame.pack(side='right', padx=8)
        self._start_btn = ctk.CTkButton(btn_frame, text="▶  Start", width=110, fg_color='#238636',
                                         hover_color='#2ea043', command=self._on_start)
        self._start_btn.pack(pady=4)
        self._stop_btn = ctk.CTkButton(btn_frame, text="⏹  Stop", width=110, fg_color='#da3633',
                                        hover_color='#f85149', state='disabled', command=self._on_stop)
        self._stop_btn.pack(pady=4)
        self._dash_btn = ctk.CTkButton(btn_frame, text="📊  Dashboard", width=110,
                                        command=self._on_dashboard)
        self._dash_btn.pack(pady=4)

        # Log area
        log_frame = ctk.CTkFrame(tab_trade)
        log_frame.pack(fill='both', expand=True, padx=8, pady=8)
        ctk.CTkLabel(log_frame, text="Log Output", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=8, pady=(6, 2))
        self._log_text = ctk.CTkTextbox(log_frame, state='disabled',
                                         font=ctk.CTkFont(family='Consolas', size=12),
                                         fg_color='#0d1117', text_color='#c9d1d9',
                                         wrap='word')
        self._log_text.pack(fill='both', expand=True, padx=4, pady=4)

        # ── Settings Tab ─────────────────────────────────────────
        self._setting_widgets = {}
        settings_grid = ctk.CTkFrame(tab_settings)
        settings_grid.pack(fill='both', expand=True, padx=16, pady=12)

        setting_defs = [
            ('ensemble_confidence_threshold', 'Confidence Threshold', 0.40, 0.80, 0.01),
            ('stop_loss_multiplier', 'SL Multiplier (×ATR)', 1.0, 5.0, 0.1),
            ('take_profit_ratio', 'TP Ratio (×SL)', 1.0, 5.0, 0.1),
            ('risk_per_trade_percent', 'Risk per Trade %', 0.5, 3.0, 0.1),
            ('max_concurrent_trades', 'Max Concurrent Trades', 1, 5, 1),
            ('daily_loss_limit_percent', 'Daily Loss Limit %', 1.0, 10.0, 0.5),
            ('max_contracts_mes', 'Max Contracts — MES', 1, 5, 1),
            ('max_contracts_mnq', 'Max Contracts — MNQ', 1, 5, 1),
            ('max_contracts_nq', 'Max Contracts — NQ', 1, 3, 1),
        ]

        for i, (key, label, lo, hi, step) in enumerate(setting_defs):
            ctk.CTkLabel(settings_grid, text=label).grid(row=i, column=0, sticky='w', padx=(12, 8), pady=6)
            val = self._settings.get(key, DEFAULT_SETTINGS[key])

            slider = ctk.CTkSlider(settings_grid, from_=lo, to=hi, number_of_steps=int((hi - lo) / step))
            slider.set(val)
            slider.grid(row=i, column=1, sticky='ew', padx=4, pady=6)

            val_label = ctk.CTkLabel(settings_grid, text=self._fmt_val(key, val), width=60)
            val_label.grid(row=i, column=2, padx=(4, 12), pady=6)

            # Bind slider movement to update label
            slider.configure(command=lambda v, k=key, lbl=val_label: lbl.configure(text=self._fmt_val(k, float(v))))
            self._setting_widgets[key] = (slider, val_label)

        settings_grid.grid_columnconfigure(1, weight=1)

        btn_row = ctk.CTkFrame(tab_settings, fg_color='transparent')
        btn_row.pack(pady=12)
        ctk.CTkButton(btn_row, text="Save Settings", width=140, command=self._save_settings).pack(side='left', padx=8)
        ctk.CTkButton(btn_row, text="Reset to Defaults", width=140, fg_color='gray',
                       command=self._reset_settings).pack(side='left', padx=8)

    @staticmethod
    def _fmt_val(key: str, val: float) -> str:
        if 'max_c' in key or key == 'max_concurrent_trades':
            return str(int(round(val)))
        if 'percent' in key:
            return f"{val:.1f}%"
        return f"{val:.2f}"

    # ── Settings Actions ─────────────────────────────────────────
    def _save_settings(self):
        for key, (slider, _) in self._setting_widgets.items():
            self._settings[key] = round(slider.get(), 4)
        save_settings(self._settings)
        self._append_log("[Settings] Saved to settings.json")

    def _reset_settings(self):
        self._settings = dict(DEFAULT_SETTINGS)
        for key, (slider, val_label) in self._setting_widgets.items():
            slider.set(DEFAULT_SETTINGS[key])
            val_label.configure(text=self._fmt_val(key, DEFAULT_SETTINGS[key]))
        save_settings(self._settings)
        self._append_log("[Settings] Reset to defaults")

    # ── Bot Lifecycle ────────────────────────────────────────────
    def _on_start(self):
        symbols = [sym for sym, var in self._sym_vars.items() if var.get()]
        if not symbols:
            self._append_log("⚠️  Select at least one instrument!")
            return

        paper = self._mode_var.get() == 'paper'

        if not paper:
            # Check credentials
            uid = os.getenv('RITHMIC_USER_ID', '').strip()
            pw = os.getenv('RITHMIC_PASSWORD', '').strip()
            if not uid or not pw:
                dlg = CredentialsDialog(self)
                self.wait_window(dlg)
                if not dlg.result:
                    self._append_log("❌ Credentials required for live trading")
                    return

        # Apply settings to env vars
        apply_settings_to_env(self._settings)
        os.environ['BROKER_TYPE'] = 'rithmic'
        os.environ['ASSET_CLASS'] = 'futures'
        os.environ['TRADING_MODE'] = 'paper' if paper else 'live'

        self._stop_event.clear()
        self._set_controls_running(True)
        self._status_label.configure(text="⏳ Connecting...", text_color='#d29922')
        mode_str = "PAPER" if paper else "LIVE"
        self._append_log(f"🚀 Starting {mode_str} trading: {', '.join(symbols)}")

        # Redirect stdout/stderr
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = QueueWriter(self._log_queue, self._orig_stdout)
        sys.stderr = QueueWriter(self._log_queue, self._orig_stderr)

        # Launch bot in background thread
        self._trader_thread = threading.Thread(
            target=self._run_bot, args=(symbols, paper), daemon=True
        )
        self._trader_thread.start()

    def _run_bot(self, symbols: list, paper: bool):
        """Runs in a background thread."""
        try:
            from start_live_rithmic import LiveRithmicTrader

            self._trader = LiveRithmicTrader(
                symbol=symbols[0],
                symbols=symbols,
                paper_mode=paper,
                skip_confirm=True,
            )
            if not self._trader.connect():
                self._log_queue.put("❌ Connection failed — check credentials")
                self._schedule_stopped()
                return

            self._schedule_status("🟢 Trading", '#3fb950')

            # Patch the while-True loop: we'll run the method but watch the stop event.
            # The run() method uses `while True` — we monkey-patch to check our event.
            self._run_with_stop_check(self._trader)

        except Exception as e:
            self._log_queue.put(f"❌ Error: {e}")
        finally:
            self._schedule_stopped()

    def _run_with_stop_check(self, trader):
        """Run the trader's main loop but break out when stop_event is set."""
        # We can't easily break into the while-True inside run().
        # Instead, we run it in a thread and use the stop_event to trigger
        # a graceful broker shutdown, which will cause the loop to error/exit.
        import types

        original_run = trader.run

        stop_event = self._stop_event

        def patched_run(self_trader):
            """Main trading loop — patched to check stop_event."""
            print("\n" + "=" * 70)
            print("  LIVE TRADING - Rithmic + Full AI Ensemble")
            print("=" * 70)
            print(f"Symbols:    {', '.join(self_trader.symbols)}")
            print(f"Max Concurrent Trades: {self_trader.max_positions}")
            print(f"Mode:       {'PAPER' if self_trader.paper_mode else '⚠️ LIVE'}")
            print(f"Strategy:   Sweep-Gate + ML (IntelligentTrader, AdvancedStrategies)")
            print("=" * 70)
            print("\n🚀 Starting trading loop...\n")

            last_bar_time = {}

            try:
                while not stop_event.is_set():
                    # Check if Rithmic connection is permanently lost
                    if hasattr(self_trader.broker, 'is_connection_dead') and self_trader.broker.is_connection_dead:
                        print("\n❌ FORCED LOGOUT — Another app is using these credentials.")
                        break

                    if not self_trader.check_daily_limits():
                        print("⏸️  Limits reached - waiting...")
                        # Wait in small increments so we can check stop_event
                        for _ in range(60):
                            if stop_event.is_set():
                                break
                            time.sleep(5)
                        continue

                    self_trader.sync_broker_position()

                    any_new_bar = False
                    cycle_candidates = []

                    for symbol in self_trader.symbols:
                        if stop_event.is_set():
                            break

                        df = self_trader.get_candles(symbol=symbol, count=250)
                        if df is None or len(df) < self_trader.lookback + self_trader.ema_len + 15:
                            continue

                        df = self_trader.calculate_indicators(df)

                        if 'datetime' in df.columns:
                            current_bar = df.iloc[-1]['datetime']
                        elif 'time' in df.columns:
                            current_bar = df.iloc[-1]['time']
                        else:
                            current_bar = datetime.now()

                        if last_bar_time.get(symbol) == current_bar:
                            continue
                        last_bar_time[symbol] = current_bar
                        any_new_bar = True

                        if self_trader.positions:
                            exits = self_trader.check_position_exit(symbol, df)
                            for order_id, exit_type in exits:
                                pos = self_trader.positions.get(order_id)
                                if pos:
                                    exit_price = pos.tp if exit_type == 'TAKE_PROFIT' else pos.sl
                                    self_trader.process_exit(order_id, exit_type, exit_price)
                            self_trader.update_trailing_stops(symbol, df)

                        if len(self_trader.positions) < self_trader.max_positions and self_trader.cooldown == 0:
                            signal = self_trader.check_entry_signal(symbol, df)
                            if signal:
                                cycle_candidates.append(signal)

                        row = df.iloc[-1]
                        pos_syms = [p.symbol for p in self_trader.positions.values()]
                        pos_str = f"{len(self_trader.positions)}/{self_trader.max_positions} {pos_syms}" if self_trader.positions else "None"
                        cooldown_str = f"(CD:{self_trader.cooldown})" if self_trader.cooldown > 0 else ""
                        print(f"[{current_bar}] {symbol} {row['close']:.2f} | Pos: {pos_str} {cooldown_str} | Daily: ${self_trader.daily_pnl:.2f}")

                    if stop_event.is_set():
                        break

                    if not any_new_bar:
                        for _ in range(5):
                            if stop_event.is_set():
                                break
                            time.sleep(1)
                        continue

                    if self_trader.cooldown > 0:
                        self_trader.cooldown -= 1

                    if len(self_trader.positions) < self_trader.max_positions and cycle_candidates:
                        priority_rank = {sym: idx for idx, sym in enumerate(self_trader.symbol_priority)}
                        sorted_candidates = sorted(
                            cycle_candidates,
                            key=lambda s: (-float(s.get('confidence', 0.0)),
                                           priority_rank.get(s.get('symbol', ''), 999)),
                        )
                        if len(sorted_candidates) > 1:
                            names = ', '.join(f"{s['symbol']}({s.get('confidence', 0.0):.0%})" for s in sorted_candidates)
                            print(f"🏆 Cycle candidates: {names}")

                        SAME_UNDERLYING = {'NQ': 'MNQ', 'MNQ': 'NQ'}
                        open_syms = {p.symbol for p in self_trader.positions.values()}
                        for candidate in sorted_candidates:
                            if len(self_trader.positions) >= self_trader.max_positions:
                                break
                            sym = candidate.get('symbol', '')
                            sibling = SAME_UNDERLYING.get(sym)
                            if sibling and sibling in open_syms:
                                print(f"   ⏭️ Skipping {sym} — {sibling} already open (same underlying)")
                                continue
                            if self_trader.place_order(candidate):
                                open_syms.add(sym)
                                spec = self_trader._spec(candidate['symbol'])
                                print(f"📊 {candidate['symbol']} ATR: {candidate['atr']:.2f}")
                                risk = abs(candidate['entry'] - candidate['sl']) * spec['point_value']
                                print(f"   Risk: ${risk:.2f} per contract")

                    for _ in range(10):
                        if stop_event.is_set():
                            break
                        if hasattr(self_trader.broker, 'is_connection_dead') and self_trader.broker.is_connection_dead:
                            break
                        time.sleep(1)

            except Exception as e:
                print(f"❌ Trading loop error: {e}")
            finally:
                self_trader.save_log()
                if self_trader.broker:
                    self_trader.broker.shutdown()
                print("\n⏹️  Trading stopped")

        # Replace the run method
        trader.run = types.MethodType(patched_run, trader)
        trader.run()

    def _schedule_stopped(self):
        """Schedule UI update back to stopped state (thread-safe)."""
        self.after(0, self._on_bot_stopped)

    def _schedule_status(self, text: str, color: str):
        self.after(0, lambda: self._status_label.configure(text=text, text_color=color))

    def _on_bot_stopped(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._running = False
        self._trader = None
        self._set_controls_running(False)
        self._status_label.configure(text="⏹ Stopped", text_color='#8b949e')

    def _on_stop(self):
        if self._trader:
            self._append_log("⏹ Stopping bot...")
            self._stop_event.set()
            self._status_label.configure(text="⏳ Stopping...", text_color='#d29922')

    def _set_controls_running(self, running: bool):
        self._running = running
        state_on = 'normal' if not running else 'disabled'
        state_off = 'disabled' if not running else 'normal'
        self._start_btn.configure(state=state_on)
        self._stop_btn.configure(state=state_off)
        for cb_var in self._sym_vars.values():
            pass  # CTkCheckBox doesn't support state directly on the var

    # ── Dashboard ────────────────────────────────────────────────
    def _on_dashboard(self):
        if self._dashboard_thread is None or not self._dashboard_thread.is_alive():
            self._dashboard_thread = threading.Thread(target=self._start_dashboard, daemon=True)
            self._dashboard_thread.start()
            self._append_log("📊 Dashboard starting at http://localhost:5000")
        webbrowser.open("http://localhost:5000")

    def _start_dashboard(self):
        try:
            from src.dashboard.app import start_dashboard
            start_dashboard(port=5000)
        except Exception as e:
            self._log_queue.put(f"⚠️ Dashboard error: {e}")

    # ── Log ──────────────────────────────────────────────────────
    def _append_log(self, text: str):
        self._log_text.configure(state='normal')
        ts = datetime.now().strftime('%H:%M:%S')
        self._log_text.insert('end', f"[{ts}] {text}\n")
        self._log_text.see('end')
        self._log_text.configure(state='disabled')

    def _poll_log(self):
        """Drain the log queue and append to the textbox. Also check for trade notifications."""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._append_log(msg)
                _check_trade_notification(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ── Update Check ─────────────────────────────────────────────
    def _check_update_async(self):
        threading.Thread(target=self._do_update_check, daemon=True).start()

    def _do_update_check(self):
        info = check_for_update()
        if info:
            self._update_url = info['url']
            self.after(0, lambda: self._show_update_banner(info['tag']))

    def _show_update_banner(self, tag: str):
        self._update_label.configure(text=f"  Update available: v{tag}")
        self._update_frame.pack(fill='x', padx=12, pady=(8, 0), before=self._tabs)

    def _open_update_url(self):
        if self._update_url:
            webbrowser.open(self._update_url)

    # ── System Tray ──────────────────────────────────────────────
    def _on_close(self):
        """Minimize to tray instead of quitting (if bot is running)."""
        if self._running:
            self._minimize_to_tray()
        else:
            self._quit_app()

    def _minimize_to_tray(self):
        self.withdraw()
        try:
            import pystray
            from PIL import Image

            # Create a simple icon (green square)
            img = Image.new('RGB', (64, 64), color=(35, 134, 54))

            menu = pystray.Menu(
                pystray.MenuItem("Show Window", self._restore_from_tray, default=True),
                pystray.MenuItem("Stop Bot", self._tray_stop),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self._tray_quit),
            )
            self._tray_icon = pystray.Icon("BottTrader", img, "BottTrader", menu)
            threading.Thread(target=self._tray_icon.run, daemon=True).start()
        except ImportError:
            # pystray not available — just minimize normally
            self.iconify()

    def _restore_from_tray(self, icon=None, item=None):
        if self._tray_icon:
            self._tray_icon.stop()
            self._tray_icon = None
        self.after(0, self.deiconify)

    def _tray_stop(self, icon=None, item=None):
        self._stop_event.set()
        self._restore_from_tray()

    def _tray_quit(self, icon=None, item=None):
        self._stop_event.set()
        if self._tray_icon:
            self._tray_icon.stop()
        self.after(0, self.destroy)

    def _quit_app(self):
        self._stop_event.set()
        if self._tray_icon:
            self._tray_icon.stop()
        self.destroy()


# ─── Entry Point ─────────────────────────────────────────────────
def main():
    app = BottTraderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
