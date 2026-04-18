#!/usr/bin/env python
"""
BottTrader — Desktop Launcher for Rithmic Futures Trading Bot

A standalone GUI application that wraps LiveRithmicTrader with:
  - Instrument selection (MES, MNQ, NQ)
  - Paper / Live toggle
  - Settings panel (confidence, SL/TP, risk, max positions)
  - Trade History tab (symbol, side, entry/exit, P&L)
  - Scrolling log output
  - System tray support (minimize to tray)
  - Trade notifications (Windows toast)
  - Auto-update check from GitHub Releases
  - Dashboard launcher (opens Flask dashboard in browser)

Usage:
    python launcher_app.py          # Launch GUI
    pyinstaller --onefile ...       # See build_exe.bat
"""

__version__ = "1.1.5"

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

# ─── PyInstaller: explicit imports so the bundler finds them ───
# These are used at runtime by start_live_rithmic / src modules.
# Without these top-level imports PyInstaller won't include them.
try:
    import pandas              # noqa: F401
    import numpy               # noqa: F401
    import pandas_ta            # noqa: F401
    import sklearn              # noqa: F401
    import sklearn.ensemble     # noqa: F401
    import sklearn.svm          # noqa: F401
    import sklearn.preprocessing  # noqa: F401
    import scipy                # noqa: F401
    import flask                # noqa: F401
    import flask_cors           # noqa: F401
    import requests             # noqa: F401
    import yfinance             # noqa: F401
    import feedparser           # noqa: F401
    import apscheduler          # noqa: F401
except ImportError:
    pass  # OK — some may be missing in dev; PyInstaller will still bundle them

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
TRADE_HISTORY_FILE = BASE_DIR / 'trade_history.json'

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


def load_trade_history() -> list:
    """Load trade history from disk."""
    if TRADE_HISTORY_FILE.exists():
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_trade_history(trades: list):
    """Save trade history to disk."""
    with open(TRADE_HISTORY_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


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
GITHUB_BRANCH = "main"
# Folders to sync from GitHub for code-only updates
CODE_FOLDERS = ["src", "config"]
CODE_FILES = ["start_live_rithmic.py", "start_live.py", "requirements.txt"]


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


def update_code_from_github(progress_callback=None) -> str:
    """Download latest src/config/scripts from GitHub and extract next to the exe.
    Returns a status message string."""
    import requests
    import zipfile
    import shutil
    import tempfile

    url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"

    if progress_callback:
        progress_callback("Downloading latest code from GitHub...")

    resp = requests.get(url, timeout=60, stream=True)
    if resp.status_code != 200:
        return f"Download failed (HTTP {resp.status_code})"

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    total = 0
    for chunk in resp.iter_content(chunk_size=65536):
        tmp.write(chunk)
        total += len(chunk)
    tmp.close()

    if progress_callback:
        progress_callback(f"Downloaded {total // 1024}KB, extracting...")

    try:
        with zipfile.ZipFile(tmp.name, 'r') as zf:
            # The zip contains a top-level folder like "bott-trader-main/"
            top_dirs = {name.split('/')[0] for name in zf.namelist() if '/' in name}
            if len(top_dirs) != 1:
                return "Unexpected zip structure"
            prefix = top_dirs.pop() + '/'

            updated = 0

            # Sync folders (src/, config/)
            for folder in CODE_FOLDERS:
                folder_prefix = prefix + folder + '/'
                members = [n for n in zf.namelist()
                           if n.startswith(folder_prefix) and not n.endswith('/')]
                if members:
                    dest_folder = BASE_DIR / folder
                    # Remove old folder and replace entirely
                    if dest_folder.exists():
                        shutil.rmtree(dest_folder)
                    for member in members:
                        rel_path = member[len(prefix):]  # e.g. "src/ai/..."
                        dest = BASE_DIR / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(dest, 'wb') as dst:
                            dst.write(src.read())
                        updated += 1

            # Sync individual files
            for fname in CODE_FILES:
                member = prefix + fname
                if member in zf.namelist():
                    with zf.open(member) as src, open(BASE_DIR / fname, 'wb') as dst:
                        dst.write(src.read())
                    updated += 1

        return f"Updated {updated} files from GitHub ({GITHUB_BRANCH} branch)"
    except Exception as e:
        return f"Extract failed: {e}"
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


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
    ('BUY', '\U0001f7e2'),
    ('SELL', '\U0001f534'),
    ('TAKE_PROFIT', '\U0001f4b0'),
    ('STOP_LOSS', '\u26d4'),
    ('trail', '\U0001f512'),
    ('Daily loss limit', '\u26a0\ufe0f'),
]


def _check_trade_notification(text: str):
    """Fire a notification if the log line indicates a trade event."""
    for keyword, icon in _TRADE_PATTERNS:
        if keyword in text:
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
        self._trade_history = load_trade_history()

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
        tab_history = self._tabs.add("Trade History")
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
        ctk.CTkRadioButton(mode_frame, text="Live \u26a0\ufe0f", variable=self._mode_var, value='live').pack(anchor='w', padx=12, pady=2)

        # Status
        status_frame = ctk.CTkFrame(top_frame)
        status_frame.pack(side='left', fill='y', padx=(0, 16))
        ctk.CTkLabel(status_frame, text="Status", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=8, pady=(6, 2))
        self._status_label = ctk.CTkLabel(status_frame, text="\u23f9 Stopped", text_color='#8b949e',
                                           font=ctk.CTkFont(size=13))
        self._status_label.pack(anchor='w', padx=12, pady=2)

        # Buttons
        btn_frame = ctk.CTkFrame(top_frame, fg_color='transparent')
        btn_frame.pack(side='right', padx=8)
        self._start_btn = ctk.CTkButton(btn_frame, text="\u25b6  Start", width=110, fg_color='#238636',
                                         hover_color='#2ea043', command=self._on_start)
        self._start_btn.pack(pady=4)
        self._stop_btn = ctk.CTkButton(btn_frame, text="\u23f9  Stop", width=110, fg_color='#da3633',
                                        hover_color='#f85149', state='disabled', command=self._on_stop)
        self._stop_btn.pack(pady=4)
        self._dash_btn = ctk.CTkButton(btn_frame, text="\U0001f4ca  Dashboard", width=110,
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

        # ── Trade History Tab ─────────────────────────────────────
        self._build_history_tab(tab_history)

        # ── Settings Tab ─────────────────────────────────────────
        self._setting_widgets = {}
        settings_grid = ctk.CTkFrame(tab_settings)
        settings_grid.pack(fill='both', expand=True, padx=16, pady=12)

        setting_defs = [
            ('ensemble_confidence_threshold', 'Confidence Threshold', 0.40, 0.80, 0.01),
            ('stop_loss_multiplier', 'SL Multiplier (\u00d7ATR)', 1.0, 5.0, 0.1),
            ('take_profit_ratio', 'TP Ratio (\u00d7SL)', 1.0, 5.0, 0.1),
            ('risk_per_trade_percent', 'Risk per Trade %', 0.5, 3.0, 0.1),
            ('max_concurrent_trades', 'Max Concurrent Trades', 1, 5, 1),
            ('daily_loss_limit_percent', 'Daily Loss Limit %', 1.0, 10.0, 0.5),
            ('max_contracts_mes', 'Max Contracts \u2014 MES', 1, 5, 1),
            ('max_contracts_mnq', 'Max Contracts \u2014 MNQ', 1, 5, 1),
            ('max_contracts_nq', 'Max Contracts \u2014 NQ', 1, 3, 1),
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
        ctk.CTkButton(btn_row, text="\U0001f511 Rithmic Login", width=140,
                       command=self._edit_credentials).pack(side='left', padx=8)

        # Second button row — updates
        btn_row2 = ctk.CTkFrame(tab_settings, fg_color='transparent')
        btn_row2.pack(pady=(0, 12))
        self._update_code_btn = ctk.CTkButton(
            btn_row2, text="\U0001f504 Update Bot Code", width=180,
            fg_color='#1f6feb', hover_color='#388bfd',
            command=self._on_update_code)
        self._update_code_btn.pack(side='left', padx=8)
        self._update_status_label = ctk.CTkLabel(btn_row2, text="", text_color='#8b949e')
        self._update_status_label.pack(side='left', padx=8)

    # ── Trade History Tab Builder ────────────────────────────────
    def _build_history_tab(self, parent):
        # Summary cards
        summary_frame = ctk.CTkFrame(parent)
        summary_frame.pack(fill='x', padx=8, pady=(8, 4))

        self._summary_labels = {}
        for i, (key, label, color) in enumerate([
            ('total', 'Total Trades', '#58a6ff'),
            ('wins', 'Wins', '#3fb950'),
            ('losses', 'Losses', '#f85149'),
            ('winrate', 'Win Rate', '#d29922'),
            ('net_pnl', 'Net P&L', '#58a6ff'),
            ('best', 'Best Trade', '#3fb950'),
            ('worst', 'Worst Trade', '#f85149'),
        ]):
            card = ctk.CTkFrame(summary_frame)
            card.grid(row=0, column=i, padx=4, pady=4, sticky='nsew')
            ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=10),
                          text_color='#8b949e').pack(pady=(6, 0))
            lbl = ctk.CTkLabel(card, text='\u2014', font=ctk.CTkFont(size=16, weight='bold'),
                                text_color=color)
            lbl.pack(pady=(0, 6))
            self._summary_labels[key] = lbl
            summary_frame.grid_columnconfigure(i, weight=1)

        # Column headers
        header_frame = ctk.CTkFrame(parent, fg_color='#161b22')
        header_frame.pack(fill='x', padx=8, pady=(8, 0))
        columns = ['Time', 'Symbol', 'Side', 'Entry', 'Exit', 'SL', 'TP', 'P&L', 'Result']
        col_widths = [130, 60, 50, 80, 80, 80, 80, 80, 70]
        for j, (col, w) in enumerate(zip(columns, col_widths)):
            ctk.CTkLabel(header_frame, text=col, width=w,
                          font=ctk.CTkFont(size=11, weight='bold'),
                          text_color='#8b949e').grid(row=0, column=j, padx=2, pady=4)

        # Scrollable trade rows
        self._history_scroll = ctk.CTkScrollableFrame(parent, fg_color='#0d1117')
        self._history_scroll.pack(fill='both', expand=True, padx=8, pady=(0, 4))

        # Buttons row
        btn_row = ctk.CTkFrame(parent, fg_color='transparent')
        btn_row.pack(fill='x', padx=8, pady=6)
        ctk.CTkButton(btn_row, text="Export CSV", width=120, command=self._export_csv).pack(side='left', padx=4)
        ctk.CTkButton(btn_row, text="Clear History", width=120, fg_color='gray',
                       command=self._clear_history).pack(side='left', padx=4)
        self._history_count_label = ctk.CTkLabel(btn_row, text="", text_color='#8b949e')
        self._history_count_label.pack(side='right', padx=8)

        self._refresh_history_display()

    def _refresh_history_display(self):
        """Rebuild the trade history rows and update summary cards."""
        for widget in self._history_scroll.winfo_children():
            widget.destroy()

        col_widths = [130, 60, 50, 80, 80, 80, 80, 80, 70]
        trades = self._trade_history

        for i, t in enumerate(reversed(trades)):
            pnl = t.get('pnl', 0)
            is_win = pnl > 0
            row_bg = '#0f1a0f' if is_win else '#1a0f0f' if pnl < 0 else '#0d1117'
            row = ctk.CTkFrame(self._history_scroll, fg_color=row_bg, height=28)
            row.pack(fill='x', pady=1)

            side = t.get('direction', t.get('side', '\u2014')).upper()
            side_color = '#3fb950' if side in ('BUY', 'LONG') else '#f85149'
            pnl_str = f"${pnl:+.2f}" if pnl != 0 else '\u2014'
            pnl_color = '#3fb950' if pnl > 0 else '#f85149' if pnl < 0 else '#8b949e'
            result = '\u2705 WIN' if pnl > 0 else '\u274c LOSS' if pnl < 0 else '\u23f3'
            result_color = '#3fb950' if pnl > 0 else '#f85149' if pnl < 0 else '#d29922'

            values = [
                (str(t.get('time', t.get('entry_time', '\u2014')))[:19], '#c9d1d9'),
                (t.get('symbol', '\u2014'), '#58a6ff'),
                (side, side_color),
                (f"{t.get('entry', t.get('entry_price', 0)):.2f}", '#c9d1d9'),
                (f"{t.get('exit', t.get('exit_price', 0)):.2f}" if t.get('exit', t.get('exit_price')) else '\u2014', '#c9d1d9'),
                (f"{t.get('sl', 0):.2f}" if t.get('sl') else '\u2014', '#c9d1d9'),
                (f"{t.get('tp', 0):.2f}" if t.get('tp') else '\u2014', '#c9d1d9'),
                (pnl_str, pnl_color),
                (result, result_color),
            ]
            for j, ((val, color), w) in enumerate(zip(values, col_widths)):
                ctk.CTkLabel(row, text=val, width=w, font=ctk.CTkFont(size=11),
                              text_color=color).grid(row=0, column=j, padx=2, pady=2)

        total = len(trades)
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
        net = sum(t.get('pnl', 0) for t in trades)
        best = max((t.get('pnl', 0) for t in trades), default=0)
        worst = min((t.get('pnl', 0) for t in trades), default=0)
        wr = f"{wins/total*100:.0f}%" if total > 0 else '\u2014'

        self._summary_labels['total'].configure(text=str(total))
        self._summary_labels['wins'].configure(text=str(wins))
        self._summary_labels['losses'].configure(text=str(losses))
        self._summary_labels['winrate'].configure(text=wr)
        net_color = '#3fb950' if net > 0 else '#f85149' if net < 0 else '#58a6ff'
        self._summary_labels['net_pnl'].configure(text=f"${net:+,.2f}", text_color=net_color)
        self._summary_labels['best'].configure(text=f"${best:+.2f}" if best else '\u2014')
        self._summary_labels['worst'].configure(text=f"${worst:+.2f}" if worst else '\u2014')

        self._history_count_label.configure(text=f"{total} trades recorded")

    def _add_trade(self, trade: dict):
        """Add a trade to history and refresh display."""
        self._trade_history.append(trade)
        save_trade_history(self._trade_history)
        self.after(0, self._refresh_history_display)

    def _export_csv(self):
        """Export trade history to CSV."""
        if not self._trade_history:
            self._append_log("\u26a0\ufe0f No trades to export")
            return
        csv_path = BASE_DIR / f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            import csv
            fields = ['time', 'symbol', 'direction', 'entry', 'exit', 'sl', 'tp', 'pnl', 'contracts']
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                for t in self._trade_history:
                    row = {
                        'time': t.get('time', t.get('entry_time', '')),
                        'symbol': t.get('symbol', ''),
                        'direction': t.get('direction', t.get('side', '')),
                        'entry': t.get('entry', t.get('entry_price', '')),
                        'exit': t.get('exit', t.get('exit_price', '')),
                        'sl': t.get('sl', ''),
                        'tp': t.get('tp', ''),
                        'pnl': t.get('pnl', 0),
                        'contracts': t.get('contracts', t.get('size', 1)),
                    }
                    writer.writerow(row)
            self._append_log(f"\U0001f4c4 Exported {len(self._trade_history)} trades to {csv_path.name}")
        except Exception as e:
            self._append_log(f"\u274c Export error: {e}")

    def _clear_history(self):
        self._trade_history = []
        save_trade_history([])
        self._refresh_history_display()
        self._append_log("\U0001f5d1\ufe0f Trade history cleared")

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

    def _edit_credentials(self):
        """Open the credentials dialog so user can view/edit Rithmic login."""
        dlg = CredentialsDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            self._append_log("\u2705 Rithmic credentials saved")

    def _on_update_code(self):
        """Pull latest bot code from GitHub without re-downloading the exe."""
        if self._running:
            self._append_log("\u26a0\ufe0f Stop the bot before updating code")
            return
        self._update_code_btn.configure(state='disabled', text="\u23f3 Updating...")
        self._update_status_label.configure(text="Downloading...", text_color='#d29922')
        threading.Thread(target=self._do_update_code, daemon=True).start()

    def _do_update_code(self):
        """Background thread for code update."""
        def progress(msg):
            self.after(0, lambda: self._update_status_label.configure(text=msg))
        try:
            result = update_code_from_github(progress_callback=progress)
            self.after(0, lambda: self._finish_update_code(result))
        except Exception as e:
            self.after(0, lambda: self._finish_update_code(f"Error: {e}"))

    def _finish_update_code(self, result: str):
        """UI callback when code update finishes."""
        self._update_code_btn.configure(state='normal', text="\U0001f504 Update Bot Code")
        is_success = result.startswith("Updated")
        color = '#3fb950' if is_success else '#f85149'
        self._update_status_label.configure(text=result, text_color=color)
        self._append_log(f"[Update] {result}")

    # ── Bot Lifecycle ────────────────────────────────────────────
    def _on_start(self):
        symbols = [sym for sym, var in self._sym_vars.items() if var.get()]
        if not symbols:
            self._append_log("\u26a0\ufe0f  Select at least one instrument!")
            return

        paper = self._mode_var.get() == 'paper'

        # Rithmic credentials needed for both paper and live
        uid = os.getenv('RITHMIC_USER_ID', '').strip()
        pw = os.getenv('RITHMIC_PASSWORD', '').strip()
        if not uid or not pw:
            dlg = CredentialsDialog(self)
            self.wait_window(dlg)
            if not dlg.result:
                self._append_log("\u274c Credentials required to connect to Rithmic")
                return

        # Apply settings to env vars
        apply_settings_to_env(self._settings)
        os.environ['BROKER_TYPE'] = 'rithmic'
        os.environ['ASSET_CLASS'] = 'futures'
        os.environ['TRADING_MODE'] = 'paper' if paper else 'live'

        self._stop_event.clear()
        self._set_controls_running(True)
        self._status_label.configure(text="\u23f3 Connecting...", text_color='#d29922')
        mode_str = "PAPER" if paper else "LIVE"
        self._append_log(f"\U0001f680 Starting {mode_str} trading: {', '.join(symbols)}")

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
                self._log_queue.put("\u274c Connection failed \u2014 check credentials")
                self._schedule_stopped()
                return

            self._schedule_status("\U0001f7e2 Trading", '#3fb950')

            # Run the patched main loop
            self._run_with_stop_check(self._trader)

        except Exception as e:
            self._log_queue.put(f"\u274c Error: {e}")
        finally:
            self._schedule_stopped()

    def _run_with_stop_check(self, trader):
        """Run the trader's main loop but break out when stop_event is set."""
        import types

        stop_event = self._stop_event

        def patched_run(self_trader):
            """Main trading loop — patched to check stop_event."""
            print("\n" + "=" * 70)
            print("  LIVE TRADING - Rithmic + Full AI Ensemble")
            print("=" * 70)
            print(f"Symbols:    {', '.join(self_trader.symbols)}")
            print(f"Max Concurrent Trades: {self_trader.max_positions}")
            mode_label = 'PAPER' if self_trader.paper_mode else '\u26a0\ufe0f LIVE'
            print(f"Mode:       {mode_label}")
            print(f"Strategy:   Sweep-Gate + ML (IntelligentTrader, AdvancedStrategies)")
            print("=" * 70)
            print("\n\U0001f680 Starting trading loop...\n")

            last_bar_time = {}

            try:
                while not stop_event.is_set():
                    # Check if Rithmic connection is permanently lost
                    if hasattr(self_trader.broker, 'is_connection_dead') and self_trader.broker.is_connection_dead:
                        print("\u274c Rithmic connection permanently lost. Attempting reconnect...")
                        if not self_trader.connect():
                            print("\u274c Reconnect failed. Waiting 60s...")
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
                            print(f"\U0001f3c6 Cycle candidates: {names}")

                        SAME_UNDERLYING = {'NQ': 'MNQ', 'MNQ': 'NQ'}
                        open_syms = {p.symbol for p in self_trader.positions.values()}
                        for candidate in sorted_candidates:
                            if len(self_trader.positions) >= self_trader.max_positions:
                                break
                            sym = candidate.get('symbol', '')
                            sibling = SAME_UNDERLYING.get(sym)
                            if sibling and sibling in open_syms:
                                print(f"   \u23ed\ufe0f Skipping {sym} \u2014 {sibling} already open (same underlying)")
                                continue
                            if self_trader.place_order(candidate):
                                open_syms.add(sym)
                                spec = self_trader._spec(candidate['symbol'])
                                print(f"\U0001f4ca {candidate['symbol']} ATR: {candidate['atr']:.2f}")
                                risk = abs(candidate['entry'] - candidate['sl']) * spec['point_value']
                                print(f"   Risk: ${risk:.2f} per contract")

                    for _ in range(10):
                        if stop_event.is_set():
                            break
                        if hasattr(self_trader.broker, 'is_connection_dead') and self_trader.broker.is_connection_dead:
                            break
                        time.sleep(1)

            except Exception as e:
                print(f"\u274c Trading loop error: {e}")
            finally:
                self_trader.save_log()
                if self_trader.broker:
                    self_trader.broker.shutdown()
                print("\n\u23f9\ufe0f  Trading stopped")

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
        self._status_label.configure(text="\u23f9 Stopped", text_color='#8b949e')

    def _on_stop(self):
        if self._trader:
            self._append_log("\u23f9 Stopping bot...")
            self._stop_event.set()
            self._status_label.configure(text="\u23f3 Stopping...", text_color='#d29922')

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
            self._append_log("\U0001f4ca Dashboard starting at http://localhost:5000")
        webbrowser.open("http://localhost:5000")

    def _start_dashboard(self):
        try:
            from src.dashboard.app import start_dashboard
            start_dashboard(port=5000)
        except Exception as e:
            self._log_queue.put(f"\u26a0\ufe0f Dashboard error: {e}")

    # ── Log ──────────────────────────────────────────────────────
    def _append_log(self, text: str):
        self._log_text.configure(state='normal')
        ts = datetime.now().strftime('%H:%M:%S')
        self._log_text.insert('end', f"[{ts}] {text}\n")
        self._log_text.see('end')
        self._log_text.configure(state='disabled')

    def _poll_log(self):
        """Drain the log queue and append to the textbox. Also check for trade notifications and record trades."""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._append_log(msg)
                _check_trade_notification(msg)
        except queue.Empty:
            pass
        # Also periodically sync trades from the bot's internal list
        self._sync_trader_trades()
        self.after(100, self._poll_log)

    def _sync_trader_trades(self):
        """Pull completed trades from LiveRithmicTrader.trades list into our history."""
        if not self._trader:
            return
        try:
            bot_trades = getattr(self._trader, 'trades', [])
            known = len(self._trade_history)
            if len(bot_trades) > known:
                for t in bot_trades[known:]:
                    self._add_trade(t)
        except Exception:
            pass

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
