from pathlib import Path

p = Path(r"C:\Users\shawa\Projects\bott-trader\src\strategy\day_hybrid.py")
text = p.read_text(encoding="utf-8")
if "day_hybrid_invert_enabled" in text:
    print("ALREADY_HAS_INVERT")
else:
    needle = '''def day_hybrid_from_env() -> bool:
    return str(os.getenv("PAPER_DAY_HYBRID", "")).strip().lower() in (
        "1", "true", "yes", "on",
    ) or str(os.getenv("DAY_HYBRID", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )
'''
    insert = '''def day_hybrid_from_env() -> bool:
    return str(os.getenv("PAPER_DAY_HYBRID", "")).strip().lower() in (
        "1", "true", "yes", "on",
    ) or str(os.getenv("DAY_HYBRID", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def day_hybrid_invert_enabled() -> bool:
    """Flip every day-hybrid signal long<->short (same SL/TP distance)."""
    return str(os.getenv("DAY_HYBRID_INVERT", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def invert_day_hybrid_signal(sig):
    """Mirror direction + SL/TP around entry when DAY_HYBRID_INVERT is on."""
    if not sig or not day_hybrid_invert_enabled():
        return sig
    out = dict(sig)
    entry = float(out["entry"])
    pts = float(out.get("sl_pts") or abs(entry - float(out["sl"])))
    orig = out.get("direction")
    if orig == "long":
        out["direction"] = "short"
        out["leg"] = "short"
        out["sl"] = entry + pts
        out["tp"] = entry - pts * TP_R
    else:
        out["direction"] = "long"
        out["leg"] = "long"
        out["sl"] = entry - pts
        out["tp"] = entry + pts * TP_R
    out["inverted"] = True
    out["orig_direction"] = orig
    return out
'''
    if needle not in text:
        raise SystemExit("needle missing")
    text = text.replace(needle, insert, 1)
    old_banner = '"Stop: 1m ATR x1.5 clip 6-25 | TP 1R",'
    new_banner = '''"Stop: 1m ATR x1.5 clip 6-25 | TP 1R",
        ("INVERT ON — every signal flips long<->short (DAY_HYBRID_INVERT=true)"
         if day_hybrid_invert_enabled() else "INVERT off"),'''
    if old_banner not in text:
        raise SystemExit("banner needle missing")
    text = text.replace(old_banner, new_banner, 1)
    p.write_text(text, encoding="utf-8")
    print("PATCHED day_hybrid.py")

# patch paper runner
r = Path(r"C:\Users\shawa\Projects\bott-trader\src\strategy\day_hybrid_paper_runner.py")
rt = r.read_text(encoding="utf-8")
if "invert_day_hybrid_signal" not in rt:
    rt = rt.replace(
        '''from src.strategy.day_hybrid import (
    SESSION_NAME,
    check_bar_exit,
    check_long_entry,
    check_short_entry,
    explain_day_hybrid_skip,
    is_scan_active_et,
    on_long_closed,
    recipe_banner_lines,
    scan_mode_label,
    should_flatten_rth,
)''',
        '''from src.strategy.day_hybrid import (
    SESSION_NAME,
    check_bar_exit,
    check_long_entry,
    check_short_entry,
    day_hybrid_invert_enabled,
    explain_day_hybrid_skip,
    invert_day_hybrid_signal,
    is_scan_active_et,
    on_long_closed,
    recipe_banner_lines,
    scan_mode_label,
    should_flatten_rth,
)''',
        1,
    )
    old_place = '''    def _place(self, sig: Dict[str, Any]) -> bool:
        if not skip_local_fake_fill() or not paper_rithmic_brackets_enabled():
            print("❌ Lucid brackets required — set PAPER_USE_RITHMIC=true PAPER_RITHMIC_BRACKETS=true")
            return False
        if self.rithmic is None or not self.rithmic.connected:
            print("⏳ No Lucid order plant — skip entry")
            return False
        ref = _last_price(self.feed, None) or float(sig["entry"])
        sl, tp = _brackets_from_pts(sig, ref)
        side = "buy" if sig["direction"] == "long" else "sell"
'''
    new_place = '''    def _place(self, sig: Dict[str, Any]) -> bool:
        if not skip_local_fake_fill() or not paper_rithmic_brackets_enabled():
            print("❌ Lucid brackets required — set PAPER_USE_RITHMIC=true PAPER_RITHMIC_BRACKETS=true")
            return False
        if self.rithmic is None or not self.rithmic.connected:
            print("⏳ No Lucid order plant — skip entry")
            return False
        sig = invert_day_hybrid_signal(sig) or sig
        if sig.get("inverted"):
            print(
                f"🔁 INVERT {sig.get('orig_direction')} → {sig['direction']} "
                f"(DAY_HYBRID_INVERT)",
                flush=True,
            )
        ref = _last_price(self.feed, None) or float(sig["entry"])
        sl, tp = _brackets_from_pts(sig, ref)
        side = "buy" if sig["direction"] == "long" else "sell"
'''
    if old_place not in rt:
        raise SystemExit("place needle missing")
    rt = rt.replace(old_place, new_place, 1)
    r.write_text(rt, encoding="utf-8")
    print("PATCHED day_hybrid_paper_runner.py")
else:
    print("RUNNER_ALREADY_HAS_INVERT")

# write START bat with invert
bat = Path(r"C:\Users\shawa\Projects\bott-trader\START_DAY_HYBRID_INVERT_PAPER.bat")
bat.write_text('''@echo off
cd /d C:\\Users\\shawa\\Projects\\bott-trader
call .\\venv\\Scripts\\activate.bat
set PAPER_DAY_HYBRID=true
set DAY_HYBRID=true
set DAY_HYBRID_GLOBEX=true
set DAY_HYBRID_RSI_GATE=true
set DAY_HYBRID_RSI_MIN_SHORT=40
set DAY_HYBRID_RSI_MAX_LONG=65
set DAY_HYBRID_INVERT=true
set PAPER_OVERNIGHT=false
set OVERNIGHT_TRADING=false
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
set RITHMIC_ALLOW_SIMULATOR=true
set RITHMIC_SKIP_HISTORY=false
set TRADING_MODE=paper
set EMA15_1M_SEED=data\\MNQ_1m.csv
echo INVERT ON — long signals become shorts and vice versa
.\\venv\\Scripts\\python.exe -u start_live_mtf_scalping.py --paper --day-hybrid --symbols MNQ --duration 0
pause
''', encoding='utf-8')
print("WROTE", bat)
