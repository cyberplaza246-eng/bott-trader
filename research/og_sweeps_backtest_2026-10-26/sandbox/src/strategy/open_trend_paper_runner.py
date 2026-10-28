"""MNQ OpenTrend 5-minute EMA/ATR Lucid PAPER runner.

This runner is intentionally paper-only. It follows the connector/history and
protective-order patterns used by the S16 mid paper runner, but has its own
session name and one-position OpenTrend state.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pytz

from src.ai.action_log import ActionLog, write_status
from src.broker.rithmic_connector import RithmicConnector, apply_lucid_plants_with_history
from src.data.paper_csv_feed import PaperCsvFeed, default_1m_csv_path, paper_rithmic_brackets_enabled
from src.strategy.mnq_15m_ema_eod import overlay_ticker_last_on_1m, parse_mnq_1m_datetime
from src.strategy.open_trend import (
    SESSION_NAME,
    check_bar_exit,
    closed_bars,
    entries_allowed_et,
    is_rth_et,
    prepare_5m_bars,
    should_flatten_rth,
    signal_on_closed_bar,
)
from src.strategy.overnight_research import day_bot_owns_lucid, skip_local_fake_fill
from src.utils.redact import redact_secrets

ET = pytz.timezone("America/New_York")
SCAN_SEC = float(os.getenv("OPEN_TREND_SCAN_SEC", "30"))
HEARTBEAT_SEC = float(os.getenv("OPEN_TREND_HEARTBEAT_SEC", "120"))


def _last_price(feed: PaperCsvFeed, df: Optional[pd.DataFrame] = None) -> Optional[float]:
    inner = getattr(feed, "rithmic", None)
    if inner is not None and getattr(inner, "connected", False):
        try:
            quote = inner.get_latest_price("MNQ") or {}
            price = float(quote.get("last") or quote.get("bid") or 0)
            if price > 0:
                return price
        except Exception:
            pass
    if df is not None and not df.empty:
        return float(df.iloc[-1]["close"])
    return None


class OpenTrendPaperEngine:
    def __init__(self) -> None:
        self.feed = PaperCsvFeed(default_1m_csv_path(), refresh_sec=60)
        self.feed.rithmic = None
        self.rithmic: Optional[RithmicConnector] = None
        self.log = ActionLog()
        self.open_pos: Optional[Dict[str, Any]] = None
        self.last_signal_bar = ""
        self.last_managed_bar = ""
        self.last_heartbeat = 0.0
        self.last_orphan_sweep = 0.0

    def _connect(self) -> None:
        os.environ["PAPER_USE_RITHMIC"] = "true"
        os.environ["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
        os.environ["TRADING_MODE"] = "paper"
        os.environ["ALLOW_LIVE"] = "false"
        try:
            self.feed.initialize()
        except Exception as exc:
            print(f"CSV/Databento seed: {redact_secrets(exc)}", flush=True)
        if day_bot_owns_lucid():
            print("Stop start_mnq_live.bat first - one Lucid session only.", flush=True)
        if not paper_rithmic_brackets_enabled():
            print("Set PAPER_RITHMIC_BRACKETS=true for Lucid sim orders", flush=True)
            return
        try:
            apply_lucid_plants_with_history()
            raw = RithmicConnector(live_mode=False, quotes_only=False)
            raw._symbols_to_watch = ["MNQ"]
            raw._disable_yahoo_fallback = True
            raw.initialize()
            if raw.connected:
                self.rithmic = raw
                self.feed.rithmic = raw
                print("Lucid TEST sim connected (history + brackets). Close R|Trader if logged in elsewhere.", flush=True)
                try:
                    cleared = int(raw.clear_chart_stops("MNQ", include_limits=True, rounds=4, only_if_flat=True) or 0)
                    if cleared:
                        print(f"startup: cleared {cleared} leftover bot stop/limit(s) (flat)", flush=True)
                except Exception as exc:
                    print(f"startup bracket check: {redact_secrets(exc)}", flush=True)
            else:
                error = redact_secrets(getattr(raw, "_last_connect_error", "") or "not connected")
                print(f"Lucid connect failed ({error})", flush=True)
                try:
                    raw.shutdown()
                except Exception:
                    pass
        except Exception as exc:
            print(f"Lucid connector error: {redact_secrets(exc)}", flush=True)

    def _merge_rithmic_1m(self, seed: pd.DataFrame) -> pd.DataFrame:
        if self.rithmic is None or not self.rithmic.connected:
            return seed
        try:
            live = self.rithmic.get_candles("MNQ", timeframe_minutes=1, num_candles=600)
        except Exception as exc:
            print(f"Rithmic 1m history fetch failed: {redact_secrets(exc)}", flush=True)
            return seed
        if live is None or live.empty:
            return seed
        live = live.copy()
        live["datetime"] = parse_mnq_1m_datetime(live["datetime"])
        live = live.dropna(subset=["datetime"]).sort_values("datetime")
        if seed is None or seed.empty:
            return live.reset_index(drop=True)
        base = seed.copy()
        base["datetime"] = parse_mnq_1m_datetime(base["datetime"])
        base = base.dropna(subset=["datetime"])
        cutoff = live["datetime"].min()
        base = base[base["datetime"] < cutoff]
        return pd.concat([base, live], ignore_index=True).drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)

    def _get_5m(self) -> Optional[pd.DataFrame]:
        self.feed.refresh(force=False)
        one_minute = self.feed._ensure_1m()
        if one_minute is None or one_minute.empty:
            return None
        one_minute = self._merge_rithmic_1m(one_minute)
        price = _last_price(self.feed, one_minute)
        if price is not None:
            one_minute = overlay_ticker_last_on_1m(one_minute, price, now=datetime.now(pytz.UTC))
        bars = prepare_5m_bars(one_minute)
        return bars if not bars.empty else None

    def _place(self, sig: Dict[str, Any], reference: Optional[float]) -> bool:
        if not skip_local_fake_fill() or not paper_rithmic_brackets_enabled():
            print("Lucid brackets required - set PAPER_USE_RITHMIC=true PAPER_RITHMIC_BRACKETS=true", flush=True)
            return False
        if self.rithmic is None or not self.rithmic.connected:
            print("No Lucid order plant - skipping entry", flush=True)
            return False
        direction = str(sig["direction"])
        side = "buy" if direction == "long" else "sell"
        atr = float(sig["atr"])
        ref = float(reference or sig["entry"])
        sl_distance = 1.5 * atr
        tp_distance = 2.0 * atr
        if direction == "long":
            sl, tp = ref - sl_distance, ref + tp_distance
        else:
            sl, tp = ref + sl_distance, ref - tp_distance
        try:
            result = self.rithmic.place_order(
                symbol="MNQ", order_type=side, size=1, entry_price=ref,
                stop_loss=sl, take_profit=tp,
            )
        except Exception as exc:
            print(f"place_order failed: {redact_secrets(exc)}", flush=True)
            return False
        if not result or not (result.get("ticket") or result.get("order_id")):
            print("Lucid order not booked", flush=True)
            return False
        ticket = str(result.get("ticket") or result.get("order_id"))
        fill = float(result.get("entry_price") or result.get("price_ref") or ref)
        sl = float(result.get("stop_loss") or sl)
        tp = float(result.get("take_profit") or tp)
        self.open_pos = {
            "direction": direction, "entry": fill, "sl": sl, "tp": tp,
            "size": 1, "lucid_ticket": ticket,
            "entry_time": datetime.now(pytz.UTC).isoformat(),
            "signal_bar": sig.get("signal_bar"), "session": SESSION_NAME,
            "strategy": "OpenTrend",
        }
        self.last_signal_bar = str(sig.get("signal_bar") or "")
        print(
            f"ENTRY {direction.upper()} @ {fill:.2f} | SL {sl:.2f} | TP {tp:.2f} | #{ticket}",
            flush=True,
        )
        self.log.record(
            "entry", title="OPENTREND ENTRY",
            lines=[f"{direction} @ {fill:.2f}", f"SL {sl:.2f} TP {tp:.2f}", f"Lucid {ticket}"],
            symbol="MNQ", paper=True, session=SESSION_NAME,
        )
        self.last_heartbeat = 0.0
        return True

    def _open_tickets(self) -> set:
        if self.open_pos and self.open_pos.get("lucid_ticket"):
            return {str(self.open_pos["lucid_ticket"])}
        return set()

    def _working_leg_flags(self, ticket: str) -> Tuple[Optional[bool], Optional[bool]]:
        if not ticket or self.rithmic is None or not self.rithmic.connected:
            return None, None
        try:
            rows = self.rithmic.list_working_orders("MNQ") or []
        except Exception:
            return None, None
        sl_working = tp_working = False
        for row in rows:
            tag = str(row.get("user_tag") or row.get("order_id") or "")
            if not tag.startswith(str(ticket)):
                continue
            sl_working |= tag.endswith("_sl") or f"{ticket}_sl" in tag
            tp_working |= tag.endswith("_tp") or f"{ticket}_tp" in tag
        return sl_working, tp_working

    def _clear_chart(self, *, reason: str, exited_ticket: str = "") -> None:
        if self.rithmic is None or not self.rithmic.connected:
            return
        count = 0
        if exited_ticket:
            try:
                count += int(self.rithmic.cancel_entry_bracket_legs(exited_ticket, "MNQ", reason=reason, force=True) or 0)
            except Exception as exc:
                print(f"cancel ticket {exited_ticket}: {redact_secrets(exc)}", flush=True)
        try:
            count += int(self.rithmic.clear_chart_stops("MNQ", include_limits=True, rounds=4, keep_entry_ids=self._open_tickets(), only_if_flat=False) or 0)
        except Exception as exc:
            print(f"clear chart: {redact_secrets(exc)}", flush=True)
        if count:
            print(f"chart cleared ({reason}): cancelled {count} bot order(s)", flush=True)

    def _market_close_one(self, direction: str) -> bool:
        if self.rithmic is None or not self.rithmic.connected:
            return False
        try:
            broker_symbol, exchange = self.rithmic._resolve_symbol("MNQ")
            net_qty = 1 if direction == "long" else -1
            return bool(self.rithmic._run_sync(self.rithmic._async_submit_flatten_market_order(symbol="MNQ", broker_sym=broker_symbol, exchange=exchange, net_qty=net_qty), timeout=20))
        except Exception as exc:
            print(f"market close: {redact_secrets(exc)}", flush=True)
            return False

    def _broker_exit_hint(self, pos: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        ticket = str(pos.get("lucid_ticket") or "")
        if not ticket:
            return None
        try:
            age = time.time() - datetime.fromisoformat(str(pos.get("entry_time") or "").replace("Z", "+00:00")).timestamp()
        except Exception:
            age = 999.0
        if age < 12.0:
            return None
        sl_working, tp_working = self._working_leg_flags(ticket)
        if sl_working is None or (sl_working and tp_working):
            return None
        if not sl_working and not tp_working:
            return "BROKER", float(pos["entry"])
        if sl_working:
            return "TP", float(pos["tp"])
        return "SL", float(pos["sl"])

    def _exit(self, reason: str, price: float) -> None:
        pos = self.open_pos
        if pos is None:
            return
        ticket = str(pos.get("lucid_ticket") or "")
        sl_working, tp_working = self._working_leg_flags(ticket)
        if reason == "EOD" or (sl_working and tp_working):
            self._market_close_one(str(pos["direction"]))
            time.sleep(0.35)
        print(f"EXIT {reason} @ {price:.2f} ({pos['direction']} from {pos['entry']:.2f})", flush=True)
        self.open_pos = None
        self._clear_chart(reason=f"exit_{reason.lower()}", exited_ticket=ticket)
        self.log.record(
            "exit", title=f"OPENTREND EXIT {reason}",
            lines=[f"{pos['direction']} @ {price:.2f}", f"Lucid {ticket}"],
            symbol="MNQ", paper=True, session=SESSION_NAME,
        )

    def _sweep_orphans(self) -> None:
        if self.rithmic is None or not self.rithmic.connected or time.time() - self.last_orphan_sweep < 10:
            return
        self.last_orphan_sweep = time.time()
        live = self._open_tickets()
        if not live:
            self._clear_chart(reason="flat_sweep")
            return
        try:
            count = int(self.rithmic.cancel_stale_bot_brackets("MNQ", live_entry_ids=live) or 0)
            if count:
                print(f"orphan sweep cancelled {count} bot order(s)", flush=True)
        except Exception as exc:
            print(f"orphan sweep: {redact_secrets(exc)}", flush=True)

    def _heartbeat(self, bars: Optional[pd.DataFrame], note: str) -> None:
        now = time.time()
        if self.open_pos is None and now - self.last_heartbeat < HEARTBEAT_SEC:
            return
        self.last_heartbeat = now
        price = _last_price(self.feed, bars)
        stamp = datetime.now(ET).strftime("%H:%M ET")
        if self.open_pos:
            pos = self.open_pos
            text = f"open {pos['direction']} @{pos['entry']:.2f} SL{pos['sl']:.2f} TP{pos['tp']:.2f}"
        else:
            text = f"flat | {note}"
        price_text = f"{price:.2f}" if price is not None else "n/a"
        print(f"{stamp} MNQ {price_text} | {text}", flush=True)
        write_status({"bot_running": True, "paper_mode": True, "session": SESSION_NAME, "open_trend": True, "open": bool(self.open_pos), "note": note, "last_price": price})

    def run(self, *, duration_minutes: int = 0) -> None:
        print("============================================")
        print("  BottTrader - MNQ OPENTREND PAPER (Lucid TEST)")
        print("  5m EMA9/EMA21 crossover | ATR14 1.5R SL / 2R TP")
        print("  Long + short | RTH entries | flatten 15:55 ET")
        print("  PAPER / TEST only - ALLOW_LIVE=false")
        print("  OpenTrend is an original open-rules bot, not Trend Catcher")
        print("============================================")
        print("Leave this window open. Ctrl+C to stop.\n")
        self._connect()
        deadline = time.time() + duration_minutes * 60 if duration_minutes > 0 else None
        try:
            while True:
                if deadline and time.time() >= deadline:
                    break
                now = datetime.now(ET)
                bars = self._get_5m()
                done = closed_bars(bars, now=now) if bars is not None else pd.DataFrame()
                latest = done.iloc[-1] if not done.empty else None
                latest_key = str(latest["datetime"]) if latest is not None else ""
                exited = False
                if self.open_pos is not None:
                    broker_hit = self._broker_exit_hint(self.open_pos)
                    hit = broker_hit
                    if not hit and latest is not None and latest_key != self.last_managed_bar:
                        self.last_managed_bar = latest_key
                        hit = check_bar_exit(self.open_pos, float(latest["high"]), float(latest["low"]), float(latest["close"]), flat_eod=should_flatten_rth(now))
                    if should_flatten_rth(now) and not hit:
                        hit = ("EOD", float(_last_price(self.feed, bars) or self.open_pos["entry"]))
                    if hit:
                        self._exit(hit[0], float(hit[1]))
                        exited = True
                self._sweep_orphans()
                if not exited and self.open_pos is None and entries_allowed_et(now) and not day_bot_owns_lucid(now) and bars is not None:
                    sig = signal_on_closed_bar(bars, now=now)
                    signal_bar = str(sig.get("signal_bar") or "") if sig else ""
                    if sig and signal_bar != self.last_signal_bar:
                        self._place(sig, _last_price(self.feed, bars))
                    self._heartbeat(bars, "RTH scanning EMA crossover")
                elif should_flatten_rth(now):
                    self._heartbeat(bars, "flatten window / outside entry window")
                elif not is_rth_et(now):
                    self._heartbeat(bars, "outside RTH")
                elif day_bot_owns_lucid(now):
                    self._heartbeat(bars, "waiting - another bot owns Lucid")
                else:
                    self._heartbeat(bars, "in trade")
                time.sleep(SCAN_SEC)
        except KeyboardInterrupt:
            print("\nStopping OpenTrend paper", flush=True)
        finally:
            if self.open_pos is not None:
                self._exit("SHUTDOWN", float(_last_price(self.feed) or self.open_pos["entry"]))
            if self.rithmic and self.rithmic.connected:
                try:
                    self._clear_chart(reason="shutdown")
                    self.rithmic.shutdown()
                except Exception:
                    pass
            write_status({"bot_running": False, "session": SESSION_NAME})


def run_open_trend_paper(*, duration_minutes: int = 0) -> None:
    OpenTrendPaperEngine().run(duration_minutes=duration_minutes)

