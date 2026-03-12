#!/usr/bin/env python
"""
Live Clean Scalper - Real-time trading with Rithmic

This script runs the optimized clean scalper strategy in live mode.
Connects to Rithmic via async_rithmic for real-time data and execution.

PAPER TRADING MODE BY DEFAULT - set LIVE_MODE=true to enable real orders

Usage:
    python scripts/live_clean_scalper.py           # Paper mode
    LIVE_MODE=true python scripts/live_clean_scalper.py   # Live mode
"""

import asyncio
import os
import sys
import signal
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Dict, Optional
import pandas as pd

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.clean_scalper import CleanScalper
from src.utils.logger import bot_logger


class LiveCleanScalper:
    """Live trading wrapper for the clean scalper strategy."""

    def __init__(self, pairs: list = None, paper_mode: bool = True):
        self.pairs = pairs or ['MES', 'MNQ']
        self.paper_mode = paper_mode
        self.strategy = CleanScalper()
        
        # Candle buffers - store last 300 candles for each timeframe
        self.bars_1m: Dict[str, deque] = {p: deque(maxlen=350) for p in self.pairs}
        self.bars_5m: Dict[str, deque] = {p: deque(maxlen=260) for p in self.pairs}
        
        # Position tracking
        self.positions: Dict[str, dict] = {}  # symbol -> position info
        self.daily_trades = 0
        self.max_daily_trades = 20
        
        # Session control
        self.running = False
        self.last_signal_time: Dict[str, datetime] = {}
        self.cooldown_seconds = 60  # 1 min between signals per symbol
        
        # Rithmic client (lazy init)
        self._client = None
        self._account_id = None
        
        # Stats
        self.signals_generated = 0
        self.orders_placed = 0
        self.paper_pnl = 0.0

    async def connect(self):
        """Connect to Rithmic."""
        from async_rithmic import RithmicClient
        
        user = os.getenv("RITHMIC_USER_ID", "")
        password = os.getenv("RITHMIC_PASSWORD", "")
        system = os.getenv("RITHMIC_SYSTEM", "LucidTrading")
        gateway = os.getenv("RITHMIC_GATEWAY", "wss://rprotocol.rithmic.com:443")
        
        if not user or not password:
            raise ValueError("RITHMIC_USER_ID and RITHMIC_PASSWORD required")
        
        bot_logger.info(f"Connecting to Rithmic: system={system}, gateway={gateway}")
        
        self._client = RithmicClient(
            user=user,
            password=password,
            system_name=system,
            app_name="CleanScalper",
            app_version="1.0",
            url=gateway,
        )
        
        await self._client.connect()
        bot_logger.info("Connected to Rithmic")
        
        # Get account
        accounts = await self._client.list_accounts()
        if accounts:
            self._account_id = accounts[0].account_id
            bot_logger.info(f"Using account: {self._account_id}")
        else:
            raise ValueError("No accounts available")

    async def subscribe_data(self):
        """Subscribe to market data for all pairs."""
        from async_rithmic import TimeBarType
        
        for pair in self.pairs:
            rith_symbol = self._get_rithmic_symbol(pair)
            exchange = "CME"
            
            bot_logger.info(f"Subscribing to {rith_symbol} ({exchange})")
            
            # Subscribe to 1-minute bars
            await self._client.subscribe_to_time_bar_data(
                symbol=rith_symbol,
                exchange=exchange,
                bar_type=TimeBarType.MINUTE_BAR,
                bar_type_periods=1,
            )
            
            # Subscribe to 5-minute bars
            await self._client.subscribe_to_time_bar_data(
                symbol=rith_symbol,
                exchange=exchange,
                bar_type=TimeBarType.MINUTE_BAR,
                bar_type_periods=5,
            )
            
            bot_logger.info(f"Subscribed to {pair} 1m and 5m bars")

    async def load_historical_data(self):
        """Load historical candles to warm up indicators."""
        from async_rithmic import TimeBarType
        
        end_time = datetime.now(timezone.utc)
        
        for pair in self.pairs:
            rith_symbol = self._get_rithmic_symbol(pair)
            
            # Load 1m candles (last 6 hours = 360 bars)
            start_1m = end_time - timedelta(hours=6)
            try:
                bars_1m = await self._client.get_historical_time_bars(
                    symbol=rith_symbol,
                    exchange="CME",
                    bar_type=TimeBarType.MINUTE_BAR,
                    bar_type_periods=1,
                    start_time=start_1m,
                    end_time=end_time,
                )
                for bar in bars_1m:
                    self.bars_1m[pair].append({
                        'datetime': bar.get('bar_end_time') or bar.get('timestamp'),
                        'open': bar.get('open_price') or bar.get('open'),
                        'high': bar.get('high_price') or bar.get('high'),
                        'low': bar.get('low_price') or bar.get('low'),
                        'close': bar.get('close_price') or bar.get('close'),
                        'volume': bar.get('volume', 0),
                    })
                bot_logger.info(f"Loaded {len(self.bars_1m[pair])} 1m bars for {pair}")
            except Exception as e:
                bot_logger.error(f"Failed to load 1m history for {pair}: {e}")
            
            # Load 5m candles (last 24 hours = 288 bars)
            start_5m = end_time - timedelta(hours=24)
            try:
                bars_5m = await self._client.get_historical_time_bars(
                    symbol=rith_symbol,
                    exchange="CME",
                    bar_type=TimeBarType.MINUTE_BAR,
                    bar_type_periods=5,
                    start_time=start_5m,
                    end_time=end_time,
                )
                for bar in bars_5m:
                    self.bars_5m[pair].append({
                        'datetime': bar.get('bar_end_time') or bar.get('timestamp'),
                        'open': bar.get('open_price') or bar.get('open'),
                        'high': bar.get('high_price') or bar.get('high'),
                        'low': bar.get('low_price') or bar.get('low'),
                        'close': bar.get('close_price') or bar.get('close'),
                        'volume': bar.get('volume', 0),
                    })
                bot_logger.info(f"Loaded {len(self.bars_5m[pair])} 5m bars for {pair}")
            except Exception as e:
                bot_logger.error(f"Failed to load 5m history for {pair}: {e}")

    def _get_rithmic_symbol(self, pair: str) -> str:
        """Convert our symbol to Rithmic format with front month."""
        # For now, use the generic symbols - Rithmic resolves to front month
        return pair

    def _bars_to_df(self, bars: deque) -> Optional[pd.DataFrame]:
        """Convert bar buffer to DataFrame."""
        if len(bars) < 210:
            return None
        df = pd.DataFrame(list(bars))
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    async def on_bar(self, symbol: str, timeframe: int, bar: dict):
        """Handle incoming bar data."""
        pair = symbol.replace("MESH6", "MES").replace("MNQH6", "MNQ")  # Normalize
        
        if pair not in self.pairs:
            return
        
        if timeframe == 1:
            self.bars_1m[pair].append(bar)
        elif timeframe == 5:
            self.bars_5m[pair].append(bar)
            # Check signal on 5m bar close
            await self.check_signal(pair)

    async def check_signal(self, pair: str):
        """Check for trading signal on this pair."""
        # Check cooldown
        now = datetime.now(timezone.utc)
        last = self.last_signal_time.get(pair)
        if last and (now - last).total_seconds() < self.cooldown_seconds:
            return
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return
        
        # Check existing position
        if pair in self.positions:
            return
        
        # Build DataFrames
        df_1m = self._bars_to_df(self.bars_1m[pair])
        df_5m = self._bars_to_df(self.bars_5m[pair])
        
        if df_1m is None or df_5m is None:
            return
        
        # Get current hour (UTC)
        candle_hour = now.hour
        
        # Generate signal
        signal = self.strategy.get_signal(
            df_1m, pair,
            df_5m=df_5m,
            candle_hour=candle_hour,
            precalculated=False
        )
        
        if signal['signal'] in ('BUY', 'SELL'):
            self.signals_generated += 1
            self.last_signal_time[pair] = now
            
            sl_tp = signal['sl_tp']
            bot_logger.info(
                f"🎯 SIGNAL: {pair} {signal['signal']} | "
                f"Conf={signal['confidence']:.2f} | "
                f"Entry={sl_tp['entry_price']:.2f} | "
                f"SL={sl_tp['stop_loss']:.2f} | TP={sl_tp['take_profit']:.2f}"
            )
            
            if self.paper_mode:
                await self.paper_trade(pair, signal)
            else:
                await self.execute_trade(pair, signal)

    async def paper_trade(self, pair: str, signal: dict):
        """Simulate trade execution for paper trading."""
        sl_tp = signal['sl_tp']
        direction = signal['signal']
        
        self.positions[pair] = {
            'direction': direction,
            'entry_price': sl_tp['entry_price'],
            'stop_loss': sl_tp['stop_loss'],
            'take_profit': sl_tp['take_profit'],
            'entry_time': datetime.now(timezone.utc),
            'contracts': 1,
        }
        
        self.daily_trades += 1
        self.orders_placed += 1
        
        bot_logger.info(
            f"📝 PAPER TRADE: {pair} {direction} @ {sl_tp['entry_price']:.2f} | "
            f"SL={sl_tp['stop_loss']:.2f} TP={sl_tp['take_profit']:.2f}"
        )

    async def execute_trade(self, pair: str, signal: dict):
        """Execute real trade via Rithmic."""
        from async_rithmic import OrderType, TransactionType
        
        sl_tp = signal['sl_tp']
        direction = signal['signal']
        rith_symbol = self._get_rithmic_symbol(pair)
        
        try:
            # Place bracket order
            tx_type = TransactionType.BUY if direction == 'BUY' else TransactionType.SELL
            
            order = await self._client.submit_order(
                symbol=rith_symbol,
                exchange="CME",
                account_id=self._account_id,
                transaction_type=tx_type,
                order_type=OrderType.MARKET,
                quantity=1,
            )
            
            if order:
                # Place stop loss
                sl_tx = TransactionType.SELL if direction == 'BUY' else TransactionType.BUY
                await self._client.submit_order(
                    symbol=rith_symbol,
                    exchange="CME",
                    account_id=self._account_id,
                    transaction_type=sl_tx,
                    order_type=OrderType.STOP_LIMIT,
                    quantity=1,
                    price=sl_tp['stop_loss'],
                    trigger_price=sl_tp['stop_loss'],
                )
                
                # Place take profit
                await self._client.submit_order(
                    symbol=rith_symbol,
                    exchange="CME",
                    account_id=self._account_id,
                    transaction_type=sl_tx,
                    order_type=OrderType.LIMIT,
                    quantity=1,
                    price=sl_tp['take_profit'],
                )
                
                self.positions[pair] = {
                    'direction': direction,
                    'entry_price': sl_tp['entry_price'],
                    'stop_loss': sl_tp['stop_loss'],
                    'take_profit': sl_tp['take_profit'],
                    'entry_time': datetime.now(timezone.utc),
                    'contracts': 1,
                }
                
                self.daily_trades += 1
                self.orders_placed += 1
                
                bot_logger.info(f"✅ LIVE ORDER: {pair} {direction} filled")
        except Exception as e:
            bot_logger.error(f"❌ Order failed for {pair}: {e}")

    async def check_paper_positions(self):
        """Check paper positions for SL/TP hits."""
        for pair, pos in list(self.positions.items()):
            if len(self.bars_1m[pair]) == 0:
                continue
            
            last_bar = self.bars_1m[pair][-1]
            high = last_bar['high']
            low = last_bar['low']
            
            exit_price = None
            exit_type = None
            
            if pos['direction'] == 'BUY':
                if low <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    exit_type = 'SL'
                elif high >= pos['take_profit']:
                    exit_price = pos['take_profit']
                    exit_type = 'TP'
            else:  # SELL
                if high >= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    exit_type = 'SL'
                elif low <= pos['take_profit']:
                    exit_price = pos['take_profit']
                    exit_type = 'TP'
            
            if exit_price:
                config = self.strategy.INSTRUMENT_CONFIG.get(pair, {})
                tick_size = config.get('tick_size', 0.25)
                tick_value = config.get('tick_value', 1.25)
                
                if pos['direction'] == 'BUY':
                    ticks = (exit_price - pos['entry_price']) / tick_size
                else:
                    ticks = (pos['entry_price'] - exit_price) / tick_size
                
                pnl = ticks * tick_value - 0.62  # minus commission
                self.paper_pnl += pnl
                
                emoji = "✅" if exit_type == 'TP' else "❌"
                bot_logger.info(
                    f"{emoji} PAPER EXIT: {pair} {exit_type} | "
                    f"P&L=${pnl:.2f} | Total=${self.paper_pnl:.2f}"
                )
                
                del self.positions[pair]

    async def run(self):
        """Main loop."""
        self.running = True
        
        mode = "PAPER" if self.paper_mode else "LIVE"
        bot_logger.info(f"Starting Clean Scalper - {mode} MODE")
        bot_logger.info(f"Pairs: {self.pairs}")
        
        # Connect and subscribe
        await self.connect()
        await self.load_historical_data()
        await self.subscribe_data()
        
        bot_logger.info("Listening for market data...")
        
        # Set up event handler for incoming bars
        bar_queue = asyncio.Queue()
        
        def on_bar_received(bar):
            bar_queue.put_nowait(bar)
        
        self._client.on_time_bar += on_bar_received
        
        # Process incoming data
        async def process_bars():
            while self.running:
                try:
                    bar = await asyncio.wait_for(bar_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Determine timeframe from bar
                bar_dict = {
                    'datetime': bar.bar_end_time,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                }
                
                # Route to appropriate handler
                await self.on_bar(bar.symbol, bar.type_periods, bar_dict)
                
                # Check paper positions
                if self.paper_mode:
                    await self.check_paper_positions()
        
        # Status logging every 5 minutes
        async def status_loop():
            while self.running:
                await asyncio.sleep(300)
                bot_logger.info(
                    f"📊 Status: Signals={self.signals_generated} | "
                    f"Orders={self.orders_placed} | "
                    f"Positions={len(self.positions)} | "
                    f"Paper P&L=${self.paper_pnl:.2f}"
                )
        
        try:
            await asyncio.gather(
                process_bars(),
                status_loop(),
            )
        except asyncio.CancelledError:
            bot_logger.info("Shutting down...")
        finally:
            if self._client:
                await self._client.disconnect()

    def stop(self):
        """Stop the live trader."""
        self.running = False


async def main():
    paper_mode = os.getenv("LIVE_MODE", "").lower() != "true"
    pairs = ['MES', 'MNQ']
    
    trader = LiveCleanScalper(pairs=pairs, paper_mode=paper_mode)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        bot_logger.info("Received shutdown signal")
        trader.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await trader.run()


if __name__ == "__main__":
    print("=" * 70)
    print("Clean Scalper - Live Trading")
    print("=" * 70)
    print()
    print("Environment Variables Required:")
    print("  RITHMIC_USER_ID    - Your Rithmic username")
    print("  RITHMIC_PASSWORD   - Your Rithmic password")
    print("  RITHMIC_SYSTEM     - System name (default: LucidTrading)")
    print("  RITHMIC_GATEWAY    - Gateway URL (default: wss://rprotocol.rithmic.com:443)")
    print()
    print("Optional:")
    print("  LIVE_MODE=true     - Enable real order execution (default: paper)")
    print()
    
    asyncio.run(main())
