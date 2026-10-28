"""
Dynamic Stop-Loss / Take-Profit Manager

Features ported from Binance-Futures-Trading-Bot:
  - SetSLTP: Dynamic SL/TP based on swing highs/lows
  - Trailing stop management
  - ATR-based TP scaling
  - Position monitoring

Source: github.com/conor19w/Binance-Futures-Trading-Bot
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from src.utils.logger import bot_logger


class DynamicSLTPManager:
    """
    Manages dynamic Stop Loss and Take Profit levels.
    
    Uses swing points and ATR for intelligent SL/TP placement.
    """
    
    def __init__(
        self,
        use_trailing_stop: bool = True,
        trailing_callback: float = 0.002,  # 0.2% callback
        atr_multiplier_sl: float = 1.5,
        atr_multiplier_tp: float = 2.5,
        lookback_period: int = 30
    ):
        self.use_trailing_stop = use_trailing_stop
        self.trailing_callback = trailing_callback
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.lookback_period = lookback_period
        
        # Track active positions for trailing stop
        self.positions: Dict[str, Dict] = {}
        
    def calculate_sl_tp(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        symbol: str = None
    ) -> Dict:
        """
        Calculate optimal SL and TP levels.
        
        Uses swing highs/lows for SL placement and ATR for TP scaling.
        
        Args:
            df: OHLCV DataFrame
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            symbol: Trading symbol
            
        Returns:
            Dict with sl_price, tp_price, and details
        """
        if len(df) < self.lookback_period + 5:
            return self._fallback_sl_tp(entry_price, direction, df)
            
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Find swing points
        swing_high, swing_low = self._find_swing_points(df)
        
        # Calculate SL based on direction
        if direction == 'BUY':
            # For longs, SL below recent swing low
            sl_from_swing = swing_low - (0.001 * entry_price)  # Small buffer
            sl_from_atr = entry_price - (atr * self.atr_multiplier_sl)
            
            # Use the tighter of the two (but not too tight)
            sl_price = max(sl_from_swing, sl_from_atr)
            
            # TP based on ATR multiple of risk
            risk = entry_price - sl_price
            tp_price = entry_price + (risk * 2.0)  # 2:1 R:R minimum
            
            # Alternative: ATR-based TP
            tp_from_atr = entry_price + (atr * self.atr_multiplier_tp)
            if tp_from_atr > tp_price:
                tp_price = tp_from_atr
                
        else:  # SELL
            # For shorts, SL above recent swing high
            sl_from_swing = swing_high + (0.001 * entry_price)
            sl_from_atr = entry_price + (atr * self.atr_multiplier_sl)
            
            # Use the tighter of the two
            sl_price = min(sl_from_swing, sl_from_atr)
            
            # TP based on ATR
            risk = sl_price - entry_price
            tp_price = entry_price - (risk * 2.0)
            
            tp_from_atr = entry_price - (atr * self.atr_multiplier_tp)
            if tp_from_atr < tp_price:
                tp_price = tp_from_atr
                
        # Calculate risk metrics
        if direction == 'BUY':
            if sl_price >= entry_price:
                sl_price = entry_price - max(atr * self.atr_multiplier_sl, entry_price * 0.001)
            if tp_price <= entry_price:
                tp_price = entry_price + max((entry_price - sl_price) * 2.0, atr * self.atr_multiplier_tp)
        else:
            if sl_price <= entry_price:
                sl_price = entry_price + max(atr * self.atr_multiplier_sl, entry_price * 0.001)
            if tp_price >= entry_price:
                tp_price = entry_price - max((sl_price - entry_price) * 2.0, atr * self.atr_multiplier_tp)

        sl_distance = abs(entry_price - sl_price)
        tp_distance = abs(tp_price - entry_price)
        risk_reward = tp_distance / sl_distance if sl_distance > 0 else 1.0
        
        return {
            'sl_price': round(sl_price, 5),
            'tp_price': round(tp_price, 5),
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'risk_reward': round(risk_reward, 2),
            'atr': atr,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'method': 'dynamic'
        }
        
    def _fallback_sl_tp(
        self,
        entry_price: float,
        direction: str,
        df: pd.DataFrame
    ) -> Dict:
        """Fallback SL/TP when insufficient data."""
        # Use simple percentage-based SL/TP
        sl_pct = 0.005  # 0.5%
        tp_pct = 0.010  # 1.0%
        
        if direction == 'BUY':
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)
            
        return {
            'sl_price': round(sl_price, 5),
            'tp_price': round(tp_price, 5),
            'sl_distance': abs(entry_price - sl_price),
            'tp_distance': abs(tp_price - entry_price),
            'risk_reward': 2.0,
            'atr': None,
            'method': 'fallback'
        }
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if 'atr' in df.columns:
            return df['atr'].iloc[-1]
            
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = []
        for i in range(1, len(df)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr.append(max(hl, hc, lc))
            
        if len(tr) < period:
            return np.mean(tr) if tr else 0.0
            
        return np.mean(tr[-period:])
        
    def _find_swing_points(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Find recent swing high and swing low."""
        high = df['high'].values[-self.lookback_period:]
        low = df['low'].values[-self.lookback_period:]
        
        # Find swing highs (local maxima)
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(high) - 2):
            # Swing high: higher than 2 bars before and after
            if high[i] > high[i-1] and high[i] > high[i+1]:
                if high[i] > high[i-2] and high[i] > high[i+2]:
                    swing_highs.append(high[i])
                    
            # Swing low: lower than 2 bars before and after
            if low[i] < low[i-1] and low[i] < low[i+1]:
                if low[i] < low[i-2] and low[i] < low[i+2]:
                    swing_lows.append(low[i])
                    
        # Get most recent significant swing points
        if swing_highs:
            recent_swing_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
        else:
            recent_swing_high = max(high)
            
        if swing_lows:
            recent_swing_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
        else:
            recent_swing_low = min(low)
            
        return recent_swing_high, recent_swing_low
        
    def start_tracking(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        quantity: float
    ):
        """Start tracking a position for trailing stop."""
        self.positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'quantity': quantity,
            'highest_price': entry_price if direction == 'BUY' else entry_price,
            'lowest_price': entry_price if direction == 'SELL' else entry_price,
            'trailing_active': False
        }
        bot_logger.info(f"Started tracking {symbol} {direction} @ {entry_price}")
        
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Dict]:
        """
        Update trailing stop for a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Dict with action if SL needs updating
        """
        if symbol not in self.positions:
            return None
            
        if not self.use_trailing_stop:
            return None
            
        pos = self.positions[symbol]
        direction = pos['direction']
        entry = pos['entry_price']
        
        action = None
        
        if direction == 'BUY':
            # Update highest price
            if current_price > pos['highest_price']:
                pos['highest_price'] = current_price
                
            # Activate trailing after reaching breakeven
            if current_price >= entry * 1.003:  # 0.3% profit minimum
                pos['trailing_active'] = True
                
            if pos['trailing_active']:
                # Calculate new trailing stop
                new_sl = pos['highest_price'] * (1 - self.trailing_callback)
                
                # Only move SL up, never down
                if new_sl > pos['sl_price']:
                    old_sl = pos['sl_price']
                    pos['sl_price'] = new_sl
                    action = {
                        'action': 'UPDATE_SL',
                        'symbol': symbol,
                        'old_sl': old_sl,
                        'new_sl': new_sl,
                        'current_price': current_price
                    }
                    bot_logger.info(f"Trailing SL moved: {old_sl:.5f} -> {new_sl:.5f}")
                    
        else:  # SELL (short)
            # Update lowest price
            if current_price < pos['lowest_price']:
                pos['lowest_price'] = current_price
                
            # Activate trailing after reaching breakeven
            if current_price <= entry * 0.997:
                pos['trailing_active'] = True
                
            if pos['trailing_active']:
                # Calculate new trailing stop (move down)
                new_sl = pos['lowest_price'] * (1 + self.trailing_callback)
                
                # Only move SL down, never up
                if new_sl < pos['sl_price']:
                    old_sl = pos['sl_price']
                    pos['sl_price'] = new_sl
                    action = {
                        'action': 'UPDATE_SL',
                        'symbol': symbol,
                        'old_sl': old_sl,
                        'new_sl': new_sl,
                        'current_price': current_price
                    }
                    bot_logger.info(f"Trailing SL moved: {old_sl:.5f} -> {new_sl:.5f}")
                    
        return action
        
    def check_exit(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Dict]:
        """
        Check if position should exit (SL or TP hit).
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Dict with exit action if triggered
        """
        if symbol not in self.positions:
            return None
            
        pos = self.positions[symbol]
        direction = pos['direction']
        sl_price = pos['sl_price']
        tp_price = pos['tp_price']
        entry = pos['entry_price']
        
        if direction == 'BUY':
            # Check SL
            if current_price <= sl_price:
                pnl = current_price - entry
                pnl_pct = (pnl / entry) * 100
                return {
                    'action': 'EXIT',
                    'reason': 'STOP_LOSS',
                    'symbol': symbol,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                
            # Check TP
            if current_price >= tp_price:
                pnl = current_price - entry
                pnl_pct = (pnl / entry) * 100
                return {
                    'action': 'EXIT',
                    'reason': 'TAKE_PROFIT',
                    'symbol': symbol,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                
        else:  # SELL (short)
            # Check SL
            if current_price >= sl_price:
                pnl = entry - current_price
                pnl_pct = (pnl / entry) * 100
                return {
                    'action': 'EXIT',
                    'reason': 'STOP_LOSS',
                    'symbol': symbol,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                
            # Check TP
            if current_price <= tp_price:
                pnl = entry - current_price
                pnl_pct = (pnl / entry) * 100
                return {
                    'action': 'EXIT',
                    'reason': 'TAKE_PROFIT',
                    'symbol': symbol,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                
        return None
        
    def stop_tracking(self, symbol: str):
        """Stop tracking a position."""
        if symbol in self.positions:
            del self.positions[symbol]
            bot_logger.info(f"Stopped tracking {symbol}")
            
    def get_position_status(self, symbol: str) -> Optional[Dict]:
        """Get current position status."""
        if symbol not in self.positions:
            return None
        return self.positions[symbol].copy()
        
    def breakeven_stop(
        self,
        symbol: str,
        current_price: float,
        breakeven_trigger: float = 0.005  # 0.5% profit to move to BE
    ) -> Optional[Dict]:
        """
        Move stop to breakeven after reaching profit threshold.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            breakeven_trigger: Profit % to trigger breakeven
            
        Returns:
            Dict with action if BE triggered
        """
        if symbol not in self.positions:
            return None
            
        pos = self.positions[symbol]
        direction = pos['direction']
        entry = pos['entry_price']
        sl = pos['sl_price']
        
        # Check if already at or past breakeven
        if direction == 'BUY':
            if sl >= entry:
                return None
                
            pnl_pct = (current_price - entry) / entry
            if pnl_pct >= breakeven_trigger:
                # Move SL to entry + small buffer
                new_sl = entry * 1.0001  # Tiny buffer for fees
                pos['sl_price'] = new_sl
                return {
                    'action': 'BREAKEVEN',
                    'symbol': symbol,
                    'old_sl': sl,
                    'new_sl': new_sl
                }
                
        else:  # SELL
            if sl <= entry:
                return None
                
            pnl_pct = (entry - current_price) / entry
            if pnl_pct >= breakeven_trigger:
                new_sl = entry * 0.9999
                pos['sl_price'] = new_sl
                return {
                    'action': 'BREAKEVEN',
                    'symbol': symbol,
                    'old_sl': sl,
                    'new_sl': new_sl
                }
                
        return None


# Global instance for convenience
_sltp_manager = None

def get_sltp_manager(**kwargs) -> DynamicSLTPManager:
    """Get or create the global SLTP manager."""
    global _sltp_manager
    if _sltp_manager is None:
        _sltp_manager = DynamicSLTPManager(**kwargs)
    return _sltp_manager


def calculate_dynamic_sl_tp(
    df: pd.DataFrame,
    direction: str,
    entry_price: float
) -> Dict:
    """Convenience function for quick SL/TP calculation."""
    manager = get_sltp_manager()
    return manager.calculate_sl_tp(df, direction, entry_price)
