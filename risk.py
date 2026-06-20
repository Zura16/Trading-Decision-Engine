import numpy as np
from config import logger, TRADING_PARAMETERS

class RiskManager:
    """Manages trading risk: positions, stop-loss levels, sizing, and global halts."""
    def __init__(self):
        self.params = TRADING_PARAMETERS
        self.peak_equity = self.params["initial_capital"]
        self.halted = False

    def check_drawdown(self, current_equity: float) -> bool:
        """Tracks peak equity and triggers a global halt if drawdown limit is breached."""
        if self.halted:
            return True
            
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown >= self.params["max_drawdown_limit"]:
            self.halted = True
            logger.warning(
                f"GLOBAL RISK HALT TRIGGERED: Max drawdown limit breached! "
                f"Peak Equity: ${self.peak_equity:,.2f}, Current Equity: ${current_equity:,.2f}, "
                f"Drawdown: {drawdown * 100:.2f}% (Limit: {self.params['max_drawdown_limit'] * 100:.2f}%)"
            )
            
        return self.halted

    def calculate_position_size(
        self, 
        signal: int, 
        prob_up: float, 
        prob_down: float, 
        current_equity: float, 
        current_price: float, 
        atr: float = None
    ) -> float:
        """
        Calculates optimal position sizing (shares count) using Kelly Criterion and ATR volatility adjustment.
        """
        if self.halted:
            logger.info("Risk Manager halted: Trade sizing returned 0.")
            return 0.0

        if signal == 0:
            return 0.0

        # 1. Kelly Sizing Fraction
        # f* = p - (1-p)/b
        # Let b = take_profit / stop_loss ratio (default = 0.06 / 0.02 = 3)
        tp_pct = self.params["take_profit_pct"]
        sl_pct = self.params["stop_loss_pct"]
        b = tp_pct / sl_pct
        
        p = prob_up if signal == 1 else prob_down
        q = 1.0 - p
        
        kelly_fraction = p - (q / b)
        
        # Apply fractional Kelly (e.g., quarter-Kelly) for capital preservation
        fractional_kelly = max(0.0, kelly_fraction * 0.25)
        
        # 2. ATR Volatility-Adjusted Risk Sizing
        # Risk 1% of equity per trade
        risk_pct = 0.01 
        risk_amount = current_equity * risk_pct
        
        if atr is not None and atr > 0:
            # Volatility-based stop distance: 2 * ATR
            stop_distance = atr * 2.0
            # Ensure stop distance is not insanely small
            stop_distance = max(stop_distance, current_price * 0.005)
        else:
            # Fallback stop distance: stop_loss_pct * current_price
            stop_distance = current_price * sl_pct
            
        vol_adjusted_qty = risk_amount / stop_distance

        # 3. Capital allocation limits
        # Find maximum allowed dollar size per trade (e.g. 20% of equity)
        max_alloc_pct = self.params["default_position_size"] * 2.0  # Cap at 20%
        max_dollar_alloc = current_equity * max_alloc_pct
        max_qty = max_dollar_alloc / current_price

        # Kelly-scaled target qty
        kelly_qty = (current_equity * fractional_kelly) / current_price if fractional_kelly > 0 else 0.0

        # Blend both sizing mechanisms
        # Use volatility sizing as baseline, but scale it down if Kelly confidence is low
        if kelly_qty > 0:
            # Take the minimum of vol-adjusted and Kelly-adjusted size
            target_qty = min(vol_adjusted_qty, kelly_qty, max_qty)
        else:
            # Fallback to standard 10% sizing scaled down if no Kelly probability
            target_qty = min(vol_adjusted_qty, max_qty)

        # Round down to 4 decimal places for precision/fractional shares
        target_qty = max(0.0, round(target_qty, 4))
        
        logger.debug(
            f"Sizing Calculation: Price={current_price:.2f}, Equity={current_equity:.2f}, "
            f"KellyFrac={fractional_kelly:.4f}, VolQty={vol_adjusted_qty:.2f}, "
            f"MaxQty={max_qty:.2f} -> TargetQty={target_qty}"
        )
        return target_qty

    def get_stops(self, side: str, entry_price: float, atr: float = None) -> tuple:
        """Returns (stop_loss, take_profit) prices based on direction and ATR/percentage."""
        sl_pct = self.params["stop_loss_pct"]
        tp_pct = self.params["take_profit_pct"]
        
        if atr is not None and atr > 0:
            # Use 2 * ATR for stop loss, 5 * ATR for take profit
            sl_dist = atr * 2.0
            tp_dist = atr * 5.0
        else:
            # Use fixed percentages
            sl_dist = entry_price * sl_pct
            tp_dist = entry_price * tp_pct
            
        if side == "BUY":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:  # SHORT/SELL
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist
            
        return round(stop_loss, 4), round(take_profit, 4)
