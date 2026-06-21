import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from database import MarketData, TradeStore, PortfolioState
from features import calculate_features_dataframe, FeaturePipeline
from model import XGBoostModel
from risk import RiskManager
from config import logger, TRADING_PARAMETERS

class BacktestEngine:
    """Event-driven backtesting engine that simulates trading over historical/simulated database records."""
    def __init__(self, symbols, initial_capital: float = None):
        self.symbols = symbols
        self.params = TRADING_PARAMETERS
        self.initial_capital = initial_capital or self.params["initial_capital"]
        
        # State
        self.cash = self.initial_capital
        self.positions = {s: 0.0 for s in symbols}  # Symbol -> Qty
        self.entry_prices = {s: 0.0 for s in symbols}  # Symbol -> Entry Price
        self.stop_losses = {s: None for s in symbols}  # Symbol -> Stop Loss Price
        self.take_profits = {s: None for s in symbols}  # Symbol -> Take Profit Price
        
        # Risk and Model
        self.risk_manager = RiskManager()
        self.model = XGBoostModel()
        self.model_loaded = False
        
        # Load model if it exists
        try:
            self.model.load()
            self.model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}. Backtester will use fallback rule-based strategy (RSI/MACD).")

    def run_backtest(self, db: Session) -> dict:
        """Runs the event-driven simulation loop over historical bar records."""
        logger.info(f"Running backtest for symbols {self.symbols}...")
        
        # Fetch all bar records (is_tick=False) ordered by time
        records = db.query(MarketData).filter(
            MarketData.symbol.in_(self.symbols),
            MarketData.is_tick == False
        ).order_by(MarketData.timestamp.asc()).all()
        
        if not records:
            logger.error("No historical bar records found in the database. Run historical ingestion first.")
            return {}

        # Convert to DataFrame to group by timestamp
        data_list = []
        for r in records:
            data_list.append({
                "timestamp": r.timestamp,
                "symbol": r.symbol,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "bid": r.bid,
                "ask": r.ask
            })
        df_all = pd.DataFrame(data_list)
        
        # Group by timestamp to process events in chronological order (cross-sectional step)
        timestamps = sorted(df_all["timestamp"].unique())
        
        # Pre-compute features for all symbols to avoid re-running calculations iteratively
        # In a real-time system, this is streaming; in backtests, we can optimize by pre-computing.
        logger.info("Pre-computing feature matrix for backtest efficiency...")
        feature_dfs = {}
        for s in self.symbols:
            symbol_df = df_all[df_all["symbol"] == s].copy()
            if len(symbol_df) >= 10:
                feature_dfs[s] = calculate_features_dataframe(symbol_df).set_index("timestamp")
            else:
                feature_dfs[s] = pd.DataFrame()

        # Trade and portfolio logging lists
        trades_to_save = []
        portfolio_history = []
        
        # Track metrics
        peak_equity = self.initial_capital
        equity_series = []
        
        logger.info(f"Starting event loop with {len(timestamps)} periods...")
        
        for ts in timestamps:
            dt_ts = pd.to_datetime(ts).to_pydatetime()
            
            # 1. Update Portfolio Valuation (Mark to Market)
            market_value = 0.0
            for symbol in self.symbols:
                qty = self.positions[symbol]
                if qty > 0:
                    # Find price at this timestamp
                    ts_data = df_all[(df_all["timestamp"] == ts) & (df_all["symbol"] == symbol)]
                    if not ts_data.empty:
                        close_price = ts_data.iloc[0]["close"]
                        market_value += qty * close_price
                        
            current_equity = self.cash + market_value
            
            # Check for risk halts
            if self.risk_manager.check_drawdown(current_equity):
                # Close all positions immediately
                for symbol in self.symbols:
                    qty = self.positions[symbol]
                    if qty > 0:
                        ts_data = df_all[(df_all["timestamp"] == ts) & (df_all["symbol"] == symbol)]
                        if not ts_data.empty:
                            close_price = ts_data.iloc[0]["close"]
                            self._execute_close(symbol, close_price, dt_ts, "RISK HALT CLOSE", trades_to_save)
                current_equity = self.cash
                break
                
            equity_series.append(current_equity)
            
            # 2. Check Stop Loss / Take Profit exits
            for symbol in self.symbols:
                qty = self.positions[symbol]
                if qty > 0:
                    ts_data = df_all[(df_all["timestamp"] == ts) & (df_all["symbol"] == symbol)]
                    if not ts_data.empty:
                        row = ts_data.iloc[0]
                        low_p = row["low"]
                        high_p = row["high"]
                        close_p = row["close"]
                        
                        # Stop Loss trigger (exits at stop level or close price, whichever is worse)
                        if self.stop_losses[symbol] and low_p <= self.stop_losses[symbol]:
                            exit_price = min(self.stop_losses[symbol], close_p)
                            self._execute_close(symbol, exit_price, dt_ts, "STOP LOSS", trades_to_save)
                        
                        # Take Profit trigger
                        elif self.take_profits[symbol] and high_p >= self.take_profits[symbol]:
                            exit_price = max(self.take_profits[symbol], close_p)
                            self._execute_close(symbol, exit_price, dt_ts, "TAKE PROFIT", trades_to_save)

            # 3. Model Inference & New Order Execution
            for symbol in self.symbols:
                # Get historical bar data up to this point for feature calculations
                if symbol not in feature_dfs or feature_dfs[symbol].empty:
                    continue
                    
                s_feat = feature_dfs[symbol]
                if ts not in s_feat.index:
                    continue
                    
                row_feat = s_feat.loc[ts]
                close_price = row_feat["close"]
                atr = row_feat.get("atr_14", 0.0)
                
                # Fetch ML signal or Fallback signal
                if self.model_loaded:
                    features_dict = row_feat.drop(["open", "high", "low", "close", "volume", "bid", "ask", "log_return"]).to_dict()
                    signal, p_down, p_neut, p_up = self.model.predict_prob(features_dict)
                else:
                    # Fallback technical rules:
                    # Buy if RSI < 35, Sell if RSI > 65
                    rsi = row_feat.get("rsi_14", 50.0)
                    p_up, p_neut, p_down = 0.33, 0.33, 0.33
                    if rsi < 35:
                        signal = 1
                        p_up = 0.60
                    elif rsi > 65:
                        signal = -1
                        p_down = 0.60
                    else:
                        signal = 0
                
                # Execution
                current_qty = self.positions[symbol]
                
                if signal == 1 and current_qty == 0:
                    # Calculate size
                    target_qty = self.risk_manager.calculate_position_size(
                        signal=1,
                        prob_up=p_up,
                        prob_down=p_down,
                        current_equity=current_equity,
                        current_price=close_price,
                        atr=atr
                    )
                    if target_qty > 0:
                        self._execute_open(symbol, close_price, target_qty, atr, dt_ts, trades_to_save)
                        
                elif signal == -1 and current_qty > 0:
                    # Close position
                    self._execute_close(symbol, close_price, dt_ts, "MODEL EXIT", trades_to_save)

            # Record portfolio state snapshot
            leverage = market_value / current_equity if current_equity > 0 else 0.0
            peak_equity = max(peak_equity, current_equity)
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
            
            portfolio_history.append(PortfolioState(
                timestamp=dt_ts,
                cash=self.cash,
                market_value=market_value,
                equity=current_equity,
                drawdown=drawdown,
                leverage=leverage
            ))

        # 4. Save simulation results to Database
        logger.info(f"Backtest execution complete. Writing {len(trades_to_save)} trades and {len(portfolio_history)} portfolio logs to DB...")
        
        db.query(TradeStore).delete()
        db.query(PortfolioState).delete()
        
        db.bulk_save_objects(trades_to_save)
        db.bulk_save_objects(portfolio_history)
        db.commit()
        
        # Calculate summary metrics
        metrics = self._calculate_metrics(equity_series, trades_to_save)
        logger.info(f"Backtest Metrics Summary: {metrics}")
        return metrics

    def _execute_open(self, symbol: str, price: float, qty: float, atr: float, ts: datetime, log_list: list):
        # Commission + slippage adjustment
        slip_pct = self.params["transaction_fee_pct"]
        exec_price = price * (1 + slip_pct)
        comm = exec_price * qty * slip_pct
        total_cost = (exec_price * qty) + comm
        
        if total_cost <= self.cash:
            self.cash -= total_cost
            self.positions[symbol] = qty
            self.entry_prices[symbol] = exec_price
            
            # Stops
            stop_loss, take_profit = self.risk_manager.get_stops("BUY", exec_price, atr)
            self.stop_losses[symbol] = stop_loss
            self.take_profits[symbol] = take_profit
            
            log_list.append(TradeStore(
                timestamp=ts,
                symbol=symbol,
                side="BUY",
                price=exec_price,
                qty=qty,
                value=exec_price * qty,
                slippage=price * slip_pct,
                commission=comm,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status="OPEN"
            ))
            logger.info(f"[{ts}] BUY {qty} {symbol} @ {exec_price:.2f} (SL: {stop_loss:.2f}, TP: {take_profit:.2f})")

    def _execute_close(self, symbol: str, price: float, ts: datetime, reason: str, log_list: list):
        qty = self.positions[symbol]
        if qty <= 0:
            return
            
        slip_pct = self.params["transaction_fee_pct"]
        exec_price = price * (1 - slip_pct)
        comm = exec_price * qty * slip_pct
        revenue = (exec_price * qty) - comm
        
        self.cash += revenue
        
        # Calculate PnL
        entry_val = self.entry_prices[symbol] * qty
        exit_val = exec_price * qty
        pnl = exit_val - entry_val - comm - (self.entry_prices[symbol] * qty * slip_pct) # PnL net of fees
        
        # Reset state
        self.positions[symbol] = 0.0
        self.entry_prices[symbol] = 0.0
        self.stop_losses[symbol] = None
        self.take_profits[symbol] = None
        
        log_list.append(TradeStore(
            timestamp=ts,
            symbol=symbol,
            side="SELL",
            price=exec_price,
            qty=qty,
            value=exec_price * qty,
            slippage=price * slip_pct,
            commission=comm,
            pnl=pnl,
            status="CLOSED"
        ))
        logger.info(f"[{ts}] SELL {qty} {symbol} @ {exec_price:.2f} | Reason: {reason} | PnL: ${pnl:.2f}")

    def _calculate_metrics(self, equity_series, closed_trades) -> dict:
        if not equity_series:
            return {}
            
        initial_eq = self.initial_capital
        final_eq = equity_series[-1]
        
        # Returns
        total_return = (final_eq - initial_eq) / initial_eq
        
        # Convert equity series to daily/bar returns to calculate risk ratios
        eq_df = pd.Series(equity_series)
        returns = eq_df.pct_change().dropna()
        
        # Sharpe ratio
        # Standard assumption: if daily/bar data, we annualize. We'll use a conservative annualization factor
        # Assuming 252 trading days per year
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        sharpe = 0.0
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        std_down = downside_returns.std()
        sortino = 0.0
        if std_down > 0:
            sortino = (mean_ret / std_down) * np.sqrt(252)
            
        # Max Drawdown
        roll_max = eq_df.cummax()
        drawdowns = (roll_max - eq_df) / roll_max
        max_dd = drawdowns.max()

        # Trade metrics
        closed_sells = [t for t in closed_trades if t.side == "SELL"]
        win_rate = 0.0
        profit_factor = 0.0
        
        if closed_sells:
            wins = [t for t in closed_sells if t.pnl > 0]
            losses = [t for t in closed_sells if t.pnl <= 0]
            
            win_rate = len(wins) / len(closed_sells)
            
            gross_profits = sum([t.pnl for t in wins])
            gross_losses = abs(sum([t.pnl for t in losses]))
            
            if gross_losses > 0:
                profit_factor = gross_profits / gross_losses
            else:
                profit_factor = gross_profits if gross_profits > 0 else 1.0

        return {
            "initial_equity": float(initial_eq),
            "final_equity": float(final_eq),
            "total_return_pct": float(total_return * 100),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_dd * 100),
            "trades_count": len(closed_sells),
            "win_rate_pct": float(win_rate * 100),
            "profit_factor": float(profit_factor)
        }
