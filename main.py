import argparse
import sys
import time
import threading
from datetime import datetime
from sqlalchemy.orm import Session
from database import init_db, SessionLocal, MarketData, TradeStore, PortfolioState, SignalStore
from data_simulator import download_historical_data, LiveMarketSimulator
from features import FeaturePipeline
from model import XGBoostModel
from risk import RiskManager
from config import logger, SYMBOLS, TRADING_PARAMETERS, DASH_HOST, DASH_PORT

def run_training(symbols, days):
    """Orchestrates downloading data, computing features, and training model."""
    logger.info(f"--- STARTING MODEL TRAINING PIPELINE FOR {symbols} ---")
    db = SessionLocal()
    try:
        # Step 1: Initialize DB tables
        init_db()
        
        # Step 2: Download data
        download_historical_data(symbols=symbols, days=days, db=db)
        
        # Step 3: Compute Features
        pipeline = FeaturePipeline()
        for s in symbols:
            pipeline.extract_and_store_features(db, s)
            
        # Step 4: Train Model
        # We'll train on the first symbol (e.g. BTC-USD or AAPL) as anchor
        anchor_symbol = symbols[0]
        model = XGBoostModel()
        model.train(db, anchor_symbol)
        
        logger.info("--- MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY ---")
    finally:
        db.close()

def run_backtesting(symbols):
    """Runs event-driven backtest on database historical records."""
    logger.info(f"--- RUNNING BACKTEST ENGINE FOR {symbols} ---")
    db = SessionLocal()
    try:
        from backtester import BacktestEngine
        engine = BacktestEngine(symbols)
        metrics = engine.run_backtest(db)
        
        print("\n" + "="*50)
        print("          BACKTEST PERFORMANCE METRICS")
        print("="*50)
        for k, v in metrics.items():
            if "pct" in k or "rate" in k:
                print(f"{k:<25}: {v:.2f}%")
            elif "equity" in k or "capital" in k:
                print(f"{k:<25}: ${v:,.2f}")
            else:
                print(f"{k:<25}: {v:.4f}" if isinstance(v, float) else f"{k:<25}: {v}")
        print("="*50 + "\n")
    finally:
        db.close()

def run_live_trading_loop(symbols, speed):
    """Listens for new ticks, computes features, calls XGBoost, applies risk controls, and trades."""
    logger.info("Starting Real-time Trading Execution Loop...")
    
    db = SessionLocal()
    pipeline = FeaturePipeline()
    risk_manager = RiskManager()
    model = XGBoostModel()
    
    # Load ML Model
    model_loaded = False
    try:
        model.load()
        model_loaded = True
        logger.info("ML Predictor initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not load ML model: {e}. Trading loop will use fallback rules.")
        
    # Reset active databases
    db.query(TradeStore).delete()
    db.query(PortfolioState).delete()
    db.query(SignalStore).delete()
    
    # Initial state
    cash = TRADING_PARAMETERS["initial_capital"]
    positions = {s: 0.0 for s in symbols}
    entry_prices = {s: 0.0 for s in symbols}
    stop_losses = {s: None for s in symbols}
    take_profits = {s: None for s in symbols}
    
    # Save initial portfolio state
    db.add(PortfolioState(
        timestamp=datetime.utcnow(),
        cash=cash,
        market_value=0.0,
        equity=cash,
        drawdown=0.0,
        leverage=0.0
    ))
    db.commit()

    # Track last processed tick timestamp per symbol to avoid double-processing
    last_processed_ts = {s: datetime.min for s in symbols}

    try:
        while True:
            # Poll for new ticks
            for symbol in symbols:
                latest_tick = db.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.is_tick == True
                ).order_by(MarketData.timestamp.desc()).first()

                if not latest_tick or latest_tick.timestamp <= last_processed_ts[symbol]:
                    continue
                
                # New tick detected!
                last_processed_ts[symbol] = latest_tick.timestamp
                tick_price = latest_tick.close
                logger.info(f"New Tick: {symbol} @ {tick_price:.2f} (Time: {latest_tick.timestamp})")
                
                # 1. Update valuation (mark to market) and check stops
                market_val = 0.0
                for s in symbols:
                    s_qty = positions[s]
                    if s_qty > 0:
                        if s == symbol:
                            market_val += s_qty * tick_price
                        else:
                            # Use last price from DB
                            last_p = db.query(MarketData).filter(MarketData.symbol == s).order_by(MarketData.timestamp.desc()).first()
                            if last_p:
                                market_val += s_qty * last_p.close

                current_equity = cash + market_val
                
                # Check Global Halts
                if risk_manager.check_drawdown(current_equity):
                    # Panic exit all
                    for s in symbols:
                        qty = positions[s]
                        if qty > 0:
                            # Close
                            slip = TRADING_PARAMETERS["transaction_fee_pct"]
                            exit_p = tick_price * (1 - slip) if s == symbol else db.query(MarketData).filter(MarketData.symbol == s).order_by(MarketData.timestamp.desc()).first().close * (1 - slip)
                            comm = exit_p * qty * slip
                            pnl = (exit_p * qty) - (entry_prices[s] * qty) - comm - (entry_prices[s] * qty * slip)
                            cash += (exit_p * qty) - comm
                            positions[s] = 0.0
                            
                            db.add(TradeStore(
                                timestamp=datetime.utcnow(),
                                symbol=s,
                                side="SELL",
                                price=exit_p,
                                qty=qty,
                                value=exit_p * qty,
                                slippage=exit_p * slip,
                                commission=comm,
                                pnl=pnl,
                                status="CLOSED"
                            ))
                    db.commit()
                    logger.warning("System halted. Active trades closed due to drawdown risk limits.")
                    return

                # Check individual Stops for active position
                if positions[symbol] > 0:
                    qty = positions[symbol]
                    sl = stop_losses[symbol]
                    tp = take_profits[symbol]
                    
                    exit_triggered = False
                    reason = ""
                    
                    if sl and tick_price <= sl:
                        exit_triggered = True
                        reason = "STOP LOSS TRIGGER"
                    elif tp and tick_price >= tp:
                        exit_triggered = True
                        reason = "TAKE PROFIT TRIGGER"
                        
                    if exit_triggered:
                        slip = TRADING_PARAMETERS["transaction_fee_pct"]
                        exit_p = tick_price * (1 - slip)
                        comm = exit_p * qty * slip
                        pnl = (exit_p * qty) - (entry_prices[symbol] * qty) - comm - (entry_prices[symbol] * qty * slip)
                        cash += (exit_p * qty) - comm
                        
                        positions[symbol] = 0.0
                        entry_prices[symbol] = 0.0
                        stop_losses[symbol] = None
                        take_profits[symbol] = None
                        
                        db.add(TradeStore(
                            timestamp=datetime.utcnow(),
                            symbol=symbol,
                            side="SELL",
                            price=exit_p,
                            qty=qty,
                            value=exit_p * qty,
                            slippage=tick_price * slip,
                            commission=comm,
                            pnl=pnl,
                            status="CLOSED"
                        ))
                        db.commit()
                        logger.info(f"[{reason}] Exit position for {symbol} @ {exit_p:.2f} | PnL: ${pnl:.2f}")
                        # Re-calculate equity
                        market_val = 0.0
                        for s in symbols:
                            market_val += positions[s] * (tick_price if s == symbol else 0.0)
                        current_equity = cash + market_val

                # 2. Extract real-time streaming features
                latest_feats = pipeline.get_latest_features(db, symbol)
                if not latest_feats:
                    continue
                
                # 3. Model Inference
                p_down, p_neut, p_up = 0.33, 0.34, 0.33
                if model_loaded:
                    signal, p_down, p_neut, p_up = model.predict_prob(latest_feats)
                else:
                    # Technical Fallback
                    rsi = latest_feats.get("rsi_14", 50)
                    if rsi < 35:
                        signal = 1
                        p_up = 0.65
                    elif rsi > 65:
                        signal = -1
                        p_down = 0.65
                    else:
                        signal = 0
                
                # Save predictions to signal store for visualization
                db.add(SignalStore(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    prediction=signal,
                    prob_up=p_up,
                    prob_down=p_down,
                    prob_neutral=p_neut
                ))
                db.commit()

                # 4. Action orders
                atr = latest_feats.get("atr_14", 0.0)
                qty = positions[symbol]
                
                if signal == 1 and qty == 0:
                    # BUY
                    target_qty = risk_manager.calculate_position_size(
                        signal=1,
                        prob_up=p_up,
                        prob_down=p_down,
                        current_equity=current_equity,
                        current_price=tick_price,
                        atr=atr
                    )
                    if target_qty > 0:
                        slip = TRADING_PARAMETERS["transaction_fee_pct"]
                        exec_p = tick_price * (1 + slip)
                        comm = exec_p * target_qty * slip
                        cost = (exec_p * target_qty) + comm
                        
                        if cost <= cash:
                            cash -= cost
                            positions[symbol] = target_qty
                            entry_prices[symbol] = exec_p
                            
                            sl, tp = risk_manager.get_stops("BUY", exec_p, atr)
                            stop_losses[symbol] = sl
                            take_profits[symbol] = tp
                            
                            db.add(TradeStore(
                                timestamp=datetime.utcnow(),
                                symbol=symbol,
                                side="BUY",
                                price=exec_p,
                                qty=target_qty,
                                value=exec_p * target_qty,
                                slippage=tick_price * slip,
                                commission=comm,
                                stop_loss=sl,
                                take_profit=tp,
                                status="OPEN"
                            ))
                            db.commit()
                            logger.info(f"[LIVE BUY ORDER] Filled {target_qty} {symbol} @ {exec_p:.2f} (SL: {sl:.2f}, TP: {tp:.2f})")
                            
                elif signal == -1 and qty > 0:
                    # SELL / Exit
                    slip = TRADING_PARAMETERS["transaction_fee_pct"]
                    exit_p = tick_price * (1 - slip)
                    comm = exit_p * qty * slip
                    pnl = (exit_p * qty) - (entry_prices[symbol] * qty) - comm - (entry_prices[symbol] * qty * slip)
                    cash += (exit_p * qty) - comm
                    
                    positions[symbol] = 0.0
                    entry_prices[symbol] = 0.0
                    stop_losses[symbol] = None
                    take_profits[symbol] = None
                    
                    db.add(TradeStore(
                        timestamp=datetime.utcnow(),
                        symbol=symbol,
                        side="SELL",
                        price=exit_p,
                        qty=qty,
                        value=exit_p * qty,
                        slippage=tick_price * slip,
                        commission=comm,
                        pnl=pnl,
                        status="CLOSED"
                    ))
                    db.commit()
                    logger.info(f"[LIVE SELL ORDER] Exit position {symbol} @ {exit_p:.2f} | PnL: ${pnl:.2f}")

                # Save portfolio snapshot
                leverage = market_val / current_equity if current_equity > 0 else 0.0
                peak_eq = risk_manager.peak_equity
                dd = (peak_eq - current_equity) / peak_eq if peak_eq > 0 else 0.0
                
                db.add(PortfolioState(
                    timestamp=datetime.utcnow(),
                    cash=cash,
                    market_value=market_val,
                    equity=current_equity,
                    drawdown=dd,
                    leverage=leverage
                ))
                db.commit()

            time.sleep(0.5)  # Poll latency-aware ticks
    except KeyboardInterrupt:
        logger.info("Exiting Live Trading loop.")
    finally:
        db.close()

def run_simulation(symbols, speed, interval):
    """Launches the GBM tick simulator in a thread, then runs the trading execution loop."""
    logger.info("Initializing Database for Live Trading Simulation...")
    init_db()
    
    # Run the Simulator in a background thread
    simulator = LiveMarketSimulator(symbols=symbols, tick_interval_sec=interval)
    sim_thread = threading.Thread(target=simulator.start, daemon=True)
    sim_thread.start()
    
    # Let the simulator generate a couple of ticks first
    time.sleep(2)
    
    # Run the Trading Strategy decision loop in the main thread
    try:
        run_live_trading_loop(symbols, speed)
    except KeyboardInterrupt:
        logger.info("Stopping Simulation...")
    finally:
        simulator.stop()
        sim_thread.join()
        logger.info("Simulator thread joined.")

def run_dashboard():
    """Starts the Dash dashboard server."""
    logger.info("Launching Visualization Dashboard...")
    import dashboard
    dashboard.app.run_server(host=DASH_HOST, port=DASH_PORT, debug=False)

def main():
    parser = argparse.ArgumentParser(description="Real-Time Trading Decision Engine CLI Manager")
    subparsers = parser.add_subparsers(dest="command", help="System command to execute")

    # Train sub-parser
    train_parser = subparsers.add_parser("train", help="Download history and train XGBoost Model")
    train_parser.add_argument("--symbols", type=str, default=",".join(SYMBOLS), help="Symbols list comma-separated")
    train_parser.add_argument("--days", type=int, default=45, help="Number of historical days for training")

    # Backtest sub-parser
    backtest_parser = subparsers.add_parser("backtest", help="Run historical event-driven simulation")
    backtest_parser.add_argument("--symbols", type=str, default=",".join(SYMBOLS), help="Symbols list comma-separated")

    # Simulate sub-parser
    simulate_parser = subparsers.add_parser("simulate", help="Start real-time GBM feeds and paper trade")
    simulate_parser.add_argument("--symbols", type=str, default=",".join(SYMBOLS), help="Symbols list comma-separated")
    simulate_parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    simulate_parser.add_argument("--interval", type=float, default=2.0, help="Ticker generation interval in seconds")

    # Dashboard sub-parser
    subparsers.add_parser("dashboard", help="Start visualization dashboard server")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    parsed_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if "symbols" in args else SYMBOLS

    if args.command == "train":
        run_training(parsed_symbols, args.days)
    elif args.command == "backtest":
        run_backtesting(parsed_symbols)
    elif args.command == "simulate":
        run_simulation(parsed_symbols, args.speed, args.interval)
    elif args.command == "dashboard":
        run_dashboard()

if __name__ == "__main__":
    main()
