import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import MarketData, SessionLocal, init_db
from config import logger, SYMBOLS

# Fallback synthetic data generator in case yfinance fails or offline
def generate_synthetic_history(symbol: str, days: int = 30, interval_mins: int = 5) -> pd.DataFrame:
    """Generates synthetic historical bar data using Geometric Brownian Motion."""
    logger.info(f"Generating synthetic history for {symbol} ({days} days, {interval_mins}m interval)...")
    
    # Define parameters based on asset type
    if "USD" in symbol or "-" in symbol:  # Crypto-like
        start_price = 60000.0 if "BTC" in symbol else 3000.0
        drift = 0.0001
        volatility = 0.02
    else:  # Equity-like
        start_price = 150.0 if symbol == "AAPL" else 400.0
        drift = 0.00005
        volatility = 0.01

    periods = int((days * 24 * 60) / interval_mins)
    end_time = datetime.utcnow()
    timestamps = [end_time - timedelta(minutes=i * interval_mins) for i in range(periods)]
    timestamps.reverse()

    # GBM simulation
    dt = 1.0 / periods
    prices = [start_price]
    for _ in range(1, periods):
        last_price = prices[-1]
        # dS = S * (mu * dt + sigma * dW)
        price_change = last_price * (drift * dt + volatility * np.sqrt(dt) * np.random.normal())
        prices.append(max(0.01, last_price + price_change))

    # Form bars (OHLCV)
    data = []
    for i, t in enumerate(timestamps):
        close_price = prices[i]
        # Add random noise to create OHLC values
        high_price = close_price * (1 + abs(np.random.normal(0, 0.002)))
        low_price = close_price * (1 - abs(np.random.normal(0, 0.002)))
        open_price = prices[i-1] if i > 0 else close_price
        
        # Keep boundary consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        volume = float(random.randint(1000, 50000))

        data.append({
            "timestamp": t,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "bid": close_price - 0.02,
            "ask": close_price + 0.02
        })

    return pd.DataFrame(data)


def download_historical_data(symbols=SYMBOLS, days=30, db: Session = None):
    """Downloads historical bar data from yfinance, with a robust fallback to synthetic data."""
    if db is None:
        db = SessionLocal()

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed. Falling back to synthetic history.")
        yf = None

    for symbol in symbols:
        df = None
        if yf is not None:
            try:
                logger.info(f"Downloading historical data for {symbol} via yfinance...")
                period_str = f"{days}d"
                # Select interval based on timeframe
                interval = "5m" if days <= 60 else "1d"
                ticker = yf.Ticker(symbol)
                history = ticker.history(period=period_str, interval=interval)
                
                if not history.empty:
                    df = history.reset_index()
                    # Standardize columns
                    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
                    df = df.rename(columns={
                        time_col: 'timestamp',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    # Add mock bid/ask
                    df['bid'] = df['close'] - 0.01
                    df['ask'] = df['close'] + 0.01
                    # Convert timestamps to naive datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask']]
                    logger.info(f"Successfully downloaded {len(df)} bars for {symbol} via yfinance.")
            except Exception as e:
                logger.error(f"yfinance download failed for {symbol}: {e}. Falling back to synthetic.")

        if df is None:
            # Fallback to synthetic
            df = generate_synthetic_history(symbol, days=days)

        # Write to Database
        logger.info(f"Writing {len(df)} historical bars for {symbol} to database...")
        db_records = []
        for _, row in df.iterrows():
            record = MarketData(
                timestamp=row['timestamp'].to_pydatetime(),
                symbol=symbol,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                bid=float(row['bid']),
                ask=float(row['ask']),
                is_tick=False
            )
            db_records.append(record)
        
        # Clear existing non-tick data for this symbol to avoid duplicates
        db.query(MarketData).filter(MarketData.symbol == symbol, MarketData.is_tick == False).delete()
        db.bulk_save_objects(db_records)
        db.commit()
        logger.info(f"Historical write complete for {symbol}.")


class LiveMarketSimulator:
    """Simulates real-time tick updates (order book and trades) for configured symbols."""
    def __init__(self, symbols=SYMBOLS, tick_interval_sec: float = 1.0):
        self.symbols = symbols
        self.tick_interval_sec = tick_interval_sec
        self.running = False
        
        # Initialize state prices based on last close or standard default
        self.prices = {}
        db = SessionLocal()
        try:
            for s in symbols:
                last_record = db.query(MarketData).filter(MarketData.symbol == s).order_by(MarketData.timestamp.desc()).first()
                if last_record:
                    self.prices[s] = last_record.close
                else:
                    self.prices[s] = 100.0 if "USD" not in s else 50000.0
        finally:
            db.close()

    def start(self):
        """Starts generating ticks in a loop."""
        self.running = True
        logger.info(f"Starting Live Market Simulator for {self.symbols} at {self.tick_interval_sec}s interval...")
        
        db = SessionLocal()
        try:
            while self.running:
                start_loop_time = time.time()
                for symbol in self.symbols:
                    # Geometric Brownian Motion simulation step
                    # Volatility scales down for sub-second interval
                    vol = 0.001 if "USD" not in symbol else 0.002
                    dt = self.tick_interval_sec / 86400.0  # Fraction of day
                    drift = 0.0001
                    
                    price_change = self.prices[symbol] * (drift * dt + vol * np.sqrt(dt) * np.random.normal())
                    new_price = max(0.01, self.prices[symbol] + price_change)
                    self.prices[symbol] = new_price
                    
                    # Order Book simulator (spread + bid/ask volume)
                    spread = 0.0005 * new_price  # 0.05% spread
                    bid = round(new_price - spread / 2.0, 4)
                    ask = round(new_price + spread / 2.0, 4)
                    volume = float(random.randint(10, 500))

                    # Exchange timestamp vs local ingestion timestamp
                    exchange_ts = datetime.utcnow()
                    
                    # Latency delay mock (randomly 2ms to 20ms)
                    simulated_latency = random.randint(2, 20) / 1000.0
                    time.sleep(simulated_latency)
                    
                    ingest_ts = datetime.utcnow()
                    latency_ms = (ingest_ts - exchange_ts).total_seconds() * 1000.0

                    # Insert tick into database
                    tick = MarketData(
                        timestamp=ingest_ts,
                        symbol=symbol,
                        open=new_price,
                        high=new_price,
                        low=new_price,
                        close=new_price,
                        volume=volume,
                        bid=bid,
                        ask=ask,
                        is_tick=True
                    )
                    db.add(tick)
                    db.commit()
                    
                    logger.debug(f"Tick simulated for {symbol}: Close={new_price:.2f}, Bid={bid:.2f}, Ask={ask:.2f}, Latency={latency_ms:.1f}ms")
                
                # Regulate tick interval
                elapsed = time.time() - start_loop_time
                sleep_time = max(0.001, self.tick_interval_sec - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Simulator stopped by user.")
        except Exception as e:
            logger.error(f"Error in LiveMarketSimulator: {e}")
        finally:
            db.close()
            self.running = False

    def stop(self):
        self.running = False
        logger.info("Live Market Simulator stopped.")

if __name__ == "__main__":
    init_db()
    # Download 30 days of data
    download_historical_data(days=30)
    # Run a test simulator for 5 seconds
    sim = LiveMarketSimulator(tick_interval_sec=0.5)
    import threading
    t = threading.Thread(target=sim.start)
    t.start()
    time.sleep(5)
    sim.stop()
    t.join()
