import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from database import MarketData, FeatureStore
from config import logger

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Wilders smoothing representation
    for i in range(period, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr.fillna(0)

def calculate_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Computes features for a batch DataFrame of historical prices."""
    df = df.sort_values("timestamp").copy()
    
    # 1. Trend Indicators
    df["sma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["sma_30"] = df["close"].rolling(window=30, min_periods=1).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False, min_periods=1).mean()
    
    # 2. Momentum
    df["rsi_14"] = compute_rsi(df["close"], period=14)
    
    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    
    # 3. Volatility
    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], period=14)
    
    # Bollinger Bands
    rolling_std = df["close"].rolling(window=20, min_periods=1).std().fillna(0)
    df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["bollinger_upper"] = df["sma_20"] + (rolling_std * 2)
    df["bollinger_lower"] = df["sma_20"] - (rolling_std * 2)
    
    # Log returns and realized volatility
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    df["volatility_10"] = df["log_return"].rolling(window=10, min_periods=1).std().fillna(0)
    
    # 4. Momentum & Order Book Features
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["momentum_5"] = df["momentum_5"].fillna(0)
    
    df["spread"] = (df["ask"] - df["bid"]).fillna(0)
    
    # Drop intermediate column
    df = df.drop(columns=["sma_20"])
    return df

class FeaturePipeline:
    """Extracts features for batch storage and latency-aware real-time feeds."""
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods

    def extract_and_store_features(self, db: Session, symbol: str) -> int:
        """Reads historical data, computes features in batch, and stores them in the db."""
        records = db.query(MarketData).filter(MarketData.symbol == symbol).order_by(MarketData.timestamp.asc()).all()
        if not records:
            logger.warning(f"No market data found for {symbol} to compute features.")
            return 0
            
        # Convert list of ORM to DataFrame
        data = []
        for r in records:
            data.append({
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
            
        df = pd.DataFrame(data)
        df_feats = calculate_features_dataframe(df)
        
        # Save features to DB
        # Delete old features for symbol to prevent duplication
        db.query(FeatureStore).filter(FeatureStore.symbol == symbol).delete()
        
        feature_records = []
        cols_to_exclude = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "bid", "ask"]
        feature_cols = [col for col in df_feats.columns if col not in cols_to_exclude]
        
        for _, row in df_feats.iterrows():
            feat_dict = {col: float(row[col]) for col in feature_cols if not pd.isna(row[col])}
            feat_store = FeatureStore(
                timestamp=row["timestamp"].to_pydatetime(),
                symbol=symbol
            )
            feat_store.features = feat_dict
            feature_records.append(feat_store)
            
        db.bulk_save_objects(feature_records)
        db.commit()
        logger.info(f"Computed and stored features for {len(feature_records)} rows for {symbol}.")
        return len(feature_records)

    def get_latest_features(self, db: Session, symbol: str) -> dict:
        """Computes features for the single latest tick by pulling just the needed lookback data."""
        # Query last `lookback_periods` market data ticks to calculate indicators
        records = db.query(MarketData).filter(
            MarketData.symbol == symbol
        ).order_by(MarketData.timestamp.desc()).limit(self.lookback_periods).all()
        
        if len(records) < 5:
            return {}
            
        # Reverse to get chronological order
        records.reverse()
        
        data = []
        for r in records:
            data.append({
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "bid": r.bid,
                "ask": r.ask
            })
            
        df = pd.DataFrame(data)
        df_feats = calculate_features_dataframe(df)
        
        # Extract features for the very last row
        latest_row = df_feats.iloc[-1]
        
        cols_to_exclude = ["timestamp", "open", "high", "low", "close", "volume", "bid", "ask"]
        feature_cols = [col for col in df_feats.columns if col not in cols_to_exclude]
        
        latest_features = {col: float(latest_row[col]) for col in feature_cols if not pd.isna(latest_row[col])}
        return latest_features
