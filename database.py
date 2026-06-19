import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL, logger

Base = declarative_base()

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    is_tick = Column(Boolean, default=False)  # True if tick data, False if OHLC bar

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "is_tick": self.is_tick
        }

class FeatureStore(Base):
    __tablename__ = "feature_store"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    features_json = Column(Text, nullable=False)  # Serialized dictionary of computed features

    @property
    def features(self):
        return json.loads(self.features_json)
        
    @features.setter
    def features(self, val):
        self.features_json = json.dumps(val)

class SignalStore(Base):
    __tablename__ = "signal_store"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    prediction = Column(Integer, nullable=False)  # 1 = Buy, 0 = Hold, -1 = Sell
    prob_up = Column(Float, nullable=False)
    prob_down = Column(Float, nullable=False)
    prob_neutral = Column(Float, nullable=False)

class TradeStore(Base):
    __tablename__ = "trade_store"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)        # BUY or SELL
    price = Column(Float, nullable=False)
    qty = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    slippage = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)
    status = Column(String(20), default="OPEN")       # OPEN or CLOSED

class PortfolioState(Base):
    __tablename__ = "portfolio_state"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    cash = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    drawdown = Column(Float, default=0.0)
    leverage = Column(Float, default=1.0)

# Database Setup
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database tables initialized successfully. URL: {DATABASE_URL}")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        raise e

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
