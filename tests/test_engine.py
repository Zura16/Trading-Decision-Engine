import unittest
import os
import shutil
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set temporary database env variable for testing
TEST_DB = "sqlite:///test_trading_engine.db"
os.environ["DATABASE_URL"] = TEST_DB

from database import Base, MarketData, FeatureStore, TradeStore, PortfolioState
from features import calculate_features_dataframe, FeaturePipeline
from risk import RiskManager

class TestTradingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test engine
        cls.engine = create_engine(TEST_DB)
        cls.SessionLocal = sessionmaker(bind=cls.engine)
        Base.metadata.create_all(bind=cls.engine)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=cls.engine)
        # Remove sqlite file
        db_file = "test_trading_engine.db"
        if os.path.exists(db_file):
            os.remove(db_file)

    def setUp(self):
        self.db = self.SessionLocal()
        # Clean up tables between tests
        self.db.query(MarketData).delete()
        self.db.query(FeatureStore).delete()
        self.db.query(TradeStore).delete()
        self.db.query(PortfolioState).delete()
        self.db.commit()

    def test_database_crud(self):
        """Verifies database insertion and querying work."""
        tick = MarketData(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.5,
            volume=10000.0,
            is_tick=True
        )
        self.db.add(tick)
        self.db.commit()

        queried = self.db.query(MarketData).filter_by(symbol="AAPL").first()
        self.assertIsNotNone(queried)
        self.assertEqual(queried.close, 151.5)
        self.assertTrue(queried.is_tick)

    def test_feature_pipeline_equivalence(self):
        """Verifies that batch and streaming feature calculations align."""
        # Insert 40 ticks for AAPL
        records = []
        base_price = 100.0
        for i in range(40):
            price = base_price + i * 0.5
            records.append(MarketData(
                timestamp=datetime(2026, 1, 1, 9, 30) + timedelta(minutes=i),
                symbol="AAPL",
                open=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
                volume=1000.0,
                bid=price - 0.05,
                ask=price + 0.05,
                is_tick=False
            ))
        self.db.bulk_save_objects(records)
        self.db.commit()

        # Batch features
        data = [{"timestamp": r.timestamp, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume, "bid": r.bid, "ask": r.ask} for r in records]
        df = pd.DataFrame(data)
        df_batch = calculate_features_dataframe(df)
        batch_last_row = df_batch.iloc[-1]

        # Streaming features
        pipeline = FeaturePipeline(lookback_periods=35)
        stream_latest = pipeline.get_latest_features(self.db, "AAPL")

        # Compare values
        self.assertIn("sma_10", stream_latest)
        self.assertIn("rsi_14", stream_latest)
        self.assertIn("macd", stream_latest)
        self.assertAlmostEqual(stream_latest["sma_10"], batch_last_row["sma_10"])
        self.assertAlmostEqual(stream_latest["rsi_14"], batch_last_row["rsi_14"])

    def test_risk_manager_stops_and_sizing(self):
        """Tests stop calculations and position sizing constraints."""
        rm = RiskManager()
        
        # 1. Stop loss / take profit buy
        sl, tp = rm.get_stops("BUY", 100.0, atr=None)
        self.assertEqual(sl, 98.0)  # default stop_loss_pct = 2%
        self.assertEqual(tp, 106.0) # default take_profit_pct = 6%

        # 2. Stop loss / take profit sell
        sl, tp = rm.get_stops("SELL", 100.0, atr=None)
        self.assertEqual(sl, 102.0)
        self.assertEqual(tp, 94.0)

        # 3. Size constraints
        qty = rm.calculate_position_size(
            signal=1,
            prob_up=0.55,
            prob_down=0.20,
            current_equity=100000.0,
            current_price=100.0,
            atr=2.0
        )
        # Kelly scale + ATR size
        # Kelly: p=0.55, b=3, q=0.45 => f* = 0.55 - 0.45/3 = 0.40. Qtr Kelly = 0.10. Dollar cap = $10k
        # Volatility size: risk 1% ($1k) / stop 2*ATR (4.0) => 250 shares.
        # Max alloc size: default 10% * 2 = 20% => $20k / 100.0 = 200 shares.
        # Min of Vol Sizing (250) and Max Alloc (200) -> 200.
        self.assertTrue(qty > 0)
        self.assertLessEqual(qty, 200.0)

    def test_risk_manager_halt(self):
        """Tests that drawdown limits trigger a halt."""
        rm = RiskManager()
        # Peak equity starts at $100k (default initial_capital)
        
        # Drops to $90k (10% drawdown, limit is 15%)
        halted = rm.check_drawdown(90000.0)
        self.assertFalse(halted)
        self.assertFalse(rm.halted)

        # Drops to $80k (20% drawdown)
        halted = rm.check_drawdown(80000.0)
        self.assertTrue(halted)
        self.assertTrue(rm.halted)
        
        # Verify sizes are 0 now
        qty = rm.calculate_position_size(
            signal=1, prob_up=0.8, prob_down=0.1, current_equity=80000.0, current_price=100.0
        )
        self.assertEqual(qty, 0.0)

if __name__ == "__main__":
    unittest.main()
