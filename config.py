import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Database configurations
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trading_engine.db")

# Trading parameters
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "AAPL,MSFT,BTC-USD").split(",") if s.strip()]

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "xgboost_model.json"))

# Logging setup
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "trading_engine.log", mode="a")
    ]
)
logger = logging.getLogger("TradingEngine")

# Dashboard Settings
DASH_HOST = os.getenv("DASH_HOST", "127.0.0.1")
DASH_PORT = int(os.getenv("DASH_PORT", "8050"))

# Quant/Trading Strategy Settings
TRADING_PARAMETERS = {
    "transaction_fee_pct": 0.0005,  # 0.05% fee per transaction (slippage + broker fee)
    "initial_capital": 100000.0,    # $100k initial cash
    "max_drawdown_limit": 0.15,     # Halted if portfolio equity drops 15% from peak
    "max_leverage": 2.0,            # Max leverage (portfolio value / equity)
    "default_position_size": 0.10,  # Default to 10% of portfolio equity per trade
    "stop_loss_pct": 0.02,          # 2% stop loss
    "take_profit_pct": 0.06,        # 6% take profit
    "forecast_horizon": 5,          # Predict direction of next 5 periods
    "model_threshold_pct": 0.0015,  # Price change threshold to trigger a Buy/Sell signal
}
