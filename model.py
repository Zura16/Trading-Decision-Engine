import os
import json
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from database import MarketData, FeatureStore, SessionLocal
from config import logger, MODEL_PATH, TRADING_PARAMETERS

# Import ML packages dynamically to handle environments without them installed yet
# We will ensure they are loaded when running training or prediction.

class XGBoostModel:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None

    def prepare_data(self, db: Session, symbol: str):
        """Loads features and prices, creates the targets, and builds clean train/test dataframes."""
        logger.info(f"Loading feature records for {symbol} to prepare ML training data...")
        
        # Load market data
        md_records = db.query(MarketData).filter(
            MarketData.symbol == symbol
        ).order_by(MarketData.timestamp.asc()).all()
        
        if not md_records:
            raise ValueError(f"No market data found for {symbol}")
            
        md_df = pd.DataFrame([{
            "timestamp": r.timestamp,
            "close": r.close
        } for r in md_records])

        # Load feature store
        feat_records = db.query(FeatureStore).filter(
            FeatureStore.symbol == symbol
        ).order_by(FeatureStore.timestamp.asc()).all()

        if not feat_records:
            raise ValueError(f"No features found for {symbol}. Run feature computation first.")

        feat_list = []
        for r in feat_records:
            row = {"timestamp": r.timestamp}
            row.update(r.features)
            feat_list.append(row)
            
        feat_df = pd.DataFrame(feat_list)
        
        # Merge prices and features
        df = pd.merge(md_df, feat_df, on="timestamp", how="inner")
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Create Multi-class Target
        horizon = TRADING_PARAMETERS["forecast_horizon"]
        threshold = TRADING_PARAMETERS["model_threshold_pct"]
        
        # Calculate future return
        df["future_return"] = (df["close"].shift(-horizon) - df["close"]) / df["close"]
        
        # Target labels: 0 = Down, 1 = Neutral, 2 = Up
        def assign_label(ret):
            if pd.isna(ret):
                return np.nan
            if ret > threshold:
                return 2  # Up (Buy)
            elif ret < -threshold:
                return 0  # Down (Sell)
            else:
                return 1  # Neutral (Hold)
                
        df["target"] = df["future_return"].apply(assign_label)
        
        # Drop rows with NaN targets (the last `horizon` rows) and NaNs in features
        df = df.dropna().reset_index(drop=True)
        
        # Columns to exclude from features
        exclude_cols = ["timestamp", "close", "future_return", "target"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        return df, feature_cols

    def train(self, db: Session, symbol: str, test_size: float = 0.2):
        """Trains an XGBoost multi-class classifier on historical data and saves the model."""
        import xgboost as xgb
        from sklearn.metrics import classification_report
        
        df, feature_cols = self.prepare_data(db, symbol)
        
        if len(df) < 100:
            raise ValueError(f"Not enough training samples. Found {len(df)}, need at least 100.")
            
        X = df[feature_cols]
        y = df["target"].astype(int)
        
        # Time-series split (no random shuffling to prevent data leakage)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training features count: {len(feature_cols)}. Columns: {feature_cols}")
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Balance classes if needed using sample weights (optional, XGBoost handles multiclass well)
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            eval_metric="mlogloss"
        )
        
        self.model.fit(X_train, y_train)
        
        # Predict and evaluate
        preds = self.model.predict(X_test)
        report = classification_report(
            y_test, preds, 
            target_names=["Down (0)", "Neutral (1)", "Up (2)"], 
            zero_division=0
        )
        logger.info(f"Model Training completed.\nClassification Report:\n{report}")
        
        # Save model and feature columns
        self.save()
        logger.info(f"Model successfully saved to {self.model_path}")
        
    def save(self):
        """Saves model booster and feature list meta-information."""
        if self.model is None:
            raise ValueError("No model trained yet to save.")
            
        # Save XGBoost booster
        booster_file = self.model_path
        self.model.save_model(booster_file)
        
        # Save feature column names in metadata
        meta_file = booster_file + ".meta"
        with open(meta_file, "w") as f:
            json.dump({"feature_columns": self.feature_columns}, f)

    def load(self):
        """Loads the saved XGBoost booster and feature list metadata."""
        import xgboost as xgb
        
        booster_file = self.model_path
        meta_file = booster_file + ".meta"
        
        if not os.path.exists(booster_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"Model files not found. Looked for {booster_file} and {meta_file}")
            
        # Load metadata
        with open(meta_file, "r") as f:
            meta = json.load(f)
            self.feature_columns = meta["feature_columns"]
            
        # Load XGBoost
        self.model = xgb.XGBClassifier()
        self.model.load_model(booster_file)
        logger.info(f"Model loaded successfully from {booster_file} with {len(self.feature_columns)} features.")

    def predict_prob(self, features_dict: dict) -> tuple:
        """Runs inference and returns predicted class and probabilities [P(Down), P(Neutral), P(Up)]."""
        if self.model is None:
            self.load()
            
        # Ensure correct features are present and ordered
        input_data = []
        for col in self.feature_columns:
            if col not in features_dict:
                # Fill missing features with neutral values
                input_data.append(0.0)
            else:
                input_data.append(features_dict[col])
                
        # Shape as row vector (1, n_features)
        X_input = pd.DataFrame([input_data], columns=self.feature_columns)
        
        # Predict probabilities
        probs = self.model.predict_proba(X_input)[0]  # Array of [P(Down), P(Neutral), P(Up)]
        pred_class = int(np.argmax(probs))  # 0, 1, or 2
        
        # Translate to signals: class 0 = -1 (Sell), class 1 = 0 (Hold), class 2 = 1 (Buy)
        signal = pred_class - 1
        
        return signal, float(probs[0]), float(probs[1]), float(probs[2])
