import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


class TechnicalAnalyser:
    def __init__(self, ohlcv_df):
        self.df = ohlcv_df.copy()

    def add_all_indicators(self):
        self._moving_averages()
        self._volatility_metrics()
        self._momentum_indicators()
        self._volume_indicators()
        self._support_resistance_trend_lines()
        self._stochastic_oscillator()
        return self.df

    def _moving_averages(self):
        # Simple Moving Averages
        self.df["SMA_10"] = self.df["close"].rolling(10).mean()
        self.df["SMA_20"] = self.df["close"].rolling(20).mean()
        self.df["SMA_50"] = self.df["close"].rolling(50).mean()

        # Exponential Moving Averages
        self.df["EMA_12"] = self.df["close"].ewm(span=12).mean()
        self.df["EMA_26"] = self.df["close"].ewm(span=26).mean()

    def _volatility_metrics(self):
        # Bollinger Bands
        rolling_mean = self.df["close"].rolling(20).mean()
        rolling_std = self.df["close"].rolling(20).std()
        self.df["BB_upper"] = rolling_mean + 2 * rolling_std
        self.df["BB_lower"] = rolling_mean - 2 * rolling_std
        self.df["BB_width"] = self.df["BB_upper"] - self.df["BB_lower"]

        # Average True Range (ATR)
        high_low = self.df["high"] - self.df["low"]
        high_close = np.abs(self.df["high"] - self.df["close"].shift())
        low_close = np.abs(self.df["low"] - self.df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df["ATR_14"] = true_range.rolling(14).mean()

    def _momentum_indicators(self):
        # RSI (Relative Strength Index)
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        self.df["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD
        self.df["MACD"] = self.df["EMA_12"] - self.df["EMA_26"]
        self.df["MACD_signal"] = self.df["MACD"].ewm(span=9).mean()
        self.df["MACD_histogram"] = self.df["MACD"] - self.df["MACD_signal"]

    def _volume_indicators(self):
        # On-Balance Volume (OBV)
        obv = (
            (np.sign(self.df["close"].diff()) * self.df["volumefrom"])
            .fillna(0)
            .cumsum()
        )
        self.df["OBV"] = obv

        # Volume Moving Average
        self.df["volume_ma_10"] = self.df["volumefrom"].rolling(10).mean()

    def _support_resistance_trend_lines(self):
        # Support and Resistance levels
        self.df["support_20"] = self.df["low"].rolling(window=20).min()
        self.df["resistance_20"] = self.df["high"].rolling(window=20).max()

        # Distance from support/resistance
        self.df["dist_from_support"] = (
            self.df["close"] - self.df["support_20"]
        ) / self.df["close"]
        self.df["dist_from_resistance"] = (
            self.df["resistance_20"] - self.df["close"]
        ) / self.df["close"]

        # Trend lines using linear regression slope
        def calculate_slope(series):
            if len(series) < 2:
                return np.nan
            x = np.arange(len(series))
            y = series.values
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return np.nan

        self.df["trend_slope_20"] = (
            self.df["close"].rolling(window=20).apply(calculate_slope, raw=False)
        )

    def _stochastic_oscillator(self):
        # Stochastic Oscillator
        low_14 = self.df["low"].rolling(window=14).min()
        high_14 = self.df["high"].rolling(window=14).max()
        self.df["stoch_k"] = 100 * ((self.df["close"] - low_14) / (high_14 - low_14))
        self.df["stoch_d"] = self.df["stoch_k"].rolling(window=3).mean()

    def _add_advanced_features(self):
        # Price momentum features
        self.df["price_momentum_3d"] = self.df["close"].pct_change(3)
        self.df["price_momentum_7d"] = self.df["close"].pct_change(7)

        # Volatility ratios
        self.df["vol_ratio_5_30"] = self.df["volatility_5d"] / self.df["volatility_30d"]

        # Volume-price relationship
        self.df["volume_price_trend"] = (
            (self.df["volumefrom"] * self.df["close"]).rolling(5).mean()
        )

        # Cross-moving average signals
        self.df["sma_cross_signal"] = np.where(
            self.df["SMA_10"] > self.df["SMA_20"], 1, 0
        )


class PricePredictor:
    def __init__(self, data_paths, model_params=None):
        self.data_paths = data_paths
        self.model_params = model_params or {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 50,
            "eval_metric": "logloss",
        }
        self.feature_df = None
        self.model = None
        self.feature_columns = None

    def calculate_market_phase(
        self,
        price_series,
        window=30,
        min_phase_days=112,
        min_cycle_days=490,
        threshold=0.20,
    ):
        """Calculate bull/bear market phases using Pagan-Sossounov algorithm"""
        df = pd.DataFrame({"price": price_series})
        df = df.dropna()

        # Find peaks and troughs
        df["peak"] = self._find_peaks(df["price"], window)
        df["trough"] = self._find_troughs(df["price"], window)

        # Identify phases
        phases = []
        current_state = None
        start_idx = 0

        for i in range(len(df)):
            if df["peak"].iloc[i]:
                if current_state != "peak":
                    phases.append(("peak", start_idx, i))
                    current_state = "peak"
                    start_idx = i
            elif df["trough"].iloc[i]:
                if current_state != "trough":
                    phases.append(("trough", start_idx, i))
                    current_state = "trough"
                    start_idx = i

        # Apply validation filters
        valid_phases = []
        for j in range(1, len(phases)):
            prev_type, prev_start, prev_end = phases[j - 1]
            curr_type, curr_start, curr_end = phases[j]

            duration = curr_start - prev_start
            if prev_start < len(df) and curr_start < len(df):
                price_change = abs(
                    (df["price"].iloc[curr_start] / df["price"].iloc[prev_start]) - 1
                )

                if duration >= min_cycle_days or price_change >= threshold:
                    valid_phases.append((prev_type, prev_start, prev_end))

        # Create market phase labels
        df["market_phase"] = 0  # Default to bear market
        for phase in valid_phases:
            phase_type, start, end = phase
            if phase_type == "trough":  # Bull markets start at troughs
                df.iloc[start : end + 1, df.columns.get_loc("market_phase")] = 1

        return df["market_phase"].reindex(price_series.index, fill_value=0)

    def _find_peaks(self, series, window):
        peaks = np.zeros(len(series), dtype=bool)
        for i in range(window, len(series) - window):
            if (
                series.iloc[i] == series.iloc[i - window : i + window + 1].max()
                and series.iloc[i] > series.iloc[i - 1]
                and series.iloc[i] > series.iloc[i + 1]
            ):
                peaks[i] = True
        return peaks

    def _find_troughs(self, series, window):
        troughs = np.zeros(len(series), dtype=bool)
        for i in range(window, len(series) - window):
            if (
                series.iloc[i] == series.iloc[i - window : i + window + 1].min()
                and series.iloc[i] < series.iloc[i - 1]
                and series.iloc[i] < series.iloc[i + 1]
            ):
                troughs[i] = True
        return troughs

    def create_features(self):
        """Feature engineering pipeline"""
        print("Loading and merging datasets...")

        # Load datasets
        btc = pd.read_csv(self.data_paths["bitcoin"], parse_dates=["date"])
        google = pd.read_csv(self.data_paths["google"], parse_dates=["date"])
        fear = pd.read_csv(self.data_paths["fear_greed"], parse_dates=["date"])

        # Merge datasets
        df = btc.merge(google, on="date", how="left")
        df = df.merge(fear, on="date", how="left")

        print("Calculating technical indicators...")
        # Calculate technical indicators
        ta = TechnicalAnalyser(btc)
        ta_df = ta.add_all_indicators()

        # Merge technical indicators (avoid duplicate columns)
        tech_cols = [
            col
            for col in ta_df.columns
            if col
            not in ["date", "open", "high", "low", "close", "volumefrom", "volumeto"]
        ]
        df = df.merge(ta_df[["date"] + tech_cols], on="date", how="left")

        print("Calculating market phases...")
        # Add market phases
        df["market_phase"] = self.calculate_market_phase(
            btc.set_index("date")["close"]
        ).values

        # Create lagged features to prevent look-ahead bias
        features_to_lag = [
            "RSI_14",
            "MACD",
            "OBV",
            "trend_slope_20",
            "stoch_k",
            "stoch_d",
            "trend_log_return",
            "fear_greed_log_return",
        ]

        for feat in features_to_lag:
            if feat in df.columns:
                df[f"{feat}_lag1"] = df[feat].shift(1)
                df[f"{feat}_lag2"] = df[feat].shift(2)

        # Create target variable: Next day's price direction (1=up, 0=down)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        # Remove rows with missing target
        df = df[:-1]
        # Convert object columns to categorical or numeric
        for col in df.columns:
            if df[col].dtype == "object":
                if col == "value_classification":
                    # Convert fear/greed classifications to numeric
                    classification_map = {
                        "Extreme Fear": 0,
                        "Fear": 1,
                        "Neutral": 2,
                        "Greed": 3,
                        "Extreme Greed": 4,
                    }
                    df[col] = (
                        df[col].map(classification_map).fillna(2)
                    )  # Default to Neutral
                else:
                    # Convert other object columns to category
                    df[col] = df[col].astype("category")

        self.feature_df = df.dropna()
        print(f"Feature engineering complete. Dataset shape: {self.feature_df.shape}")
        return self

    def train_model(self, n_splits=5):
        """Train XGBoost classifier with temporal validation"""
        if self.feature_df is None:
            raise ValueError("Run create_features() first")

        # Define feature columns (exclude non-predictive columns)
        exclude_cols = [
            "date",
            "target",
            "open",
            "high",
            "low",
            "close",
            "volumefrom",
            "volumeto",
        ]
        self.feature_columns = [
            col for col in self.feature_df.columns if col not in exclude_cols
        ]

        X = self.feature_df[self.feature_columns]
        y = self.feature_df["target"]

        print(f"Training with {len(self.feature_columns)} features...")
        print(f"Features: {self.feature_columns[:10]}...")  # Show first 10 features

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"Training fold {fold + 1}/{n_splits}...")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model with proper early stopping
            model = XGBClassifier(
                **self.model_params,
                early_stopping_rounds=50,  # Move to constructor
                eval_metric="logloss",  # Required for early stopping
            )

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            # Evaluate
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            scores.append(accuracy)

            print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

        print(f"\nCross-validation results:")
        print(f"Mean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")

        # Train final model on all data
        print("Training final model on full dataset...")
        self.model = XGBClassifier(
            **{
                k: v
                for k, v in self.model_params.items()
                if k not in ["early_stopping_rounds", "eval_metric"]
            }  # Remove these for final training
        )
        self.model.fit(X, y)

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        return self

    def predict(self, new_data):
        """Generate predictions on new data"""
        if self.model is None:
            raise ValueError("Train model first using train_model()")

        # Ensure new_data has the same features
        X_new = new_data[self.feature_columns]
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)

        return predictions, probabilities

    def save_model(self, filepath):
        """Save trained model"""
        import joblib

        if self.model is None:
            raise ValueError("No trained model to save")
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns}, filepath
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        import joblib

        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        print(f"Model loaded from {filepath}")
