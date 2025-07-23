import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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
        self._add_advanced_features()
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


class BitcoinPricePredictor:
    def __init__(self, data_paths, model_params=None):
        self.data_paths = data_paths
        self.model_params = model_params or {
            "n_estimators": 1500,
            "learning_rate": 0.005,
            "max_depth": 8,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
        }
        self.feature_df = None
        self.model = None
        self.feature_columns = None

    @staticmethod
    def create_target_classes(price_change):
        """Create multi-class targets based on price change magnitude"""
        if price_change > 0.02:  # >2% up
            return 3  # Strong up
        elif price_change > 0:  # 0-2% up
            return 2  # Weak up
        elif price_change > -0.02:  # 0-2% down
            return 1  # Weak down
        else:  # >2% down
            return 0  # Strong down

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

    def create_lagged_features(self, df):
        """Create multiple lag periods for external features"""
        external_features = [
            "trend_log_return",
            "fear_greed_log_return",
            "fear_greed_index",
            "value_classification",
        ]

        for feature in external_features:
            if feature in df.columns:
                for lag in range(1, 8):  # 1-7 day lags
                    df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
        return df

    def _evaluate_with_cv(self, temp_df, y_col="target"):
        """Fixed helper method for cross-validation evaluation"""
        # Ensure target column exists and exclude it from features
        if y_col not in temp_df.columns:
            print(f"Warning: {y_col} not found in dataframe")
            return 0.0

        # Properly exclude all non-feature columns
        exclude_cols = [
            "target",
            "target_multiclass",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volumefrom",
            "volumeto",
            "timestamp",
        ]
        feature_cols = [col for col in temp_df.columns if col not in exclude_cols]

        X_features = temp_df[feature_cols].fillna(0)  # Handle NaN values
        y = temp_df[y_col].fillna(0)

        # Ensure we have enough data
        if len(X_features) < 100:
            return 0.0

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, test_idx in tscv.split(X_features):
            X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Skip if classes are missing
            if len(np.unique(y_train)) < 2:
                continue

            model = XGBClassifier(
                n_estimators=50,  # Fast training
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            )

            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                scores.append(accuracy_score(y_test, preds))
            except Exception as e:
                print(f"Error in CV fold: {e}")
                continue

        return np.mean(scores) if scores else 0.0

    def find_optimal_lags_simple(self):
        """Simplified lag optimization without data leakage"""
        if self.feature_df is None:
            raise ValueError("Run create_features() first")
        
        # Use correlation-based approach instead of accuracy
        external_features = ['trend_log_return', 'fear_greed_log_return']
        target_col = 'price_log_return_x'  # Use existing price returns
        
        optimal_lags = {}
        
        for feature in external_features:
            if feature in self.feature_df.columns:
                best_corr = 0
                best_lag = 0
                
                for lag in range(0, 8):
                    lagged_feature = self.feature_df[feature].shift(lag)
                    corr = abs(self.feature_df[target_col].corr(lagged_feature))
                    
                    if not np.isnan(corr) and corr > best_corr:
                        best_corr = corr
                        best_lag = lag
                
                optimal_lags[feature] = best_lag
                print(f"{feature}: Optimal lag = {best_lag} days, Correlation = {best_corr:.4f}")
        
        return optimal_lags


    def analyze_lead_lag_correlation(self, target_col="price_log_return", max_lag=10):
        """Analyze cross-correlation at different lags"""
        if self.feature_df is None:
            return

        external_features = ["trend_log_return", "fear_greed_log_return"]

        for feature in external_features:
            if feature in self.feature_df.columns:
                correlations = []
                lags = range(-max_lag, max_lag + 1)

                for lag in lags:
                    try:
                        if lag == 0:
                            corr = self.feature_df[target_col].corr(
                                self.feature_df[feature]
                            )
                        elif lag > 0:
                            corr = self.feature_df[target_col].corr(
                                self.feature_df[feature].shift(lag)
                            )
                        else:
                            corr = (
                                self.feature_df[target_col]
                                .shift(-lag)
                                .corr(self.feature_df[feature])
                            )
                        correlations.append(corr)
                    except:
                        correlations.append(0)

                # Plot cross-correlation
                plt.figure(figsize=(10, 6))
                plt.plot(lags, correlations, marker="o")
                plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
                plt.title(f"Cross-correlation: {feature} vs {target_col}")
                plt.xlabel("Lag (days)")
                plt.ylabel("Correlation")
                plt.grid(True, alpha=0.3)
                plt.show()

                # Find optimal lag
                optimal_idx = np.argmax(np.abs(correlations))
                optimal_lag = lags[optimal_idx]
                print(
                    f"{feature}: Optimal lag = {optimal_lag} days, Correlation = {correlations[optimal_idx]:.4f}"
                )

    def create_dynamic_lag_features(self, df):
        """Create features with market-regime dependent lags"""
        # Different lags for bull vs bear markets
        bull_mask = df["market_phase"] == 1
        bear_mask = df["market_phase"] == 0

        # Fear/Greed may have different impact timing in different market phases
        if "fear_greed_log_return" in df.columns:
            df["fear_greed_dynamic_lag"] = np.nan
            df.loc[bull_mask, "fear_greed_dynamic_lag"] = df.loc[
                bull_mask, "fear_greed_log_return"
            ].shift(
                1
            )  # 1-day lag in bull
            df.loc[bear_mask, "fear_greed_dynamic_lag"] = df.loc[
                bear_mask, "fear_greed_log_return"
            ].shift(
                3
            )  # 3-day lag in bear

        return df

    def optimize_feature_lags(self):
        """Main method to optimize all lag features"""
        try:
            print("Analyzing lead-lag correlations...")
            self.analyze_lead_lag_correlation()
            
            print("Finding optimal lags...")
            optimal_lag, lag_results = self.find_optimal_lags()
            
            print("Creating optimized lag features...")
            self.feature_df = self.create_dynamic_lag_features(self.feature_df)
            
            return self, optimal_lag, lag_results
            
        except Exception as e:
            print(f"Error in lag optimization: {e}")
            # Return safe defaults
            return self, 0, {}


    def create_features(self, target_type="binary"):
        """Feature engineering pipeline with configurable target"""
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

        # Merge technical indicators
        tech_cols = [
            col
            for col in ta_df.columns
            if col
            not in ["date", "open", "high", "low", "close", "volumefrom", "volumeto"]
        ]
        df = df.merge(ta_df[["date"] + tech_cols], on="date", how="left")

        print("Processing categorical variables...")
        # Handle categorical variables
        if "value_classification" in df.columns:
            classification_map = {
                "Extreme Fear": 0,
                "Fear": 1,
                "Neutral": 2,
                "Greed": 3,
                "Extreme Greed": 4,
            }
            df["value_classification"] = (
                df["value_classification"].map(classification_map).fillna(2)
            )

        # Convert remaining object columns to categorical
        for col in df.columns:
            if df[col].dtype == "object" and col != "date":
                df[col] = df[col].astype("category")

        print("Calculating market phases...")
        # Add market phases
        df["market_phase"] = self.calculate_market_phase(
            btc.set_index("date")["close"]
        ).values

        # Create lagged features
        features_to_lag = [
            "RSI_14",
            "MACD",
            "OBV",
            "trend_slope_20",
            "stoch_k",
            "stoch_d",
        ]
        if "trend_log_return" in df.columns:
            features_to_lag.append("trend_log_return")
        if "fear_greed_log_return" in df.columns:
            features_to_lag.append("fear_greed_log_return")

        for feat in features_to_lag:
            if feat in df.columns:
                df[f"{feat}_lag1"] = df[feat].shift(1)
                df[f"{feat}_lag2"] = df[feat].shift(2)

        # Create target variables
        if target_type == "binary":
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        elif target_type == "multiclass":
            price_changes = df["close"].pct_change().shift(-1)
            df["target_multiclass"] = price_changes.apply(self.create_target_classes)
            df["target"] = df["target_multiclass"]  # Use multiclass as primary target

        # Remove last row with NaN target
        df = df[:-1]

        self.feature_df = df.dropna()
        print(f"Feature engineering complete. Dataset shape: {self.feature_df.shape}")
        return self

    def create_features_simple(self, target_type='binary'):
        """Simplified feature engineering without problematic lag optimization"""
        print("Loading and merging datasets...")
        
        # Load datasets
        btc = pd.read_csv(self.data_paths['bitcoin'], parse_dates=['date'])
        google = pd.read_csv(self.data_paths['google'], parse_dates=['date'])
        fear = pd.read_csv(self.data_paths['fear_greed'], parse_dates=['date'])
        
        # Merge datasets
        df = btc.merge(google, on='date', how='left')
        df = df.merge(fear, on='date', how='left')
        
        # Calculate technical indicators
        ta = TechnicalAnalyser(btc)
        ta_df = ta.add_all_indicators()
        
        # Merge technical indicators
        tech_cols = [col for col in ta_df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto']]
        df = df.merge(ta_df[['date'] + tech_cols], on='date', how='left')
        
        # Handle categorical variables
        if 'value_classification' in df.columns:
            classification_map = {
                'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4
            }
            df['value_classification'] = df['value_classification'].map(classification_map).fillna(2)
        
        # Add market phases
        df['market_phase'] = self.calculate_market_phase(btc.set_index('date')['close']).values
        
        # Simple lag features (1-2 days only)
        simple_lag_features = ['RSI_14', 'MACD', 'trend_log_return', 'fear_greed_log_return']
        for feat in simple_lag_features:
            if feat in df.columns:
                df[f'{feat}_lag1'] = df[feat].shift(1)
        
        # Create target
        if target_type == 'binary':
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        else:  # multiclass
            price_changes = df['close'].pct_change().shift(-1)
            df['target'] = price_changes.apply(self.create_target_classes)
        
        df = df[:-1]  # Remove last row
        self.feature_df = df.dropna()
        return self


    def train_model(self, n_splits=5):
        """Train XGBoost classifier with temporal validation"""
        if self.feature_df is None:
            raise ValueError("Run create_features() first")

        # Define feature columns (exclude non-predictive columns)
        exclude_cols = [
            "date",
            "target",
            "target_multiclass",
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

            # Train model
            model = XGBClassifier(**self.model_params)
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
        self.model = XGBClassifier(**self.model_params)
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
