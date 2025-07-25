import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

warnings.filterwarnings("ignore")


class SentimentAnalyzer:
    def __init__(self, fear_greed_df=None, google_trends_df=None):
        self.fear_greed_df = fear_greed_df
        self.google_trends_df = google_trends_df

    def add_fear_greed_features(self, df):
        if self.fear_greed_df is not None:
            df = df.merge(
                self.fear_greed_df[["date", "fear_greed_index"]], on="date", how="left"
            )
            # Add log return
            df["fear_greed_log_return"] = np.log(
                df["fear_greed_index"] / df["fear_greed_index"].shift(1)
            )

            df["fear_greed_ma_7"] = df["fear_greed_index"].rolling(7).mean()
            df["fear_greed_ma_30"] = df["fear_greed_index"].rolling(30).mean()
            df["fear_greed_change"] = df["fear_greed_index"].diff()
            df["fear_greed_momentum"] = df["fear_greed_index"].rolling(5).mean().diff()
            df["extreme_fear"] = (df["fear_greed_index"] < 25).astype(int)
            df["extreme_greed"] = (df["fear_greed_index"] > 75).astype(int)
            df["fear_contrarian"] = (df["fear_greed_index"] < 30).astype(int)
            df["greed_contrarian"] = (df["fear_greed_index"] > 70).astype(int)
            df["price_sentiment_divergence"] = df["price_momentum_5d"] - (
                df["fear_greed_change"] / 100
            )
        return df

    def add_google_trends_log_return_features(self, df):
        """Add Google Trends features using log return data"""
        if self.google_trends_df is not None:
            # Merge Google Trends log return data
            df = df.merge(
                self.google_trends_df[["date", "trend_log_return"]],
                on="date",
                how="left",
            )

            # Use log return directly as momentum indicator
            df["search_log_return"] = df["trend_log_return"]

            # Moving averages of log returns (smoothed momentum)
            df["search_log_return_ma_7"] = df["search_log_return"].rolling(7).mean()
            df["search_log_return_ma_30"] = df["search_log_return"].rolling(30).mean()

            # Momentum of momentum (acceleration)
            df["search_log_return_momentum"] = df["search_log_return_ma_7"].diff()
            df["search_acceleration"] = df["search_log_return"].diff()

            # Regime detection based on log returns
            df["high_search_momentum"] = (
                df["search_log_return"]
                > df["search_log_return"].rolling(90).quantile(0.8)
            ).astype(int)

            df["low_search_momentum"] = (
                df["search_log_return"]
                < df["search_log_return"].rolling(90).quantile(0.2)
            ).astype(int)

            # Extreme search interest changes
            df["extreme_search_spike"] = (
                df["search_log_return"]
                > df["search_log_return"].rolling(90).quantile(0.95)
            ).astype(int)

            df["extreme_search_drop"] = (
                df["search_log_return"]
                < df["search_log_return"].rolling(90).quantile(0.05)
            ).astype(int)

            # Search-price momentum correlation
            df["search_price_momentum_corr"] = (
                df["search_log_return"].rolling(30).corr(df["price_momentum_5d"])
            )

            # Volatility of search interest
            df["search_volatility"] = df["search_log_return"].rolling(30).std()
            df["search_volatility_regime"] = (
                df["search_volatility"] > df["search_volatility"].rolling(90).mean()
            ).astype(int)

        return df

    def add_combined_sentiment_features(self, df):
        """Combine sentiment indicators including log return features"""
        sentiment_components = []

        if "fear_greed_index" in df.columns:
            df["fear_greed_normalized"] = (df["fear_greed_index"] - 50) / 50
            sentiment_components.append("fear_greed_normalized")

        if "search_log_return" in df.columns:
            # Normalize search log return (z-score)
            df["search_log_return_normalized"] = (
                df["search_log_return"] - df["search_log_return"].rolling(90).mean()
            ) / df["search_log_return"].rolling(90).std()
            sentiment_components.append("search_log_return_normalized")

        # Combined sentiment score
        if sentiment_components:
            df["combined_sentiment"] = df[sentiment_components].mean(axis=1)

            # Sentiment regime classification
            df["sentiment_regime"] = pd.cut(
                df["combined_sentiment"],
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=["bearish_sentiment", "neutral_sentiment", "bullish_sentiment"],
            )
        return df

    def add_lagged_sentiment(self, df, cols=None, lags=range(0, 8)):
        """
        Create lagged versions of selected sentiment columns.
        Avoid look-ahead leakage by only shifting backward.
        """
        if cols is None:
            cols = [
                c
                for c in df.columns
                if c.startswith(("fear_greed", "search_", "combined_sentiment"))
            ]

        for c in cols:
            # For lag 0, use the current value with a suffix
            df[f"{c}_lag0"] = df[c]
            # For other lags, shift as usual
            for l in [l for l in lags if l > 0]:
                df[f"{c}_lag{l}"] = df[c].shift(l)
        return df


class MacroAnalyzer:
    def __init__(
        self, gold_df=None, silver_df=None, stock_fear_greed_df=None, vix_df=None
    ):
        self.gold_df = gold_df
        self.silver_df = silver_df
        self.stock_fear_greed_df = stock_fear_greed_df
        self.vix_df = vix_df

    def add_metal_features(self, df):
        """Add gold and silver price features"""
        df = df.copy()  # Create a copy to avoid modifying the original

        if self.gold_df is not None:
            # Ensure dates are aligned
            gold_data = self.gold_df[["date", "usd"]].copy()
            gold_data["date"] = pd.to_datetime(gold_data["date"])
            print(
                f"Gold data date range: {gold_data['date'].min()} to {gold_data['date'].max()}"
            )

            # Merge gold price data
            df_before = len(df)
            df = df.merge(gold_data, on="date", how="left")
            df_after = len(df)
            if df_after != df_before:
                print(
                    f"Warning: Gold merge changed row count from {df_before} to {df_after}"
                )

            df.rename(columns={"usd": "gold_price"}, inplace=True)

            # Gold price features
            df["gold_ma_30"] = df["gold_price"].rolling(30).mean()
            df["gold_ma_90"] = df["gold_price"].rolling(90).mean()
            df["gold_volatility"] = df["gold_price"].rolling(30).std()
            df["gold_momentum"] = df["gold_price"].pct_change(30)
            df["gold_trend"] = (df["gold_price"] / df["gold_ma_90"] - 1) * 100
            df["gold_log_return"] = np.log(df["gold_price"] / df["gold_price"].shift(1))

        if self.silver_df is not None:
            # Ensure dates are aligned
            silver_data = self.silver_df[["date", "usd"]].copy()
            silver_data["date"] = pd.to_datetime(silver_data["date"])
            print(
                f"Silver data date range: {silver_data['date'].min()} to {silver_data['date'].max()}"
            )

            # Merge silver price data
            df_before = len(df)
            df = df.merge(silver_data, on="date", how="left")
            df_after = len(df)
            if df_after != df_before:
                print(
                    f"Warning: Silver merge changed row count from {df_before} to {df_after}"
                )

            df.rename(columns={"usd": "silver_price"}, inplace=True)

            # Silver price features
            df["silver_ma_30"] = df["silver_price"].rolling(30).mean()
            df["silver_ma_90"] = df["silver_price"].rolling(90).mean()
            df["silver_volatility"] = df["silver_price"].rolling(30).std()
            df["silver_momentum"] = df["silver_price"].pct_change(30)
            df["silver_trend"] = (df["silver_price"] / df["silver_ma_90"] - 1) * 100
            df["silver_log_return"] = np.log(
                df["silver_price"] / df["silver_price"].shift(1)
            )

            # Gold/Silver ratio features
            if "gold_price" in df.columns:
                df["gold_silver_ratio"] = df["gold_price"] / df["silver_price"]
                df["gold_silver_ratio_ma_30"] = (
                    df["gold_silver_ratio"].rolling(30).mean()
                )
                df["gold_silver_ratio_trend"] = (
                    df["gold_silver_ratio"] / df["gold_silver_ratio_ma_30"] - 1
                ) * 100

        return df

    def add_stock_fear_greed_features(self, df):
        """Add stock market fear and greed index features"""
        df = df.copy()  # Create a copy to avoid modifying the original

        if self.stock_fear_greed_df is not None:
            # Ensure dates are aligned
            stock_fg_data = self.stock_fear_greed_df[["date", "value"]].copy()
            stock_fg_data["date"] = pd.to_datetime(stock_fg_data["date"])
            print(
                f"Stock F&G data date range: {stock_fg_data['date'].min()} to {stock_fg_data['date'].max()}"
            )

            # Merge stock fear & greed data
            df_before = len(df)
            df = df.merge(stock_fg_data, on="date", how="left")
            df_after = len(df)
            if df_after != df_before:
                print(
                    f"Warning: Stock F&G merge changed row count from {df_before} to {df_after}"
                )

            df.rename(columns={"value": "stock_fear_greed"}, inplace=True)

            # Stock fear & greed features
            df["stock_fear_greed_ma_7"] = df["stock_fear_greed"].rolling(7).mean()
            df["stock_fear_greed_ma_30"] = df["stock_fear_greed"].rolling(30).mean()
            df["stock_fear_greed_momentum"] = (
                df["stock_fear_greed"].rolling(5).mean().diff()
            )
            df["stock_extreme_fear"] = (df["stock_fear_greed"] < 25).astype(int)
            df["stock_extreme_greed"] = (df["stock_fear_greed"] > 75).astype(int)
            df["stock_fear_greed_log_return"] = np.log(
                df["stock_fear_greed"] / df["stock_fear_greed"].shift(1)
            )

            # Compare with crypto fear & greed if available
            if "fear_greed_index" in df.columns:
                df["fear_greed_divergence"] = (
                    df["fear_greed_index"] - df["stock_fear_greed"]
                )
                df["fear_greed_correlation"] = (
                    df["fear_greed_index"].rolling(30).corr(df["stock_fear_greed"])
                )

        return df

    def add_vix_features(self, df):
        """Add VIX volatility index features"""
        df = df.copy()  # Create a copy to avoid modifying the original

        if self.vix_df is not None:
            # Clean and prepare VIX data
            vix_clean = self.vix_df[["date", "CLOSE"]].copy()
            vix_clean["date"] = pd.to_datetime(vix_clean["date"])
            print(
                f"VIX data date range: {vix_clean['date'].min()} to {vix_clean['date'].max()}"
            )

            # Merge VIX data
            df_before = len(df)
            df = df.merge(vix_clean, on="date", how="left")
            df_after = len(df)
            if df_after != df_before:
                print(
                    f"Warning: VIX merge changed row count from {df_before} to {df_after}"
                )

            df.rename(columns={"CLOSE": "vix"}, inplace=True)

            # VIX features
            df["vix_ma_10"] = df["vix"].rolling(10).mean()
            df["vix_ma_30"] = df["vix"].rolling(30).mean()
            df["vix_trend"] = (df["vix"] / df["vix_ma_30"] - 1) * 100
            df["vix_momentum"] = df["vix"].pct_change(5)
            df["vix_log_return"] = np.log(df["vix"] / df["vix"].shift(1))
            df["extreme_volatility"] = (
                df["vix"] > df["vix"].rolling(90).quantile(0.9)
            ).astype(int)
            df["low_volatility"] = (
                df["vix"] < df["vix"].rolling(90).quantile(0.1)
            ).astype(int)

            # VIX regime features
            df["vix_regime"] = pd.cut(
                df["vix"],
                bins=[-np.inf, 15, 25, 35, np.inf],
                labels=["low_vol", "normal_vol", "high_vol", "extreme_vol"],
            )

            # Compare with crypto volatility if available
            if "volatility_20" in df.columns:
                df["volatility_divergence"] = df["volatility_20"] - df["vix"]
                df["volatility_correlation"] = (
                    df["volatility_20"].rolling(30).corr(df["vix"])
                )

        return df

    def add_macro_correlation_features(self, df):
        """Add correlation features between different macro indicators"""
        # Initialize correlation features if we have the necessary data
        if all(col in df.columns for col in ["gold_price", "silver_price", "vix"]):
            # Correlation between metals
            df["metals_correlation"] = (
                df["gold_price"].rolling(30).corr(df["silver_price"])
            )

            # Correlation with VIX
            df["gold_vix_correlation"] = df["gold_price"].rolling(30).corr(df["vix"])
            df["silver_vix_correlation"] = (
                df["silver_price"].rolling(30).corr(df["vix"])
            )

            # Combined trend indicators
            df["macro_risk_indicator"] = (
                df["vix_trend"] + df["gold_trend"] + df["silver_trend"]
            ) / 3

            # Extreme conditions indicator
            df["macro_extreme_conditions"] = (
                df["extreme_volatility"].astype(int)
                + (
                    abs(df["gold_trend"]) > df["gold_trend"].rolling(90).std() * 2
                ).astype(int)
                + (
                    abs(df["silver_trend"]) > df["silver_trend"].rolling(90).std() * 2
                ).astype(int)
            )

        return df

    def add_all_macro_features(self, df):
        """Add all macro features in the correct order"""
        df = self.add_metal_features(df)
        df = self.add_stock_fear_greed_features(df)
        df = self.add_vix_features(df)
        df = self.add_macro_correlation_features(df)

        # Add lagged versions of important macro features
        macro_cols = [
            c
            for c in df.columns
            if c.startswith(("gold_", "silver_", "vix_", "stock_fear_greed", "macro_"))
            and not c.endswith(
                ("_ma_30", "_ma_90", "_ma_7", "_ma_10")
            )  # Exclude moving averages
        ]
        for c in macro_cols:
            # For lag 0, use the current value with a suffix
            df[f"{c}_lag0"] = df[c]
            # For other lags, shift as usual
            for l in range(1, 8):
                df[f"{c}_lag{l}"] = df[c].shift(l)

        return df


class EnhancedTechnicalAnalyser:
    def __init__(
        self,
        ohlcv_df,
        sentiment_analyzer=None,
        macro_analyzer=None,
        btc_df=None,
        is_btc=False,
        lag_days=range(0, 8),  # Changed to include 0
    ):
        self.df = ohlcv_df.copy()
        self.sentiment_analyzer = sentiment_analyzer
        self.macro_analyzer = macro_analyzer
        self.btc_df = btc_df
        self.is_btc = is_btc
        self.lag_days = lag_days

    def add_bitcoin_features(self):
        """Add Bitcoin-related features for altcoins"""
        if not self.is_btc and self.btc_df is not None:
            # Ensure dates are aligned
            btc_data = self.btc_df[["date", "close"]].copy()
            btc_data["date"] = pd.to_datetime(btc_data["date"])
            print(
                f"Bitcoin data date range: {btc_data['date'].min()} to {btc_data['date'].max()}"
            )

            # Merge Bitcoin price data
            df_before = len(self.df)
            self.df = self.df.merge(
                btc_data, on="date", how="left", suffixes=("", "_btc")
            )
            df_after = len(self.df)
            if df_after != df_before:
                print(
                    f"Warning: Bitcoin merge changed row count from {df_before} to {df_after}"
                )

            # Bitcoin price features
            self.df["btc_raw_price"] = self.df[
                "close_btc"
            ]  # Keep raw price for reference
            self.df["btc_price"] = np.log(self.df["close_btc"])  # Log price
            self.df["btc_ma_7"] = self.df["btc_price"].rolling(7).mean()
            self.df["btc_ma_30"] = self.df["btc_price"].rolling(30).mean()
            self.df["btc_volatility"] = self.df["btc_price"].rolling(30).std()
            self.df["btc_momentum"] = self.df["btc_price"].diff(
                30
            )  # Changed to diff for log price
            self.df["btc_trend"] = (
                self.df["btc_price"] - self.df["btc_ma_30"]
            ) * 100  # Changed for log price
            self.df["btc_log_return"] = self.df[
                "btc_price"
            ].diff()  # Simpler with log price

            # Add lagged versions of Bitcoin features
            btc_cols = [
                "btc_price",
                "btc_momentum",
                "btc_trend",
                "btc_log_return",
                "btc_volatility",
            ]
            for c in btc_cols:
                # For lag 0, use the current value with a suffix
                self.df[f"{c}_lag0"] = self.df[c]
                # For other lags, shift as usual
                for l in [l for l in self.lag_days if l > 0]:
                    self.df[f"{c}_lag{l}"] = self.df[c].shift(l)

            # Add price differences between lags
            # For example: lag0-lag1, lag1-lag2, etc.
            for i in range(len(self.lag_days) - 1):
                current_lag = i
                next_lag = i + 1
                self.df[f"btc_price_diff_lag{current_lag}_{next_lag}"] = (
                    self.df[f"btc_price_lag{current_lag}"]
                    - self.df[f"btc_price_lag{next_lag}"]
                )

            # Add rolling means of lagged prices
            for window in [3, 5]:
                lag_cols = [f"btc_price_lag{l}" for l in self.lag_days]
                self.df[f"btc_price_lag_ma{window}"] = self.df[lag_cols].mean(axis=1)

            # Add correlation features
            self.df["btc_correlation"] = (
                self.df["close"]
                .rolling(30)
                .corr(self.df["btc_raw_price"])  # Use raw price for correlation
            )
            self.df["btc_beta"] = (
                self.df["close"]
                .pct_change()
                .rolling(30)
                .cov(self.df["btc_raw_price"].pct_change())
                / self.df["btc_raw_price"].pct_change().rolling(30).var()
            )
            self.df["btc_relative_strength"] = self.df["close"].pct_change(
                30
            ) - self.df["btc_raw_price"].pct_change(30)

            # Drop the temporary btc_price column to avoid confusion
            self.df = self.df.drop(["close_btc"], axis=1)

    def add_core_indicators(self):
        self.add_basic_indicators()
        self.add_advanced_momentum_features()
        self.add_onchain_proxy_features()
        self.add_macro_proxy_features()

        if not self.is_btc:
            self.add_bitcoin_features()

        if self.sentiment_analyzer:
            self.df = self.sentiment_analyzer.add_fear_greed_features(self.df)
            self.df = self.sentiment_analyzer.add_google_trends_log_return_features(
                self.df
            )
            self.df = self.sentiment_analyzer.add_combined_sentiment_features(self.df)

            lag_cols = [
                c
                for c in self.df.columns
                if c.startswith(("fear_greed", "search_", "combined_sentiment"))
            ]
            self.df = self.sentiment_analyzer.add_lagged_sentiment(
                self.df, cols=lag_cols, lags=self.lag_days
            )

        if self.macro_analyzer:
            self.df = self.macro_analyzer.add_all_macro_features(self.df)
            # Add lagged versions of important macro features
            macro_cols = [
                c
                for c in self.df.columns
                if c.startswith(
                    ("gold_", "silver_", "vix_", "stock_fear_greed", "macro_")
                )
                and not c.endswith(
                    ("_ma_30", "_ma_90", "_ma_7", "_ma_10")
                )  # Exclude moving averages
            ]
            for c in macro_cols:
                for l in self.lag_days:
                    self.df[f"{c}_lag{l}"] = self.df[c].shift(l)

        return self.df

    def add_basic_indicators(self):
        # Moving averages
        for period in [10, 20, 50, 200]:
            self.df[f"SMA_{period}"] = self.df["close"].rolling(period).mean()
        self.df["EMA_12"] = self.df["close"].ewm(span=12).mean()
        self.df["EMA_26"] = self.df["close"].ewm(span=26).mean()

        # Volume indicators
        self.df["volume_ma_10"] = self.df["volumefrom"].rolling(10).mean()
        self.df["volume_ratio"] = self.df["volumefrom"] / self.df["volume_ma_10"]

        # RSI
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        self.df["RSI_14"] = 100 - (100 / (1 + rs))

        # Price position
        self.df["price_vs_sma"] = self.df["close"] / self.df["SMA_20"]

        # Support and Resistance
        self.df["support_20"] = self.df["low"].rolling(20).min()
        self.df["resistance_20"] = self.df["high"].rolling(20).max()

        # Volatility
        self.df["volatility_20"] = self.df["close"].rolling(20).std()
        self.df["volatility_ratio"] = (
            self.df["volatility_20"] / self.df["volatility_20"].rolling(50).mean()
        )

        # Momentum
        for period in [1, 3, 5, 7, 14]:
            self.df[f"price_momentum_{period}d"] = self.df["close"].pct_change(period)

    def add_advanced_momentum_features(self):
        # Multiple timeframe RSI
        for period in [7, 21, 30]:
            delta = self.df["close"].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss
            self.df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        self.df["bb_middle"] = self.df["close"].rolling(20).mean()
        bb_std = self.df["close"].rolling(20).std()
        self.df["bb_upper"] = self.df["bb_middle"] + (bb_std * 2)
        self.df["bb_lower"] = self.df["bb_middle"] - (bb_std * 2)
        self.df["bb_position"] = (self.df["close"] - self.df["bb_lower"]) / (
            self.df["bb_upper"] - self.df["bb_lower"]
        )

        # MACD
        self.df["macd"] = self.df["EMA_12"] - self.df["EMA_26"]
        self.df["macd_signal"] = self.df["macd"].ewm(span=9).mean()
        self.df["macd_histogram"] = self.df["macd"] - self.df["macd_signal"]

        # Trend strength
        for period in [10, 30, 90]:
            self.df[f"trend_{period}d"] = (
                self.df["close"] / self.df["close"].shift(period) - 1
            ) * 100

    def add_onchain_proxy_features(self):
        # MVRV Z-Score proxy
        self.df["mvrv_zscore_proxy"] = (
            self.df["close"] - self.df["close"].rolling(365).mean()
        ) / self.df["close"].rolling(365).std()

        # Puell Multiple proxy
        daily_revenue_proxy = self.df["close"] * self.df["volumefrom"]
        self.df["puell_multiple_proxy"] = (
            daily_revenue_proxy / daily_revenue_proxy.rolling(365).mean()
        )

        # NVT proxy
        self.df["nvt_proxy"] = (
            self.df["close"] / self.df["volumefrom"].rolling(30).mean()
        )

        # Network activity proxy
        self.df["active_address_proxy"] = (
            self.df["volumefrom"] / self.df["volumefrom"].rolling(90).mean()
        )

        # Terminal price proxy
        coin_days_proxy = (self.df["close"] * self.df["volumefrom"]).rolling(30).sum()
        self.df["terminal_price_proxy"] = coin_days_proxy.rolling(90).mean()

        # Hash rate proxy
        self.df["hash_rate_proxy"] = (
            self.df["close"].rolling(14).mean() / self.df["volatility_20"]
        )

    def add_macro_proxy_features(self):
        # Market stress
        self.df["market_stress"] = (
            self.df["volatility_20"] / self.df["volatility_20"].rolling(100).mean()
        )

        # Risk sentiment
        self.df["risk_sentiment"] = self.df["volume_ratio"] * self.df["price_vs_sma"]

        # Momentum divergence
        self.df["price_momentum_14"] = self.df["close"].pct_change(14)
        self.df["volume_momentum_14"] = self.df["volumefrom"].pct_change(14)
        self.df["momentum_divergence"] = (
            self.df["price_momentum_14"] - self.df["volume_momentum_14"]
        )

        # Liquidity proxy
        self.df["liquidity_proxy"] = self.df["volumefrom"] / self.df["volatility_20"]


class EnhancedRegimeSpecificPredictor:
    def __init__(self, data_paths, model_params=None, btc_data=None, is_btc=False):
        self.data_paths = data_paths
        self.btc_data = btc_data
        self.is_btc = is_btc
        self.bull_model = None
        self.bear_model = None
        self.unified_model = None
        self.ensemble_model = None
        self.bull_ensemble = None
        self.bear_ensemble = None
        self.feature_df = None
        self.bull_data = None
        self.bear_data = None
        self.bull_features = None
        self.bear_features = None
        self.unified_features = None

        self.model_params = model_params or {
            "n_estimators": 100,
            "learning_rate": 0.01,
            "max_depth": 3,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "min_child_weight": 5,
            "gamma": 0.5,
            "random_state": 42,
        }

    def create_features(self):
        """Create enhanced features with sentiment analysis"""
        print("\nLoading and merging data sources...")

        # Load coin data
        coin_df = pd.read_csv(self.data_paths["coin"], parse_dates=["date"])
        print(
            f"Coin data shape: {coin_df.shape}, Date range: {coin_df['date'].min()} to {coin_df['date'].max()}"
        )

        # Load sentiment data
        fear_greed_df = None
        google_trends_df = None
        gold_df = None
        silver_df = None
        stock_fear_greed_df = None
        vix_df = None

        try:
            if "fear_greed" in self.data_paths:
                fear_greed_df = pd.read_csv(
                    self.data_paths["fear_greed"], parse_dates=["date"]
                )
                print(
                    f"Crypto Fear & Greed data shape: {fear_greed_df.shape}, Date range: {fear_greed_df['date'].min()} to {fear_greed_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load fear/greed data: {e}")

        try:
            if "google_trends" in self.data_paths:
                google_trends_df = pd.read_csv(
                    self.data_paths["google_trends"], parse_dates=["date"]
                )
                print(
                    f"Google Trends data shape: {google_trends_df.shape}, Date range: {google_trends_df['date'].min()} to {google_trends_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load Google Trends data: {e}")

        try:
            if "gold" in self.data_paths:
                gold_df = pd.read_csv(self.data_paths["gold"], parse_dates=["date"])
                print(
                    f"Gold price data shape: {gold_df.shape}, Date range: {gold_df['date'].min()} to {gold_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load gold data: {e}")

        try:
            if "silver" in self.data_paths:
                silver_df = pd.read_csv(self.data_paths["silver"], parse_dates=["date"])
                print(
                    f"Silver price data shape: {silver_df.shape}, Date range: {silver_df['date'].min()} to {silver_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load silver data: {e}")

        try:
            if "stock_fear_greed" in self.data_paths:
                stock_fear_greed_df = pd.read_csv(
                    self.data_paths["stock_fear_greed"], parse_dates=["date"]
                )
                print(
                    f"Stock Fear & Greed data shape: {stock_fear_greed_df.shape}, Date range: {stock_fear_greed_df['date'].min()} to {stock_fear_greed_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load stock fear/greed data: {e}")

        try:
            if "vix" in self.data_paths:
                vix_df = pd.read_csv(self.data_paths["vix"])
                vix_df["date"] = pd.to_datetime(vix_df["DATE"])
                vix_df = vix_df.drop("DATE", axis=1)
                print(
                    f"VIX data shape: {vix_df.shape}, Date range: {vix_df['date'].min()} to {vix_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load VIX data: {e}")

        # Create analyzers
        sentiment_analyzer = SentimentAnalyzer(fear_greed_df, google_trends_df)
        macro_analyzer = MacroAnalyzer(gold_df, silver_df, stock_fear_greed_df, vix_df)

        print("\nCalculating technical indicators and features...")
        # Calculate enhanced technical indicators
        ta = EnhancedTechnicalAnalyser(
            coin_df,
            sentiment_analyzer,
            macro_analyzer,
            btc_df=self.btc_data,
            is_btc=self.is_btc,
        )
        ta_df = ta.add_core_indicators()
        print(f"Feature DataFrame shape after indicators: {ta_df.shape}")

        # Merge all features
        self.feature_df = ta_df.copy()

    def enhanced_market_regime_detection(self, df):
        """Enhanced regime detection"""
        df["trend_strength_short"] = (df["close"] / df["SMA_20"] - 1) * 100
        df["trend_strength_medium"] = (df["close"] / df["SMA_50"] - 1) * 100
        df["volume_trend"] = df["volume_ratio"].rolling(5).mean()

        conditions = [
            (df["trend_strength_short"] > 2)
            & (df["trend_strength_medium"] > 0)
            & (df["volume_trend"] > 1.05),
            (df["trend_strength_short"] < -2) & (df["trend_strength_medium"] < 0),
            (df["trend_strength_short"].abs() <= 2),
        ]

        choices = ["bull", "bear", "neutral"]
        df["detailed_regime"] = np.select(conditions, choices, default="neutral")
        df["bull_market"] = (df["detailed_regime"] == "bull").astype(int)

        return df

    def create_enhanced_bull_market_target(self):
        """Create target predicting bull market within next 7 days"""
        print("Creating enhanced bull market target...")

        if "detailed_regime" not in self.feature_df.columns:
            self.feature_df = self.enhanced_market_regime_detection(self.feature_df)

        self.feature_df["bull_next_week"] = 0

        for i in range(len(self.feature_df)):
            end_idx = min(i + 8, len(self.feature_df))
            future_window = self.feature_df["detailed_regime"].iloc[i + 1 : end_idx]

            if "bull" in future_window.values or "strong_bull" in future_window.values:
                self.feature_df.at[self.feature_df.index[i], "bull_next_week"] = 1

        self.feature_df["target"] = self.feature_df["bull_next_week"]

        target_balance = self.feature_df["target"].mean()
        print(f"Bull market next week target balance: {target_balance:.3f}")

        return self

    def create_features(self):
        """Create enhanced features with sentiment analysis"""
        print("\nLoading and merging data sources...")

        # Load coin data
        coin_df = pd.read_csv(self.data_paths["coin"], parse_dates=["date"])
        print(
            f"Coin data shape: {coin_df.shape}, Date range: {coin_df['date'].min()} to {coin_df['date'].max()}"
        )

        # Load sentiment data
        fear_greed_df = None
        google_trends_df = None
        gold_df = None
        silver_df = None
        stock_fear_greed_df = None
        vix_df = None

        try:
            if "fear_greed" in self.data_paths:
                fear_greed_df = pd.read_csv(
                    self.data_paths["fear_greed"], parse_dates=["date"]
                )
                print(
                    f"Crypto Fear & Greed data shape: {fear_greed_df.shape}, Date range: {fear_greed_df['date'].min()} to {fear_greed_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load fear/greed data: {e}")

        try:
            if "google_trends" in self.data_paths:
                google_trends_df = pd.read_csv(
                    self.data_paths["google_trends"], parse_dates=["date"]
                )
                print(
                    f"Google Trends data shape: {google_trends_df.shape}, Date range: {google_trends_df['date'].min()} to {google_trends_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load Google Trends data: {e}")

        try:
            if "gold" in self.data_paths:
                gold_df = pd.read_csv(self.data_paths["gold"], parse_dates=["date"])
                print(
                    f"Gold price data shape: {gold_df.shape}, Date range: {gold_df['date'].min()} to {gold_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load gold data: {e}")

        try:
            if "silver" in self.data_paths:
                silver_df = pd.read_csv(self.data_paths["silver"], parse_dates=["date"])
                print(
                    f"Silver price data shape: {silver_df.shape}, Date range: {silver_df['date'].min()} to {silver_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load silver data: {e}")

        try:
            if "stock_fear_greed" in self.data_paths:
                stock_fear_greed_df = pd.read_csv(
                    self.data_paths["stock_fear_greed"], parse_dates=["date"]
                )
                print(
                    f"Stock Fear & Greed data shape: {stock_fear_greed_df.shape}, Date range: {stock_fear_greed_df['date'].min()} to {stock_fear_greed_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load stock fear/greed data: {e}")

        try:
            if "vix" in self.data_paths:
                vix_df = pd.read_csv(self.data_paths["vix"])
                vix_df["date"] = pd.to_datetime(vix_df["DATE"])
                vix_df = vix_df.drop("DATE", axis=1)
                print(
                    f"VIX data shape: {vix_df.shape}, Date range: {vix_df['date'].min()} to {vix_df['date'].max()}"
                )
        except Exception as e:
            print(f"Could not load VIX data: {e}")

        # Create analyzers
        sentiment_analyzer = SentimentAnalyzer(fear_greed_df, google_trends_df)
        macro_analyzer = MacroAnalyzer(gold_df, silver_df, stock_fear_greed_df, vix_df)

        print("\nCalculating technical indicators and features...")
        # Calculate enhanced technical indicators
        ta = EnhancedTechnicalAnalyser(
            coin_df,
            sentiment_analyzer,
            macro_analyzer,
            btc_df=self.btc_data,
            is_btc=self.is_btc,
        )
        ta_df = ta.add_core_indicators()
        print(f"Feature DataFrame shape after indicators: {ta_df.shape}")

        # Merge all features
        self.feature_df = ta_df.copy()

    def select_robust_features(self):
        """Enhanced feature selection including log return sentiment"""
        # Define feature groups
        feature_groups = {
            "technical": [
                "RSI_7",
                "RSI_14",
                "price_vs_sma",
                "volume_ratio",
                "price_momentum_5d",
                "bb_position",
                "macd_histogram",
                "volatility_20",
                "volatility_ratio",
            ],
            "onchain": [
                "mvrv_zscore_proxy",
                "puell_multiple_proxy",
                "nvt_proxy",
                "active_address_proxy",
                "terminal_price_proxy",
            ],
            "sentiment": [
                "fear_greed_index",
                "fear_greed_log_return",
                "fear_greed_momentum",
                "extreme_fear",
                "extreme_greed",
                "fear_greed_normalized",
                "combined_sentiment",
            ],
            "search_trends": [
                "search_log_return",
                "search_log_return_ma_7",
                "search_log_return_ma_30",
                "search_log_return_momentum",
                "search_price_momentum_corr",
            ],
            "metals": [
                "gold_price",
                "gold_momentum",
                "gold_trend",
                "gold_log_return",
                "silver_price",
                "silver_momentum",
                "silver_trend",
                "silver_log_return",
                "gold_silver_ratio",
                "gold_silver_ratio_trend",
            ],
            "stock_sentiment": [
                "stock_fear_greed",
                "stock_fear_greed_momentum",
                "stock_fear_greed_log_return",
                "stock_extreme_fear",
                "stock_extreme_greed",
                "fear_greed_divergence",
                "fear_greed_correlation",
            ],
            "market_volatility": [
                "vix",
                "vix_trend",
                "vix_momentum",
                "vix_log_return",
                "extreme_volatility",
                "low_volatility",
                "volatility_divergence",
                "volatility_correlation",
            ],
            "macro_combined": [
                "macro_risk_indicator",
                "macro_extreme_conditions",
                "metals_correlation",
                "gold_vix_correlation",
                "silver_vix_correlation",
            ],
        }

        if not self.is_btc:
            feature_groups["bitcoin_features"] = [
                "btc_price",
                "btc_momentum",
                "btc_trend",
                "btc_log_return",
                "btc_volatility",
                "btc_correlation",
                "btc_beta",
                "btc_relative_strength",
            ]

        # Get lagged features for each group
        for group_name, features in feature_groups.items():
            lagged_features = []
            for feature in features:
                lagged_features.extend(
                    [
                        f"{feature}_lag{i}"
                        for i in range(1, 8)
                        if f"{feature}_lag{i}" in self.feature_df.columns
                    ]
                )
            feature_groups[group_name].extend(lagged_features)

        # Filter out non-numeric and non-existent features
        numeric_features_by_group = {}
        for group_name, features in feature_groups.items():
            numeric_features = []
            for feature in features:
                if feature in self.feature_df.columns:
                    dtype = self.feature_df[feature].dtype
                    is_numeric = (
                        pd.api.types.is_numeric_dtype(dtype)
                        and not pd.api.types.is_categorical_dtype(dtype)
                        and not pd.api.types.is_bool_dtype(dtype)
                    )
                    if is_numeric:
                        numeric_features.append(feature)
                    else:
                        print(
                            f"Skipping non-numeric feature: {feature} (type: {dtype})"
                        )
            if numeric_features:
                numeric_features_by_group[group_name] = numeric_features
                print(
                    f"\nFound {len(numeric_features)} numeric features in {group_name} group"
                )

        # Select top features from each group using mutual information
        from sklearn.feature_selection import mutual_info_classif

        selected_features = []

        # Define minimum features per group based on group size
        min_features_per_group = {
            "technical": 4,
            "onchain": 2,
            "sentiment": 2,
            "search_trends": 2,
            "metals": 3,
            "stock_sentiment": 2,
            "market_volatility": 2,
            "macro_combined": 2,
            "bitcoin_features": 3,  # If present
        }

        print("\nSelecting features from each group:")
        for group_name, features in numeric_features_by_group.items():
            if not features:
                continue

            # Create a clean subset of features for selection
            X = self.feature_df[features].copy()
            y = self.feature_df["target"].copy()

            # Handle any remaining NaN values
            for col in X.columns:
                if X[col].isna().any():
                    print(f"Handling NaN values in {col}")
                    # Replace inf/-inf with NaN
                    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                    # Fill NaN with median
                    X[col] = X[col].fillna(X[col].median())

            # Verify no NaN values remain
            if X.isna().any().any():
                print(
                    f"Warning: NaN values still present in {group_name} group after handling"
                )
                # Drop rows with NaN as a last resort
                clean_idx = ~X.isna().any(axis=1)
                X = X[clean_idx]
                y = y[clean_idx]

            if len(X) == 0:
                print(
                    f"No valid data remaining for {group_name} group after NaN handling"
                )
                continue

            # Calculate mutual information scores
            try:
                mi_scores = mutual_info_classif(X, y, random_state=42)
                # Create feature importance ranking
                feature_importance = pd.Series(mi_scores, index=features)
                feature_importance = feature_importance.sort_values(ascending=False)

                # Select top N features from this group
                n_select = min_features_per_group.get(group_name, 2)
                top_features = feature_importance.head(n_select).index.tolist()

                print(f"\n{group_name} - Top {n_select} features:")
                for i, (feature, score) in enumerate(
                    feature_importance.head(n_select).items(), 1
                ):
                    print(f"{i}. {feature}: {score:.4f}")

                selected_features.extend(top_features)
            except Exception as e:
                print(f"Error selecting features for {group_name} group: {str(e)}")
                continue

        self.unified_features = selected_features
        print(f"\nTotal selected features: {len(self.unified_features)}")

        return self

    def create_ensemble_models(self):
        """Create ensemble of different algorithms"""
        xgb_params = {**self.model_params}
        xgb_params.update(
            {
                "enable_categorical": False,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }
        )

        self.estimators = {
            "xgb": XGBClassifier(**xgb_params),
            "rf": RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight="balanced",
            ),
            "lr": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
        }
        return self

    def train_ensemble_models(self):
        """Train ensemble models"""
        print("Training ensemble models...")

        X_unified = self.feature_df[self.unified_features]
        y_unified = self.feature_df["target"]

        self.create_ensemble_models()

        # Train each model separately
        for name, model in self.estimators.items():
            model.fit(X_unified, y_unified)
            score = accuracy_score(y_unified, model.predict(X_unified))
            print(f"{name} training accuracy: {score:.4f}")

        # Create ensemble prediction function
        def ensemble_predict(X):
            predictions = []
            for model in self.estimators.values():
                try:
                    pred_proba = model.predict_proba(X)[:, 1]
                except:
                    pred = model.predict(X)
                    pred_proba = pred.astype(float)
                predictions.append(pred_proba)

            # Average predictions
            avg_pred = np.mean(predictions, axis=0)
            return (avg_pred > 0.5).astype(int)

        self.ensemble_predict = ensemble_predict

        # Calculate ensemble score
        ensemble_preds = self.ensemble_predict(X_unified)
        ensemble_score = accuracy_score(y_unified, ensemble_preds)
        print(f"Unified ensemble training accuracy: {ensemble_score:.4f}")

        return self

    def train_regime_specific_models(self):
        """
        Train models to predict regime transitions with balanced sampling
        """
        # Create proper regime transition target
        df = self.create_regime_transition_target()

        if len(df) == 0:
            print("❌ No valid regime transition data available")
            return

        if self.unified_features is None:
            print("❌ Features not selected yet")
            return

        # Initialize regime models
        self.regime_models = {"bull": None, "bear": None}

        X = df[self.unified_features]
        y = df["regime_target"]  # Use regime transition target
        current_regimes = df["detailed_regime"]

        print(f"\nTraining regime transition models:")

        for regime in ["bull", "bear"]:
            regime_mask = current_regimes == regime
            if regime_mask.sum() < 30:  # Lower minimum for testing
                print(f"  ⚠ Insufficient {regime} data: {regime_mask.sum()} samples")
                continue

            X_regime = X[regime_mask]
            y_regime = y[regime_mask]

            # Check class balance
            class_counts = y_regime.value_counts()
            print(f"  {regime.upper()} regime training data:")
            print(f"    - Total samples: {len(y_regime)}")
            print(f"    - Future bull: {class_counts.get(1, 0)}")
            print(f"    - Future bear: {class_counts.get(0, 0)}")

            if len(y_regime.unique()) < 2:
                print(f"    ❌ Only one class present - skipping")
                continue

            # Create preprocessing and model pipeline
            try:
                # Adjust XGBoost parameters for better handling of imbalanced data
                model = XGBClassifier(
                    n_estimators=100,  # Increased from 30
                    max_depth=4,  # Slightly increased from 3
                    learning_rate=0.03,  # Reduced from 0.05
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_weight=3,
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=class_counts.get(0, 1)
                    / class_counts.get(1, 1),  # Balance classes
                )

                # Create pipeline with SMOTE
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("smote", SMOTE(random_state=42, sampling_strategy="auto")),
                        ("classifier", model),
                    ]
                )

                # Cross-validation to check for overfitting
                if len(X_regime) >= 20:  # Minimum for CV
                    cv_scores = cross_val_score(
                        pipeline,
                        X_regime,
                        y_regime,
                        cv=3,
                        scoring="f1",  # Using F1 score instead of accuracy
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    print(f"    Cross-validation F1: {cv_mean:.1%} ± {cv_std:.1%}")

                # Train final model
                pipeline.fit(X_regime, y_regime)
                self.regime_models[regime] = pipeline

                # Test prediction on training data
                train_pred = pipeline.predict(X_regime)
                train_acc = accuracy_score(y_regime, train_pred)
                train_precision = precision_score(y_regime, train_pred)
                train_recall = recall_score(y_regime, train_pred)
                train_f1 = f1_score(y_regime, train_pred)

                print(f"    Training metrics:")
                print(f"    - Accuracy:  {train_acc:.1%}")
                print(f"    - Precision: {train_precision:.3f}")
                print(f"    - Recall:    {train_recall:.3f}")
                print(f"    - F1 Score:  {train_f1:.3f}")

                # Detailed class prediction analysis
                print("\n    Detailed Classification Report:")
                print(
                    classification_report(
                        y_regime,
                        train_pred,
                        target_names=["Future Bear", "Future Bull"],
                    )
                )

                # Warning for overfitting
                if hasattr(locals(), "cv_mean") and train_f1 - cv_mean > 0.15:
                    print(
                        f"    ⚠️  WARNING: Possible overfitting (train-CV gap: {train_f1 - cv_mean:.1%})"
                    )

            except Exception as e:
                print(f"    ❌ Training failed: {e}")

        # Store the regime transition data for validation
        self.regime_transition_df = df
        return self

    def train_regime_models(self):
        """Train unified regime models"""
        print("Training unified regime models...")

        # First select features if not already done
        if self.unified_features is None:
            self.select_robust_features()

        # Split data by regime
        bull_mask = self.feature_df["bull_market"] == 1
        bear_mask = self.feature_df["bull_market"] == 0

        self.bull_data = self.feature_df[bull_mask].copy()
        self.bear_data = self.feature_df[bear_mask].copy()

        print(f"Bull market samples: {len(self.bull_data)}")
        print(f"Bear market samples: {len(self.bear_data)}")

        # Train unified model with class balancing
        X_unified = self.feature_df[self.unified_features]
        y_unified = self.feature_df["target"]

        # Calculate class weights
        n_samples = len(y_unified)
        n_bull = y_unified.sum()
        n_bear = n_samples - n_bull

        # Update model parameters with class balancing
        model_params = {**self.model_params}
        model_params.update(
            {
                "scale_pos_weight": n_bear / n_bull if n_bull > 0 else 1,
                "eval_metric": "logloss",
                "enable_categorical": False,
                "objective": "binary:logistic",
            }
        )

        self.unified_model = XGBClassifier(**model_params)
        self.unified_model.fit(X_unified, y_unified)

        # Evaluate on training data
        unified_pred = self.unified_model.predict(X_unified)
        unified_score = accuracy_score(y_unified, unified_pred)
        unified_precision = precision_score(y_unified, unified_pred)
        unified_recall = recall_score(y_unified, unified_pred)
        unified_f1 = f1_score(y_unified, unified_pred)

        print("\nUnified Model Training Metrics:")
        print(f"Accuracy:  {unified_score:.4f}")
        print(f"Precision: {unified_precision:.4f}")
        print(f"Recall:    {unified_recall:.4f}")
        print(f"F1 Score:  {unified_f1:.4f}")

        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(
            classification_report(
                y_unified, unified_pred, target_names=["Bear", "Bull"]
            )
        )

        return self

    def walk_forward_validation(
        self, window_size=365, step_size=30, min_train_size=500
    ):
        """Walk-forward cross-validation with result printing"""
        print("Starting Walk-Forward Cross-Validation...")

        df_sorted = self.feature_df.sort_values("date").reset_index(drop=True)

        results = {
            "unified_model": [],
            "ensemble_model": [],
            "dates": [],
            "actual_values": [],
            "regimes": [],
        }

        start_idx = min_train_size
        prediction_count = 0

        while start_idx + step_size < len(df_sorted):
            if window_size is None:
                train_start = 0
            else:
                train_start = max(0, start_idx - window_size)

            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + step_size, len(df_sorted))

            train_data = df_sorted.iloc[train_start:train_end]
            test_data = df_sorted.iloc[test_start:test_end]

            if len(train_data) < min_train_size or len(test_data) == 0:
                start_idx += step_size
                continue

            try:
                # Retrain models
                X_train = train_data[self.unified_features]
                y_train = train_data["target"]

                # Train unified model
                unified_model = XGBClassifier(**self.model_params)
                unified_model.fit(X_train, y_train)

                # Train ensemble models
                estimators = {}
                xgb_params = {**self.model_params}
                xgb_params.update(
                    {
                        "enable_categorical": False,
                        "objective": "binary:logistic",
                        "eval_metric": "logloss",
                    }
                )

                estimators["xgb"] = XGBClassifier(**xgb_params)
                estimators["rf"] = RandomForestClassifier(
                    n_estimators=50, max_depth=3, random_state=42
                )

                # Train each model
                for name, model in estimators.items():
                    model.fit(X_train, y_train)

                # Make predictions
                for i in range(len(test_data)):
                    current_row = test_data.iloc[i : i + 1]
                    actual = current_row["target"].iloc[0]
                    current_date = current_row["date"].iloc[0]

                    regime_data = self.enhanced_market_regime_detection(
                        current_row.copy()
                    )
                    current_regime = regime_data["detailed_regime"].iloc[0]

                    results["actual_values"].append(actual)
                    results["dates"].append(current_date)
                    results["regimes"].append(current_regime)

                    # Unified model prediction
                    try:
                        unified_pred = unified_model.predict(
                            current_row[self.unified_features]
                        )[0]
                        results["unified_model"].append(unified_pred == actual)
                    except Exception as e:
                        print(f"Unified model error: {e}")
                        results["unified_model"].append(False)

                    # Ensemble prediction
                    try:
                        # Get predictions from each model
                        predictions = []
                        for model in estimators.values():
                            try:
                                pred_proba = model.predict_proba(
                                    current_row[self.unified_features]
                                )[:, 1]
                            except:
                                pred = model.predict(current_row[self.unified_features])
                                pred_proba = pred.astype(float)
                            predictions.append(pred_proba)

                        # Average predictions
                        avg_pred = np.mean(predictions, axis=0)
                        ensemble_pred = (avg_pred > 0.5).astype(int)[0]
                        results["ensemble_model"].append(ensemble_pred == actual)
                    except Exception as e:
                        print(f"Ensemble error: {e}")
                        results["ensemble_model"].append(False)

                prediction_count += len(test_data)

            except Exception as e:
                print(f"Error in walk-forward validation: {e}")

            start_idx += step_size

        print(
            f"\nCompleted walk-forward validation with {prediction_count} predictions"
        )
        print("\nSample Results (first 10):")
        for i in range(min(10, len(results["dates"]))):
            print(
                f"  {results['dates'][i]}: Actual={results['actual_values'][i]}, "
                f"Unified={results['unified_model'][i]}, "
                f"Ensemble={results['ensemble_model'][i]}, "
                f"Regime={results['regimes'][i]}"
            )

        print("=" * 60)
        return results

    def validate_data_for_walkforward(self):
        """Validate data before walk-forward validation"""
        print("Validating data for walk-forward...")

        # Check basic data structure
        print(f"Feature dataframe shape: {self.feature_df.shape}")
        print(
            f"Date range: {self.feature_df['date'].min()} to {self.feature_df['date'].max()}"
        )

        # Check required columns
        required_cols = ["date", "target"] + self.unified_features
        missing_cols = [
            col for col in required_cols if col not in self.feature_df.columns
        ]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False

        # Check for sufficient data
        if len(self.feature_df) < 1000:
            print(
                f"Warning: Only {len(self.feature_df)} data points (recommended: >1000)"
            )

        # Check target distribution
        if "target" in self.feature_df.columns:
            target_balance = self.feature_df["target"].mean()
            print(f"Target balance: {target_balance:.3f}")
            if target_balance < 0.1 or target_balance > 0.9:
                print(f"Warning: Imbalanced target ({target_balance:.1%})")

        # Check for NaN values
        nan_counts = self.feature_df[required_cols].isnull().sum()
        if nan_counts.sum() > 0:
            print(f"NaN values found:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count} NaN values")
            return False

        print("Data validation passed")
        return True

    def analyze_walk_forward_results(self, wf_results):
        """Analyze walk-forward results"""
        print("=" * 60)
        print("WALK-FORWARD CROSS-VALIDATION RESULTS")
        print("=" * 60)

        for model_name in ["unified_model", "ensemble_model"]:
            if wf_results[model_name]:
                accuracy = np.mean(wf_results[model_name])
                std_dev = np.std(wf_results[model_name])
                n_predictions = len(wf_results[model_name])

                print(
                    f"{model_name}: {accuracy:.4f} ± {std_dev:.4f} ({n_predictions} predictions)"
                )

        # Regime distribution
        if "regimes" in wf_results:
            print(f"\nRegime Distribution:")
            regime_counts = pd.Series(wf_results["regimes"]).value_counts()
            # for regime, count in regime_counts.items():
            #     print(f"  {regime}: {count} predictions ({count/len(wf_results['regimes'])*100:.1f}%)")

        # Regime-specific performance
        results_df = pd.DataFrame(
            {
                "date": pd.to_datetime(wf_results["dates"]),
                "unified": wf_results["unified_model"],
                "ensemble": wf_results["ensemble_model"],
                "actual": wf_results["actual_values"],
                "regime": wf_results.get(
                    "regimes", ["unknown"] * len(wf_results["dates"])
                ),
            }
        )

        print(f"\nRegime-Specific Performance (Average Accuracy):")
        regime_performance = results_df.groupby("regime")[
            ["unified", "ensemble"]
        ].mean()
        print(regime_performance)

        return results_df

    def get_feature_importance_analysis(self):
        """Analyze feature importance across models"""
        print("\nAnalyzing Feature Importance...")

        importance_data = []

        # Get unified model importance
        if hasattr(self, "unified_model") and self.unified_model is not None:
            unified_importance = self.unified_model.feature_importances_
            for feat, imp in zip(self.unified_features, unified_importance):
                importance_data.append(
                    {"model": "Unified", "feature": feat, "importance": imp}
                )

        # Get XGBoost importance from our custom ensemble
        if hasattr(self, "estimators") and "xgb" in self.estimators:
            xgb_model = self.estimators["xgb"]
            if hasattr(xgb_model, "feature_importances_"):
                xgb_importance = xgb_model.feature_importances_
                for feat, imp in zip(self.unified_features, xgb_importance):
                    importance_data.append(
                        {"model": "Ensemble_XGB", "feature": feat, "importance": imp}
                    )

        if not importance_data:
            print("No feature importance data available!")
            return pd.DataFrame(columns=["model", "feature", "importance"])

        importance_df = pd.DataFrame(importance_data)

        # Calculate average importance across models
        avg_importance = importance_df.groupby("feature")["importance"].mean()

        print("\nTop 10 Most Important Features (Average):")
        print(avg_importance.sort_values(ascending=False).head(10))

        # Create visualization if there's data
        if len(importance_df) > 0:
            plt.figure(figsize=(12, 8))

            # Plot unified model importance if available
            if "Unified" in importance_df["model"].unique():
                plt.subplot(2, 1, 1)
                unified_importance = importance_df[
                    importance_df["model"] == "Unified"
                ].set_index("feature")["importance"]
                if len(unified_importance) > 0:
                    unified_importance.sort_values(ascending=True).plot(kind="barh")
                    plt.title("Unified Model Feature Importance")
                    plt.xlabel("Importance (Gain)")

            # Plot ensemble XGBoost importance if available
            if "Ensemble_XGB" in importance_df["model"].unique():
                plt.subplot(2, 1, 2)
                ensemble_importance = importance_df[
                    importance_df["model"] == "Ensemble_XGB"
                ].set_index("feature")["importance"]
                if len(ensemble_importance) > 0:
                    ensemble_importance.sort_values(ascending=True).plot(kind="barh")
                    plt.title("Ensemble Model (XGBoost) Feature Importance")
                    plt.xlabel("Importance (Gain)")

            plt.tight_layout()
            plt.show()

            # Feature categories analysis
            print(f"\nFeature Category Analysis:")
            self.analyze_feature_categories(avg_importance)

        return importance_df

    def summarize_lag_importance(self, importance_df):
        # importance_df from get_feature_importance_analysis()
        lag_rows = importance_df[importance_df.feature.str.contains(r"_lag[1-7]$")]
        lag_rows["root"] = lag_rows.feature.str.replace(r"_lag[1-7]$", "", regex=True)
        best_lags = (
            lag_rows.sort_values("importance", ascending=False)
            .groupby("root")
            .head(1)[["root", "feature", "importance"]]
        )
        return best_lags

    def analyze_feature_categories(self, importance_series):
        """Analyze importance by feature categories"""

        # Define feature categories
        categories = {
            "Technical_RSI": ["RSI_7", "RSI_14", "RSI_21", "RSI_30"],
            "Price_Position": ["price_vs_sma", "bb_position"],
            "Momentum": [
                "price_momentum_1d",
                "price_momentum_3d",
                "price_momentum_5d",
                "price_momentum_7d",
                "price_momentum_14d",
            ],
            "Volume": ["volume_ratio", "volume_momentum_14"],
            "Advanced_Technical": ["macd_histogram", "macd", "macd_signal"],
            "OnChain_Proxies": [
                "mvrv_zscore_proxy",
                "puell_multiple_proxy",
                "nvt_proxy",
                "active_address_proxy",
            ],
            "Market_Stress": ["market_stress", "volatility_ratio", "volatility_20"],
            "Sentiment_FearGreed": [
                "fear_greed_index",
                "fear_greed_change",
                "fear_greed_momentum",
                "extreme_fear",
                "extreme_greed",
                "fear_contrarian",
                "greed_contrarian",
            ],
            "Sentiment_Search": [
                "search_log_return",
                "search_log_return_ma_7",
                "search_acceleration",
                "search_price_momentum_corr",
            ],
            "Combined_Sentiment": ["combined_sentiment", "price_sentiment_divergence"],
        }

        category_importance = {}

        for category, features in categories.items():
            category_total = 0
            category_count = 0

            for feature in features:
                if feature in importance_series.index:
                    category_total += importance_series[feature]
                    category_count += 1

            if category_count > 0:
                category_importance[category] = category_total / category_count

        # Sort and display
        sorted_categories = sorted(
            category_importance.items(), key=lambda x: x[1], reverse=True
        )

        print("Feature Category Importance (Average):")
        for category, avg_importance in sorted_categories:
            print(f"  {category:<20} {avg_importance:.4f}")

    def detect_current_regime(self, current_data):
        """
        Detect if current market condition is bull or bear
        """
        # Example regime detection logic - customize based on your criteria
        if "ma_20" in current_data and "ma_50" in current_data:
            if current_data["ma_20"] > current_data["ma_50"] * 1.02:
                return "bull"
            elif current_data["ma_20"] < current_data["ma_50"] * 0.98:
                return "bear"

        # Add more sophisticated regime detection here
        # Could use RSI, sentiment, volatility, etc.
        return "neutral"  # Default fallback

    def create_regime_target(self):
        """
        Create target variable using EXISTING regime detection
        """
        df = self.feature_df.copy()

        # Use your existing detailed_regime column (already created in your pipeline)
        if "detailed_regime" not in df.columns:
            df = self.enhanced_market_regime_detection(df)

        # Create target: regime 7 days ahead using existing regime
        df["future_regime"] = df["detailed_regime"].shift(-7)  # 7 days ahead

        # Convert to binary: bull=1, bear=0 (exclude neutral)
        mask = df["future_regime"].isin(["bull", "bear"])
        df = df[mask].copy()
        df["target"] = (df["future_regime"] == "bull").astype(int)
        df["current_regime"] = df["detailed_regime"]  # Use existing regime

        self.feature_df = df
        print(f"Regime target distribution: {df['target'].value_counts()}")
        print(f"Current regime distribution: {df['current_regime'].value_counts()}")
        return df

    def predict_next_week_regime(self, current_features_row):
        """
        Predict using existing regime from the data row
        """
        # Extract current regime from the data
        if hasattr(current_features_row, "get"):
            current_regime = current_features_row.get("detailed_regime", "neutral")
        else:
            current_regime = "neutral"  # Fallback

        if (
            current_regime in self.regime_models
            and self.regime_models[current_regime] is not None
        ):
            # Get feature values for prediction
            feature_values = current_features_row[self.unified_features].values

            model = self.regime_models[current_regime]
            prediction = model.predict_proba(feature_values.reshape(1, -1))[0]

            return {
                "current_regime": current_regime,
                "model_used": f"{current_regime}_model",
                "bull_probability": prediction[1] if len(prediction) > 1 else 0.5,
                "bear_probability": prediction[0] if len(prediction) > 1 else 0.5,
                "prediction": (
                    "bull" if (len(prediction) > 1 and prediction[1] > 0.5) else "bear"
                ),
            }
        else:
            return {
                "current_regime": current_regime,
                "model_used": "none",
                "prediction": "neutral",
                "error": f"No model available for {current_regime} regime",
            }

    def walk_forward_validation_regime_specific(
        self, window_size=600, step_size=30, min_train_size=300
    ):
        """
        Walk-forward validation for regime-specific models
        """
        if not hasattr(self, "regime_transition_df"):
            print(
                "❌ No regime transition data. Run train_regime_specific_models() first."
            )
            return {
                "predictions": [],
                "actual": [],
                "regimes_used": [],
                "models_used": [],
            }

        df = self.regime_transition_df.copy()
        results = {
            "predictions": [],
            "actual": [],
            "regimes_used": [],
            "models_used": [],
        }

        print(f"Starting regime-specific validation with {len(df)} samples")

        n_samples = len(df)
        start_idx = min_train_size

        while start_idx + step_size < n_samples:
            # Training window
            if window_size is None:
                train_start = 0
            else:
                train_start = max(0, start_idx - window_size)

            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + step_size, n_samples)

            train_data = df.iloc[train_start:train_end].copy()
            test_data = df.iloc[test_start:test_end].copy()

            if len(train_data) < min_train_size or len(test_data) == 0:
                start_idx += step_size
                continue

            # Temporarily train models on this window
            temp_regime_models = {"bull": None, "bear": None}

            X_train = train_data[self.unified_features]
            y_train = train_data["regime_target"]
            train_regimes = train_data["detailed_regime"]

            # Train models for this window
            for regime in ["bull", "bear"]:
                regime_mask = train_regimes == regime
                if regime_mask.sum() < 10:  # Very low minimum for testing
                    continue

                X_regime = X_train[regime_mask]
                y_regime = y_train[regime_mask]

                if len(y_regime.unique()) < 2:
                    continue

                try:
                    model = XGBClassifier(
                        n_estimators=30,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric="logloss",
                    )
                    model.fit(X_regime, y_regime)
                    temp_regime_models[regime] = model
                except:
                    continue

            # Make predictions on test data
            for _, row in test_data.iterrows():
                current_regime = row["detailed_regime"]
                actual_future = "bull" if row["regime_target"] == 1 else "bear"

                if (
                    current_regime in temp_regime_models
                    and temp_regime_models[current_regime] is not None
                ):
                    try:
                        features = row[self.unified_features].values.reshape(1, -1)
                        pred_proba = temp_regime_models[current_regime].predict_proba(
                            features
                        )[0]
                        predicted = "bull" if pred_proba[1] > 0.5 else "bear"
                        model_used = f"{current_regime}_model"
                    except:
                        predicted = "bear"  # Default fallback
                        model_used = "fallback"
                else:
                    predicted = "bear"  # Default fallback
                    model_used = "none"

                results["predictions"].append(predicted)
                results["actual"].append(actual_future)
                results["regimes_used"].append(current_regime)
                results["models_used"].append(model_used)

            start_idx += step_size

        # Calculate accuracy
        if results["predictions"]:
            correct = sum(
                1 for p, a in zip(results["predictions"], results["actual"]) if p == a
            )
            accuracy = correct / len(results["predictions"])
            print(f"Regime-Specific Model Accuracy: {accuracy:.1%}")
            print(f"Total predictions: {len(results['predictions'])}")
            print(f"Correct predictions: {correct}")
        else:
            print("❌ No predictions made!")

        return results

    def create_regime_transition_target(self):
        """
        Create a proper target for regime prediction (bull/bear next week)
        """
        df = self.feature_df.copy()

        # Create future regime target (7 days ahead)
        df["future_regime"] = df["detailed_regime"].shift(-7)

        # Only keep rows where we have both current and future regime
        df = df.dropna(subset=["detailed_regime", "future_regime"])

        # Filter to only bull/bear transitions (exclude neutral)
        current_bull_bear = df["detailed_regime"].isin(["bull", "bear"])
        future_bull_bear = df["future_regime"].isin(["bull", "bear"])
        mask = current_bull_bear & future_bull_bear

        df = df[mask].copy()

        # Create binary target: 1 if future is bull, 0 if future is bear
        df["regime_target"] = (df["future_regime"] == "bull").astype(int)

        print(f"Regime transition data:")
        print(f"  Total samples: {len(df)}")
        print(f"  Current regime distribution: {df['detailed_regime'].value_counts()}")
        print(f"  Future regime distribution: {df['future_regime'].value_counts()}")
        print(f"  Target distribution: {df['regime_target'].value_counts()}")

        return df

    def analyze_regime_model_feature_importance(self):
        """
        Analyze feature importance for bull and bear regime models
        """
        print("\n" + "=" * 70)
        print("REGIME-SPECIFIC MODEL FEATURE IMPORTANCE")
        print("=" * 70)

        regime_importance = {}

        for regime in ["bull", "bear"]:
            if regime in self.regime_models and self.regime_models[regime] is not None:
                pipeline = self.regime_models[regime]
                # Get the XGBoost classifier from the pipeline
                model = pipeline.named_steps["classifier"]

                # Get feature importance
                if hasattr(model, "feature_importances_"):
                    importance_scores = model.feature_importances_

                    # Create importance DataFrame
                    importance_df = pd.DataFrame(
                        {
                            "feature": self.unified_features,
                            "importance": importance_scores,
                            "regime": regime,
                        }
                    ).sort_values("importance", ascending=False)

                    regime_importance[regime] = importance_df

                    # Print top features for this regime
                    print(
                        f"\n{regime.upper()} REGIME MODEL - Top 20 Most Important Features:"
                    )
                    print("-" * 65)
                    for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
                        print(f"{i:2d}. {row['feature']:35s} {row['importance']:.4f}")

                    # Feature category analysis for this regime
                    print(f"\n{regime.upper()} REGIME - Feature Category Analysis:")
                    print("-" * 50)

                    # Categorize features
                    categories = {
                        "Price_Position": [
                            "price_vs_sma",
                            "bb_position",
                            "support_20",
                            "resistance_20",
                        ],
                        "Technical_RSI": ["RSI_14", "RSI_7", "RSI_21", "RSI_30"],
                        "Momentum": [
                            "price_momentum_1d",
                            "price_momentum_3d",
                            "price_momentum_5d",
                            "price_momentum_7d",
                            "price_momentum_14d",
                            "volume_momentum_14",
                        ],
                        "Moving_Averages": [
                            "SMA_10",
                            "SMA_20",
                            "SMA_50",
                            "SMA_200",
                            "EMA_12",
                            "EMA_26",
                        ],
                        "Volatility": [
                            "volatility_5d",
                            "volatility_10d",
                            "volatility_30d",
                            "volatility_20",
                        ],
                        "Bitcoin_Features": [
                            "btc_price",
                            "btc_momentum",
                            "btc_trend",
                            "btc_log_return",
                            "btc_volatility",
                            "btc_correlation",
                            "btc_beta",
                            "btc_relative_strength",
                        ],
                        "Sentiment_FearGreed": [
                            "fear_greed_index",
                            "fear_greed_ma_7",
                            "fear_greed_ma_30",
                            "extreme_fear",
                            "extreme_greed",
                            "fear_greed_normalized",
                        ],
                        "Sentiment_Search": [
                            "search_log_return",
                            "search_log_return_ma_7",
                            "search_volatility",
                        ],
                        "Metals": [
                            "gold_price",
                            "gold_momentum",
                            "gold_trend",
                            "silver_price",
                            "silver_momentum",
                            "silver_trend",
                            "gold_silver_ratio",
                        ],
                        "Stock_Sentiment": [
                            "stock_fear_greed",
                            "stock_fear_greed_momentum",
                            "fear_greed_divergence",
                            "fear_greed_correlation",
                        ],
                        "Market_Volatility": [
                            "vix",
                            "vix_trend",
                            "vix_momentum",
                            "volatility_divergence",
                            "volatility_correlation",
                        ],
                        "Macro_Combined": [
                            "macro_risk_indicator",
                            "macro_extreme_conditions",
                            "metals_correlation",
                            "gold_vix_correlation",
                            "silver_vix_correlation",
                        ],
                    }

                    category_importance = {}
                    for category, features in categories.items():
                        category_features = [
                            f for f in features if f in importance_df["feature"].values
                        ]
                        if category_features:
                            avg_importance = importance_df[
                                importance_df["feature"].isin(category_features)
                            ]["importance"].mean()
                            category_importance[category] = avg_importance

                    # Sort and display category importance
                    sorted_categories = sorted(
                        category_importance.items(), key=lambda x: x[1], reverse=True
                    )
                    for category, avg_imp in sorted_categories:
                        print(f"  {category:20s}: {avg_imp:.4f}")

                else:
                    print(
                        f"Cannot extract feature importance for {regime} regime model"
                    )

        # Create comparison visualization
        if len(regime_importance) >= 2:
            self._create_regime_comparison_plot(regime_importance)

        # Save detailed results
        if regime_importance:
            all_regime_importance = pd.DataFrame()
            for regime, df in regime_importance.items():
                df_copy = df.copy()
                df_copy["regime"] = regime
                all_regime_importance = pd.concat(
                    [all_regime_importance, df_copy], ignore_index=True
                )

            all_regime_importance.to_csv(
                "regime_model_feature_importance.csv", index=False
            )
            print(f"\n📁 Saved: regime_model_feature_importance.csv")

        return regime_importance

    def _create_regime_comparison_plot(self, regime_importance):
        """
        Create side-by-side comparison of regime feature importance
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        colors = {"bull": "green", "bear": "red"}

        for idx, (regime, importance_df) in enumerate(regime_importance.items()):
            ax = axes[idx]

            # Plot top 20 features
            top_features = importance_df.head(20)
            bars = ax.barh(
                range(len(top_features)),
                top_features["importance"],
                color=colors.get(regime, "blue"),
                alpha=0.7,
            )

            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features["feature"], fontsize=10)
            ax.set_xlabel("Feature Importance", fontsize=12)
            ax.set_title(
                f"{regime.upper()} Regime Model\nTop 20 Features",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(
                    width + width * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}",
                    ha="left",
                    va="center",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig("regime_model_feature_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(f"📊 Saved: regime_model_feature_comparison.png")
