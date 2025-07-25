from nodiensenv.analyser_bull_bear_test import EnhancedRegimeSpecificPredictor
from nodiensenv.constants import DATA_DIR
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

coin_name = "DOGE"
data_paths = {
    "coin": DATA_DIR / f"price_mcap_{coin_name}.csv",
    "bitcoin": DATA_DIR / "price_mcap_BTC.csv",  # Always include BTC data
    "google_trends": DATA_DIR / f"trend_log_returns_{coin_name}.csv",
    "fear_greed": DATA_DIR / "index_crypto_fear_greed.csv",
    "gold": DATA_DIR / "metal_price_gold.csv",
    "silver": DATA_DIR / "metal_price_silver.csv",
    "stock_fear_greed": DATA_DIR / "index_stock_fear_greed.csv",
    "vix": DATA_DIR / "index_vix.csv",
}

print(f"Initializing ENHANCED {coin_name} Bull Market Predictor...")

# Load Bitcoin data if analyzing an altcoin
btc_df = None
if coin_name != "BTC":
    try:
        btc_df = pd.read_csv(data_paths["bitcoin"], parse_dates=["date"])
        print(f"Loaded Bitcoin data for {coin_name} analysis")
    except Exception as e:
        print(f"Warning: Could not load Bitcoin data: {e}")

predictor = EnhancedRegimeSpecificPredictor(
    data_paths=data_paths, btc_data=btc_df, is_btc=(coin_name == "BTC")
)

# Execute enhanced pipeline
print("Starting enhanced feature engineering...")
predictor.create_features()

print("Detecting market regimes...")
predictor.feature_df = predictor.enhanced_market_regime_detection(predictor.feature_df)
print("Creating bull market target...")
predictor.create_enhanced_bull_market_target()

# Handle NaN values
print("\nHandling missing values...")
# First, identify essential columns that must not have NaN values
essential_columns = [
    "date",
    "close",
    "target",  # Basic price and target
    "RSI_14",
    "volume_ratio",
    "price_vs_sma",  # Core technical indicators
    "volatility_20",
    "price_momentum_5d",  # Core volatility and momentum
    "detailed_regime",  # Regime information
]

# Drop rows where essential columns have NaN values
original_shape = predictor.feature_df.shape
predictor.feature_df = predictor.feature_df.dropna(subset=essential_columns)
print(f"Shape changed from {original_shape} to {predictor.feature_df.shape}")
print(f"Rows removed: {original_shape[0] - predictor.feature_df.shape[0]}")

# For non-essential features, fill NaN values with appropriate methods
# Moving averages and rolling calculations - forward fill then backward fill
rolling_cols = [
    col
    for col in predictor.feature_df.columns
    if any(x in col for x in ["_ma_", "_correlation", "momentum", "trend"])
]
predictor.feature_df[rolling_cols] = (
    predictor.feature_df[rolling_cols].fillna(method="ffill").fillna(method="bfill")
)

# Handle categorical columns first
categorical_cols = predictor.feature_df.select_dtypes(include=["category"]).columns
for col in categorical_cols:
    # Get existing categories
    categories = predictor.feature_df[col].cat.categories
    # Add a default category for NaN values if needed
    if "neutral" in categories:
        fill_value = "neutral"
    elif "normal_vol" in categories:
        fill_value = "normal_vol"
    else:
        fill_value = categories[0]  # Use first category as default
    # Fill NaN values
    predictor.feature_df[col] = predictor.feature_df[col].fillna(fill_value)

# Binary indicators - fill with 0 (excluding categorical columns)
binary_cols = [
    col
    for col in predictor.feature_df.columns
    if any(x in col for x in ["extreme_", "_regime", "_contrarian"])
    and col not in categorical_cols
]
predictor.feature_df[binary_cols] = predictor.feature_df[binary_cols].fillna(0)

# Log returns - fill with 0
log_return_cols = [col for col in predictor.feature_df.columns if "log_return" in col]
predictor.feature_df[log_return_cols] = predictor.feature_df[log_return_cols].fillna(0)

# Handle remaining numeric columns
numeric_cols = predictor.feature_df.select_dtypes(include=["float64", "int64"]).columns
for col in numeric_cols:
    # Replace infinite values with NaN first
    predictor.feature_df[col] = predictor.feature_df[col].replace(
        [np.inf, -np.inf], np.nan
    )

    # Calculate reasonable bounds for each feature
    if (
        predictor.feature_df[col].notna().any()
    ):  # Only process if we have any non-NaN values
        q1 = predictor.feature_df[col].quantile(0.01)
        q99 = predictor.feature_df[col].quantile(0.99)
        iqr = q99 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q99 + 3 * iqr

        # Clip values to bounds
        predictor.feature_df[col] = predictor.feature_df[col].clip(
            lower_bound, upper_bound
        )

        # Fill any remaining NaN values with median for that column
        median_val = predictor.feature_df[col].median()
        predictor.feature_df[col] = predictor.feature_df[col].fillna(median_val)

# Verify no infinite values remain
inf_check = (
    np.isinf(predictor.feature_df.select_dtypes(include=["float64", "int64"]))
    .sum()
    .sum()
)
if inf_check > 0:
    print("\nWarning: Some infinite values remain. Replacing with 0...")
    predictor.feature_df = predictor.feature_df.replace([np.inf, -np.inf], 0)

print(f"\nFinal Feature DataFrame shape: {predictor.feature_df.shape}")

# Check for any remaining NaN values
nan_counts = predictor.feature_df.isna().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print("\nWarning: Found columns with NaN values:")
    for col, count in nan_cols.items():
        print(f"  {col}: {count} NaN values")
        # Get column type
        col_type = predictor.feature_df[col].dtype
        print(f"  Column type: {col_type}")
        # Fill these remaining NaNs with appropriate values based on column type
        if pd.api.types.is_numeric_dtype(col_type):
            predictor.feature_df[col] = predictor.feature_df[col].fillna(
                predictor.feature_df[col].median()
            )
        elif pd.api.types.is_categorical_dtype(col_type):
            # For categorical, fill with mode
            predictor.feature_df[col] = predictor.feature_df[col].fillna(
                predictor.feature_df[col].mode()[0]
            )
        else:
            # For other types, fill with 0
            predictor.feature_df[col] = predictor.feature_df[col].fillna(0)
    print("\nAfter final NaN handling:")
    remaining_nans = predictor.feature_df.isna().sum()
    remaining_nan_cols = remaining_nans[remaining_nans > 0]
    if len(remaining_nan_cols) > 0:
        print("Still have NaN values in:")
        print(remaining_nan_cols)
    else:
        print("All NaN values have been handled")

# Select features first
print("\nSelecting robust features...")
# Print the feature groups being used
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
}
print("\nChecking feature availability:")
for group, features in feature_groups.items():
    available = [f for f in features if f in predictor.feature_df.columns]
    missing = [f for f in features if f not in predictor.feature_df.columns]
    print(f"\n{group}:")
    print(f"  Available: {available}")
    if missing:
        print(f"  Missing: {missing}")
    # Check for NaN in available features
    if available:
        nan_in_group = predictor.feature_df[available].isna().sum()
        nan_features = nan_in_group[nan_in_group > 0]
        if not nan_features.empty:
            print(f"  NaN values found in: {nan_features.index.tolist()}")

predictor.select_robust_features()

if predictor.unified_features is None:
    print("❌ Feature selection failed!")
    exit()
else:
    print(f"✅ Selected {len(predictor.unified_features)} features")


# NEW: Train regime-specific models (bull/bear prediction models)
print("Available columns in feature_df:")
if predictor.feature_df is not None:
    print(list(predictor.feature_df.columns))
else:
    print("feature_df is None!")
print("\nSelected unified_features:")
print(predictor.unified_features)

print("Training regime-specific models...")
predictor.train_regime_specific_models()

# Keep your original unified models as well
print("Training unified models...")
predictor.train_regime_models()  # Your original method
predictor.train_ensemble_models()

# Walk-forward validation - BOTH approaches
print("\n" + "=" * 70)
print("WALK-FORWARD CROSS-VALIDATION")
print("=" * 70)

# Configuration 1: Original unified model approach
print("Testing Original Unified Models...")
print("Rolling Window (600 days)...")
wf_results_rolling = predictor.walk_forward_validation(
    window_size=600, step_size=30, min_train_size=300
)

print("Expanding Window...")
wf_results_expanding = predictor.walk_forward_validation(
    window_size=None, step_size=30, min_train_size=300
)

# Configuration 2: NEW regime-specific approach
print("\n" + "=" * 50)
print("REGIME-SPECIFIC MODEL VALIDATION")
print("=" * 50)

print("Testing Regime-Specific Models (Rolling Window)...")
regime_results_rolling = predictor.walk_forward_validation_regime_specific(
    window_size=600, step_size=30, min_train_size=300
)

print("Testing Regime-Specific Models (Expanding Window)...")
regime_results_expanding = predictor.walk_forward_validation_regime_specific(
    window_size=None, step_size=30, min_train_size=300
)

# Organize all results
all_configurations = {
    "Unified_Rolling": wf_results_rolling,
    "Unified_Expanding": wf_results_expanding,
    "Regime_Rolling": regime_results_rolling,
    "Regime_Expanding": regime_results_expanding,
}

# Analyze each configuration
for configuration, results in all_configurations.items():
    print("\n" + "=" * 70)
    print(f"RESULTS ANALYSIS FOR {configuration}")
    print("=" * 70)

    if configuration.startswith("Regime"):
        # Handle regime-specific results
        if results.get("predictions") and len(results["predictions"]) > 0:
            total_predictions = len(results["predictions"])
            correct_predictions = sum(
                1 for p, a in zip(results["predictions"], results["actual"]) if p == a
            )
            accuracy = correct_predictions / total_predictions
            y_true = [1 if a == "bull" else 0 for a in results["actual"]]
            y_pred = [1 if p == "bull" else 0 for p in results["predictions"]]
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            print(f"Total predictions made: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Regime-Specific Model Accuracy: {accuracy:.3f}")
            print(f"Regime-Specific Model Precision: {precision:.3f}")
            print(f"Regime-Specific Model Recall: {recall:.3f}")
            print(f"Regime-Specific Model F1 Score: {f1:.3f}")

            # Regime usage analysis
            regime_usage = pd.Series(results["regimes_used"]).value_counts()
            print(f"\nRegime Model Usage:")
            for regime, count in regime_usage.items():
                percentage = count / total_predictions * 100
                print(
                    f"  • {regime.title()} model used: {count} times ({percentage:.1f}%)"
                )

            # Model usage analysis
            model_usage = pd.Series(results["models_used"]).value_counts()
            print(f"\nModel Type Usage:")
            for model, count in model_usage.items():
                percentage = count / total_predictions * 100
                print(f"  • {model}: {count} times ({percentage:.1f}%)")

            # Performance assessment
            print(f"\nPERFORMANCE ASSESSMENT for {configuration}:")
            if accuracy > 0.70:
                print(
                    f"EXCELLENT: {accuracy:.1%} accuracy is exceptional for regime prediction!"
                )
            elif accuracy > 0.60:
                print(
                    f"VERY GOOD: {accuracy:.1%} accuracy is strong for regime transitions!"
                )
            elif accuracy > 0.55:
                print(
                    f"GOOD: {accuracy:.1%} accuracy shows meaningful predictive value!"
                )
            elif accuracy > 0.52:
                print(
                    f"MODERATE: {accuracy:.1%} accuracy is above random but could be improved"
                )
            else:
                print(f"POOR: {accuracy:.1%} accuracy needs significant improvement")

            regime_df = pd.DataFrame(
                {
                    "prediction": results["predictions"],
                    "actual": results["actual"],
                    "regime_used": results["regimes_used"],
                    "model_used": results["models_used"],
                }
            )
            regime_df.to_csv(
                f"{coin_name}_regime_specific_results_{configuration}.csv", index=False
            )

        else:
            print("No regime-specific predictions were generated!")

    else:
        # Handle unified model results (your original code)
        if results.get("unified_model") and len(results["unified_model"]) > 0:
            # Your existing analysis code
            results_df = predictor.analyze_walk_forward_results(results)
            results_df.to_csv(
                f"{coin_name}_enhanced_walk_forward_results_{configuration}.csv",
                index=False,
            )

            importance_df = predictor.get_feature_importance_analysis()
            best_lags = predictor.summarize_lag_importance(importance_df)
            print(best_lags)  # quick view

            unified_acc = sum(results["unified_model"]) / len(results["unified_model"])
            ensemble_acc = sum(results["ensemble_model"]) / len(
                results["ensemble_model"]
            )

            print(f"WALK-FORWARD VALIDATION RESULTS for {configuration}:")
            print(
                f"  • Unified Model:  {unified_acc:.1%} accuracy ({len(results['unified_model'])} predictions)"
            )
            print(
                f"  • Ensemble Model: {ensemble_acc:.1%} accuracy ({len(results['ensemble_model'])} predictions)"
            )

            # Performance assessment
            print(f"\nPERFORMANCE ASSESSMENT for {configuration}:")
            if unified_acc > 0.70:
                print(
                    f"EXCELLENT: {unified_acc:.1%} accuracy is exceptional for financial prediction!"
                )
            elif unified_acc > 0.60:
                print(
                    f"VERY GOOD: {unified_acc:.1%} accuracy is strong for {coin_name} prediction!"
                )
            elif unified_acc > 0.55:
                print(
                    f"GOOD: {unified_acc:.1%} accuracy shows meaningful predictive value!"
                )
            elif unified_acc > 0.52:
                print(
                    f"MODERATE: {unified_acc:.1%} accuracy is above random but could be improved"
                )
            else:
                print(f"POOR: {unified_acc:.1%} accuracy needs significant improvement")

            # Baseline comparisons
            target_balance = predictor.feature_df["target"].mean()
            majority_baseline = max(target_balance, 1 - target_balance)

            print(f"\nBASELINE COMPARISONS for {configuration}:")
            print(f"  • Random Baseline:     50.0%")
            print(f"  • Majority Class:      {majority_baseline:.1%}")
            print(f"  • Your Model:          {unified_acc:.1%}")
            print(f"  • Edge over Random:    +{(unified_acc - 0.5)*100:.1f}%")
            print(
                f"  • Edge over Majority:  {(unified_acc - majority_baseline)*100:+.1f}%"
            )

        else:
            print("No unified model predictions were generated!")

# FINAL COMPREHENSIVE COMPARISON
print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

print("\nModel Performance Summary:")
print("-" * 50)

for config_name, results in all_configurations.items():
    if config_name.startswith("Regime"):
        if results.get("predictions"):
            accuracy = sum(
                1 for p, a in zip(results["predictions"], results["actual"]) if p == a
            ) / len(results["predictions"])
            print(
                f"{config_name:20s}: {accuracy:.1%} accuracy ({len(results['predictions'])} predictions)"
            )
        else:
            print(f"{config_name:20s}: No predictions generated")
    else:
        if results.get("unified_model"):
            unified_acc = sum(results["unified_model"]) / len(results["unified_model"])
            print(
                f"{config_name:20s}: {unified_acc:.1%} accuracy ({len(results['unified_model'])} predictions)"
            )
        else:
            print(f"{config_name:20s}: No predictions generated")


print("\n" + "=" * 70)
print("ANALYZING REGIME MODEL FEATURE IMPORTANCE")
print("=" * 70)

# Analyze feature importance for bull/bear models
regime_feature_importance = predictor.analyze_regime_model_feature_importance()


print(f"\n ENHANCED REGIME-SPECIFIC {coin_name} PREDICTION ANALYSIS COMPLETE!")
