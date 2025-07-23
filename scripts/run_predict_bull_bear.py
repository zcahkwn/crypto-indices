from nodiensenv.analyser_bull_bear import EnhancedRegimeSpecificPredictor
from nodiensenv.constants import DATA_DIR
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data_paths = {
    "bitcoin": DATA_DIR / "price_mcap_BTC.csv",
    "google_trends": DATA_DIR / "BTC_trend_log_returns_2018-2025.csv",
    "fear_greed": DATA_DIR / "fear_greed_index.csv",
}

print("Initializing ENHANCED Bitcoin Bull Market Predictor...")
predictor = EnhancedRegimeSpecificPredictor(data_paths=data_paths)

# Execute enhanced pipeline
print("Starting enhanced feature engineering...")
predictor.create_features()

# Select features first
print("Selecting robust features...")
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

            # Save regime-specific results
            regime_df = pd.DataFrame(
                {
                    "prediction": results["predictions"],
                    "actual": results["actual"],
                    "regime_used": results["regimes_used"],
                    "model_used": results["models_used"],
                }
            )
            regime_df.to_csv(
                f"regime_specific_results_{configuration}.csv", index=False
            )

        else:
            print("No regime-specific predictions were generated!")

    else:
        # Handle unified model results (your original code)
        if results.get("unified_model") and len(results["unified_model"]) > 0:
            # Your existing analysis code
            results_df = predictor.analyze_walk_forward_results(results)
            results_df.to_csv(
                f"enhanced_walk_forward_results_{configuration}.csv", index=False
            )

            importance_df = predictor.get_feature_importance_analysis()
            importance_df.to_csv(
                f"feature_importance_detailed_{configuration}.csv", index=False
            )

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
                    f"VERY GOOD: {unified_acc:.1%} accuracy is strong for Bitcoin prediction!"
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


print(f"\n ENHANCED REGIME-SPECIFIC BITCOIN PREDICTION ANALYSIS COMPLETE!")
