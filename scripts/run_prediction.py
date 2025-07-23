from nodiensenv.analyser_ohlc import PricePredictor
from nodiensenv.constants import DATA_DIR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

coin_name = "BTC"

data_paths = {
    "bitcoin": DATA_DIR / f"{coin_name}_price_mcap.csv",  # OHLCV data
    "google": DATA_DIR
    / f"{coin_name}_trend_log_returns_2019-2025.csv",  # Google trends data
    "fear_greed": DATA_DIR / "fear_greed_index.csv",  # Fear & Greed index data
}

# Initialize predictor
print("Initializing Bitcoin Price Predictor...")
predictor = PricePredictor(
    data_paths=data_paths,
    # model_params={
    #     "n_estimators": 1000,
    #     "learning_rate": 0.01,
    #     "max_depth": 6,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8,
    #     "random_state": 42,
    # },
    # # Try these parameters for better performance
    model_params={
        "n_estimators": 1500,
        "learning_rate": 0.005,  # Lower learning rate
        "max_depth": 8,  # Deeper trees
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 0.1,  # L2 regularization
        "random_state": 42,
    },
)

try:
    # Execute full pipeline
    print("Starting feature engineering...")
    predictor.create_features()

    print("\nStarting model training...")
    predictor.train_model(n_splits=5)

    # Save the trained model
    predictor.save_model("bitcoin_predictor_model.pkl")

    # Generate predictions on the last 30 days for demonstration
    print("\nGenerating sample predictions...")
    recent_data = predictor.feature_df.tail(30)
    predictions, probabilities = predictor.predict(recent_data)

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "date": recent_data["date"].values,
            "actual_price": recent_data["close"].values,
            "predicted_direction": predictions,
            "prob_down": probabilities[:, 0],
            "prob_up": probabilities[:, 1],
        }
    )

    print("\nSample Predictions (Last 30 days):")
    print(results_df.head(10))

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot 1: Price and predictions
    plt.subplot(2, 2, 1)
    plt.plot(
        results_df["date"],
        results_df["actual_price"],
        label="BTC Price",
        linewidth=2,
    )
    up_days = results_df[results_df["predicted_direction"] == 1]
    down_days = results_df[results_df["predicted_direction"] == 0]
    plt.scatter(
        up_days["date"],
        up_days["actual_price"],
        color="green",
        alpha=0.7,
        label="Predicted Up",
    )
    plt.scatter(
        down_days["date"],
        down_days["actual_price"],
        color="red",
        alpha=0.7,
        label="Predicted Down",
    )
    plt.title("Bitcoin Price with Predictions")
    plt.legend()
    plt.xticks(rotation=45)

    # Plot 2: Prediction probabilities
    plt.subplot(2, 2, 2)
    plt.plot(results_df["date"], results_df["prob_up"], label="Prob Up", color="green")
    plt.plot(
        results_df["date"], results_df["prob_down"], label="Prob Down", color="red"
    )
    plt.title("Prediction Probabilities")
    plt.legend()
    plt.xticks(rotation=45)

    # Plot 3: Feature importance (top 15)
    plt.subplot(2, 2, 3)
    feature_importance = (
        pd.DataFrame(
            {
                "feature": predictor.feature_columns,
                "importance": predictor.model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    sns.barplot(data=feature_importance, y="feature", x="importance")
    plt.title("Top 15 Feature Importance")

    # Plot 4: Market phases
    plt.subplot(2, 2, 4)
    market_data = predictor.feature_df.tail(200)  # Last 200 days
    bull_periods = market_data[market_data["market_phase"] == 1]
    bear_periods = market_data[market_data["market_phase"] == 0]

    plt.plot(market_data["date"], market_data["close"], color="black", alpha=0.7)
    plt.scatter(
        bull_periods["date"],
        bull_periods["close"],
        color="green",
        alpha=0.5,
        s=10,
        label="Bull Market",
    )
    plt.scatter(
        bear_periods["date"],
        bear_periods["close"],
        color="red",
        alpha=0.5,
        s=10,
        label="Bear Market",
    )
    plt.title("Market Phases (Last 200 days)")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("bitcoin_prediction_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Save results
    results_df.to_csv("prediction_results.csv", index=False)
    print(f"\nResults saved to 'prediction_results.csv'")
    print(f"Visualization saved to 'bitcoin_prediction_results.png'")

except FileNotFoundError as e:
    print(f"Error: Could not find data file. Please check your file paths.")
    print(f"Expected files: {list(data_paths.values())}")
    print(f"Error details: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()
