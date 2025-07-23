import pandas as pd
import numpy as np
from functools import reduce
from nodiensenv.constants import DATA_DIR, FIGURE_DIR
from nodiensenv.analyser_trendprice import TrendAnalyser

keydates = [
    "2013-04-18",
    "2014-01-30",
    "2015-07-07",
    "2018-01-14",
    "2019-04-02",
    "2021-05-19",
    "2023-01-19",
    "2025-06-01",
]

coin_name = "BTC"

trend_df = pd.read_csv(
    DATA_DIR / f"{coin_name}_trend_log_returns_2013-2025.csv", parse_dates=["date"]
)

price_df = pd.read_csv(
    DATA_DIR / f"{coin_name}_price_mcap_2013-2025.csv", parse_dates=["date"]
)

price_df["abs_price_log_return"] = price_df["price_log_return"].abs()
trend_df["abs_trend_log_return"] = trend_df["trend_log_return"].abs()

# fng_df = pd.read_csv(DATA_DIR / "fear_greed_index.csv", parse_dates=["date"])

# price_df["price_log_return"] = (
#     price_df["close"].pct_change().apply(lambda x: np.log(1 + x))
# )
# price_df["volumefrom_log_return"] = (
#     price_df["volumefrom"].pct_change().apply(lambda x: np.log(1 + x))
# )
# price_df["market_cap_log_return"] = (
#     price_df["market_cap"].pct_change().apply(lambda x: np.log(1 + x))
# )
# volatility_days = [5, 10, 30]  # days for rolling volatility
# for days in volatility_days:
#     price_df[f"volatility_{days}d"] = (
#         price_df["price_log_return"].rolling(window=days).std()
#     )  # rolling volatility
#     price_df[f"volatility_{days}d_log_return"] = (
#         price_df[f"volatility_{days}d"].pct_change().apply(lambda x: np.log(1 + x))
#     )
# # save the updated DataFrame
# price_df.to_csv(DATA_DIR / f"{coin_name}_price_mcap.csv", index=False)

transaction_col = [
    "price_log_return",
    "volumefrom_log_return",
    "abs_price_log_return",
    # "market_cap_log_return",
    "volatility_5d_log_return",
    "volatility_10d_log_return",
    "volatility_30d_log_return",
]

sentiment_col = [
    "trend_log_return",
    "abs_trend_log_return",
    # "fear_greed_index",
    # "fear_greed_log_return",
]

for idx, (start_date, end_date) in enumerate(zip(keydates[:-1], keydates[1:])):
    print(f"\nProcessing period {idx + 1}: {start_date} to {end_date}")

    # Filter price to the period
    price_period = price_df[
        (price_df["date"] >= start_date) & (price_df["date"] <= end_date)
    ].sort_values("date")

    trend_period = trend_df[
        (trend_df["date"] >= start_date) & (trend_df["date"] <= end_date)
    ].sort_values("date")

    # fng_period = fng_df[
    #     (fng_df["date"] >= start_date) & (fng_df["date"] <= end_date)
    # ].sort_values("date")

    dfs = [
        trend_period,
        price_period[["date"] + transaction_col],
        # fng_period[["date", "fear_greed_index", "fear_greed_log_return"]],
    ]
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs
    )

    df_merged = df_merged.dropna(
        subset=transaction_col
        + ["trend_log_return"]
        # , "fear_greed_log_return"]
    )

    # pairs = [(senti, col) for senti in sentiment_col for col in transaction_col]
    pairs = [
        ("trend_log_return", "volumefrom_log_return"),
        ("trend_log_return", "price_log_return"),
        ("trend_log_return", "abs_price_log_return"),
        ("abs_trend_log_return", "abs_price_log_return"),
        ("trend_log_return", "volatility_30d_log_return"),
        ("trend_log_return", "volatility_10d_log_return"),
    ]

    analyser = TrendAnalyser(df_merged, pairs=pairs)
    rolling_window = [1]
    for window in rolling_window:
        # for s in range(-3, 4):
        #     smoothed_corrs = analyser.compute_smoothed_correlations(
        #         smooth_window=window, lag=s
        #     )
        #     print(
        #         f"Correlations with {s}-day shift and {window}-day rolling window in period {idx + 1}:",
        #         smoothed_corrs,
        #     )

        for x_col, y_col in pairs:
            analyser.plot_smoothed(
                x_col=x_col,
                y_col=y_col,
                smooth_window=window,
                x_label=f"{coin_name} {x_col.replace('_', ' ').title()}",
                y_label=f"{coin_name} {y_col.replace('_', ' ').title()}",
                title=f"Period {idx + 1}: {coin_name} {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()} (rolling window={window})",
            )
            for lag in range(-1, 2):
                analyser.plot_scatter(
                    x_col=x_col,
                    y_col=y_col,
                    window=window,
                    lag=lag,
                    title=f"Scatter: Period {idx + 1}: {coin_name} {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()} (rolling window={window}, lag={lag})",
                )

    # analyser.run_granger(max_lag=5)
