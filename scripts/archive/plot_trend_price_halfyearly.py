import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nodiensenv.constants import DATA_DIR, FIGURE_DIR
from nodiensenv.analyser import TrendPriceAnalyser

years = [2021, 2022, 2023, 2024]

for year in years:
    trend = "dogecoin"
    coin_name = "DOGE"
    trend_h1 = pd.read_csv(
        DATA_DIR / f"trend_{coin_name}_{year}_H1.csv", parse_dates=["date"]
    )
    trend_h2 = pd.read_csv(
        DATA_DIR / f"trend_{coin_name}_{year}_H2.csv", parse_dates=["date"]
    )
    price = pd.read_csv(DATA_DIR / f"{coin_name}_price_mcap.csv", parse_dates=["date"])

    # Filter price to {year}, split H1 & H2, and compute daily log-returns
    price_year = price[price["date"].dt.year == year].sort_values("date")
    price_h1 = price_year[(price_year["date"] <= f"{year}-06-30")].copy()
    price_h2 = price_year[(price_year["date"] >= f"{year}-07-01")].copy()

    for df in (price_h1, price_h2):
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100
        df["abs_log_return"] = abs(df["log_return"])

    for df in (trend_h1, trend_h2):
        df["trend_log_return"] = np.log(df[f"{trend}"] / df[f"{trend}"].shift(1)) * 100
        df["abs_trend_log_return"] = abs(df["trend_log_return"])

    # Merge each half’s trends with its prices
    df_h1 = pd.merge(
        trend_h1,
        price_h1[["date", "close", "log_return", "abs_log_return"]],
        on="date",
        how="inner",
    )
    df_h2 = pd.merge(
        trend_h2,
        price_h2[["date", "close", "log_return", "abs_log_return"]],
        on="date",
        how="inner",
    )

    analyser = TrendPriceAnalyser(trend, df_h1, df_h2)
    shift = [-3, -2, -1, 0, 1, 2, 3]

    for s in shift:
        rolling_window = [3]
        for window in rolling_window:
            smoothed_corrs = analyser.compute_smoothed_correlations(
                smooth_window=window, lag=s
            )
            print(
                f"Correlations with {s}-day shift and {window} rolling window in {year}:",
                smoothed_corrs,
            )
        # rolling_corr = analyser.compute_rolling_correlations(window=30, lag=s)
        # rolling_corr["H1"].plot()
        # plt.title(f"H1 Rolling Correlation with {trend} ({s}-day shift)")
        # rolling_corr["H2"].plot()
        # plt.title(f"H2 Rolling Correlation with {trend} ({s}-day shift)")

    for smooth_window in [3]:
        analyser.plot_smoothed(
            "H1",
            "close",
            f"{trend}",
            smooth_window,
            "Close Price",
            f"Google Trend ({trend})",
            f"H1 {year}: {coin_name} Close vs Trend (rolling window={smooth_window})",
        )
        analyser.plot_smoothed(
            "H2",
            "close",
            f"{trend}",
            smooth_window,
            "Close Price",
            f"Google Trend ({trend})",
            f"H2 {year}: {coin_name} Close vs Trend (rolling window={smooth_window})",
        )
        #     analyser.plot_smoothed(
        #         "H1",
        #         "log_return",
        #         f"{trend}",
        #         smooth_window,
        #         "Log Return",
        #         f"Google Trend ({trend})",
        #         f"H1 {year}: {coin_name} Log Return vs Trend (rolling window={smooth_window})",
        #     )
        #     analyser.plot_smoothed(
        #         "H2",
        #         "log_return",
        #         f"{trend}",
        #         smooth_window,
        #         "Log Return",
        #         f"Google Trend ({trend})",
        #         f"H2 {year}: {coin_name} Log Return vs Trend (rolling window={smooth_window})",
        #     )
        #     analyser.plot_smoothed(
        #         "H1",
        #         "abs_log_return",
        #         f"{trend}",
        #         smooth_window,
        #         "Log Return",
        #         f"Google Trend ({trend})",
        #         f"H1 {year}: {coin_name} abs Log Return vs Trend (rolling window={smooth_window})",
        #     )
        #     analyser.plot_smoothed(
        #         "H2",
        #         "abs_log_return",
        #         f"{trend}",
        #         smooth_window,
        #         "Log Return",
        #         f"Google Trend ({trend})",
        #         f"H2 {year}: {coin_name} abs Log Return vs Trend (rolling window={smooth_window})",
        #     )
        #     analyser.plot_smoothed(
        #         "H1",
        #         "log_return",
        #         "trend_log_return",
        #         smooth_window,
        #         "Price Log Return",
        #         "Trend Log Return",
        #         f"H1 {year}: {coin_name} Price Log Return vs Trend Log Return (rolling window={smooth_window})",
        #     )
        #     analyser.plot_smoothed(
        #         "H2",
        #         "log_return",
        #         "trend_log_return",
        #         smooth_window,
        #         "Price Log Return",
        #         "Trend Log Return",
        #         f"H2 {year}: {coin_name} Price Log Return vs Trend Log Return (rolling window={smooth_window})",
        #     )
        analyser.plot_smoothed(
            "H1",
            "abs_log_return",
            "abs_trend_log_return",
            smooth_window,
            "Price Abs Log Return",
            f"Trend Abs Log Return({trend})",
            f"H1 {year}: {coin_name} Price Abs Log Return vs Trend Abs Log Return (rolling window={smooth_window})",
        )
        analyser.plot_smoothed(
            "H2",
            "abs_log_return",
            "abs_trend_log_return",
            smooth_window,
            "Price Abs Log Return",
            f"Trend Abs Log Return ({trend})",
            f"H2 {year}: {coin_name} Price Abs Log Return vs Trend Abs Log Return (rolling window={smooth_window})",
        )
