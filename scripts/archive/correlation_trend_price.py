import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from nodiensenv.constants import DATA_DIR, FIGURE_DIR
from nodiensenv.analyser import TrendPriceAnalyser

years = [2021, 2022, 2023, 2024]

for year in years:
    trend = "bitcoin"
    coin_name = "BTC"
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

    def run_granger(df, max_lag):
        print(f"\n=== Testing Granger causality up to lag={max_lag} in {year} ===\n")

        # trend --> price
        print(">> Does TREND Granger-cause PRICE?")
        test_result_1 = grangercausalitytests(
            df[["price_ret", "trend_ret"]], maxlag=max_lag, verbose=True
        )
        # price --> trend
        print("\n>> Does PRICE Granger-cause TREND?")
        test_result_2 = grangercausalitytests(
            df[["trend_ret", "price_ret"]], maxlag=max_lag, verbose=True
        )

        return test_result_1, test_result_2


    for h in ("H1", "H2"):
        df = df_h1 if h == "H1" else df_h2
        series_price = df[["close"]].dropna()
        series_trend = df[[f"{trend}"]].dropna()
        # series_price = df_h1[["log_return"]].dropna()
        # series_trend = df_h1[["trend_log_return"]].dropna()
        df_gc = pd.concat([series_price, series_trend], axis=1).dropna()
        df_gc.columns = ["price_ret", "trend_ret"]
        print(f"\nGranger causality results for {h} in {year}:")
        res_trend_to_price, res_price_to_trend = run_granger(df_gc, max_lag=3)


    # analyser = TrendPriceAnalyser(trend, df_h1, df_h2)

    # for smooth_window in [3]:
    #         analyser.plot_smoothed(
    #             "H1",
    #             "close",
    #             f"{trend}",
    #             smooth_window,
    #             "Close Price",
    #             f"Google Trend ({trend})",
    #             f"H1 {year}: {coin_name} Close vs Trend (rolling window={smooth_window})",
    #         )
    #         analyser.plot_smoothed(
    #             "H2",
    #             "close",
    #             f"{trend}",
    #             smooth_window,
    #             "Close Price",
    #             f"Google Trend ({trend})",
    #             f"H2 {year}: {coin_name} Close vs Trend (rolling window={smooth_window})",
    #         )
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
    # analyser.plot_smoothed(
    #     "H1",
    #     "abs_log_return",
    #     "abs_trend_log_return",
    #     smooth_window,
    #     "Price Abs Log Return",
    #     f"Trend Abs Log Return({trend})",
    #     f"H1 {year}: {coin_name} Price Abs Log Return vs Trend Abs Log Return (rolling window={smooth_window})",
    # )
    # analyser.plot_smoothed(
    #     "H2",
    #     "abs_log_return",
    #     "abs_trend_log_return",
    #     smooth_window,
    #     "Price Abs Log Return",
    #     f"Trend Abs Log Return ({trend})",
    #     f"H2 {year}: {coin_name} Price Abs Log Return vs Trend Abs Log Return (rolling window={smooth_window})",
    # )
