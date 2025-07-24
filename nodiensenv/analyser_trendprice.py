"""
This class is used to analyse the correlation between trend and price,
and the granger causality test for each pair.
It also plots the graph of trend and price with different lag, and the scatter plot of the pairs. 

This class is used in scripts/plot_analyse_bullbear.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests


class TrendAnalyser:
    def __init__(self, df: pd.DataFrame, pairs: list[tuple[str, str]] = None):
        """
        df should contain columns:
        ['date', 'price_log_return', 'trend_log_return', 'abs_trend_log_return', 'abs_price_log_return']
        """
        self.data = df.set_index("date")
        # the few pairs we care about
        self._pairs = pairs

    def compute_correlations(self, lag: int = 0) -> pd.DataFrame:
        """
        Compute correlations for the few specified pairs at a given lag:

        Any "trend" column (trend_log_return or abs_trend_log_return) is shifted RIGHT by `lag` days
        before correlating, so you're asking "how well does trend at day t - lag
        correlate with price at day t?"

        Returns a DataFrame indexed by (idx, Pair) with columns:
          - lag: the lag you used
          - correlation: the Pearson r
        """
        results = []

        for x, y in self._pairs:
            # decide which series to shift
            sx = self.data[x].shift(lag)
            sy = self.data[y]

            # align and drop NaNs
            common = pd.concat([sx, sy], axis=1).dropna()
            corr = common.iloc[:, 0].corr(common.iloc[:, 1])

            results.append(
                {
                    "Pair": f"{x} vs {y}",
                    "lag": lag,
                    "correlation": corr,
                }
            )

        return pd.DataFrame(results).set_index(["Pair", "lag"])

    def compute_rolling_correlations(
        self, window: int, lag: int = 0
    ) -> dict[str, pd.DataFrame]:
        """
        Compute rolling-window Pearson r for each pair.
        - window: number of days in the rolling window
        - lag: shift applied to any trend column before rolling
        Returns a dict {'H1': df1, 'H2': df2}, where each df has:
          index: date of window end
          columns: one per pair, e.g. 'eth vs close'
        """
        out = {}
        df2 = self.data.copy()
        # apply lag to sentiment columns
        # columns_to_shift = list(set([pair[0] for pair in self._pairs]))
        # df2[columns_to_shift] = df2[columns_to_shift].shift(lag)

        # compute rolling correlations
        roll_dfs = []
        for x, y in self._pairs:
            pair_name = f"{x} vs {y}"
            # rolling apply paired correlation
            df2[x] = df2[x].shift(lag)
            roll_r = df2[x].rolling(window).corr(df2[y])
            roll_dfs.append(roll_r.rename(pair_name))
        out = pd.concat(roll_dfs, axis=1).dropna(how="all")
        return out

    def compute_smoothed_correlations(
        self, smooth_window: int, lag: int = 0
    ) -> pd.DataFrame:
        """
        Smooth each series with a centered moving average of length `smooth_window`,
        then compute one Pearson r for each pair (at given lag).
        Returns a DataFrame indexed by (Pair, lag).
        """
        results = []

        # apply smoothing
        # df2 = self.data.rolling(smooth_window, center=True).mean()
        # # then shift trend cols if needed
        # columns_to_shift = list(set([pair[0] for pair in self._pairs]))
        # df2[columns_to_shift] = df2[columns_to_shift].shift(lag)
        # compute correlations on smoothed data
        for x, y in self._pairs:
            # df2 = self.data.rolling(smooth_window, center=False).mean()
            # df2[x] = df2[x].shift(lag)
            df2 = self.data[[x, y]].copy()
            df2[x] = df2[x].shift(lag)
            df2 = df2.rolling(smooth_window, center=False).mean().dropna()

            common = df2[[x, y]].dropna()
            r = common[x].corr(common[y])
            results.append(
                {
                    "Pair": f"{x} vs {y}",
                    "lag": lag,
                    "smooth_window": smooth_window,
                    "correlation": r,
                }
            )
        return pd.DataFrame(results).set_index(["Pair", "lag", "smooth_window"])

    def run_granger(self, max_lag):
        for x, y in self._pairs:
            df_granger = self.data[[x, y]]
            df_granger.columns = [f"{x}_ret", f"{y}_ret"]
            print(
                f"\n=== Testing Granger causality for {x} and {y} up to lag={max_lag} ===\n"
            )
            # trend --> price
            print(f">> Does {y} Granger-cause {x}?")
            grangercausalitytests(
                df_granger[[f"{x}_ret", f"{y}_ret"]], maxlag=max_lag, verbose=True
            )
            # price --> trend
            print(f"\n>> Does {x} Granger-cause {y}?")
            grangercausalitytests(
                df_granger[[f"{y}_ret", f"{x}_ret"]], maxlag=max_lag, verbose=True
            )

    def plot_smoothed(
        self,
        x_col: str,
        y_col: str,
        smooth_window: int,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
    ):
        """
        Smooth both series with a centered moving average (smooth_window) and plot them
        on dual y-axes.
        """
        df = self.data.copy()
        # apply smoothing
        df[x_col] = df[x_col].rolling(smooth_window, center=False).mean()
        df[y_col] = df[y_col].rolling(smooth_window, center=False).mean()

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        ax1.plot(df.index, df[x_col], label=x_label or x_col, color="blue")
        ax2.plot(df.index, df[y_col], label=y_label or y_col, color="orange")

        ax1.set_xlabel("Date")
        ax1.set_ylabel(x_label or x_col)
        ax2.set_ylabel(y_label or y_col)
        plt.title(
            title
            or f"Smoothed {x_label or x_col} vs {y_label or y_col} (window={smooth_window})"
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_scatter(
        self, x_col: str, y_col: str, window: int, lag: int, title: str = None
    ):
        """
        Smooth both series with a centered moving average (window),
        shift x_col by lag, and plot x_col vs y_col as a scatter plot.
        """
        df = self.data.copy()
        # Apply smoothing
        df[x_col] = df[x_col].rolling(window, center=False).mean()
        df[y_col] = df[y_col].rolling(window, center=False).mean()
        # Apply lag to x_col
        df[x_col] = df[x_col].shift(lag)
        # Drop rows with NaN values in either column
        df_plot = df[[x_col, y_col]].dropna()
        plt.figure(figsize=(7, 5))
        plt.scatter(df_plot[x_col], df_plot[y_col], alpha=0.7)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.tight_layout()
        plt.show()
