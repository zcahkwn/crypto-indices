import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TrendPriceAnalyser:
    def __init__(self, trend_word: str, df_h1: pd.DataFrame, df_h2: pd.DataFrame):
        """
        df_h1, df_h2 should each contain columns:
        ['date', '{trend_word}', 'close', 'log_return', 'trend_log_return']
        """
        self.trend_word = trend_word
        self.data = {"H1": df_h1.set_index("date"), "H2": df_h2.set_index("date")}
        # which columns to shift when applying lag
        self._trend_cols = {f"{trend_word}", "trend_log_return", "abs_trend_log_return"}
        # the six pairs we care about
        self._pairs = [
            (f"{trend_word}", "close"),
            (f"{trend_word}", "log_return"),
            (f"{trend_word}", "abs_log_return"),
            ("log_return", "trend_log_return"),
            ("abs_log_return", "trend_log_return"),
            ("log_return", "abs_trend_log_return"),
            ("abs_log_return", "abs_trend_log_return"),
        ]

    def compute_correlations(self, lag: int = 0) -> pd.DataFrame:
        """
        Compute correlations for the six specified pairs at a given lag:
          1) {trend_word} vs close
          2) {trend_word} vs log_return
          3) log_return vs trend_log_return

        Any “trend” column ({trend_word} or trend_log_return) is shifted RIGHT by `lag` days
        before correlating, so you’re asking “how well does trend at day t - lag
        correlate with price at day t?”

        Returns a DataFrame indexed by (Half, Pair) with columns:
          - lag: the lag you used
          - correlation: the Pearson r
        """
        results = []
        trend_cols = {f"{self.trend_word}", "trend_log_return"}

        for half, df in self.data.items():
            for x, y in self._pairs:
                # decide which series to shift
                sx = df[x].shift(lag) if x in trend_cols else df[x]
                sy = df[y].shift(lag) if y in trend_cols else df[y]

                # align and drop NaNs
                common = pd.concat([sx, sy], axis=1).dropna()
                corr = common.iloc[:, 0].corr(common.iloc[:, 1])

                results.append(
                    {
                        "Half": half,
                        "Pair": f"{x} vs {y}",
                        "lag": lag,
                        "correlation": corr,
                    }
                )

        return pd.DataFrame(results).set_index(["Half", "Pair", "lag"])

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
        for half, df in self.data.items():
            df2 = df.copy()
            # apply lag to trend columns
            for col in self._trend_cols:
                if col in df2:
                    df2[col] = df2[col].shift(lag)
            # compute rolling correlations
            roll_dfs = []
            for x, y in self._pairs:
                pair_name = f"{x} vs {y}"
                # rolling apply paired correlation
                roll_r = df2[x].rolling(window).corr(df2[y])
                roll_dfs.append(roll_r.rename(pair_name))
            out[half] = pd.concat(roll_dfs, axis=1).dropna(how="all")
        return out

    def compute_smoothed_correlations(
        self, smooth_window: int, lag: int = 0
    ) -> pd.DataFrame:
        """
        Smooth each series with a centered moving average of length `smooth_window`,
        then compute one Pearson r for each pair (at given lag).
        Returns a DataFrame indexed by (Half, Pair, lag).
        """
        results = []
        for half, df in self.data.items():
            # apply smoothing
            df2 = df.rolling(smooth_window, center=True).mean()
            # then shift trend cols if needed
            for col in self._trend_cols:
                if col in df2:
                    df2[col] = df2[col].shift(lag)
            # compute correlations on smoothed data
            for x, y in self._pairs:
                common = df2[[x, y]].dropna()
                r = common[x].corr(common[y])
                results.append(
                    {
                        "Half": half,
                        "Pair": f"{x} vs {y}",
                        "lag": lag,
                        "smooth_window": smooth_window,
                        "correlation": r,
                    }
                )
        return pd.DataFrame(results).set_index(["Half", "Pair", "lag", "smooth_window"])

    def plot_smoothed(
        self,
        half: str,
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
        df = self.data[half].copy()
        # apply smoothing
        df[x_col] = df[x_col].rolling(smooth_window, center=True).mean()
        df[y_col] = df[y_col].rolling(smooth_window, center=True).mean()

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        ax1.plot(df.index, df[x_col], label=x_label or x_col, color="blue")
        ax2.plot(df.index, df[y_col], label=y_label or y_col, color="orange")

        ax1.set_xlabel("Date")
        ax1.set_ylabel(x_label or x_col)
        ax2.set_ylabel(y_label or y_col)
        plt.title(
            title
            or f"{half}: Smoothed {x_label or x_col} vs {y_label or y_col} (window={smooth_window})"
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        plt.show()
