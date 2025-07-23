import pandas as pd
from nodiensenv.constants import DATA_DIR
import numpy as np

# 1) Load & resample BTC price to weekly (Sunday-ending) mean
df_price = pd.read_csv(DATA_DIR / "btc_prices_2024.csv", parse_dates=["date"])

df_price["log_returns"] = abs(
    np.log(df_price["close"] / df_price["close"].shift(1))
).dropna()

weekly_price = (
    df_price.set_index("date")
    .resample("W-SUN")["log_returns"]
    .mean()
    .rename("weekly_close")
)


# 2) Load your Trends (already weekly, same W-SUN dates)
df_bitcoin_trends = pd.read_csv(
    DATA_DIR / "trend_bitcoin_2024.csv", parse_dates=["date"]
)
weekly_bitcoin_trends = df_bitcoin_trends.set_index("date")[["bitcoin"]]

df_btc_trends = pd.read_csv(DATA_DIR / "trend_btc_2024.csv", parse_dates=["date"])
weekly_btc_trends = df_btc_trends.set_index("date")[["btc"]]

# 3) Shift the Trends *forward* by 2 periods (i.e. 2 weeks)
shift = 1
trends_bitcoin_shifted = weekly_bitcoin_trends.shift(shift)
trends_btc_shifted = weekly_btc_trends.shift(shift)

# 4) Combine & drop any NaNs
df_lagged_bitcoin = pd.concat([weekly_price, trends_bitcoin_shifted], axis=1).dropna()
df_lagged_btc = pd.concat([weekly_price, trends_btc_shifted], axis=1).dropna()
df_bitcoin_btc = pd.concat([weekly_bitcoin_trends, weekly_btc_trends], axis=1).dropna()

# 5) Recompute correlation
corr_lagged_bitcoin = df_lagged_bitcoin.corr().loc["weekly_close", ["bitcoin"]]
print(f"BTC vs. bitcoin Trends (with {shift}-week Trend lag) in 2024:")
print(corr_lagged_bitcoin)
corr_lagged_btc = df_lagged_btc.corr().loc["weekly_close", ["btc"]]
print(f"BTC vs. btc Trends (with {shift}-week Trend lag) in 2024:")
print(corr_lagged_btc)

corr_bitcoin_btc = df_bitcoin_btc.corr().loc["bitcoin", ["btc"]]
print("bitcoin vs. btc Trends in 2023:")
print(corr_bitcoin_btc)


# corr_lagged_normalised = df_lagged_normalised.corr().loc[
#     "weekly_normalised_close", ["bitcoin"]
# ]
# print("BTC(normalised) vs. Trends (with 2-week Trend lag) in 2023:")
# print(corr_lagged_normalised)
