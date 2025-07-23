import pandas as pd
from nodiensenv.constants import DATA_DIR

# Load & resample BTC price to weekly (Sunday-ending) mean
df_price = pd.read_csv(DATA_DIR / "eth_prices_2023.csv", parse_dates=["date"])
weekly_price = (
    df_price.set_index("date").resample("W-SUN")["close"].mean().rename("weekly_close")
)

# Load your Trends (already weekly, same W-SUN dates)
df_ethereum_trends = pd.read_csv(
    DATA_DIR / "trend_ethereum_2023.csv", parse_dates=["date"]
)
weekly_ethereum_trends = df_ethereum_trends.set_index("date")[["ethereum"]]

df_eth_trends = pd.read_csv(DATA_DIR / "trend_eth_2023.csv", parse_dates=["date"])
weekly_eth_trends = df_eth_trends.set_index("date")[["eth"]]

# Shift the Trends
trends_ethereum_shifted = weekly_ethereum_trends.shift(0)
trends_eth_shifted = weekly_eth_trends.shift(0)

# Combine & drop any NaNs
df_lagged_ethereum = pd.concat([weekly_price, trends_ethereum_shifted], axis=1).dropna()
df_lagged_eth = pd.concat([weekly_price, trends_eth_shifted], axis=1).dropna()
df_ethereum_eth = pd.concat(
    [weekly_ethereum_trends, weekly_eth_trends], axis=1
).dropna()

# Recompute correlation
corr_lagged_ethereum = df_lagged_ethereum.corr().loc["weekly_close", ["ethereum"]]
print("ETH vs. ethereum Trends (with 0-week Trend lag) in 2023:")
print(corr_lagged_ethereum)
corr_lagged_eth = df_lagged_eth.corr().loc["weekly_close", ["eth"]]
print("ETH vs. eth Trends (with 0-week Trend lag) in 2023:")
print(corr_lagged_eth)

corr_ethereum_eth = df_ethereum_eth.corr().loc["ethereum", ["eth"]]
print("ethereum vs. eth Trends in 2023:")
print(corr_ethereum_eth)
