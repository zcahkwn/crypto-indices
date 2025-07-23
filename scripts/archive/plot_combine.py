from nodiensenv.constants import DATA_DIR, FIGURE_DIR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_btc_price = pd.read_csv(DATA_DIR / "btc_prices_2024.csv")
# df_btc_normalised = pd.read_csv(DATA_DIR / "btc_prices_2023_normalised.csv")
# df_trends_daily = pd.read_csv(DATA_DIR / "trends_daily.csv")
df_trends_weekly = pd.read_csv(DATA_DIR / "trend_btc_2024.csv")

df_btc_price["log_returns"] = abs(
    np.log(df_btc_price["close"] / df_btc_price["close"].shift(1))
).dropna()

df_btc_price["weekly_returns"] = df_btc_price["log_returns"].rolling(7).mean()


# plot bitcoin price
# plt.figure(figsize=(10, 5))
# plt.plot(
#     df_btc_price_2024["date"],
#     df_btc_price_2024["close"],
#     label="BTC Price",
#     color="blue",
# )
# plt.title("Bitcoin Price in 2024")
# plt.xlabel("Date")
# plt.ylabel("Price (USD)")
# plt.legend()
# plt.tight_layout()
# plt.savefig(FIGURE_DIR / "btc_price_2024.pdf")
# plt.show()

# Plotting daily Google Trends

# plt.figure(figsize=(10, 5))
# plt.plot(df_trends_daily["date"], df_trends_daily["bitcoin"], label="bitcoin")
# plt.title("Google Trends for Bitcoin")
# plt.xlabel("Date")
# plt.ylabel("Interest")
# plt.legend()
# plt.show()
# plt.savefig(FIGURE_DIR / "trends_weekly.pdf")
# plt.close()

# plot weekly trends
# plt.figure(figsize=(10, 5))
# plt.plot(
#     df_trends_2024_weekly["date"], df_trends_2024_weekly["bitcoin"], label="bitcoin"
# )
# plt.title("Google Trends for Bitcoin in 2024")
# plt.xlabel("Date")
# plt.ylabel("Interest")
# plt.legend()
# plt.show()
# plt.savefig(FIGURE_DIR / "trends_2024_weekly.pdf")
# plt.close()


# # Merge daily with BTC price
# fig, ax1 = plt.subplots(figsize=(12, 6))
# ax2 = ax1.twinx()

# # BTC daily (left axis)
# (p1,) = ax1.plot(
#     df_btc_price["date"],
#     df_btc_price["close"],
#     label="BTC Price",
#     color="royalblue",
#     linewidth=2,
# )
# ax1.set_ylabel("Price (USD)", fontsize=12)

# # Google Trends (right axis)
# (p2,) = ax2.plot(
#     df_trends_daily["date"],
#     df_trends_daily["bitcoin"],
#     label="Trend: bitcoin",
#     color="orange",
#     linewidth=2,
# )

# ax2.set_ylabel("Google Trends Index", fontsize=12)

# lines = [p1, p2]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc="upper left")

# ax1.set_xlabel("Date", fontsize=12)
# plt.title("BTC Price vs. Search Interest (2023)", fontsize=14)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.savefig(FIGURE_DIR / "btc_price_vs_trends_daily.png")
# plt.close()

# Merge weekly
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
# BTC weekly (left axis)
(p1,) = ax1.plot(
    df_btc_price["date"],
    df_btc_price["weekly_returns"],
    label="BTC Price",
    color="royalblue",
    linewidth=2,
)
ax1.set_ylabel("Price (USD)", fontsize=12)
# Google Trends (right axis)
(p2,) = ax2.plot(
    df_trends_weekly["date"],
    df_trends_weekly["btc"],
    label="Trend: btc",
    color="orange",
    linewidth=2,
)
ax2.set_ylabel("Google Trends Index", fontsize=12)
lines = [p1, p2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")
ax1.set_xlabel("Date", fontsize=12)
plt.title("BTC Price vs. btc Search Interest (2024)", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig(FIGURE_DIR / "btc_return_price_vs_btc_trends_weekly_2024.pdf")
