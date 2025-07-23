from nodiensenv.constants import DATA_DIR, FIGURE_DIR
import matplotlib.pyplot as plt
import pandas as pd


df_price = pd.read_csv(DATA_DIR / "eth_prices_2024.csv")
df_trends_ethereum = pd.read_csv(DATA_DIR / "trend_ethereum_2024.csv")
df_trends_eth = pd.read_csv(DATA_DIR / "trend_eth_2024.csv")

# Merge weekly
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
# BTC weekly (left axis)
(p1,) = ax1.plot(
    df_price["date"],
    df_price["close"],
    label="ETH Price",
    color="royalblue",
    linewidth=2,
)
ax1.set_ylabel("Price (USD)", fontsize=12)
# Google Trends (right axis)
(p2,) = ax2.plot(
    df_trends_ethereum["date"],
    df_trends_ethereum["ethereum"],
    label="Trend: ethereum",
    color="orange",
    linewidth=2,
)
ax2.set_ylabel("Google Trends Index", fontsize=12)
lines = [p1, p2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")
ax1.set_xlabel("Date", fontsize=12)
plt.title("ETH Price vs. ethereum Search Interest (2024)", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig(FIGURE_DIR / "eth_price_vs_ethereum_trends_weekly_2024.pdf")
