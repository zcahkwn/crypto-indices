import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from nodiensenv.constants import DATA_DIR, FIGURE_DIR

coingecko_coin = "binancecoin"
trend_item = "bnb"

coingecko_path = DATA_DIR / f"metrics_{coingecko_coin}_2024-06-01_to_2025-05-27.csv"
google_trends_path = DATA_DIR / f"trends_2024-06-01_to_2025-05-27.json"

df_coingecko = pd.read_csv(coingecko_path, parse_dates=["date"])
df_coingecko["date"] = pd.to_datetime(df_coingecko["date"])
df_coingecko["log_return"] = abs(
    df_coingecko["log_return"] * 100
)  # Convert to percentage


with open(google_trends_path, "r") as f:
    trends_json = json.load(f)

# Extract trend_item values
trend_items = trends_json["result"][0]["items"][0]["data"]
coingecko_data = pd.DataFrame(
    [
        {"date": pd.to_datetime(item["date_from"]), "google_trend": item["values"][3]}
        for item in trend_items
    ]
)

# Merge on date
merged = pd.merge(
    df_coingecko[["date", "log_return"]], coingecko_data, on="date", how="inner"
)

# # Plotting
plt.figure()
plt.plot(merged["date"], merged["log_return"], label="Log Return")
plt.plot(merged["date"], merged["google_trend"], label=f"{trend_item} Trend")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title(f"{trend_item} Log Return vs Google Trend Over the past year")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(FIGURE_DIR / f"{coingecko_coin}_seo_plot.pdf")

# Shift the trend series forward by shift_size
shift_size = 0
merged["google_trend_shifted"] = merged["google_trend"].shift(shift_size)

# Calculate rolling averages
window_size = 1  # rolling average
merged["log_return_roll_avg"] = merged["log_return"].rolling(window=window_size).mean()
merged["trend_roll_avg_shifted"] = (
    merged["google_trend_shifted"].rolling(window=window_size).mean()
)

clean = merged.dropna(subset=["log_return_roll_avg", "trend_roll_avg_shifted"])

# Pearson correlation
corr, pval = pearsonr(clean["trend_roll_avg_shifted"], clean["log_return_roll_avg"])
print(
    f"When there is a {shift_size} day shift, rolling window size is {window_size} \n Pearson r = {corr:.4f}, p-value = {pval:.2e}"
)
