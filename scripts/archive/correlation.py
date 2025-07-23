import pandas as pd
from nodiensenv.constants import DATA_DIR, FIGURE_DIR


# 1) Load your data
df_price = pd.read_csv(DATA_DIR / "btc_prices_2023.csv", parse_dates=["date"])
df_trends = pd.read_csv(DATA_DIR / "trends_daily.csv", parse_dates=["date"])

# 2) Convert price to a weekly frequency (e.g. mean or last-value)
#    Here we take the weekly mean, ending each week on Sunday:
weekly_price = (
    df_price.set_index("date").resample("W-SUN")["close"].mean().rename("weekly_close")
)

# 3) Prepare trends (already weekly, same 'W-SUN' dates)
weekly_trends = (
    df_trends.set_index("date")
    # ensure it's also indexed on Sundays; if Google Trends dates
    # are on Sundays by default, you’re good—otherwise you can
    # .resample('W-SUN').first() or similar.
)

# 4) Combine into one DataFrame
df_weekly = pd.concat([weekly_price, weekly_trends], axis=1).dropna()

# 5) Compute correlations
corr_matrix = df_weekly.corr()
print("Full correlation matrix:\n", corr_matrix)

# If you just want BTC price vs. the two trends:
btc_vs_trends = corr_matrix.loc["weekly_close", ["bitcoin", "blockchain"]]
print("\nBTC vs. Trends:\n", btc_vs_trends)

#daily correlation

df_trends_daily= pd.read_csv(DATA_DIR / "trends_daily.csv", parse_dates=["date"])
# merge daily trend with daily price
df_daily = df_price.merge(df_trends_daily, on="date", how="inner")
# compute daily correlation
daily_corr_matrix = df_daily.corr()
print("Daily correlation matrix:\n", daily_corr_matrix)



