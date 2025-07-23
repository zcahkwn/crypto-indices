from serpapi import GoogleSearch
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from nodiensenv.constants import DATA_DIR

API_KEY = "cc03de0acd0af0110207ab0d6a06f06f48f02d1cd4e480ca4e2086865d810814"
coin = "eth"
keywords = ["eth", "ethereum"]
year = 2023
start_date, end_date = f"{year}-01-01", f"{year}-06-30"


all_trends = []
for kw in keywords:
    params = {
        "engine": "google_trends",
        "q": kw,
        "api_key": API_KEY,
        "date": f"{start_date} {end_date}",
        "hl": "en",
        "tz": 0,
    }
    result = GoogleSearch(params).get_dict()

    # Pull out the weekly buckets
    data_dict = result.get("interest_over_time", {})
    timeline = data_dict.get("timeline_data", [])

    # Build DataFrame from timeline_data
    rows = []
    for pt in timeline:
        # Use the Unix timestamp (start of the week) for a clear, regular date
        date = pd.to_datetime(int(pt["timestamp"]), unit="s")
        value = int(pt["values"][0]["extracted_value"])
        rows.append({"date": date, kw: value})

    df_kw = pd.DataFrame(rows)
    all_trends.append(df_kw)

# Filter out any empty frames
valid_trends = [df for df in all_trends if not df.empty and "date" in df.columns]
if not valid_trends:
    raise RuntimeError("No trend data fetched; check your key and keywords.")

# Merge on the date
trends_df = valid_trends[0]
for df_kw in valid_trends[1:]:
    trends_df = trends_df.merge(df_kw, on="date", how="inner")


print("Merged trends (weekly) shape:", trends_df.shape)
print(trends_df.head())

# save to CSV
trends_df.to_csv(DATA_DIR / "trend_eth_2023.csv", index=False)

# plot
plt.figure(figsize=(10, 5))
plt.plot(trends_df["date"], trends_df["eth"], label="eth google trend")
plt.title(f"Google Trends for {coin} in {year}")
plt.xlabel("Date")
plt.ylabel("Interest")
plt.legend()
plt.show()
plt.savefig(f"trend_{coin}_2023.png")
plt.close()
