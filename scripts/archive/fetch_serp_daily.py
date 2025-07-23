from serpapi import GoogleSearch
import pandas as pd
from datetime import datetime, timedelta
from nodiensenv.constants import DATA_DIR


def make_chunks(start, end, days=90):
    chunks = []
    s = start
    while s < end:
        e = min(s + timedelta(days=days - 1), end)
        chunks.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        s = e + timedelta(days=1)
    return chunks


# Example for 2023
start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31)
chunks = make_chunks(start, end)


def fetch_daily_trends(kw, chunks, api_key):
    all_dfs = []
    for s, e in chunks:
        params = {
            "engine": "google_trends",
            "q": kw,
            "api_key": "cc03de0acd0af0110207ab0d6a06f06f48f02d1cd4e480ca4e2086865d810814",
            "date": f"{s} {e}",
            "hl": "en",
            "tz": 0,
        }
        result = GoogleSearch(params).get_dict()
        timeline = result["interest_over_time"]["timeline_data"]
        # timeline now has daily buckets when span ≤90d
        rows = [
            {
                "date": pd.to_datetime(int(pt["timestamp"]), unit="s"),
                kw: int(pt["values"][0]["extracted_value"]),
            }
            for pt in timeline
        ]
        all_dfs.append(pd.DataFrame(rows))
    # concatenate
    df = (
        pd.concat(all_dfs)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


# Usage
API_KEY = "YOUR_KEY"
btc_kw = fetch_daily_trends("bitcoin", chunks, API_KEY)
bcn_kw = fetch_daily_trends("blockchain", chunks, API_KEY)

# Merge them
trends_daily = btc_kw.merge(bcn_kw, on="date")
print("Daily Trends shape:", trends_daily.shape)
print(trends_daily.head())

# Save to CSV
trends_daily.to_csv(DATA_DIR / "trends_daily.csv", index=False)