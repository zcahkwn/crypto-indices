from serpapi import GoogleSearch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nodiensenv.constants import DATA_DIR, FIGURE_DIR
from nodiensenv.settings import SERP_API

coin = "BTC"
keywords = ["bitcoin"]
years = list(range(2013, 2025))  # 2013-2024

last_chunk_start = "2024-12-31"
last_chunk_end = "2025-06-04"

chunked_frames = []

for year in years:

    # Define two halves of the year
    date_ranges = {
        "H1": (f"{year-1}-12-31", f"{year}-06-30"),
        "H2": (f"{year}-06-30", f"{year}-12-31"),
    }

    for half_label, (start_date, end_date) in date_ranges.items():
        print(f"\nProcessing {half_label} of {year}: {start_date} to {end_date}")

        keyword_dfs = []
        for kw in keywords:
            params = {
                "engine": "google_trends",
                "q": kw,
                "api_key": SERP_API,
                "date": f"{start_date} {end_date}",
                "hl": "en",
                "tz": 0,
            }
            result = GoogleSearch(params).get_dict()
            timeline = result.get("interest_over_time", {}).get("timeline_data", [])

            rows = []
            for pt in timeline:
                date = pd.to_datetime(int(pt["timestamp"]), unit="s")
                value = int(pt["values"][0]["extracted_value"])
                rows.append({"date": date, kw: value})
            df_kw = pd.DataFrame(rows)
            keyword_dfs.append(df_kw)

        # Merge only non-empty frames
        valid = [df for df in keyword_dfs if not df.empty]
        if not valid:
            print(f"No data fetched for {half_label}. Check your API key or keywords.")
            continue

        # Merge all keyword DataFrames on "date" (if you have multiple keywords)
        merged = keyword_dfs[0]
        for df_kw in keyword_dfs[1:]:
            merged = merged.merge(df_kw, on="date", how="inner")

        merged = merged.sort_values("date").reset_index(drop=True)
        value_col = keywords[0]

        # log_return = ln(v_t / v_{t-1}). The first row will be NaN.
        merged["trend_log_return"] = np.log(
            merged[value_col] / merged[value_col].shift(1)
        )

        merged = merged.iloc[1:].reset_index(drop=True)

        # Append this chunk (with its log_return) to our list
        chunked_frames.append(merged)

print(f"Fetching H1 of 2025: {last_chunk_start} → {last_chunk_end}")
keyword_dfs_2025 = []
for kw in keywords:
    params = {
        "engine": "google_trends",
        "q": kw,
        "api_key": SERP_API,
        "date": f"{last_chunk_start} {last_chunk_end}",
        "hl": "en",
        "tz": 0,
    }
    result = GoogleSearch(params).get_dict()
    timeline = result.get("interest_over_time", {}).get("timeline_data", [])

    rows = []
    for pt in timeline:
        date = pd.to_datetime(int(pt["timestamp"]), unit="s")
        value = int(pt["values"][0]["extracted_value"])
        rows.append({"date": date, kw: value})
    df_kw = pd.DataFrame(rows)
    keyword_dfs_2025.append(df_kw)

if keyword_dfs_2025:
    merged_2025 = keyword_dfs_2025[0]
    for df_kw in keyword_dfs_2025[1:]:
        merged_2025 = merged_2025.merge(df_kw, on="date", how="inner")

    merged_2025 = merged_2025.sort_values("date").reset_index(drop=True)
    value_col = keywords[0]
    merged_2025["trend_log_return"] = np.log(
        merged_2025[value_col] / merged_2025[value_col].shift(1)
    )

    merged_2025 = merged_2025.iloc[1:].reset_index(drop=True)

    chunked_frames.append(merged_2025)


full_df = pd.concat(chunked_frames, ignore_index=True)

full_df = full_df.sort_values("date").reset_index(drop=True)

mask = (full_df["date"] >= pd.to_datetime("2019-01-01")) & (
    full_df["date"] <= pd.to_datetime("2025-06-04")
)
full_df = full_df.loc[mask].reset_index(drop=True)

output_df = full_df.loc[:, ["date", f"{value_col}", "trend_log_return"]]

output_path = DATA_DIR / f"{coin}_trend_log_returns_2018-2025.csv"
output_df.to_csv(output_path, index=False)

print(f"Saved daily log‐returns to: {output_path}")
