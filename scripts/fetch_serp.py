from serpapi import GoogleSearch
import pandas as pd
import matplotlib.pyplot as plt
from nodiensenv.constants import DATA_DIR, FIGURE_DIR
from nodiensenv.settings import SERP_API

coin = "DOGE"
keywords = ["dogecoin"]
years = [2021, 2022, 2023, 2024]

for year in years:

    # Define two halves of the year
    date_ranges = {
        "H1": (f"{year}-01-01", f"{year}-06-30"),
        "H2": (f"{year}-07-01", f"{year}-12-31"),
    }

    for half_label, (start_date, end_date) in date_ranges.items():
        print(f"\nProcessing {half_label}: {start_date} to {end_date}")

        all_trends = []
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
            all_trends.append(df_kw)

        # Merge only non-empty frames
        valid = [df for df in all_trends if not df.empty]
        if not valid:
            print(f"No data fetched for {half_label}. Check your API key or keywords.")
            continue

        trends_df = valid[0]
        for df_kw in valid[1:]:
            trends_df = trends_df.merge(df_kw, on="date", how="inner")

        print(f"Merged shape: {trends_df.shape}")

        # Save CSV
        out_csv = DATA_DIR / f"trend_{coin}_{year}_{half_label}.csv"
        trends_df.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")

        # Plot
        plt.figure(figsize=(10, 5))
        for kw in keywords:
            plt.plot(trends_df["date"], trends_df[kw], label=f"{kw} trend")
        plt.title(f"Google Trends for {coin.upper()} ({half_label} {year})")
        plt.xlabel("Date")
        plt.ylabel("Interest")
        plt.legend()
        plt.tight_layout()
        png_path = FIGURE_DIR / f"trend_{coin}_{year}_{half_label}.png"
        plt.savefig(png_path)
        plt.close()
        print(f"Plot saved as {png_path}")
