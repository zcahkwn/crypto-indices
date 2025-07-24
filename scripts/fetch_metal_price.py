"""
This script fetches daily gold price data from LBMA (https://www.lbma.org.uk/prices-and-data/precious-metal-prices#/table), and saves the data to a CSV file.
"""

import datetime as dt, requests, pandas as pd
from nodiensenv.constants import DATA_DIR

START_DATE = pd.Timestamp("2000-01-01")
END_DATE = pd.Timestamp("today")

# Define metal endpoints
metal_endpoints = {
    "gold": "https://prices.lbma.org.uk/json/gold_pm.json",
    "silver": "https://prices.lbma.org.uk/json/silver.json",
}

# Fetch and process each metal's data
for metal_name, endpoint in metal_endpoints.items():
    metal_data = requests.get(endpoint, timeout=15).json()

    df = (
        pd.DataFrame(metal_data)
        .rename(columns={"d": "date"})
        .assign(
            usd=lambda x: x["v"].str[0],
            gbp=lambda x: x["v"].str[1],
            eur=lambda x: x["v"].str[2],
        )
        .drop(columns=["v", "is_cms_locked"])
        .loc[
            lambda x: (START_DATE <= pd.to_datetime(x["date"]))
            & (pd.to_datetime(x["date"]) <= END_DATE)
        ]
    )

    df.to_csv(DATA_DIR / f"price_{metal_name}.csv", index=False)
    print(f"\nProcessed {metal_name} data:")
    print(df.head(), "\n…\n", df.tail())
