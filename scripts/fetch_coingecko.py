import requests
import pandas as pd
import numpy as np
from datetime import datetime
from nodiensenv.constants import DATA_DIR

"""
Access to historical data via the Public API (Demo plan) is restricted to the past 365 days only. 
To access the complete range of historical data, please subscribe to one of our paid plans to obtain a Pro-API key.
"""

# Configuration
coin_id = "hedera-hashgraph"  # Change this to any CoinGecko ID (e.g., "ethereum", "ripple", "bitcoin")
vs_currency = "usd"
start_date = "2024-06-01"
end_date = datetime.utcnow().strftime("%Y-%m-%d")

# Convert dates to UNIX timestamps
start_ts = int(pd.to_datetime(start_date).timestamp())
end_ts = int(pd.to_datetime(end_date).timestamp())

# Fetch market chart data
url = (
    f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    f"/market_chart/range?vs_currency={vs_currency}"
    f"&from={start_ts}&to={end_ts}"
)
resp = requests.get(url, timeout=30).json()

# Build DataFrame
prices = pd.DataFrame(resp["prices"], columns=["ts", "price"])
market_caps = pd.DataFrame(resp["market_caps"], columns=["ts", "market_cap"])
volumes = pd.DataFrame(resp["total_volumes"], columns=["ts", "total_volume"])

# Merge into one DataFrame
df = prices.merge(market_caps, on="ts").merge(volumes, on="ts")
df["date"] = pd.to_datetime(df["ts"], unit="ms")
df.set_index("date", inplace=True)
df.drop(columns="ts", inplace=True)

# Compute circulating supply
df["circulating_supply"] = df["market_cap"] / df["price"]

# Compute volatility (annualized 30-day rolling)
df["log_return"] = np.log(df["price"] / df["price"].shift(1))
df["volatility_30d"] = df["log_return"].rolling(window=30).std() * np.sqrt(365)


# Save to CSV
output_file = DATA_DIR / f"metrics_{coin_id}_{start_date}_to_{end_date}.csv"
df.to_csv(output_file)
