import os, requests, pandas as pd
from datetime import datetime, timedelta
from nodiensenv.settings import COINMARKETCAP_API

BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
BTC_ID = 1  # CoinMarketCap ID for Bitcoin
HEADERS = {"X-CMC_PRO_API_KEY": COINMARKETCAP_API}


# -------------------------------------------------------------
# Helper: download one window (max 365 d) and return a DataFrame
# -------------------------------------------------------------
def get_window(start_dt, end_dt):
    params = {
        "id": BTC_ID,
        "convert": "USD",
        "interval": "daily",
        "time_start": start_dt.strftime("%Y-%m-%d"),
        "time_end": end_dt.strftime("%Y-%m-%d"),
    }
    r = requests.get(BASE_URL, headers=HEADERS, params=params)
    r.raise_for_status()
    quotes = r.json()["data"]["quotes"]

    # Flatten the JSON into rows: date | close | market_cap
    rows = [
        {
            "date": q["time_open"][:10],
            "close": q["quote"]["USD"]["close"],
            "market_cap": q["quote"]["USD"]["market_cap"],
        }
        for q in quotes
    ]
    return pd.DataFrame(rows)


# -------------------------------------------------------------
# 1 year per call → stitch together 2014-today
# -------------------------------------------------------------
start = datetime(2014, 1, 1)
today = datetime.utcnow()
chunk = timedelta(days=365)

frames = []
while start <= today:
    end = min(start + chunk, today)
    frames.append(get_window(start, end))
    start = end + timedelta(days=1)  # move to next day to avoid overlap

btc_df = pd.concat(frames, ignore_index=True)
btc_df.to_csv("bitcoin_price_mcap_2014-today.csv", index=False)
print(btc_df.head())
