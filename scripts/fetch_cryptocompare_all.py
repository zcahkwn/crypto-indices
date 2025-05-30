import os
import time
import requests
import pandas as pd
from datetime import datetime
from nodiensenv.constants import DATA_DIR
from nodiensenv.settings import CRYPTOCOMPARE_API

if not CRYPTOCOMPARE_API:
    raise RuntimeError("Please set your CRYPTOCOMPARE_API environment variable")

HEADERS = {"authorization": f"Apikey {CRYPTOCOMPARE_API}"}
BASE_PRICE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"
BASE_SUPPLY_URL = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
SYMBOL = "BTC"
VS_CURRENCY = "USD"
# Date range: 2021-01-01 → today
START_TS = int(datetime(2021, 1, 1).timestamp())
END_TS = int(datetime.utcnow().timestamp())


def get_histoday(start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Page daily OHLCV price data in 2 000-day chunks.
    Returns DataFrame with columns: time, open, high, low, close, volumefrom, volumeto.
    """
    frames = []
    to_ts = end_ts
    while to_ts > start_ts:
        params = {"fsym": SYMBOL, "tsym": VS_CURRENCY, "limit": 2000, "toTs": to_ts}
        r = requests.get(
            BASE_PRICE_URL, headers=HEADERS, params=params, timeout=30
        ).json()
        data = r["Data"]["Data"]
        df = pd.DataFrame(data)
        frames.append(df)
        to_ts = df["time"].min() - 86400  # step back one day
        time.sleep(0.25)
    df_all = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    return df_all[(df_all.time >= start_ts) & (df_all.time <= end_ts)]


def get_supply(start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Page daily circulating-supply in 2 000-day chunks.
    Returns DataFrame with columns: time, supply.
    """
    frames = []
    to_ts = end_ts
    while to_ts > start_ts:
        url = f"{BASE_SUPPLY_URL}?fsym={SYMBOL}" f"&limit=2000&toTs={to_ts}"
        r = requests.get(url, headers=HEADERS, timeout=30).json()
        if r.get("Response") == "Error":
            raise RuntimeError(f"CryptoCompare error: {r.get('Message')}")
        data = r["Data"]["Data"]
        df = pd.DataFrame(data).rename(columns={"current_supply": "supply"})
        frames.append(df[["time", "supply"]])
        to_ts = df["time"].min() - 86400
        time.sleep(0.25)
    df_all = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    return df_all[(df_all.time >= start_ts) & (df_all.time <= end_ts)]


if __name__ == "__main__":
    # Fetch price & supply
    price_df = get_histoday(START_TS, END_TS)
    supply_df = get_supply(START_TS, END_TS)

    # Merge & compute market-cap
    df = (
        price_df.merge(supply_df, on="time", how="left")
        .assign(
            date=lambda d: pd.to_datetime(d.time, unit="s"),
            market_cap=lambda d: d.close * d.supply,
        )
        .set_index("date")
        .sort_index()
    )

    # Save to CSV
    out_file = DATA_DIR / f"{SYMBOL}_price_mcap.csv"
    df[
        [
            "open",
            "high",
            "low",
            "close",
            "volumefrom",
            "volumeto",
            "supply",
            "market_cap",
        ]
    ].to_csv(out_file)
    print(f"Saved {len(df)} rows to {out_file}")
print(df.head())
