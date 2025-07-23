import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timezone
from nodiensenv.constants import DATA_DIR


def fetch_2023_prices_v2():
    API_KEY = "cc03de0acd0af0110207ab0d6a06f06f48f02d1cd4e480ca4e2086865d810814"
    # UNIX timestamp for end of 2023-12-31 UTC
    end_2023_ts = int(datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc).timestamp())
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": "ETH",
        "tsym": "USD",
        "limit": 365,  # 0 → 364 = 365 points (one full year)
        "toTs": end_2023_ts,
        "api_key": API_KEY,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["Data"]["Data"]
    return [
        {
            "date": datetime.utcfromtimestamp(pt["time"]).strftime("%Y-%m-%d"),
            "close": pt["close"],
        }
        for pt in data
    ]


if __name__ == "__main__":
    for d in fetch_2023_prices_v2():
        print(f"{d['date']}: ${d['close']:.2f}")

    # Save to CSV
    df = pd.DataFrame(fetch_2023_prices_v2())
    df.to_csv(DATA_DIR / "eth_prices_2023.csv", index=False)
    print("Data saved to eth_prices_2023.csv")
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["close"], label="ETH Price")
    plt.title("ETH Price in 2023")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()
