"""
fetch US stock marketfear and greed index from ttps://www.finhacker.cz/fear-and-greed-index-historical-data-and-chart/
"""

import requests
import pandas as pd
from nodiensenv.constants import DATA_DIR


def fetch_fear_greed_data():
    url = "https://www.finhacker.cz/wp-content/custom-api/fear-greed-data2.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Referer": "https://www.finhacker.cz/fear-and-greed-index-historical-data-and-chart/",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Check if response contains JSON
        if response.headers.get("content-type") == "application/json; charset=utf-8":
            return response.json()
        else:
            print("Response is not JSON format")
            return response.text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


data = fetch_fear_greed_data()
df = pd.DataFrame(data["daily"])
df.to_csv(DATA_DIR / "fear_greed_stock.csv", columns=["date", "value"], index=False)
