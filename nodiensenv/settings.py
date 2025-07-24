import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SERP_API = os.environ.get("SERP_API")
COINMARKETCAP_API = os.environ.get("COINMARKETCAP_API")
CRYPTOCOMPARE_API = os.environ.get("CRYPTOCOMPARE_API")
