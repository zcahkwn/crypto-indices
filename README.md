# Crypto Indices

## Setup

```
git clone https://github.com/zcahkwn/crypto-indices.git
cd crypto-indices
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
pyenv local 3.11.6
python -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

---

## Set up the environmental variables

put your APIs in `.env`:

```
SERP_API = "xxx"

COINMARKETCAP_API = "xxx"

CRYPTOCOMPARE_API = "xxx"
```


## Fetch data 

Fetch Google search data from SerpAPI:
```
python scripts/fetch_serp.py
```

Fetch OHLC data from cyptocompare (only for BTC, ETH, LTC, DOGE, BCH):
```
python scripts/fetch_cryptocompare.py
```

Fetch OHLC data from Coingecko (only data from the past 1 year is available)(for XRP, SOL, BNB, etc):
```
python scripts/fetch_coingecko.py  
```

Fetch crypto fear and greed data:
```
python scripts/fetch_fear_greed.py
```

Fetch US stock market fear and greed index:
```
python scripts/fetch_finhacker.py
```

Fetch metals price from LMBA (gold, silver)
```
python scripts/fetch_metal_price.py
```

VIX index (index_vix.csv) downloaded directly from CBOE website: https://www.cboe.com/tradable_products/vix/vix_historical_data/ 



## Plot and analyse data

Plot and calculate correlation (pearson, granger causalty test) between Google search trend and market data 
```
python scripts/plot_analyse_bullbear.py
```

## Machine learning model
