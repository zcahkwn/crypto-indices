# Nodiens Indices

## Setup

```
git clone https://github.com/zcahkwn/nodiens_index.git
cd nodiens_index
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
python3 -m venv venv
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
```
python scripts/fetch_serp.py

python scripts/fetch_fear_greed.py

python scripts/fetch_cryptocompare.py
```

Coingecko (only data from the past 1 year is available):
```
python scripts/fetch_coingecko.py  
```

## Plot and analyse data
```
python scripts/plot_analyse_bullbear.py
```