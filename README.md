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


## Set up the environmental variables

put your APIs in `.env`:

```
GOOGLE_TREND_API="cc03de0acd0af0110207ab0d6a06f06f48f02d1cd4e480ca4e2086865d810814"
BTC_PRICE_API="39197365d7df7f30de03f1e2b0db9e6808f30a23cc73b621ce5cebe3ec576352"
```


## Fetch data 
```
python scripts/fetch_bitcoin.py
```
