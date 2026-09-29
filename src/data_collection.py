# Download daily adjusted close prices and save them to data/prices.csv.
# Use the existing file if the required columns are already present.

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Downloading price data...")

data_path = "data/prices.csv"
tickers = PARAMS["tickers"]

need_download = True
if os.path.exists(data_path):
    cached = pd.read_csv(data_path, index_col=0, parse_dates=True)
    if set(tickers).issubset(cached.columns):
        prices = cached[tickers].loc[PARAMS["start_date"]:PARAMS["end_date"]]
        need_download = False
        print(f"Using cached data/prices.csv ({len(prices)} rows)")

if need_download:
    import yfinance as yf
    raw = yf.download(tickers, start=PARAMS["start_date"], end=PARAMS["end_date"],
                       auto_adjust=True, progress=False, group_by="ticker")
    prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers})
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    
# Fill small gaps and remove rows that still contain missing prices.
prices = prices.ffill(limit=5).dropna()
prices = prices[tickers]

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
prices.to_csv(data_path)

n_formation = (prices.index <= PARAMS["formation_end"]).sum()
n_test = (prices.index >= PARAMS["test_start"]).sum()

print(f"Saved {len(prices)} rows to {data_path}")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Formation period: {PARAMS['start_date']} to {PARAMS['formation_end']} ({n_formation} days)")
print(f"Test period: {PARAMS['test_start']} onward ({n_test} days)")
