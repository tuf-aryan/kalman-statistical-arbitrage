# Re-estimate the hedge ratio each day using the previous roll_window days.
# The window ends at t-1 to avoid look-ahead.
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Computing rolling OLS hedge ratio...")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
with open("data/selected_pair.txt") as f:
    y_name, x_name, _ = f.read().strip().split(",")

log_prices = np.log(prices)
y = log_prices[y_name]
x = log_prices[x_name]

W = PARAMS["roll_window"]

# rolling beta from the standard formula cov(y,x)/var(x), using the window
# ENDING AT t-1 (shift by one day so today's price doesn't enter the fit)
beta = y.rolling(W).cov(x) / x.rolling(W).var()
alpha = y.rolling(W).mean() - beta * x.rolling(W).mean()

beta = beta.shift(1)
alpha = alpha.shift(1)

spread = y - alpha - beta * x

out = pd.DataFrame({"spread": spread, "beta": beta, "alpha": alpha}, index=y.index)
out.to_csv("data/rolling_spread.csv")

print(f"Rolling window = {W} days")
print(f"Beta range over test period: "
      f"{beta.loc[PARAMS['test_start']:].min():.3f} to {beta.loc[PARAMS['test_start']:].max():.3f}")
print("Saved: data/rolling_spread.csv")
