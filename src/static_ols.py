# Fit the static hedge ratio using the formation period only.
# Keep alpha and beta fixed during the test period.
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Fitting static OLS hedge ratio (formation period only)...")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
with open("data/selected_pair.txt") as f:
    y_name, x_name, _ = f.read().strip().split(",")

log_prices = np.log(prices)
y = log_prices[y_name]
x = log_prices[x_name]

formation = y.index <= PARAMS["formation_end"]
X = np.column_stack([np.ones(formation.sum()), x[formation].values])
beta_hat, *_ = np.linalg.lstsq(X, y[formation].values, rcond=None)
alpha, beta = beta_hat

resid = y[formation].values - X @ beta_hat
resid_var = np.var(resid, ddof=2)

print(f"Pair: {y_name} (y) vs {x_name} (x)")
print(f"alpha = {alpha:.4f}   beta = {beta:.4f}")
print(f"residual variance (formation) = {resid_var:.6f}")

# spread over the WHOLE sample using the frozen formation coefficients
spread = y - alpha - beta * x

out = pd.DataFrame({"spread": spread, "beta": beta}, index=y.index)
out.to_csv("data/static_spread.csv")

with open("data/static_ols_params.txt", "w") as f:
    f.write(f"{alpha},{beta},{resid_var}\n")

print("Saved: data/static_spread.csv, data/static_ols_params.txt")
