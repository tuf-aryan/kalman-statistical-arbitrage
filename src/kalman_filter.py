# src/kalman_filter.py
#
# Kalman filter for a hedge ratio that is allowed to drift over time.
#
# State:        theta_t = [beta_t, alpha_t]'        (random walk)
# Observation:  y_t = H_t theta_t + e_t,  H_t = [x_t, 1]
#
# Keep the prior and posterior states separate.
# The traded spread uses the prior state, so today's observation is
# not used in the same day's hedge ratio.
# Posterior values are saved for reference only.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Running the Kalman filter...")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
with open("data/selected_pair.txt") as f:
    y_name, x_name, _ = f.read().strip().split(",")
with open("data/static_ols_params.txt") as f:
    _, _, resid_var = f.read().strip().split(",")
resid_var = float(resid_var)   # this becomes R, calibrated on formation data

log_prices = np.log(prices)
y = log_prices[y_name].values
x = log_prices[x_name].values
n = len(y)

delta = PARAMS["kalman_delta"]
burnin = PARAMS["kalman_burnin"]

Q = delta / (1 - delta) * np.eye(2)   # process noise: how much theta can move per day
R = resid_var                          # observation noise, from formation-period OLS residuals

# initial state: OLS on the first `burnin` formation observations
X0 = np.column_stack([x[:burnin], np.ones(burnin)])
theta, *_ = np.linalg.lstsq(X0, y[:burnin], rcond=None)   # theta = [beta, alpha]
P = R * np.eye(2)

innovation = np.full(n, np.nan)        # one-step-ahead spread we actually trade
beta_prior = np.full(n, np.nan)        # beta_(t|t-1), used to size the trade at t
beta_posterior = np.full(n, np.nan)    # beta_(t|t), saved for reference only
alpha_posterior = np.full(n, np.nan)

beta_posterior[burnin - 1] = theta[0]
alpha_posterior[burnin - 1] = theta[1]

for t in range(burnin, n):
    H = np.array([x[t], 1.0])

    # --- prediction step (prior) ---
    theta_prior = theta                  # random walk: E[theta_t] = theta_(t-1)
    P_prior = P + Q

    # --- innovation: this is the spread we actually trade ---
    y_pred = H @ theta_prior
    e = y[t] - y_pred
    S = H @ P_prior @ H + R

    # --- Kalman gain and posterior (update) step ---
    K = P_prior @ H / S
    theta = theta_prior + K * e
    P = P_prior - np.outer(K, H @ P_prior)

    innovation[t] = e
    beta_prior[t] = theta_prior[0]
    beta_posterior[t] = theta[0]
    alpha_posterior[t] = theta[1]

out = pd.DataFrame({
    "spread": innovation,        # traded spread = one-step-ahead innovation
    "beta": beta_prior,          # hedge ratio used to size the trade at t
    "beta_posterior": beta_posterior,   # saved for reference only (not used for trading or plotting)
    "alpha_posterior": alpha_posterior,
}, index=prices.index)

out.to_csv("data/kalman_spread.csv")

print(f"Pair: {y_name} (y) vs {x_name} (x)")
print(f"delta = {delta:.0e}   Q = delta/(1-delta) * I   R = {R:.6f} (formation resid. var.)")
print(f"burn-in = {burnin} observations")
print("Saved: data/kalman_spread.csv")
