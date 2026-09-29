# src/robustness.py
#
# Small sensitivity checks around the baseline parameters that were fixed
# before we ever looked at test-period performance (z_window=60, entry=2.0,
# exit=0.5, cost=10bps). We vary one parameter at a time and re-run the
# same backtest logic used in backtesting.py, so we can see whether the
# Kalman-is-weakest pattern (and the static-vs-rolling ordering) depends on
# these specific choices. This is a diagnostic, not a search for better
# numbers -- we never touch the primary result based on what we see here.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Running sensitivity checks...")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
with open("data/selected_pair.txt") as f:
    y_name, x_name, _ = f.read().strip().split(",")

ret_y = prices[y_name].pct_change()
ret_x = prices[x_name].pct_change()
test_start = PARAMS["test_start"]

spreads = {
    "Static OLS": pd.read_csv("data/static_spread.csv", index_col=0, parse_dates=True),
    "Rolling OLS": pd.read_csv("data/rolling_spread.csv", index_col=0, parse_dates=True),
    "Kalman filter": pd.read_csv("data/kalman_spread.csv", index_col=0, parse_dates=True),
}


def run_backtest(spread_df, z_window, z_entry, z_exit, z_stop, cost_bps):
    s = spread_df["spread"]
    mean = s.shift(1).rolling(z_window).mean()
    std = s.shift(1).rolling(z_window).std()
    z = (s - mean) / std

    pos = np.zeros(len(z))
    current = 0
    zvals = z.values
    for t in range(len(zvals)):
        zt = zvals[t]
        if np.isnan(zt):
            current = 0
        elif current == 0:
            if zt < -z_entry:
                current = 1
            elif zt > z_entry:
                current = -1
        elif current == 1:
            if zt > -z_exit or zt < -z_stop:
                current = 0
        elif current == -1:
            if zt < z_exit or zt > z_stop:
                current = 0
        pos[t] = current

    pos = pd.Series(pos, index=z.index).shift(1)
    test = spread_df.loc[test_start:].copy()
    pos = pos.loc[test_start:].fillna(0.0)
    pos.iloc[0] = 0.0
    pos.iloc[-1] = 0.0

    beta = test["beta"]
    w_y = pos / (1 + beta.abs())
    w_x = -pos * beta / (1 + beta.abs())

    ry = ret_y.loc[test.index]
    rx = ret_x.loc[test.index]
    gross = w_y * ry + w_x * rx
    turnover = w_y.diff().abs().fillna(w_y.abs()) + w_x.diff().abs().fillna(w_x.abs())
    net = gross - cost_bps / 10000.0 * turnover

    net = net.dropna()
    if net.std() == 0 or len(net) == 0:
        return np.nan
    return net.mean() / net.std() * np.sqrt(PARAMS["trading_days"])


# Consistency check: the sensitivity backtest re-implements the same logic as
# backtesting.py, so at the baseline parameters it must reproduce the main
# net Sharpe ratios exactly.
_main = pd.read_csv("data/performance_metrics.csv", index_col=0)
for _name, _df in spreads.items():
    _s = run_backtest(_df, PARAMS["z_window"], PARAMS["z_entry"], PARAMS["z_exit"],
                      PARAMS["z_stop"], PARAMS["cost_bps"])
    assert np.isclose(_s, _main.loc[_name, "net_sharpe"], atol=1e-10), \
        f"robustness baseline != main backtest for {_name}"

base = dict(z_window=PARAMS["z_window"], z_entry=PARAMS["z_entry"],
            z_exit=PARAMS["z_exit"], z_stop=PARAMS["z_stop"], cost_bps=PARAMS["cost_bps"])

# 1. transaction cost sensitivity
print("\n[1] Transaction cost sensitivity (bps)")
rows = []
for name, df in spreads.items():
    for c in [0, 5, 10, 20]:
        kwargs = dict(base)
        kwargs["cost_bps"] = c
        sharpe = run_backtest(df, **kwargs)
        rows.append({"model": name, "cost_bps": c, "sharpe": sharpe})
    line = "  ".join(f"{c}bp:{s['sharpe']:.2f}" for s, c in
                      zip([r for r in rows if r['model'] == name], [0, 5, 10, 20]))
    print(f"  {name:14s}: {line}")
cost_sens = pd.DataFrame(rows)
cost_sens.to_csv("data/sensitivity_cost.csv", index=False)

# 2. z-score window sensitivity
print("\n[2] Z-score window sensitivity (days)")
rows = []
for name, df in spreads.items():
    for w in [40, 60, 120]:
        kwargs = dict(base)
        kwargs["z_window"] = w
        sharpe = run_backtest(df, **kwargs)
        rows.append({"model": name, "z_window": w, "sharpe": sharpe})
    line = "  ".join(f"{r['z_window']}d:{r['sharpe']:.2f}" for r in rows if r['model'] == name)
    print(f"  {name:14s}: {line}")
window_sens = pd.DataFrame(rows)
window_sens.to_csv("data/sensitivity_window.csv", index=False)

# 3. entry threshold sensitivity
print("\n[3] Entry threshold sensitivity")
rows = []
for name, df in spreads.items():
    for e in [1.5, 2.0, 2.5]:
        kwargs = dict(base)
        kwargs["z_entry"] = e
        sharpe = run_backtest(df, **kwargs)
        rows.append({"model": name, "z_entry": e, "sharpe": sharpe})
    line = "  ".join(f"{r['z_entry']}:{r['sharpe']:.2f}" for r in rows if r['model'] == name)
    print(f"  {name:14s}: {line}")
entry_sens = pd.DataFrame(rows)
entry_sens.to_csv("data/sensitivity_entry.csv", index=False)

# 4. exit threshold sensitivity
print("\n[4] Exit threshold sensitivity")
rows = []
for name, df in spreads.items():
    for ex in [0.0, 0.5, 1.0]:
        kwargs = dict(base)
        kwargs["z_exit"] = ex
        sharpe = run_backtest(df, **kwargs)
        rows.append({"model": name, "z_exit": ex, "sharpe": sharpe})
    line = "  ".join(f"{r['z_exit']}:{r['sharpe']:.2f}" for r in rows if r['model'] == name)
    print(f"  {name:14s}: {line}")
exit_sens = pd.DataFrame(rows)
exit_sens.to_csv("data/sensitivity_exit.csv", index=False)

print("\nSaved: data/sensitivity_cost.csv, data/sensitivity_window.csv, "
      "data/sensitivity_entry.csv, data/sensitivity_exit.csv")
