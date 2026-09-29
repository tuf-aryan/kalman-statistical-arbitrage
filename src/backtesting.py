# src/backtesting.py
#
# Turns positions into portfolio returns.
#
# We use SIMPLE returns for each leg (not log returns), since that is what
# actually gets paid out: r_t = P_t / P_(t-1) - 1.
#
# Portfolio weights (same convention for all three models):
#   w_y =  position / (1 + |beta|)
#   w_x = -position * beta / (1 + |beta|)
# so the two legs' weights always add up to at most 1 in absolute value.
# This is the same normalization for static, rolling and Kalman -- the
# only thing that differs between the three strategies is beta itself.
#
# Transaction costs are charged on actual turnover (the change in weights
# from one day to the next), not a flat fee per trade:
#   cost_t = cost_bps/10000 * (|w_y_t - w_y_(t-1)| + |w_x_t - w_x_(t-1)|)
#
# We evaluate the test period only. Position is forced to zero on the
# first and last day of the test period, so the strategy starts flat and
# ends flat.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Running the backtest...")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
with open("data/selected_pair.txt") as f:
    y_name, x_name, _ = f.read().strip().split(",")

ret_y = prices[y_name].pct_change()
ret_x = prices[x_name].pct_change()

cost_rate = PARAMS["cost_bps"] / 10000.0
test_start = PARAMS["test_start"]

files = {
    "Static OLS": "data/static_spread.csv",
    "Rolling OLS": "data/rolling_spread.csv",
    "Kalman filter": "data/kalman_spread.csv",
}

all_gross = {}
all_net = {}
all_turnover = {}
all_trades = []

for name, path in files.items():
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    test = df.loc[test_start:].copy()

    pos = test["position_lagged"].fillna(0.0).copy()
    pos.iloc[0] = 0.0     # start flat on the first test day
    pos.iloc[-1] = 0.0    # end flat on the last test day

    beta = test["beta"]

    w_y = pos / (1 + beta.abs())
    w_x = -pos * beta / (1 + beta.abs())

    ry = ret_y.loc[test.index]
    rx = ret_x.loc[test.index]

    gross_return = w_y * ry + w_x * rx
    turnover = w_y.diff().abs().fillna(w_y.abs()) + w_x.diff().abs().fillna(w_x.abs())
    cost = cost_rate * turnover
    net_return = gross_return - cost

    all_gross[name] = gross_return
    all_net[name] = net_return
    all_turnover[name] = turnover

    # Trade log. `pos` here is ALREADY the executed (lagged) position, i.e.
    # pos[i] is the position held DURING day i and earning day i's return.
    #   entry_i = first day with pos != 0  -> first day the trade earns a
    #             return, and the day the entry turnover/cost is charged.
    #   i       = first day with pos == 0 after entry -> gross_return[i] = 0,
    #             but the exit turnover/cost is charged on this day.
    # So the trade's P&L window is days entry_i .. i INCLUSIVE, i.e.
    # iloc[entry_i : i + 1]. (An earlier version used entry_i + 1, which
    # dropped the first day's return and the entry cost from every trade.)
    trade_side = 0
    entry_i = None
    for i in range(len(pos)):
        if trade_side == 0 and pos.iloc[i] != 0:
            trade_side = pos.iloc[i]
            entry_i = i
        elif trade_side != 0 and pos.iloc[i] == 0:
            gp = gross_return.iloc[entry_i: i + 1].sum()
            cp = cost.iloc[entry_i: i + 1].sum()
            all_trades.append({
                "model": name, "entry": test.index[entry_i], "exit": test.index[i],
                "side": int(trade_side), "days_held": i - entry_i,
                "gross_pnl": gp, "cost": cp, "net_pnl": gp - cp,
            })
            trade_side = 0

    # Reconciliation: the strategy starts and ends flat, so every daily
    # return and every cost belongs to exactly one trade.
    _sub = [t for t in all_trades if t["model"] == name]
    assert np.isclose(sum(t["net_pnl"] for t in _sub), net_return.sum(), atol=1e-12), \
        f"{name}: trade net P&L does not reconcile with daily net returns"
    assert np.isclose(sum(t["gross_pnl"] for t in _sub), gross_return.sum(), atol=1e-12), \
        f"{name}: trade gross P&L does not reconcile with daily gross returns"

    print(f"{name}: {sum(t['model'] == name for t in all_trades)} completed trades, "
          f"mean daily net return = {net_return.mean()*1e4:.2f} bp")

gross_df = pd.DataFrame(all_gross)
net_df = pd.DataFrame(all_net)

# buy-and-hold benchmark: just hold the y leg, for context (not a fair
# market-neutral comparison, shown for reference only)
buy_hold = ret_y.loc[gross_df.index]
gross_df["Buy & Hold"] = buy_hold
net_df["Buy & Hold"] = buy_hold

pd.DataFrame(all_turnover).to_csv("data/turnover.csv")
gross_df.to_csv("data/returns_gross.csv")
net_df.to_csv("data/returns_net.csv")

trades = pd.DataFrame(all_trades)
trades.to_csv("data/trades.csv", index=False)

print(f"\nBuy & hold ({y_name}) mean daily return over test period: "
      f"{buy_hold.mean()*1e4:.2f} bp")
print("Saved: data/returns_gross.csv, data/returns_net.csv, data/turnover.csv, data/trades.csv")
