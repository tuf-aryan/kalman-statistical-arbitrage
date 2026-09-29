
# Independently recompute the main results.
# This script does not import the project's backtest or metric functions.
# The checks compare the recomputed values with the saved outputs.
# A mismatch causes the script to fail.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Independent verification...")
lines = []
fails = []


def check(label, a, b, tol=1e-9):
    ok = bool(np.isclose(a, b, atol=tol, rtol=1e-9))
    lines.append(f"{'OK  ' if ok else 'FAIL'} {label}: independent={a:.10g}  generated={b:.10g}")
    if not ok:
        fails.append(label)


prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
with open("data/selected_pair.txt") as f:
    yn, xn, _ = f.read().strip().split(",")
perf = pd.read_csv("data/performance_metrics.csv", index_col=0)
trades = pd.read_csv("data/trades.csv", parse_dates=["entry", "exit"])
net_csv = pd.read_csv("data/returns_net.csv", index_col=0, parse_dates=True)
gross_csv = pd.read_csv("data/returns_gross.csv", index_col=0, parse_dates=True)
stat = pd.read_csv("data/stationarity_results.csv").set_index("model")

Py = prices[yn].values
Px = prices[xn].values
idx = prices.index
test_mask = idx >= pd.Timestamp(PARAMS["test_start"])
ti = np.where(test_mask)[0]
c = PARAMS["cost_bps"] / 1e4
TD = PARAMS["trading_days"]

files = {"Static OLS": "data/static_spread.csv", "Rolling OLS": "data/rolling_spread.csv",
         "Kalman filter": "data/kalman_spread.csv"}

for name, path in files.items():
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    pos_all = d["position_lagged"].values
    beta_all = d["beta"].values

    # ---- rebuild daily returns with explicit loops
    n = len(ti)
    pos = np.nan_to_num(pos_all[ti], nan=0.0)
    pos[0] = 0.0
    pos[-1] = 0.0
    beta = beta_all[ti]
    gross = np.zeros(n); cost = np.zeros(n); turn = np.zeros(n)
    wy_prev = wx_prev = 0.0
    for k in range(n):
        t = ti[k]
        ry = Py[t] / Py[t - 1] - 1.0
        rx = Px[t] / Px[t - 1] - 1.0
        wy = pos[k] / (1.0 + abs(beta[k]))
        wx = -pos[k] * beta[k] / (1.0 + abs(beta[k]))
        # equivalent form: (r_y - beta r_x)/(1+|beta|) * position
        assert np.isclose(wy * ry + wx * rx, pos[k] * (ry - beta[k] * rx) / (1 + abs(beta[k])), atol=1e-15)
        gross[k] = wy * ry + wx * rx
        turn[k] = abs(wy - wy_prev) + abs(wx - wx_prev)
        cost[k] = c * turn[k]
        wy_prev, wx_prev = wy, wx
    net = gross - cost
    check(f"{name}: max |net - returns_net.csv|", np.abs(net - net_csv[name].values).max(), 0.0, 1e-13)
    check(f"{name}: max |gross - returns_gross.csv|", np.abs(gross - gross_csv[name].values).max(), 0.0, 1e-13)
    assert (cost >= 0).all()

    # ---- performance metrics from the net series (numpy only)
    mu = net.mean() * TD
    sd = net.std(ddof=1) * np.sqrt(TD)
    dd_dev = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)) * np.sqrt(TD)
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    mdd = (wealth / np.maximum.accumulate(wealth) - 1.0).min()
    p = perf.loc[name]
    check(f"{name}: cumulative net return", wealth[-1] - 1.0, p["net_total_return"])
    check(f"{name}: annualised net return", mu, p["net_ann_return"])
    check(f"{name}: annualised net vol", sd, p["net_ann_vol"])
    check(f"{name}: net Sharpe", mu / sd, p["net_sharpe"])
    check(f"{name}: net Sortino", mu / dd_dev, p["net_sortino"])
    check(f"{name}: net max drawdown", mdd, p["net_max_dd"])
    check(f"{name}: gross Sharpe", gross.mean() * TD / (gross.std(ddof=1) * np.sqrt(TD)), p["gross_sharpe"])

    # ---- trade statistics from daily positions + returns (run-length path)
    held = pos != 0
    starts = [k for k in range(n) if held[k] and (k == 0 or not held[k - 1])]
    ends = [k for k in range(n) if (not held[k]) and k > 0 and held[k - 1]]   # first flat day
    assert len(starts) == len(ends), "open trade at end of sample"
    pnl = np.array([net[s:e + 1].sum() for s, e in zip(starts, ends)])
    holds = np.array([e - s for s, e in zip(starts, ends)])
    wins = pnl[pnl > 0].sum(); losses = -pnl[pnl < 0].sum()
    sub = trades[trades.model == name]
    check(f"{name}: number of trades", len(pnl), len(sub))
    check(f"{name}: winning trades", int((pnl > 0).sum()), int((sub.net_pnl > 0).sum()))
    check(f"{name}: hit rate", (pnl > 0).mean(), p["hit_rate"])
    check(f"{name}: average trade P&L", pnl.mean(), p["avg_trade_pnl"])
    check(f"{name}: profit factor", wins / losses, p["profit_factor"])
    check(f"{name}: average holding days", holds.mean(), p["avg_holding_days"])
    check(f"{name}: total turnover", turn.sum(), p["total_turnover"])
    check(f"{name}: sum of trade P&L == sum of daily net returns", pnl.sum(), net.sum())
    # hand-checked single example: the first trade
    s0, e0 = starts[0], ends[0]
    hand = sum(gross[k] - c * turn[k] for k in range(s0, e0 + 1))
    check(f"{name}: first trade P&L by hand vs trades.csv", hand, sub.net_pnl.iloc[0])

# ---- Kalman replay (independent re-implementation of the recursion)
kal = pd.read_csv("data/kalman_spread.csv", index_col=0, parse_dates=True)
lp = np.log(prices.values)
yl = lp[:, list(prices.columns).index(yn)]
xl = lp[:, list(prices.columns).index(xn)]
_, _, rv = open("data/static_ols_params.txt").read().strip().split(",")
R = float(rv); dl = PARAMS["kalman_delta"]; B = PARAMS["kalman_burnin"]
A = np.c_[xl[:B], np.ones(B)]
th = np.linalg.solve(A.T @ A, A.T @ yl[:B])
P = R * np.eye(2)
e_rep = np.full(len(yl), np.nan); b_rep = np.full(len(yl), np.nan)
for t in range(B, len(yl)):
    H = np.array([xl[t], 1.0])
    Pp = P + dl / (1 - dl) * np.eye(2)
    b_rep[t] = th[0]                       # prior beta: state BEFORE seeing y_t
    e_rep[t] = yl[t] - H @ th              # innovation with the prior state
    S = H @ Pp @ H + R
    K = Pp @ H / S
    th = th + K * e_rep[t]
    P = (np.eye(2) - np.outer(K, H)) @ Pp
m = ~np.isnan(e_rep)
check("Kalman: max |replayed innovation - saved spread|", np.abs(e_rep[m] - kal["spread"].values[m]).max(), 0.0, 1e-10)
check("Kalman: max |replayed prior beta - saved beta|", np.abs(b_rep[m] - kal["beta"].values[m]).max(), 0.0, 1e-10)

# ---- Rolling OLS replay with np.polyfit on the window ending at t-1
rol = pd.read_csv("data/rolling_spread.csv", index_col=0, parse_dates=True)
W = PARAMS["roll_window"]
worst = 0.0
for t in ti[::37]:                                   # every 37th test day
    b_hat, a_hat = np.polyfit(xl[t - W:t], yl[t - W:t], 1)     # rows t-W .. t-1 only
    worst = max(worst, abs(b_hat - rol["beta"].values[t]), abs(a_hat - rol["alpha"].values[t]))
check("Rolling OLS: max |polyfit(window ending t-1) - saved beta/alpha|", worst, 0.0, 1e-8)

# ---- Stationarity: half-life and KPSS from scratch
for name, path in files.items():
    s = pd.read_csv(path, index_col=0, parse_dates=True)["spread"].loc[PARAMS["test_start"]:].dropna().values
    ds = np.diff(s); lag = s[:-1]
    slope = np.cov(lag, ds, ddof=0)[0, 1] / np.var(lag)
    hl = -np.log(2) / np.log(1 + slope) if -1 < slope < 0 else np.nan
    check(f"{name}: AR(1) b", slope, stat.loc[name, "ar1_b"])
    check(f"{name}: half-life", hl, stat.loc[name, "half_life"])
    n = len(s); e = s - s.mean(); L = int(np.ceil(12 * (n / 100) ** 0.25))
    lr = sum((1 - abs(j) / (L + 1)) * (e[abs(j):] @ e[:n - abs(j)]) / n for j in range(-L, L + 1))
    check(f"{name}: KPSS statistic", (np.cumsum(e) ** 2).sum() / (n ** 2 * lr), stat.loc[name, "kpss_stat"])

# ---- BH q-values with a different algorithm (step-up by explicit search)
co = pd.read_csv("data/cointegration_results.csv")
p = co["pvalue"].values; m_ = len(p)
q_alt = np.array([min(1.0, min(p[j] * m_ / (sum(p <= p[j])) if False else
                               min(p2 * m_ / r for r, p2 in enumerate(sorted(p), 1) if p2 >= p[j])
                               for _ in [0])) for j in range(m_)])
check("BH: max |q (independent) - q (generated)|", np.abs(q_alt - co["bh_qvalue"].values).max(), 0.0, 1e-12)
lines.append(f"INFO pairs with BH q < {PARAMS['fdr_q']}: {(co['bh_qvalue'] < PARAMS['fdr_q']).sum()} of {len(co)}")

with open("data/independent_verification.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
if fails:
    print(f"\n{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print(f"\nAll {sum(l.startswith('OK') for l in lines)} independent checks match the generated outputs.")
