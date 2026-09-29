# src/visualizations.py
#
# Makes all the plots used in the paper, straight from the CSV files the
# other scripts saved. Nothing here is hand-edited.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Making figures...")

os.makedirs("results/figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
with open("data/selected_pair.txt") as f:
    y_name, x_name, _ = f.read().strip().split(",")

static = pd.read_csv("data/static_spread.csv", index_col=0, parse_dates=True)
rolling = pd.read_csv("data/rolling_spread.csv", index_col=0, parse_dates=True)
kalman = pd.read_csv("data/kalman_spread.csv", index_col=0, parse_dates=True)
gross = pd.read_csv("data/returns_gross.csv", index_col=0, parse_dates=True)
net = pd.read_csv("data/returns_net.csv", index_col=0, parse_dates=True)

formation_end = pd.Timestamp(PARAMS["formation_end"])
test_start = PARAMS["test_start"]

# 1. normalized prices
fig, ax = plt.subplots(figsize=(9, 4.5))
norm = prices / prices.iloc[0] * 100
for col in prices.columns:
    ax.plot(norm.index, norm[col], label=col.replace(".NS", ""), linewidth=1.1)
ax.axvline(formation_end, color="black", linestyle="--", linewidth=1.2, label="formation / test split")
ax.set_title("Normalized prices, Indian banking universe")
ax.set_ylabel("Price (base = 100)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/fig1_normalized_prices.png", dpi=150)
plt.close()

# 2. hedge ratio comparison, test period
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(rolling.loc[test_start:].index, rolling.loc[test_start:, "beta"],
        label=f"Rolling OLS (W={PARAMS['roll_window']})", color="#9467bd")
ax.plot(kalman.loc[test_start:].index, kalman.loc[test_start:, "beta"],
        label="Kalman filter (prior, used for sizing)", color="#d62728")
ax.axhline(static["beta"].iloc[0], color="#2ca02c", linestyle="--", label="Static OLS")
ax.set_title(f"Hedge ratio over the test period: {y_name.replace('.NS','')} vs {x_name.replace('.NS','')}")
ax.set_ylabel("beta")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/fig2_hedge_ratio_comparison.png", dpi=150)
plt.close()

# 3. spread comparison (3 panels)
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ax, (df, name, color) in zip(axes, [
        (static, "Static OLS", "#2ca02c"), (rolling, "Rolling OLS", "#9467bd"),
        (kalman, "Kalman filter", "#d62728")]):
    s = df.loc[test_start:, "spread"]
    ax.plot(s.index, s.values, color=color, linewidth=0.8)
    ax.axhline(s.mean(), color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{name} spread, test period", fontsize=10)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("results/figures/fig3_spreads_comparison.png", dpi=150)
plt.close()

# 4. ADF stat / half-life bar chart
stat_df = pd.read_csv("data/stationarity_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
colors = ["#2ca02c", "#9467bd", "#d62728"]
axes[0].bar(stat_df["model"], stat_df["adf_stat"], color=colors)
axes[0].set_title("ADF statistic (test period; more negative = stronger rejection of unit root)", fontsize=8)
axes[0].set_ylabel("ADF statistic")
axes[0].tick_params(axis="x", labelrotation=15)
hl = stat_df["half_life"]
axes[1].bar(stat_df["model"], hl, color=colors)
axes[1].set_title("AR(1) half-life (days)", fontsize=9)
axes[1].set_ylabel("days")
axes[1].tick_params(axis="x", labelrotation=15)
plt.tight_layout()
plt.savefig("results/figures/fig4_adf_halflife.png", dpi=150)
plt.close()

# 5. z-score signal plot (Kalman)
z = kalman.loc[test_start:, "zscore"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(z.index, z.values, linewidth=0.7, color="#1f77b4")
for lv in [PARAMS["z_entry"], -PARAMS["z_entry"]]:
    ax.axhline(lv, color="gray", linestyle="--", linewidth=1)
for lv in [PARAMS["z_stop"], -PARAMS["z_stop"]]:
    ax.axhline(lv, color="gray", linestyle="-.", linewidth=1)
ax.set_ylim(-5, 5)
ax.set_title("Kalman spread z-score, test period (dashed: entry, dash-dot: stop-loss)")
ax.set_ylabel("z-score")
plt.tight_layout()
plt.savefig("results/figures/fig5_zscore_signals.png", dpi=150)
plt.close()

# 6. equity curves (net of costs)
equity = (1 + net).cumprod()
fig, ax = plt.subplots(figsize=(9, 4.5))
colors_map = {"Static OLS": "#2ca02c", "Rolling OLS": "#9467bd",
              "Kalman filter": "#d62728", "Buy & Hold": "#1f77b4"}
for col in equity.columns:
    ax.plot(equity.index, equity[col], label=col, color=colors_map[col])
ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
ax.set_title("Equity curves, net of transaction costs (test period, start = 1)")
ax.set_ylabel("Cumulative wealth")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/fig6_equity_curves.png", dpi=150)
plt.close()

# 7. drawdowns
fig, ax = plt.subplots(figsize=(9, 4))
for col in equity.columns:
    # wealth starts at 1, so the running peak is at least 1 (same as performance_metrics.py)
    dd = equity[col] / np.maximum(equity[col].cummax(), 1.0) - 1
    ax.plot(dd.index, dd.values * 100, label=col, color=colors_map[col])
ax.set_title("Drawdowns, net of transaction costs")
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/fig7_drawdowns.png", dpi=150)
plt.close()

# 8. sensitivity panels
cost_sens = pd.read_csv("data/sensitivity_cost.csv")
window_sens = pd.read_csv("data/sensitivity_window.csv")
entry_sens = pd.read_csv("data/sensitivity_entry.csv")
exit_sens = pd.read_csv("data/sensitivity_exit.csv")

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for name, color in zip(["Static OLS", "Rolling OLS", "Kalman filter"], colors):
    sub = cost_sens[cost_sens.model == name]
    axes[0, 0].plot(sub.cost_bps, sub.sharpe, marker="o", label=name, color=color)
    sub = window_sens[window_sens.model == name]
    axes[0, 1].plot(sub.z_window, sub.sharpe, marker="o", label=name, color=color)
    sub = entry_sens[entry_sens.model == name]
    axes[1, 0].plot(sub.z_entry, sub.sharpe, marker="o", label=name, color=color)
    sub = exit_sens[exit_sens.model == name]
    axes[1, 1].plot(sub.z_exit, sub.sharpe, marker="o", label=name, color=color)

axes[0, 0].set_title("Net Sharpe vs. transaction cost (bps)", fontsize=9)
axes[0, 1].set_title("Net Sharpe vs. z-score window (days)", fontsize=9)
axes[1, 0].set_title("Net Sharpe vs. entry threshold", fontsize=9)
axes[1, 1].set_title("Net Sharpe vs. exit threshold", fontsize=9)
for ax in axes.flat:
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Net Sharpe ratio")
axes[0, 0].legend(fontsize=7)
plt.tight_layout()
plt.savefig("results/figures/fig8_sensitivity.png", dpi=150)
plt.close()

print("Saved figures to results/figures/")
