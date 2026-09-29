# src/performance_metrics.py
#
# Standard performance metrics computed from the daily return series and
# the trade log produced by backtesting.py. Risk-free rate is set to 0
# because the strategy is a self-financing long-short spread (the short
# leg's proceeds fund the long leg); this is stated in the paper.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Computing performance metrics...")

gross = pd.read_csv("data/returns_gross.csv", index_col=0, parse_dates=True)
net = pd.read_csv("data/returns_net.csv", index_col=0, parse_dates=True)
trades = pd.read_csv("data/trades.csv", parse_dates=["entry", "exit"])
turnover = pd.read_csv("data/turnover.csv", index_col=0, parse_dates=True)

td = PARAMS["trading_days"]


def summarize(r):
    r = r.dropna()
    ann_return = r.mean() * td
    ann_vol = r.std() * np.sqrt(td)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    # Sortino: downside deviation relative to a 0 target return, computed
    # over ALL observations: sqrt(mean(min(r, 0)^2)) * sqrt(252).
    down_dev = np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2)) * np.sqrt(td)
    sortino = ann_return / down_dev if down_dev > 0 else np.nan
    # wealth path starts at 1 so a loss on the very first day counts as a drawdown
    wealth = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)],
                       ignore_index=True)
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    total_return = wealth.iloc[-1] - 1
    return dict(ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, calmar=calmar,
                total_return=total_return)


rows = []
for model in gross.columns:
    g = summarize(gross[model])
    n = summarize(net[model])
    sub = trades[trades.model == model]
    n_trades = len(sub)
    hit_rate = (sub.net_pnl > 0).mean() if n_trades > 0 else np.nan
    avg_hold = sub.days_held.mean() if n_trades > 0 else np.nan
    avg_trade = sub.net_pnl.mean() if n_trades > 0 else np.nan
    wins = sub.loc[sub.net_pnl > 0, "net_pnl"].sum()
    losses = -sub.loc[sub.net_pnl < 0, "net_pnl"].sum()
    profit_factor = wins / losses if losses > 0 else np.nan

    rows.append({
        "model": model,
        "gross_ann_return": g["ann_return"], "gross_sharpe": g["sharpe"],
        "net_ann_return": n["ann_return"], "net_ann_vol": n["ann_vol"],
        "net_sharpe": n["sharpe"], "net_sortino": n["sortino"],
        "net_max_dd": n["max_dd"], "net_calmar": n["calmar"],
        "net_total_return": n["total_return"],
        "n_trades": n_trades, "hit_rate": hit_rate,
        "avg_holding_days": avg_hold, "avg_trade_pnl": avg_trade,
        "profit_factor": profit_factor,
        "total_turnover": turnover[model].sum() if model in turnover.columns else np.nan,
    })

    print(f"\n{model}")
    print(f"  gross: ann. return = {g['ann_return']:+.2%}   Sharpe = {g['sharpe']:.3f}")
    print(f"  net:   ann. return = {n['ann_return']:+.2%}   vol = {n['ann_vol']:.2%}   "
          f"Sharpe = {n['sharpe']:.3f}   max DD = {n['max_dd']:.2%}")
    if n_trades > 0:
        print(f"  trades = {n_trades}   hit rate = {hit_rate:.1%}   "
              f"avg hold = {avg_hold:.1f} days   profit factor = {profit_factor:.2f}")

metrics = pd.DataFrame(rows).set_index("model")
metrics.to_csv("data/performance_metrics.csv")

with open("results/tables/performance.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
    f.write("Method & Net ann. ret. & Net vol. & Sharpe & Sortino & Max DD & Hit rate & \\# trades\\\\\n\\midrule\n")
    for m, r in metrics.iterrows():
        hr = f"{r['hit_rate']:.1%}" if pd.notna(r["hit_rate"]) else "--"
        nt = f"{int(r['n_trades'])}" if r["n_trades"] > 0 else "--"
        f.write(f"{m.replace('&', chr(92)+'&')} & {r['net_ann_return']:+.2%} & "
                f"{r['net_ann_vol']:.2%} & {r['net_sharpe']:.3f} & {r['net_sortino']:.3f} & "
                f"{r['net_max_dd']:.2%} & {hr} & {nt}\\\\\n".replace("%", "\\%"))
    f.write("\\bottomrule\n\\end{tabular}\n")

with open("results/tables/performance_gross_net.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("Method & Gross ann. ret. & Gross Sharpe & Net ann. ret. & Net Sharpe\\\\\n\\midrule\n")
    for m, r in metrics.iterrows():
        f.write((f"{m.replace('&', chr(92)+'&')} & {r['gross_ann_return']:+.2%} & "
                 f"{r['gross_sharpe']:.3f} & {r['net_ann_return']:+.2%} & "
                 f"{r['net_sharpe']:.3f}\\\\\n").replace("%", "\\%"))
    f.write("\\bottomrule\n\\end{tabular}\n")

with open("results/tables/trade_stats.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    f.write("Method & \\# trades & Hit rate & Avg. hold (d) & Avg. trade P\\&L & Profit factor & Total turnover\\\\\n\\midrule\n")
    for m, r in metrics.iterrows():
        if r["n_trades"] == 0:
            continue
        f.write((f"{m} & {int(r['n_trades'])} & {r['hit_rate']:.1%} & {r['avg_holding_days']:.1f} & "
                 f"{r['avg_trade_pnl']:+.2%} & {r['profit_factor']:.2f} & {r['total_turnover']:.1f}\\\\\n").replace("%", "\\%"))
    f.write("\\bottomrule\n\\end{tabular}\n")

print("\nSaved: data/performance_metrics.csv, results/tables/performance.tex, "
      "results/tables/performance_gross_net.tex, results/tables/trade_stats.tex")
