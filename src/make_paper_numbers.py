# src/make_paper_numbers.py
#
# Pulls every number the paper needs out of the CSV files the other
# scripts produced, and writes them as LaTeX \newcommand macros into
# paper/numbers.tex. This way nothing in the paper is typed in by hand --
# if you change a parameter and re-run run_all.py, the paper updates too.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS
from utils import format_mc_p

print("Collecting numbers for the paper...")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
coint = pd.read_csv("data/cointegration_results.csv")
with open("data/selected_pair.txt") as f:
    y_name, x_name, survives = f.read().strip().split(",")
with open("data/static_ols_params.txt") as f:
    alpha, beta, resid_var = [float(v) for v in f.read().strip().split(",")]
stat = pd.read_csv("data/stationarity_results.csv").set_index("model")
perf = pd.read_csv("data/performance_metrics.csv", index_col=0)
trades = pd.read_csv("data/trades.csv")
cost_sens = pd.read_csv("data/sensitivity_cost.csv")
win_sens = pd.read_csv("data/sensitivity_window.csv")
ent_sens = pd.read_csv("data/sensitivity_entry.csv")
ext_sens = pd.read_csv("data/sensitivity_exit.csv")
rolling = pd.read_csv("data/rolling_spread.csv", index_col=0, parse_dates=True)
NSIM = PARAMS["n_sim"]

M = {}
M["NTickers"] = len(PARAMS["tickers"])
M["DataStart"] = PARAMS["start_date"]
M["DataEnd"] = str(prices.index[-1].date())
M["FormationEnd"] = PARAMS["formation_end"]
M["TestStart"] = PARAMS["test_start"]
M["NFormation"] = int((prices.index <= PARAMS["formation_end"]).sum())
M["NTest"] = int((prices.index >= PARAMS["test_start"]).sum())
M["NPairs"] = len(coint)
M["PairY"] = y_name.replace(".NS", "")
M["PairX"] = x_name.replace(".NS", "")
M["FDRq"] = PARAMS["fdr_q"]
M["Exploratory"] = "true" if survives == "False" else "false"
M["RollWindow"] = PARAMS["roll_window"]
_mant, _exp = f"{PARAMS['kalman_delta']:.0e}".split("e")
M["KalmanDelta"] = (f"10^{{{int(_exp)}}}" if float(_mant) == 1 else f"{_mant}\\times 10^{{{int(_exp)}}}")
M["KalmanBurnin"] = PARAMS["kalman_burnin"]
M["ZWindow"] = PARAMS["z_window"]
M["ZEntry"] = PARAMS["z_entry"]
M["ZExit"] = PARAMS["z_exit"]
M["ZStop"] = PARAMS["z_stop"]
M["CostBps"] = f"{PARAMS['cost_bps']:g}"
M["StaticAlpha"] = f"{alpha:.4f}"
M["StaticBeta"] = f"{beta:.4f}"
M["NSim"] = NSIM
M["Seed"] = PARAMS["seed"]
M["MCResolution"] = f"{1.0 / NSIM:.4f}"
M["BestP"] = format_mc_p(coint["pvalue"].min(), NSIM)
M["BestQ"] = format_mc_p(coint.loc[coint["pvalue"].idxmin(), "bh_qvalue"], NSIM)
M["MinQ"] = format_mc_p(coint["bh_qvalue"].min(), NSIM)
M["NSurviveBH"] = int((coint["bh_qvalue"] < PARAMS["fdr_q"]).sum())
M["FormationFirst"] = str(prices.index[0].date())
M["FormationLast"] = str(prices.index[prices.index <= PARAMS["formation_end"]][-1].date())
M["TestFirst"] = str(prices.index[prices.index >= PARAMS["test_start"]][0].date())
M["TestLast"] = str(prices.index[-1].date())
M["KPSSCrit"] = "0.463"

# rolling-OLS beta instability, test period
rb = rolling.loc[PARAMS["test_start"]:, "beta"]
M["RollBetaNTest"] = f"{len(rb):,}"
M["RollBetaNeg"] = int((rb < 0).sum())
M["RollBetaNegPct"] = f"{(rb < 0).mean()*100:.1f}"
M["RollBetaAbsGtTwo"] = int((rb.abs() > 1.8).sum())
M["RollBetaMin"] = f"{rb.min():.2f}"
M["RollBetaMax"] = f"{rb.max():.2f}"

# hedge-ratio dynamics over the test period
kal = pd.read_csv("data/kalman_spread.csv", index_col=0, parse_dates=True)
kb = kal.loc[PARAMS["test_start"]:, "beta"]
M["KalBetaMin"] = f"{kb.min():.2f}"
M["KalBetaMax"] = f"{kb.max():.2f}"
M["KalBetaMAD"] = f"{kb.diff().abs().mean():.4f}"
M["RollBetaMAD"] = f"{rb.diff().abs().mean():.3f}"
M["KalStd"] = f"{kb.std():.3f}"

# sensitivity summary: one row per DISTINCT specification (baseline counted once)
b = dict(z_window=PARAMS["z_window"], z_entry=PARAMS["z_entry"], z_exit=PARAMS["z_exit"], cost_bps=PARAMS["cost_bps"])
recs = {}
def _add(df, col):
    for _, r in df.iterrows():
        spec = dict(b); spec[col] = r[col]
        key = (spec["z_window"], spec["z_entry"], spec["z_exit"], spec["cost_bps"])
        recs.setdefault(key, {})[r["model"]] = r["sharpe"]
_add(cost_sens, "cost_bps"); _add(win_sens, "z_window"); _add(ent_sens, "z_entry"); _add(ext_sens, "z_exit")
sens = pd.DataFrame(recs).T
sens.columns = [c for c in sens.columns]
n_spec = len(sens)
M["SensNSpecs"] = n_spec
M["SensKalmanNeg"] = int((sens["Kalman filter"] < 0).sum())
M["SensKalmanWeakest"] = int(((sens["Kalman filter"] < sens["Static OLS"]) & (sens["Kalman filter"] < sens["Rolling OLS"])).sum())
M["SensStaticGtRolling"] = int((sens["Static OLS"] > sens["Rolling OLS"]).sum())
M["SensRollingGtStatic"] = int((sens["Rolling OLS"] > sens["Static OLS"]).sum())
M["SensStaticPos"] = int((sens["Static OLS"] > 0).sum())
M["SensRollingPos"] = int((sens["Rolling OLS"] > 0).sum())
rg = sens[sens["Rolling OLS"] > sens["Static OLS"]]
M["SensRollingGtStaticWhere"] = "; ".join(
    f"z-window={int(k[0])}, entry={k[1]}, exit={k[2]}, cost={int(k[3])} bp" for k in rg.index) or "none"
M["KalmanSensMax"] = f"{sens['Kalman filter'].max():.2f}"
M["KalmanSensMin"] = f"{sens['Kalman filter'].min():.2f}"
sens.to_csv("data/sensitivity_summary.csv", index_label=["z_window", "z_entry", "z_exit", "cost_bps"])

key_map = {"Static OLS": "Static", "Rolling OLS": "Rolling", "Kalman filter": "Kalman"}
for model, key in key_map.items():
    r = stat.loc[model]
    M[f"ADF{key}"] = f"{r['adf_stat']:.3f}"
    M[f"ADFp{key}"] = format_mc_p(r["adf_pvalue"], NSIM)
    M[f"KPSS{key}"] = f"{r['kpss_stat']:.3f}"
    M[f"KPSSReject{key}"] = "rejects" if r["kpss_reject_5pct"] else "does not reject"
    M[f"ARb{key}"] = f"{r['ar1_b']:.4f}"
    M[f"HL{key}"] = f"{r['half_life']:.1f}" if pd.notna(r["half_life"]) else "n/a"
    M[f"Std{key}"] = f"{r['spread_std']:.4f}"

for model, key in list(key_map.items()) + [("Buy & Hold", "BH")]:
    if model not in perf.index:
        continue
    r = perf.loc[model]
    M[f"NetRet{key}"] = f"{r['net_ann_return']*100:.2f}"
    M[f"NetVol{key}"] = f"{r['net_ann_vol']*100:.2f}"
    M[f"NetSharpe{key}"] = f"{r['net_sharpe']:.3f}"
    M[f"NetSortino{key}"] = f"{r['net_sortino']:.3f}"
    M[f"NetTotRet{key}"] = f"{r['net_total_return']*100:.2f}"
    M[f"GrossRet{key}"] = f"{r['gross_ann_return']*100:.2f}"
    M[f"NetMDD{key}"] = f"{r['net_max_dd']*100:.2f}"
    M[f"GrossSharpe{key}"] = f"{r['gross_sharpe']:.3f}"
    if r["n_trades"] > 0:
        M[f"NTrades{key}"] = int(r["n_trades"])
        M[f"HitRate{key}"] = f"{r['hit_rate']*100:.1f}"
        M[f"AvgHold{key}"] = f"{r['avg_holding_days']:.1f}"
        M[f"ProfitFactor{key}"] = f"{r['profit_factor']:.2f}"
        M[f"AvgTradePnl{key}"] = f"{r['avg_trade_pnl']*100:.2f}"
        M[f"Turnover{key}"] = f"{r['total_turnover']:.1f}"

# break-even cost for Kalman (linear interpolation across the swept grid).
# If the Kalman Sharpe is already negative at 0 bps (gross, no costs at
# all), there is no positive break-even cost -- the strategy is unprofitable
# before any trading costs are even applied. We flag that case explicitly
# instead of reporting a made-up number.
sub = cost_sens[cost_sens.model == "Kalman filter"].sort_values("cost_bps")
vals = sub["sharpe"].values
costs = sub["cost_bps"].values

kalman_gross_sharpe = perf.loc["Kalman filter", "gross_sharpe"]
kalman_negative_gross = kalman_gross_sharpe < 0

be = np.nan
for i in range(len(vals) - 1):
    if np.sign(vals[i]) != np.sign(vals[i + 1]):
        be = costs[i] + (0 - vals[i]) * (costs[i + 1] - costs[i]) / (vals[i + 1] - vals[i])
        break

M["KalmanGrossSharpe"] = f"{kalman_gross_sharpe:.3f}"
M["KalmanNegativeGross"] = "true" if kalman_negative_gross else "false"
if kalman_negative_gross:
    # no positive break-even cost exists; report the (negative) gross Sharpe instead
    M["BreakevenKalman"] = "not applicable"
elif not np.isnan(be):
    M["BreakevenKalman"] = f"{be:.1f}"
else:
    M["BreakevenKalman"] = "outside the tested 0--20 bps range"

os.makedirs("paper", exist_ok=True)
with open("paper/numbers.tex", "w") as f:
    f.write("% auto-generated by src/make_paper_numbers.py -- do not edit by hand\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

print(f"Saved {len(M)} macros to paper/numbers.tex")
