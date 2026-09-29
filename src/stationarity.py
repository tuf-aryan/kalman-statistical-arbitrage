# src/stationarity.py
#
# Checks whether each of the three spreads (static, rolling, Kalman) looks
# stationary/mean-reverting DURING THE TEST PERIOD. This is the "spread
# quality" question, separate from whether the spread is actually
# profitable to trade (that comes later in backtesting.py).

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS
from utils import adf_test, kpss_test, half_life, format_mc_p

print("Checking spread stationarity over the test period...")

static = pd.read_csv("data/static_spread.csv", index_col=0, parse_dates=True)
rolling = pd.read_csv("data/rolling_spread.csv", index_col=0, parse_dates=True)
kalman = pd.read_csv("data/kalman_spread.csv", index_col=0, parse_dates=True)

test_start = PARAMS["test_start"]
spreads = {
    "Static OLS": static["spread"].loc[test_start:].dropna(),
    "Rolling OLS": rolling["spread"].loc[test_start:].dropna(),
    "Kalman filter": kalman["spread"].loc[test_start:].dropna(),
}

rows = []
for name, s in spreads.items():
    adf_stat, adf_p = adf_test(s.values, n_sim=PARAMS["n_sim"], seed=PARAMS["seed"])
    kpss_stat, kpss_reject = kpss_test(s.values)
    b, hl = half_life(s.values)
    rows.append({
        "model": name, "n": len(s), "adf_stat": adf_stat, "adf_pvalue": adf_p,
        "kpss_stat": kpss_stat, "kpss_reject_5pct": bool(kpss_reject), "ar1_b": b, "half_life": hl,
        "spread_std": s.std(),
    })
    hl_text = f"{hl:.1f} days" if pd.notna(hl) else "n/a (not mean reverting)"
    print(f"\n{name}")
    print(f"  n = {len(s)}   ADF stat = {adf_stat:.3f}   ADF MC p = {format_mc_p(adf_p, PARAMS['n_sim'])}")
    print(f"  KPSS stat = {kpss_stat:.3f}   (reject stationarity at 5% if > 0.463: {bool(kpss_reject)})")
    print(f"  AR(1) coefficient b = {b:.4f}   half-life = {hl_text}")
    print(f"  spread std. dev. = {s.std():.4f}")

results = pd.DataFrame(rows)
results.to_csv("data/stationarity_results.csv", index=False)

with open("results/tables/stationarity.tex", "w") as f:
    f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
    f.write("Method & ADF stat. & ADF p-value & KPSS stat. & AR(1) $b$ & Half-life\\\\\n\\midrule\n")
    for _, r in results.iterrows():
        hl = f"{r['half_life']:.1f} d" if pd.notna(r["half_life"]) else "n/a"
        f.write(f"{r['model']} & {r['adf_stat']:.3f} & {format_mc_p(r['adf_pvalue'], PARAMS['n_sim'])} & "
                f"{r['kpss_stat']:.3f} & {r['ar1_b']:.4f} & {hl}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("\nSaved: data/stationarity_results.csv, results/tables/stationarity.tex")
