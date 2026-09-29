# Test all six pairs using the formation period only.
# The test is run on log prices.
#
# Select the pair with the lowest formation-period p-value.

import os
import sys
from itertools import combinations
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS
from utils import engle_granger_test, bh_adjust, format_mc_p

print("Running Engle-Granger cointegration tests on the formation period...")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
formation = prices.loc[PARAMS["start_date"]:PARAMS["formation_end"]]
log_prices = np.log(formation)

tickers = sorted(PARAMS["tickers"])          # fix the ordering ahead of time
pairs = list(combinations(tickers, 2))

print(f"Formation sample: {formation.index[0].date()} to {formation.index[-1].date()} "
      f"({len(formation)} days)")
print(f"Testing all {len(pairs)} pairs\n")

rows = []
for y_name, x_name in pairs:
    y = log_prices[y_name].values
    x = log_prices[x_name].values
    stat, pval, alpha, beta = engle_granger_test(y, x, n_sim=PARAMS["n_sim"], seed=PARAMS["seed"])
    rows.append({"y": y_name, "x": x_name, "eg_stat": stat, "pvalue": pval,
                 "alpha": alpha, "beta": beta})
    print(f"  {y_name:14s} vs {x_name:14s}  EG stat = {stat:.3f}   p = {pval:.4f}")

results = pd.DataFrame(rows)
results["bh_qvalue"] = bh_adjust(results["pvalue"].values)
results = results.sort_values("pvalue").reset_index(drop=True)

print("\nBenjamini-Hochberg adjusted p-values (q = {:.2f}):".format(PARAMS["fdr_q"]))
print(results[["y", "x", "pvalue", "bh_qvalue"]].to_string(index=False))

best = results.iloc[0]
survives_bh = best["bh_qvalue"] < PARAMS["fdr_q"]

print(f"\nSelected pair (smallest formation p-value): {best['y']} vs {best['x']}")
print(f"  raw p-value = {best['pvalue']:.4f}, BH q-value = {best['bh_qvalue']:.4f}")
if survives_bh:
    print("  This pair survives the BH-FDR correction.")
else:
    print("  This pair does NOT survive BH-FDR correction -> selection is EXPLORATORY.")

results.to_csv("data/cointegration_results.csv", index=False)

with open("data/selected_pair.txt", "w") as f:
    f.write(f"{best['y']},{best['x']},{survives_bh}\n")

# simple LaTeX table for the paper
with open("results/tables/cointegration.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrl}\n\\toprule\n")
    f.write("y & x & EG stat. & p-value & BH q-value & Selected\\\\\n\\midrule\n")
    for _, r in results.iterrows():
        mark = "Yes" if (r["y"] == best["y"] and r["x"] == best["x"]) else "No"
        yname = r["y"].replace(".NS", "")
        xname = r["x"].replace(".NS", "")
        f.write(f"{yname} & {xname} & {r['eg_stat']:.3f} & {format_mc_p(r['pvalue'], PARAMS['n_sim'])} & "
                f"{format_mc_p(r['bh_qvalue'], PARAMS['n_sim'])} & {mark}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("\nSaved: data/cointegration_results.csv, data/selected_pair.txt, "
      "results/tables/cointegration.tex")
