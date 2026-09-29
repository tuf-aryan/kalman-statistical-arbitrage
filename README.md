# Kalman Filter Statistical Arbitrage

Undergraduate research project comparing three ways of estimating the hedge ratio
for pairs trading -- static OLS, 60-day rolling OLS, and a Kalman filter -- on four
Indian banking stocks (HDFC Bank, ICICI Bank, Kotak Mahindra Bank, SBI).

Research question: does a Kalman-filter dynamic hedge ratio generate a more
stationary spread and superior risk-adjusted performance than static OLS and
rolling OLS? In this sample, on one exploratory pair (HDFCBANK vs KOTAKBANK, which
does **not** survive Benjamini-Hochberg correction), the Kalman spread has the
strongest stationarity diagnostics but the weakest net-of-cost trading results.
See `paper/main.pdf`. The figures reported there are read from the files in `data/`
and `results/` -- do not copy numbers from this README.

## Setup

```
pip install -r requirements.txt
```

Python 3.9+ (the pinned versions in `requirements.txt` were used for the final run).
Dependencies: `numpy`, `pandas`, `matplotlib`, and `yfinance` (imported only if
`data/prices.csv` is missing; with the shipped file no network access is needed).
**statsmodels and scipy are not used.** Building the PDF needs a LaTeX
installation (`pdflatex`, `bibtex`).

## Reproduce the analysis

```
python run_all.py
```

Runs, in order: data loading -> Engle-Granger pair screening (all 6 pairs) ->
static / rolling / Kalman hedge ratios -> stationarity diagnostics -> z-score
signals -> backtest -> performance metrics -> robustness sweeps -> independent
re-computation of the headline numbers (`src/verify_independent.py`) -> figures
-> `paper/numbers.tex` -> unit tests. Any failed check makes `run_all.py` exit
with an error. It writes `data/*.csv`, `results/tables/*.tex`,
`results/figures/*.png` and `paper/numbers.tex`.

**`run_all.py` does not compile the PDF.** Build the paper separately:

```
cd paper
latexmk -pdf main.tex
# or by hand (three pdflatex passes after bibtex were needed for stable cross-references):
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

The paper reads figures and tables directly from `../results/` and every number
in its text from `numbers.tex`; there are no duplicate copies.

## Statistical implementation (single, deterministic)

All tests live in `utils.py` and are the only implementation in the project:
ADF (constant, AIC lag choice on a common sample), two-step Engle-Granger,
KPSS (Bartlett kernel, 5% critical value 0.463), AR(1) half-life, and
Benjamini-Hochberg. ADF and Engle-Granger p-values are **Monte-Carlo p-values**
from `n_sim = 2000` simulated random-walk null draws with fixed seed 42
(`params.py`). Their resolution is 1/2000, so a value of 0 is reported as
`< 0.0005`. They are not the MacKinnon response-surface p-values from
`statsmodels` and are not numerically identical to them. There is no fallback
to another implementation.

## What each script does

| Script | What it does |
|---|---|
| `src/data_collection.py` | loads (or downloads) daily adjusted closes into `data/prices.csv` |
| `src/cointegration.py` | Engle-Granger on the 2015-2019 formation sample for all 6 pairs, BH q-values, picks the lowest-p pair (exploratory if q >= 0.05) |
| `src/static_ols.py` | fixed hedge ratio fitted on formation data only |
| `src/rolling_ols.py` | 60-day rolling hedge ratio using the window ending at t-1 |
| `src/kalman_filter.py` | Kalman hedge ratio; traded spread is the prior-based innovation |
| `src/stationarity.py` | ADF / KPSS / half-life of each spread, 2020-2024 only |
| `src/signal_generation.py` | causal z-score (mean/std of the previous 60 days) and position; position applied from t+1 |
| `src/backtesting.py` | portfolio returns, turnover-based costs, trade log (with a reconciliation assertion) |
| `src/performance_metrics.py` | Sharpe, Sortino, drawdown, trade statistics, turnover |
| `src/robustness.py` | one-at-a-time sensitivity sweeps (cost, z-window, entry, exit) |
| `src/verify_independent.py` | independent numpy re-computation of returns, metrics, trade stats, Kalman/rolling replays, KPSS, half-life, BH |
| `src/visualizations.py` | all figures |
| `src/make_paper_numbers.py` | writes every number used in the paper text to `paper/numbers.tex` |

`params.py` holds every parameter. `tests/test_basic.py` holds the unit and
output-level tests (`python tests/test_basic.py`).

## Data and split

Daily adjusted closes from Yahoo Finance (`auto_adjust=True`), cached in
`data/prices.csv`. Formation period 2015-01-01 to 2019-12-31 (pair selection, static
OLS, Kalman calibration); test period 2020-01-01 onward (evaluation only). Exact sample
sizes are in `paper/numbers.tex`.

## Key parameters (all in `params.py`)

rolling window 60 days; Kalman delta 1e-5, burn-in 60; z-score window 60, entry 2.0,
exit 0.5, stop 4.0; transaction cost 10 bps of traded notional (robustness: 0, 5, 10, 20).
Exposure convention (all three methods): w_y = pos/(1+|beta|), w_x = -pos*beta/(1+|beta|).

## Reproducibility statement

`data/` outputs are deterministic (fixed seed, no other randomness). A clean-directory
rerun was compared file by file with the delivered outputs; see `FINAL_AUDIT_REPORT.md`
for exactly what was compared and what matched.
