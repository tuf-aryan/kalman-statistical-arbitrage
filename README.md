# Kalman Filter Statistical Arbitrage

This is an undergraduate research project on pairs trading.

I wanted to compare three ways of estimating the hedge ratio:

- Static OLS
- 60-day Rolling OLS
- Kalman Filter

I tested them on four large Indian banking stocks: HDFC Bank,
ICICI Bank, SBI and Kotak Mahindra Bank.

The main question I was interested in was whether allowing the
hedge ratio to change over time actually improves the statistical
properties of the spread and the trading performance.

The main test is out-of-sample (2020–2024). Transaction costs are
also included in the backtest.

The Kalman filter produced the most mean-reverting spread in the
test sample, but this did not translate into better trading
performance. In fact, after costs, the Kalman strategy performed
worse than both OLS benchmarks.

This was not the result I initially expected, but I kept it because
it is an important part of the comparison.


## Data

The project uses daily adjusted closing prices of four NSE-listed
banking stocks:

- HDFC Bank
- ICICI Bank
- State Bank of India (SBI)
- Kotak Mahindra Bank

The data covers 1 January 2015 to 30 December 2024.

The analysis is split into a formation period and an out-of-sample
test period:

- Formation period: 2015–2019
- Test period: 2020–2024

The formation period is used for pair selection and model setup.
The trading results are evaluated on the separate test period.


## Methods

For the selected pair, I compare three hedge-ratio methods.

### 1. Static OLS

The hedge ratio is estimated using the formation period and then
kept fixed during the test period.

### 2. Rolling OLS

The hedge ratio is re-estimated using a 60-day rolling window.

### 3. Kalman Filter

The hedge ratio and intercept are allowed to change over time using
a state-space model.

The Kalman filter uses a one-step-ahead estimate for the trading
signals so that the current observation is not used to make the
same-day trading decision.


## Analysis

The project follows these main steps:

1. Collect and clean the price data.
2. Test the possible stock pairs using the Engle-Granger approach.
3. Apply the Benjamini-Hochberg correction for multiple testing.
4. Estimate hedge ratios using static OLS, rolling OLS and the
   Kalman filter.
5. Construct the corresponding spreads.
6. Check spread stationarity using ADF and KPSS tests.
7. Estimate the AR(1) half-life of the spreads.
8. Generate trading signals using rolling z-scores.
9. Run the strategy on the out-of-sample period.
10. Include transaction costs in the backtest.
11. Compare the performance of the three methods.
12. Run a few robustness checks.


## Project Structure

```text
Kalman_stat_arb/
│
├── data/
├── figures/
├── results/
├── src/
│   ├── data_collection.py
│   ├── eda.py
│   ├── cointergration.py
│   ├── static_ols.py
│   ├── rolling_ols.py
│   ├── kalman_filter.py
│   ├── stationarity.py
│   ├── signal_generation.py
│   ├── backtesting.py
│   ├── performance_metrices.py
│   ├── robustness.py
│   └── 12_visualizations.py
│
├── tests/
├── paper/
├── run_all.py
├── requirements.txt
└── README.md