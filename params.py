# params.py
#
# All the settings for the project live in one place so the paper and the
# code always use the same numbers. Change something here and re-run
# run_all.py.

PARAMS = {
    # data
    "tickers": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",

    # formation / test split (formation = fit everything, test = trade it)
    "formation_end": "2019-12-31",
    "test_start": "2020-01-01",

    # pair selection
    "fdr_q": 0.05,          # Benjamini-Hochberg level
    "n_sim": 2000,          # Monte Carlo draws for the Engle-Granger p-value

    # rolling OLS
    "roll_window": 60,

    # Kalman filter
    "kalman_delta": 1e-5,   # controls how much the state is allowed to move each day
    "kalman_burnin": 60,    # observations used to set the initial state

    # trading signal
    "z_window": 60,
    "z_entry": 2.0,
    "z_exit": 0.5,
    "z_stop": 4.0,

    # transaction costs, in basis points of notional traded (per side)
    "cost_bps": 10.0,

    "trading_days": 252,
    "seed": 42,
}
