# utils.py
#
# A few statistics helper functions shared by cointegration.py and
# stationarity.py.
#
# This file is the ONLY statistical implementation in the project. It does
# not import statsmodels or scipy, and there is no fallback path. The ADF and
# Engle-Granger p-values are Monte Carlo p-values: the null distribution of
# the test statistic is simulated (n_sim random-walk draws, fixed seed) and
# the p-value is the fraction of simulated statistics that are <= the
# observed one. They are NOT the MacKinnon response-surface p-values that
# statsmodels reports, and are not numerically identical to them. With
# n_sim draws the resolution of the estimator is 1/n_sim; an estimate of 0
# means "p < 1/n_sim", not zero probability (see format_mc_p).
# Note: the simulated null uses driftless Gaussian random walks and a fixed
# n_sim = params.PARAMS["n_sim"]; it does not model the specific
# heteroskedasticity/autocorrelation of the tested series.

import numpy as np


def adf_test(series, n_sim=2000, seed=42):
    """
    Augmented Dickey-Fuller test with a constant. Returns (stat, pvalue).
    p-value comes from simulating n_sim random walks of the same length and
    seeing how extreme our statistic is relative to theirs.
    """
    x = np.asarray(series, dtype=float)
    stat = _adf_stat(x, has_const=True)
    n = len(x)
    rng = np.random.default_rng(seed)
    sim_stats = np.empty(n_sim)
    for i in range(n_sim):
        w = np.cumsum(rng.standard_normal(n))
        sim_stats[i] = _adf_stat(w, has_const=True)
    pval = (sim_stats <= stat).mean()
    return stat, pval


def _adf_stat(x, has_const=True, maxlag=None):
    # ADF regression: delta x_t = [a] + g*x_(t-1) + sum_j phi_j*delta x_(t-j) + e_t
    # Lag length p is chosen by AIC with every candidate p scored on the SAME
    # sample (the one that leaves room for maxlag lags); the regression is then
    # re-estimated at the chosen p on the largest available sample and the
    # t-statistic on g is returned. (Scoring different p on different sample
    # lengths makes the AICs incomparable and biases the choice towards long
    # lags -- an earlier version of this file did exactly that.)
    n = len(x)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
    maxlag = min(maxlag, n // 2 - 3)
    dx = np.diff(x)

    def design(p, start):
        # rows t = start .. len(dx)-1 ; columns: x_(t) [level, i.e. x_{t-1} in x-time],
        # lagged differences dx_(t-1..t-p), optional constant
        y = dx[start:]
        cols = [x[start:-1]]
        for j in range(1, p + 1):
            cols.append(dx[start - j:len(dx) - j])
        if has_const:
            cols.append(np.ones(len(y)))
        return y, np.column_stack(cols)

    best_aic, best_p = np.inf, 0
    for p in range(maxlag + 1):
        y, X = design(p, maxlag)                 # common sample for all p
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ssr = float(resid @ resid)
        aic = len(y) * np.log(ssr / len(y)) + 2 * X.shape[1]
        if aic < best_aic:
            best_aic, best_p = aic, p

    y, X = design(best_p, best_p)                # re-estimate on the full sample
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    k = X.shape[1]
    se = np.sqrt(float(resid @ resid) / (len(y) - k) * np.linalg.inv(X.T @ X)[0, 0])
    return beta[0] / se


def engle_granger_test(y, x, n_sim=2000, seed=42):
    """
    Two-step Engle-Granger cointegration test.
    Step 1: OLS  y = alpha + beta*x + u
    Step 2: ADF test on the residuals u (no constant in the ADF regression)
    Returns (eg_stat, pvalue, alpha, beta)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = beta_hat
    resid = y - X @ beta_hat
    stat = _adf_stat(resid, has_const=False)

    rng = np.random.default_rng(seed)
    n = len(y)
    sim_stats = np.empty(n_sim)
    for i in range(n_sim):
        w1 = np.cumsum(rng.standard_normal(n))
        w2 = np.cumsum(rng.standard_normal(n))
        Xs = np.column_stack([np.ones(n), w2])
        b, *_ = np.linalg.lstsq(Xs, w1, rcond=None)
        r = w1 - Xs @ b
        sim_stats[i] = _adf_stat(r, has_const=False)
    pval = (sim_stats <= stat).mean()
    return stat, pval, alpha, beta


def kpss_test(series):
    """
    KPSS test for level stationarity (null hypothesis: stationary).
    Bartlett-kernel long-run variance estimate, standard KPSS statistic.
    Returns (stat, reject_at_5pct) using the Kwiatkowski et al. (1992)
    asymptotic critical value of 0.463 at the 5% level.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    e = x - x.mean()
    lags = int(np.ceil(12 * (n / 100) ** 0.25))
    s = np.cumsum(e)
    lrvar = e @ e / n
    for i in range(1, lags + 1):
        w = 1 - i / (lags + 1)
        lrvar += 2 * w * (e[i:] @ e[:-i]) / n
    stat = (s @ s) / (n ** 2 * lrvar)
    crit_5pct = 0.463
    return stat, stat > crit_5pct


def half_life(spread):
    """
    Fit  delta s_t = a + b*s_(t-1) + e_t  and convert b into a half-life.
    Only meaningful if -1 < b < 0 (mean reverting). Returns (b, half_life)
    where half_life is np.nan if b is outside that range.
    """
    s = np.asarray(spread, dtype=float)
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([np.ones(len(lag)), lag])
    beta, *_ = np.linalg.lstsq(X, ds, rcond=None)
    b = beta[1]
    if -1 < b < 0:
        hl = -np.log(2) / np.log(1 + b)
    else:
        hl = np.nan
    return b, hl


def format_mc_p(p, n_sim, digits=4):
    """
    Format a Monte-Carlo p-value honestly. With n_sim draws the smallest
    non-zero value the estimator can return is 1/n_sim, so an estimate of
    exactly 0 means "no simulated statistic was as extreme", i.e. p < 1/n_sim
    -- NOT that the probability is literally zero.
    """
    if p <= 0:
        return f"<{1.0 / n_sim:.{digits}f}"
    return f"{p:.{digits}f}"


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjusted p-values (a.k.a. q-values)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q
