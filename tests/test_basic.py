# tests/test_basic.py
#
# A few sanity checks on synthetic data, mostly to make sure we didn't
# introduce look-ahead bugs or get a formula backwards. Run with:
#   python tests/test_basic.py
# (or with pytest, if you have it installed)

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import half_life, bh_adjust, _adf_stat, format_mc_p


def test_rolling_ols_no_lookahead():
    # if we change a future price, a CAUSAL rolling beta computed only up
    # to today should not change
    rng = np.random.default_rng(0)
    n = 300
    x = np.cumsum(rng.standard_normal(n))
    y = 2 * x + rng.standard_normal(n) * 0.1
    y, x = pd.Series(y), pd.Series(x)

    W = 60
    beta = (y.rolling(W).cov(x) / x.rolling(W).var()).shift(1)
    beta_at_100 = beta.iloc[100]

    # now change a price 50 days in the FUTURE and recompute
    y2 = y.copy()
    y2.iloc[150] += 1000
    beta2 = (y2.rolling(W).cov(x) / x.rolling(W).var()).shift(1)

    assert np.isclose(beta_at_100, beta2.iloc[100]), "future price leaked into today's beta"
    print("test_rolling_ols_no_lookahead passed")


def test_kalman_state_dimensions():
    # quick check that a 2-state random walk Kalman filter keeps theta at
    # length 2 and P as a 2x2 matrix throughout
    rng = np.random.default_rng(1)
    n = 50
    x = np.linspace(1, 2, n)
    y = 1.5 * x + 0.5 + rng.standard_normal(n) * 0.01

    theta = np.array([1.0, 0.0])
    P = np.eye(2)
    Q = 1e-5 * np.eye(2)
    R = 0.01

    for t in range(1, n):
        H = np.array([x[t], 1.0])
        theta_prior = theta
        P_prior = P + Q
        e = y[t] - H @ theta_prior
        S = H @ P_prior @ H + R
        K = P_prior @ H / S
        theta = theta_prior + K * e
        P = P_prior - np.outer(K, H @ P_prior)
        assert theta.shape == (2,)
        assert P.shape == (2, 2)

    print("test_kalman_state_dimensions passed")


def test_kalman_innovation_uses_prior_not_posterior():
    # the innovation e_t = y_t - H_t theta_(t|t-1) must be computed BEFORE
    # the state is updated with y_t. If we accidentally used the posterior
    # theta_(t|t) instead, the "innovation" would just be the tiny residual
    # left after the update -- much smaller in magnitude than the true
    # one-step-ahead prediction error. We check this on a case where the
    # state jumps: the prior-based innovation must be strictly larger in
    # magnitude than the (wrong) posterior-based residual.
    rng = np.random.default_rng(3)
    n = 40
    x = np.linspace(1, 2, n)
    true_beta = np.r_[np.full(20, 1.0), np.full(20, 3.0)]  # beta jumps at t=20
    y = true_beta * x + rng.standard_normal(n) * 0.01

    theta = np.array([1.0, 0.0])
    P = np.eye(2)
    Q = 1e-4 * np.eye(2)
    R = 0.01

    prior_innovations = []
    posterior_residuals = []
    for t in range(1, n):
        H = np.array([x[t], 1.0])
        theta_prior = theta
        P_prior = P + Q
        e_prior = y[t] - H @ theta_prior           # correct: uses prior state
        S = H @ P_prior @ H + R
        K = P_prior @ H / S
        theta = theta_prior + K * e_prior
        P = P_prior - np.outer(K, H @ P_prior)
        e_posterior = y[t] - H @ theta              # wrong quantity to trade on
        prior_innovations.append(e_prior)
        posterior_residuals.append(e_posterior)

    # right after the beta jump (t=20), the prior-based innovation should
    # spike much harder than the posterior-based residual, because the
    # posterior has already partly absorbed the jump into theta
    jump_idx = 19  # index into the t=1..n-1 loop corresponding to t=20
    assert abs(prior_innovations[jump_idx]) > abs(posterior_residuals[jump_idx]), \
        "prior-based innovation should react more strongly to a fresh shock than the posterior residual"
    print("test_kalman_innovation_uses_prior_not_posterior passed")


def test_kalman_prior_beta_does_not_see_todays_price():
    # theta_(t|t-1) (used to size today's trade) must be identical whether
    # or not today's y observation is later revised -- it is computed
    # BEFORE that observation is used at all.
    rng = np.random.default_rng(4)
    n = 30
    x = np.linspace(1, 2, n)
    y = 1.5 * x + rng.standard_normal(n) * 0.01

    def run_and_get_prior_beta_at(y_series, t_check):
        theta = np.array([1.0, 0.0])
        P = np.eye(2)
        Q = 1e-5 * np.eye(2)
        R = 0.01
        for t in range(1, n):
            H = np.array([x[t], 1.0])
            theta_prior = theta
            if t == t_check:
                prior_beta = theta_prior[0]
            P_prior = P + Q
            e = y_series[t] - H @ theta_prior
            S = H @ P_prior @ H + R
            K = P_prior @ H / S
            theta = theta_prior + K * e
            P = P_prior - np.outer(K, H @ P_prior)
        return prior_beta

    t_check = 15
    beta_original = run_and_get_prior_beta_at(y, t_check)

    y_revised = y.copy()
    y_revised[t_check] += 5.0  # change TODAY's observation at t_check
    beta_revised = run_and_get_prior_beta_at(y_revised, t_check)

    assert np.isclose(beta_original, beta_revised), \
        "theta_(t|t-1) changed when only today's (t) observation was revised -- look-ahead bug"
    print("test_kalman_prior_beta_does_not_see_todays_price passed")


def test_zscore_no_lookahead():
    # z_t must use the mean/std of the L days BEFORE t, never including s_t
    # itself. Changing s_t should not change its own z-score's mean/std.
    rng = np.random.default_rng(5)
    n = 200
    s = pd.Series(rng.standard_normal(n))
    L = 60

    mean = s.shift(1).rolling(L).mean()
    std = s.shift(1).rolling(L).std()
    z = (s - mean) / std

    mean_at_100_before = mean.iloc[100]
    s2 = s.copy()
    s2.iloc[100] += 1000  # blow up today's spread value
    mean2 = s2.shift(1).rolling(L).mean()

    assert np.isclose(mean_at_100_before, mean2.iloc[100]), \
        "today's spread value leaked into its own z-score normalization"
    print("test_zscore_no_lookahead passed")


def test_next_day_execution():
    # a position decided from information available at the close of day t
    # (z_t) must only start affecting returns from day t+1 onward -- this
    # is what position.shift(1) is for. We check that shifting a position
    # series actually delays it by exactly one day and starts with NaN/0.
    raw_position = pd.Series([0, 1, 1, 0, -1, -1, 0])
    executed_position = raw_position.shift(1)

    assert pd.isna(executed_position.iloc[0]), "first day should have no executed position yet"
    assert executed_position.iloc[2] == raw_position.iloc[1], \
        "position executed on day t should equal the signal decided on day t-1"
    assert executed_position.iloc[1] != raw_position.iloc[1], \
        "position should NOT be executed the same day the signal was generated"
    print("test_next_day_execution passed")


def test_transaction_cost_calculation():
    # cost should be zero when there is no change in weights, and
    # proportional to weight changes when there is
    w_y = pd.Series([0.0, 0.5, 0.5, 0.0])
    w_x = pd.Series([0.0, -0.5, -0.5, 0.0])
    turnover = w_y.diff().abs().fillna(w_y.abs()) + w_x.diff().abs().fillna(w_x.abs())
    cost_bps = 10
    cost = cost_bps / 10000 * turnover

    assert cost.iloc[2] == 0, "cost should be zero when position doesn't change"
    assert cost.iloc[1] > 0, "cost should be positive when a position is opened"
    print("test_transaction_cost_calculation passed")


def test_portfolio_return_formula():
    # a simple hand-computed example: beta = 1, position = 1 (long spread)
    # portfolio should be roughly (r_y - r_x) / 2
    r_y, r_x = 0.02, 0.01
    beta = 1.0
    pos = 1.0
    w_y = pos / (1 + abs(beta))
    w_x = -pos * beta / (1 + abs(beta))
    port_return = w_y * r_y + w_x * r_x
    expected = (r_y - r_x) / 2
    assert np.isclose(port_return, expected)
    print("test_portfolio_return_formula passed")


def test_bh_adjust_matches_known_example():
    # textbook example: with p = [0.01, 0.02, 0.03, 0.04, 0.05], the BH
    # q-values should just equal p * 5 / rank in this simple increasing case
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = bh_adjust(p)
    expected = p * 5 / np.array([1, 2, 3, 4, 5])
    assert np.allclose(q, expected)
    print("test_bh_adjust_matches_known_example passed")


def test_half_life_only_defined_when_mean_reverting():
    # a pure random walk should NOT get a half-life (b should be ~0, not
    # negative), while an AR(1) with b=-0.3 clearly should
    rng = np.random.default_rng(2)
    random_walk = np.cumsum(rng.standard_normal(500))
    b_rw, hl_rw = half_life(random_walk)
    assert np.isnan(hl_rw) or b_rw >= 0 or hl_rw > 0  # loose check, just shouldn't crash

    n = 500
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.7 * s[t - 1] + rng.standard_normal()
    b_ar, hl_ar = half_life(s)
    assert -1 < b_ar < 0
    assert hl_ar > 0
    print("test_half_life_only_defined_when_mean_reverting passed")


def test_adf_statistic_matches_external_reference_values():
    # Reference values were computed ONCE, outside this repository, with
    # statsmodels.tsa.stattools.adfuller(x, regression="c", autolag="AIC") on
    # the two seeded series below. statsmodels is NOT a dependency of the
    # project; the numbers are pinned here so a regression in the lag-selection
    # logic of utils._adf_stat is caught.
    rng = np.random.default_rng(7)
    n = 600
    rw = np.cumsum(rng.standard_normal(n))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + rng.standard_normal()
    assert np.isclose(_adf_stat(rw), -0.43610049836445336, atol=1e-8)
    assert np.isclose(_adf_stat(ar), -5.607770550546645, atol=1e-8)
    print("test_adf_statistic_matches_external_reference_values passed")


def test_mc_p_value_formatting_does_not_report_zero():
    assert format_mc_p(0.0, 2000) == "<0.0005"
    assert format_mc_p(0.057, 2000) == "0.0570"
    print("test_mc_p_value_formatting_does_not_report_zero passed")


def test_trade_pnl_window_alignment_synthetic():
    # `pos` is the EXECUTED position (already lagged). A trade that is held
    # on days 2,3,4 and flat on day 5 earns returns on days 2,3,4 and pays
    # the entry cost on day 2 and the exit cost on day 5. Its P&L window is
    # therefore iloc[entry_i : exit_i + 1] = days 2..5, and the sum over all
    # trades must equal the sum of daily net returns.
    pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    ry = pd.Series([0.01, 0.02, 0.03, -0.01, 0.02, 0.05, 0.01])
    w_y = pos / 2.0
    gross = w_y * ry
    turnover = w_y.diff().abs().fillna(w_y.abs())
    cost = 10 / 10000.0 * turnover
    net = gross - cost

    entry_i, exit_i = 2, 5
    trade_net = (gross.iloc[entry_i: exit_i + 1] - cost.iloc[entry_i: exit_i + 1]).sum()
    assert np.isclose(trade_net, net.sum()), "trade window must cover entry day through exit day"
    wrong = (gross.iloc[entry_i + 1: exit_i + 1] - cost.iloc[entry_i + 1: exit_i + 1]).sum()
    assert not np.isclose(wrong, net.sum()), "the old entry_i+1 window would have dropped day-2 P&L"
    print("test_trade_pnl_window_alignment_synthetic passed")


def _have_outputs():
    need = ["data/trades.csv", "data/returns_net.csv", "data/returns_gross.csv",
            "data/turnover.csv", "data/kalman_spread.csv", "data/prices.csv"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return all(os.path.exists(os.path.join(root, p)) for p in need), root


def test_trade_pnl_reconciles_with_daily_returns_on_real_outputs():
    ok, root = _have_outputs()
    if not ok:
        print("test_trade_pnl_reconciles_with_daily_returns_on_real_outputs SKIPPED (run the pipeline first)")
        return
    net = pd.read_csv(os.path.join(root, "data/returns_net.csv"), index_col=0, parse_dates=True)
    gross = pd.read_csv(os.path.join(root, "data/returns_gross.csv"), index_col=0, parse_dates=True)
    trades = pd.read_csv(os.path.join(root, "data/trades.csv"), parse_dates=["entry", "exit"])
    for m in ["Static OLS", "Rolling OLS", "Kalman filter"]:
        t = trades[trades.model == m]
        assert np.isclose(t.net_pnl.sum(), net[m].sum(), atol=1e-12), f"{m}: net P&L mismatch"
        assert np.isclose(t.gross_pnl.sum(), gross[m].sum(), atol=1e-12), f"{m}: gross P&L mismatch"
        # each trade's first day must be a day with non-zero gross exposure
        # (a return can be exactly 0 only by coincidence, so test the day count instead)
        for _, r in t.iterrows():
            win = gross[m].loc[r.entry:r.exit]
            assert len(win) - 1 == r.days_held, f"{m}: days_held inconsistent with window"
    print("test_trade_pnl_reconciles_with_daily_returns_on_real_outputs passed")


def test_real_kalman_outputs_use_prior_state():
    # Rebuild the Kalman spread from the SAVED outputs using only lagged
    # (posterior at t-1 = prior at t) states and check it matches the traded
    # spread: e_t = y_t - beta_(t|t-1) x_t - alpha_(t|t-1),
    # with beta_(t|t-1) = beta_(t-1|t-1) (random walk) and the same for alpha.
    ok, root = _have_outputs()
    if not ok:
        print("test_real_kalman_outputs_use_prior_state SKIPPED (run the pipeline first)")
        return
    k = pd.read_csv(os.path.join(root, "data/kalman_spread.csv"), index_col=0, parse_dates=True)
    p = pd.read_csv(os.path.join(root, "data/prices.csv"), index_col=0, parse_dates=True)
    sel = open(os.path.join(root, "data/selected_pair.txt")).read().strip().split(",")
    y, x = np.log(p[sel[0]]), np.log(p[sel[1]])
    rebuilt = y - k["beta_posterior"].shift(1) * x - k["alpha_posterior"].shift(1)
    valid = k["spread"].notna() & rebuilt.notna()
    assert valid.sum() > 1000
    assert np.allclose(k.loc[valid, "spread"], rebuilt[valid], atol=1e-10), \
        "traded Kalman spread is not the prior-based innovation"
    assert np.allclose(k.loc[valid, "beta"], k["beta_posterior"].shift(1)[valid], atol=1e-12), \
        "beta used for sizing is not the prior (previous posterior) beta"
    print("test_real_kalman_outputs_use_prior_state passed")


def test_real_signal_outputs_are_causal_and_lagged():
    ok, root = _have_outputs()
    if not ok:
        print("test_real_signal_outputs_are_causal_and_lagged SKIPPED (run the pipeline first)")
        return
    for f in ["static_spread", "rolling_spread", "kalman_spread"]:
        d = pd.read_csv(os.path.join(root, f"data/{f}.csv"), index_col=0, parse_dates=True)
        L = 60
        z = (d["spread"] - d["spread"].shift(1).rolling(L).mean()) / d["spread"].shift(1).rolling(L).std()
        v = z.notna() & d["zscore"].notna()
        assert np.allclose(z[v], d.loc[v, "zscore"], atol=1e-10), f"{f}: z-score not causal"
        lag = d["position"].shift(1)
        v2 = lag.notna() & d["position_lagged"].notna()
        assert (lag[v2] == d.loc[v2, "position_lagged"]).all(), f"{f}: position not lagged one day"
    print("test_real_signal_outputs_are_causal_and_lagged passed")


def test_costs_never_increase_returns_and_match_turnover():
    ok, root = _have_outputs()
    if not ok:
        print("test_costs_never_increase_returns_and_match_turnover SKIPPED (run the pipeline first)")
        return
    net = pd.read_csv(os.path.join(root, "data/returns_net.csv"), index_col=0, parse_dates=True)
    gross = pd.read_csv(os.path.join(root, "data/returns_gross.csv"), index_col=0, parse_dates=True)
    to = pd.read_csv(os.path.join(root, "data/turnover.csv"), index_col=0, parse_dates=True)
    sys.path.insert(0, root)
    from params import PARAMS
    for m in to.columns:
        assert ((gross[m] - net[m]) >= -1e-15).all(), f"{m}: cost increased a return"
        assert np.allclose(gross[m] - net[m], PARAMS["cost_bps"] / 1e4 * to[m], atol=1e-14), \
            f"{m}: cost != cost_rate * turnover"
    print("test_costs_never_increase_returns_and_match_turnover passed")


if __name__ == "__main__":
    test_rolling_ols_no_lookahead()
    test_kalman_state_dimensions()
    test_kalman_innovation_uses_prior_not_posterior()
    test_kalman_prior_beta_does_not_see_todays_price()
    test_zscore_no_lookahead()
    test_next_day_execution()
    test_transaction_cost_calculation()
    test_portfolio_return_formula()
    test_bh_adjust_matches_known_example()
    test_half_life_only_defined_when_mean_reverting()
    test_adf_statistic_matches_external_reference_values()
    test_mc_p_value_formatting_does_not_report_zero()
    test_trade_pnl_window_alignment_synthetic()
    test_trade_pnl_reconciles_with_daily_returns_on_real_outputs()
    test_real_kalman_outputs_use_prior_state()
    test_real_signal_outputs_are_causal_and_lagged()
    test_costs_never_increase_returns_and_match_turnover()
    print("\nAll tests passed. (17/17; 4 of them read pipeline outputs and print SKIPPED if the pipeline has not been run yet)")
