# Convert each spread into a trading position using the same z-score
# rule and thresholds.
#
# The z-score at t uses the previous z_window observations to avoid
# look-ahead.
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAMS

print("Generating trading signals...")

L = PARAMS["z_window"]
z_entry = PARAMS["z_entry"]
z_exit = PARAMS["z_exit"]
z_stop = PARAMS["z_stop"]


def zscore(spread):
    mean = spread.shift(1).rolling(L).mean()
    std = spread.shift(1).rolling(L).std()
    return (spread - mean) / std


def positions_from_zscore(z):
    # simple state machine, one position at a time
    z = z.values
    pos = np.zeros(len(z))
    current = 0
    for t in range(len(z)):
        zt = z[t]
        if np.isnan(zt):
            current = 0
        elif current == 0:
            if zt < -z_entry:
                current = 1          # long the spread
            elif zt > z_entry:
                current = -1         # short the spread
        elif current == 1:
            if zt > -z_exit or zt < -z_stop:
                current = 0
        elif current == -1:
            if zt < z_exit or zt > z_stop:
                current = 0
        pos[t] = current
    return pos


files = {
    "Static OLS": "data/static_spread.csv",
    "Rolling OLS": "data/rolling_spread.csv",
    "Kalman filter": "data/kalman_spread.csv",
}

for name, path in files.items():
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    z = zscore(df["spread"])
    pos = positions_from_zscore(z)
    # position is decided using info up to close of day t, so we trade it
    # starting the NEXT day -> shift by one before it is used for returns
    df["zscore"] = z
    df["position"] = pos
    df["position_lagged"] = pd.Series(pos, index=df.index).shift(1)
    df.to_csv(path)
    n_signals = int((pd.Series(pos).diff().fillna(0) != 0).sum())
    print(f"{name}: {n_signals} position changes over the whole sample")

print("Saved signals back into data/static_spread.csv, data/rolling_spread.csv, "
      "data/kalman_spread.csv")
