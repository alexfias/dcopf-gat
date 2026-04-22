# dcopf_gat/windowing.py
from __future__ import annotations
import numpy as np

def make_windows_concat(x: np.ndarray, y: np.ndarray, window: int):
    """
    Concatenate a lookback window into the feature axis.

    Input:
      x: (T, N, F)
      y: (T, Y)
      window: int, number of past steps (0 = no windowing)

    Output:
      xw: (T-window, N, F*(window+1))
      yw: (T-window, Y)

    Assumes x/y are time-ordered within each split.
    """
    if window <= 0:
        return x, y

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (T,N,F). Got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (T,Y). Got {y.shape}")

    T, N, F = x.shape
    if T <= window:
        raise ValueError(f"Not enough timesteps: T={T} <= window={window}")

    xs = []
    for t in range(window, T):
        # collect [t-window, ..., t] and concat features
        xcat = np.concatenate([x[t - i] for i in range(window, -1, -1)], axis=-1)
        xs.append(xcat)

    xw = np.stack(xs, axis=0)
    yw = y[window:]
    return xw, yw

def make_windows_sequence(x: np.ndarray, y: np.ndarray, window: int):
    """
    Create sequence windows preserving the time axis.

    Input:
      x: (T, N, F)
      y: (T, Y)
      window: int, number of past steps (0 = no windowing)

    Output:
      xw: (T-window, window+1, N, F)
      yw: (T-window, Y)

    For each t, uses [t-window, ..., t] as input and y[t] as target.
    """
    if window <= 0:
        return x, y

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (T,N,F). Got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (T,Y). Got {y.shape}")

    T, N, F = x.shape
    if T <= window:
        raise ValueError(f"Not enough timesteps: T={T} <= window={window}")

    xs = []
    for t in range(window, T):
        # sequence: [t-window, ..., t]
        xseq = x[t - window : t + 1]  # shape (window+1, N, F)
        xs.append(xseq)

    xw = np.stack(xs, axis=0)   # (T-window, window+1, N, F)
    yw = y[window:]             # (T-window, Y)

    return xw, yw