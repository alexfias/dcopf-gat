from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EraaDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    metadata: dict


def _standardize_train_val_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    eps: float = 1e-3,
):
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std_raw = X_train.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std_raw < eps, 1.0, x_std_raw)

    y_mean = y_train.mean(axis=(0, 1), keepdims=True)
    y_std_raw = y_train.std(axis=(0, 1), keepdims=True)
    y_std = np.where(y_std_raw < eps, 1.0, y_std_raw)

    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return X_train, X_val, X_test, y_train, y_val, y_test, x_mean, x_std, y_mean, y_std


def load_eraa_dataset(
    data_dir: str | Path = "data_eraa_ml",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 1234,
    max_samples: int | None = None,
) -> EraaDataset:
    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    metadata = json.loads(meta_path.read_text())
    samples = metadata["samples"]

    if max_samples is not None:
        samples = samples[:max_samples]

    X_list = []
    y_list = []

    for sample in samples:
        z = np.load(data_dir / sample["file"])

        load = z["load"]              # [168, n_buses]
        gen_avail = z["gen_avail"]    # [168, n_bus_carrier]
        flows = z["flows"]            # [168, n_flows]

        X = np.concatenate([load, gen_avail], axis=-1)

        X_list.append(X.astype(np.float32))
        y_list.append(flows.astype(np.float32))

    X_all = np.stack(X_list, axis=0)
    y_all = np.stack(y_list, axis=0)

    if np.isnan(X_all).any():
        raise ValueError("NaNs found in X_all")
    if np.isnan(y_all).any():
        raise ValueError("NaNs found in y_all")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X_all))
    rng.shuffle(indices)

    X_all = X_all[indices]
    y_all = y_all[indices]

    n = len(X_all)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]

    X_val = X_all[n_train : n_train + n_val]
    y_val = y_all[n_train : n_train + n_val]

    X_test = X_all[n_train + n_val :]
    y_test = y_all[n_train + n_val :]

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = _standardize_train_val_test(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    return EraaDataset(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        metadata=metadata,
    )