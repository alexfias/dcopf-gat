from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypsa

@dataclass
class EraaGraphDataset:
    X_nodes_train: np.ndarray
    y_edges_train: np.ndarray
    X_nodes_val: np.ndarray
    y_edges_val: np.ndarray
    X_nodes_test: np.ndarray
    y_edges_test: np.ndarray
    edge_index: np.ndarray
    bus_names: list[str]
    edge_names: list[str]
    node_feature_names: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    metadata: dict

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

def _safe_std(x: np.ndarray, axis, keepdims: bool = True, eps: float = 1e-3):
    std_raw = x.std(axis=axis, keepdims=keepdims)
    return np.where(std_raw < eps, 1.0, std_raw)


def load_eraa_graph_dataset(
    data_dir: str | Path = "data_eraa_ml",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 1234,
    max_samples: int | None = None,
) -> EraaGraphDataset:
    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    metadata = json.loads(meta_path.read_text())
    samples = metadata["samples"]

    if max_samples is not None:
        samples = samples[:max_samples]

    # Load reference network for full PyPSA graph structure
    source_file = samples[0]["source_file"]
    network_path = data_dir / "solved_networks" / source_file
    n = pypsa.Network(network_path)

    bus_names = list(n.buses.index)
    bus_to_i = {b: i for i, b in enumerate(bus_names)}

    flow_cols = metadata["columns"]["flows"]
    edge_names = flow_cols

    edge_index = np.zeros((2, len(flow_cols)), dtype=np.int64)

    for j, c in enumerate(flow_cols):
        kind, name = c.split("::", 1)

        if kind == "Line":
            row = n.lines.loc[name]
        elif kind == "Link":
            row = n.links.loc[name]
        else:
            raise ValueError(f"Unknown flow column kind: {kind}")

        edge_index[0, j] = bus_to_i[row.bus0]
        edge_index[1, j] = bus_to_i[row.bus1]

    # Generator availability columns are stored as [bus, carrier]
    gen_cols = [tuple(c) for c in metadata["columns"]["gen_avail"]]
    carriers = sorted({carrier for _, carrier in gen_cols})

    node_feature_names = ["load"] + [f"gen_avail::{c}" for c in carriers]

    n_buses = len(bus_names)
    n_features = len(node_feature_names)

    X_list = []
    y_list = []

    for sample in samples:
        z = np.load(data_dir / sample["file"])

        load = z["load"]              # [168, n_load_buses]
        gen_avail = z["gen_avail"]    # [168, n_bus_carrier]
        flows = z["flows"]            # [168, n_edges]

        T = load.shape[0]
        X_nodes = np.zeros((T, n_buses, n_features), dtype=np.float32)

        # Load features
        load_cols = metadata["columns"]["load"]
        for k, bus in enumerate(load_cols):
            if bus in bus_to_i:
                X_nodes[:, bus_to_i[bus], 0] = load[:, k]

        # Generator availability by carrier
        carrier_to_feature = {
            carrier: i + 1 for i, carrier in enumerate(carriers)
        }

        for k, (bus, carrier) in enumerate(gen_cols):
            if bus in bus_to_i:
                X_nodes[:, bus_to_i[bus], carrier_to_feature[carrier]] += gen_avail[:, k]

        X_list.append(X_nodes.astype(np.float32))
        y_list.append(flows.astype(np.float32))

    X_all = np.stack(X_list, axis=0)  # [samples, 168, buses, features]
    y_all = np.stack(y_list, axis=0)  # [samples, 168, edges]

    if np.isnan(X_all).any():
        raise ValueError("NaNs found in graph X_all")
    if np.isnan(y_all).any():
        raise ValueError("NaNs found in graph y_all")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X_all))
    rng.shuffle(indices)

    X_all = X_all[indices]
    y_all = y_all[indices]

    n_samples = len(X_all)
    n_train = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]

    X_val = X_all[n_train : n_train + n_val]
    y_val = y_all[n_train : n_train + n_val]

    X_test = X_all[n_train + n_val :]
    y_test = y_all[n_train + n_val :]

    # Normalize node features and edge targets using train split only
    x_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    x_std = _safe_std(X_train, axis=(0, 1, 2), keepdims=True)

    y_mean = y_train.mean(axis=(0, 1), keepdims=True)
    y_std = _safe_std(y_train, axis=(0, 1), keepdims=True)

    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Flatten weeks into hourly graph samples for Architecture A
    X_nodes_train = X_train.reshape(-1, n_buses, n_features)
    y_edges_train = y_train.reshape(-1, len(flow_cols))

    X_nodes_val = X_val.reshape(-1, n_buses, n_features)
    y_edges_val = y_val.reshape(-1, len(flow_cols))

    X_nodes_test = X_test.reshape(-1, n_buses, n_features)
    y_edges_test = y_test.reshape(-1, len(flow_cols))

    return EraaGraphDataset(
        X_nodes_train=X_nodes_train.astype(np.float32),
        y_edges_train=y_edges_train.astype(np.float32),
        X_nodes_val=X_nodes_val.astype(np.float32),
        y_edges_val=y_edges_val.astype(np.float32),
        X_nodes_test=X_nodes_test.astype(np.float32),
        y_edges_test=y_edges_test.astype(np.float32),
        edge_index=edge_index,
        bus_names=bus_names,
        edge_names=edge_names,
        node_feature_names=node_feature_names,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        metadata=metadata,
    )