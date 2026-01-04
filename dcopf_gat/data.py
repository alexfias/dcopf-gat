# dcopf_gat/data.py
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .utils import minmax_zero_max
from .graph import (
    build_adjacency,
    laplacian_pe,
    power_incidence_matrices,
    build_edge_list_for_multi_hop,
    build_link_pe,
)


def load_raw_data(data_dir: str | pathlib.Path):
    """
    Load the CSVs in the same way as the notebook.

    Expected filenames (adapt if different):
      - generators_t_p.csv
      - p_max_pu.csv
      - loads-p_set.csv
      - linkf.csv
      - links.csv
      - stores_t_e.csv
      - generators.csv
      - buses.csv
      - nodes_orig.csv
    """
    data_dir = pathlib.Path(data_dir)

    p_t = pd.read_csv(data_dir / "generators_t_p.csv", index_col=0)
    weather = pd.read_csv(data_dir / "p_max_pu.csv", index_col=0)
    demand = pd.read_csv(data_dir / "loads-p_set.csv", index_col=0)
    flow = pd.read_csv(data_dir / "linkf.csv", index_col=0)
    links = pd.read_csv(data_dir / "links.csv", index_col=0)
    soc = pd.read_csv(data_dir / "stores_t_e.csv", index_col=0)

    gen_setting = pd.read_csv(data_dir / "generators.csv", index_col=0)
    buses = pd.read_csv(data_dir / "buses.csv", index_col=0)
    nodes_orig_df = pd.read_csv(data_dir / "nodes_orig.csv")
    if "name" in nodes_orig_df.columns:
        nodes_orig = nodes_orig_df["name"]
    else:
        # fallback: first column
        nodes_orig = nodes_orig_df.iloc[:, 0]

    return {
        "p_t": p_t,
        "weather": weather,
        "demand": demand,
        "flow": flow,
        "links": links,
        "soc": soc,
        "gen_setting": gen_setting,
        "buses": buses,
        "nodes_orig": nodes_orig,
    }


def prepare_dataset(
    data_dir: str | pathlib.Path,
    pca_flag: bool = False,
    pca_variance: float = 0.95,
    multi_window: int = 3,
    first_window: int = 1,
    steps_per_window: int = 2,
    train_fraction: float = 0.95,
    seed: int = 1234,
):
    """
    High-level helper that:
      - loads data
      - builds adjacency, PE, incidence matrices
      - normalizes inputs/outputs
      - does a random train/val/test split
    Returns:
      train_x, train_y, val_x, val_y, test_x, test_y, meta
    """
    rng = np.random.default_rng(seed)

    raw = load_raw_data(data_dir)
    p_t = raw["p_t"]
    weather = raw["weather"]
    demand = raw["demand"]
    flow = raw["flow"]
    links = raw["links"]
    gen_setting = raw["gen_setting"]
    buses = raw["buses"]
    nodes_orig = raw["nodes_orig"].values  # array of node names

    # Basic dimensions
    nodes = buses.index.values
    num_nodes = len(nodes)
    num_nodes_orig = len(nodes_orig)
    num_links = links.shape[0]

    # Filter tiny generators as in notebook
    power_precision = 1e-3
    gen_setting_reduced = gen_setting.loc[gen_setting["p_nom"] > power_precision]
    p_t_reduced = p_t.loc[:, gen_setting_reduced.index]
    # Mask tiny dispatch values
    p_t_reduced = (p_t_reduced > power_precision).astype(int) * p_t_reduced

    # Group generators by bus
    gen_setting_reduced = gen_setting_reduced.copy()
    if "bus" not in gen_setting_reduced.columns:
        raise ValueError("generators.csv must contain a 'bus' column.")

    # reorder generators according to nodes_orig and sum per bus
    gen_bus = p_t_reduced.T
    gen_bus["bus"] = gen_setting_reduced["bus"]
    gen_bus = gen_bus.groupby("bus").sum().T
    gen_bus = gen_bus.reindex(columns=nodes_orig, fill_value=0.0)

    # max installed capacity per bus for scaling
    p_nom_reorder = gen_setting_reduced[["p_nom", "bus"]].groupby("bus").sum()
    p_nom_reorder = p_nom_reorder.reindex(index=nodes_orig, fill_value=0.0)
    p_nom_bus = p_nom_reorder["p_nom"].values.astype(np.float32)

    # reorder demand by nodes_orig
    # assume demand columns are f"{bus} total_demand" as in notebook
    demand_cols = [f"{bus} total_demand" for bus in nodes_orig]
    demand_reorder = demand[demand_cols]

    # build normalization
    gen_bus_norm, _ = minmax_zero_max(gen_bus.values, p_nom_bus)
    demand_norm, demand_max = minmax_zero_max(demand_reorder.values)
    flow_max = links["p_nom"].values.astype(np.float32)
    flow_norm, _ = minmax_zero_max(flow.values, flow_max)

    # weather: assume already in [0,1], or you can also normalize here
    weather_reorder = weather.copy()
    weather_reorder = weather_reorder.reindex(index=p_t.index)  # align with time
    weather_input = weather_reorder.values.astype(np.float32)

    # adjacency + graph structures
    adjacency = build_adjacency(buses.index, links)
    nodes_list = list(buses.index)
    nodes_orig_index = [nodes_list.index(n) for n in nodes_orig]

    node_pe_full = laplacian_pe(adjacency, num_codes=8 if num_nodes <= 40 else 16)
    node_pe_orig = node_pe_full[nodes_orig_index]

    withd_m, injec_m = power_incidence_matrices(nodes_orig, links)

    edge_list = build_edge_list_for_multi_hop(
        adjacency,
        nodes_orig_index,
        multi_window=multi_window,
        first_window=first_window,
        steps_per_window=steps_per_window,
    )

    link_pe = build_link_pe(node_pe_full, buses.index, links)

    # ------------------------------------------------------------------
    # NEW: build link_edges in nodes_orig indexing for NLAT-like decoder
    # link_edges[l] = (from_bus_idx, to_bus_idx) where indices refer to nodes_orig order.
    # Ordering must match flow targets, which match "links" row ordering.
    # ------------------------------------------------------------------
    # detect column names (most common are bus0/bus1)
    if "bus0" in links.columns and "bus1" in links.columns:
        col0, col1 = "bus0", "bus1"
    elif "from_bus" in links.columns and "to_bus" in links.columns:
        col0, col1 = "from_bus", "to_bus"
    elif "from" in links.columns and "to" in links.columns:
        col0, col1 = "from", "to"
    else:
        raise ValueError(
            f"links.csv must contain endpoint columns (e.g. bus0/bus1). "
            f"Found columns: {list(links.columns)}"
        )

    bus_to_orig = {bus: i for i, bus in enumerate(nodes_orig)}
    bus0_idx = links[col0].map(bus_to_orig)
    bus1_idx = links[col1].map(bus_to_orig)

    if bus0_idx.isna().any() or bus1_idx.isna().any():
        bad_rows = links[bus0_idx.isna() | bus1_idx.isna()].head(10)
        raise ValueError(
            "Some link endpoints are not contained in nodes_orig. "
            "Check nodes_orig.csv vs links.csv endpoints.\n"
            f"Example problematic rows:\n{bad_rows}"
        )

    link_edges = np.stack([bus0_idx.values, bus1_idx.values], axis=1).astype(np.int32)

    assert link_edges.shape == (num_links, 2)
    assert link_edges.min() >= 0
    assert link_edges.max() < num_nodes_orig
    # ------------------------------------------------------------------

    # PCA on flow if requested
    pca = None
    flow_target = flow_norm
    output_weight = None
    if pca_flag:
        pca = PCA()
        pca.fit(flow_norm)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        pc_num = np.where(cum_var >= pca_variance)[0][0]
        pca = PCA(pc_num)
        flow_pca = pca.fit_transform(flow_norm)
        var_pc = np.var(flow_pca, axis=0)
        weight_pc = 1.0 / var_pc
        weight_pc /= np.sum(weight_pc)
        flow_target = flow_pca.astype(np.float32)
        output_weight = weight_pc.astype(np.float32)

    # build input and output tensors
    # X: [T, num_nodes_orig, features=(weather + demand)]
    # here we broadcast weather features per node; notebook did something similar.
    T = gen_bus_norm.shape[0]
    num_weather_feat = weather_input.shape[1]
    # repeat same weather features for all nodes_orig
    weather_nodes = np.repeat(weather_input[:, None, :], num_nodes_orig, axis=1)
    demand_nodes = demand_norm[:, :, None]  # [T, num_nodes_orig, 1]
    X = np.concatenate([weather_nodes, demand_nodes], axis=-1).astype(np.float32)

    # Y: [T, num_nodes_orig + num_links]
    Y_gen = gen_bus_norm.astype(np.float32)
    Y = np.concatenate([Y_gen, flow_target.astype(np.float32)], axis=-1)

    # split train / test
    idx = np.arange(T)
    rng.shuffle(idx)
    train_size = int(T * train_fraction)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    # make validation as a slice of train
    val_fraction = 0.05
    val_size = max(1, int(train_size * val_fraction))
    val_idx = train_idx[:val_size]
    train_idx = train_idx[val_size:]

    train_x, train_y = X[train_idx], Y[train_idx]
    val_x, val_y = X[val_idx], Y[val_idx]
    test_x, test_y = X[test_idx], Y[test_idx]

    meta = {
        "nodes": nodes,
        "nodes_orig": nodes_orig,
        "num_nodes": num_nodes,
        "num_nodes_orig": num_nodes_orig,
        "num_links": num_links,
        "withd_m": withd_m.astype(np.float32),
        "injec_m": injec_m.astype(np.float32),
        "flow_max": flow_max,
        "p_nom_bus": p_nom_bus,
        "demand_max": demand_max.astype(np.float32),
        "adjacency": adjacency.astype(np.float32),
        "edge_list": edge_list,
        "node_pe_orig": node_pe_orig,
        "link_pe": link_pe,
        "link_edges": link_edges,  # <-- NEW
        "pca": pca,
        "output_weight": output_weight,
    }

    return train_x, train_y, val_x, val_y, test_x, test_y, meta
