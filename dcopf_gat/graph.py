# dcopf_gat/graph.py
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import eig


def build_adjacency(nodes: pd.Index, links: pd.DataFrame) -> np.ndarray:
    """
    Build an undirected adjacency matrix for the full node set from PyPSA-style links.
    nodes: index of all buses (strings).
    links: DataFrame with columns ['bus0', 'bus1'] (and maybe others).
    """
    num_nodes = len(nodes)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    nodes_list = list(nodes)

    for i in links.index:
        bus0 = nodes_list.index(links.loc[i, "bus0"])
        bus1 = nodes_list.index(links.loc[i, "bus1"])
        adjacency[bus0, bus1] = 1.0
        adjacency[bus1, bus0] = 1.0

    return adjacency


def power_incidence_matrices(nodes: pd.Index, links: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build withdraw and injection matrices (withd_m, injec_m) such that:
      line_flows @ withd_m  +  line_flows @ injec_m
    gives net inflow at each node.

    withd_m[i, j] = 1 if link i withdraws power at node j (bus0)
    injec_m[i, j] = -efficiency if link i injects power at node j (bus1)
    """
    num_links = links.shape[0]
    num_nodes = len(nodes)
    withd_m = np.zeros((num_links, num_nodes), dtype=np.float32)
    injec_m = np.zeros((num_links, num_nodes), dtype=np.float32)

    for idx, link in enumerate(links.index.values):
        b0 = links.loc[link, "bus0"]
        b1 = links.loc[link, "bus1"]
        eff = links.loc[link, "efficiency"] if "efficiency" in links.columns else 1.0

        withd_m[idx, np.where(nodes == b0)[0]] = 1.0
        injec_m[idx, np.where(nodes == b1)[0]] = -eff

    return withd_m, injec_m


def laplacian_pe(adjacency: np.ndarray, num_codes: int = 8) -> np.ndarray:
    """
    Compute Laplacian eigenvector-based positional encodings (node PE).
    adjacency: full adjacency matrix (no self-loops).
    num_codes: number of eigenvectors to keep (excluding trivial one).
    """
    num_nodes = adjacency.shape[0]
    degree = np.diag(adjacency.sum(axis=0))
    laplac = degree - adjacency

    # normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}
    degree_inv = np.diag(np.power(np.maximum(adjacency.sum(axis=0), 1e-12), -0.5))
    laplac_norm = degree_inv @ laplac @ degree_inv

    eigenval, eigenvec = eig(laplac_norm)
    idx = np.argsort(eigenval.real)
    eigenvec = eigenvec[:, idx].real

    # skip first (constant) eigenvector
    num_codes = min(num_codes, num_nodes - 1)
    return eigenvec[:, 1 : num_codes + 1].astype(np.float32)


def build_edge_list_for_multi_hop(
    adjacency: np.ndarray,
    nodes_orig_index: list[int],
    multi_window: int = 3,
    first_window: int = 1,
    steps_per_window: int = 2,
) -> list[np.ndarray]:
    """
    Build list of edge index pairs for multi-hop attention windows, following your notebook.
    adjacency: full adjacency (without self-loops)
    nodes_orig_index: indices of the reduced/original nodes subset
    """
    adjacency_full = adjacency.copy()
    num_nodes_full = adjacency_full.shape[0]

    # restrict to orig nodes
    adjacency_orig = adjacency_full[np.ix_(nodes_orig_index, nodes_orig_index)]
    num_nodes_orig = adjacency_orig.shape[0]

    # add self loops
    adjacency_orig = adjacency_orig + np.eye(num_nodes_orig, dtype=np.float32)

    edge_list: list[np.ndarray] = []

    first_adj = adjacency_orig.copy()
    for _ in range(first_window - 1):
        first_adj = adjacency_orig @ first_adj

    update = adjacency_orig.copy()
    for _ in range(steps_per_window - 1):
        update = adjacency_orig @ update

    new_adj = first_adj
    for _ in range(multi_window):
        new_edges: list[list[int]] = []
        for j in range(num_nodes_orig):
            nodes_con = set(np.where(new_adj[j] > 0)[0])
            for k in sorted(list(nodes_con)):
                new_edges.append([j, k])
        edge_list.append(np.array(new_edges, dtype=np.int32))
        new_adj = update @ new_adj

    return edge_list


def build_link_pe(node_pe_full: np.ndarray, nodes: pd.Index, links: pd.DataFrame) -> np.ndarray:
    """
    Build link positional encodings by concatenating PE of bus0 and bus1.
    node_pe_full: [num_nodes, d]
    """
    nodes_list = list(nodes)
    bus0_pe = []
    bus1_pe = []
    for i in links.index:
        b0 = nodes_list.index(links.loc[i, "bus0"])
        b1 = nodes_list.index(links.loc[i, "bus1"])
        bus0_pe.append(node_pe_full[b0])
        bus1_pe.append(node_pe_full[b1])
    bus0_pe = np.stack(bus0_pe, axis=0)
    bus1_pe = np.stack(bus1_pe, axis=0)
    return np.concatenate([bus0_pe, bus1_pe], axis=-1).astype(np.float32)
