# dcopf_gat/architectures/common.py
from __future__ import annotations
from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras

from ..model import GraphAttentionNetwork


def build_gat_from_meta(meta: Dict[str, Any], lamb: float) -> keras.Model:
    """Build a GraphAttentionNetwork from dataset meta information."""
    return GraphAttentionNetwork(
        num_nodes_orig=meta["num_nodes_orig"],
        num_links=meta["num_links"],
        node_pe_orig=meta["node_pe_orig"],
        link_pe=meta["link_pe"],
        link_edges=meta["link_edges"],
        edge_list=meta["edge_list"],
        g_max=meta["p_nom_bus"],
        d_max=meta["demand_max"],
        f_max=meta["flow_max"],
        withd_m=meta["withd_m"],
        injec_m=meta["injec_m"],
        pca_obj=meta["pca"],
        lamb=lamb,
        output_weight=meta["output_weight"],
        hidden_units=64,
        num_heads=3,
        num_layers=3,
    )


def build_and_compile(
    model: keras.Model,
    sample_x,
    learning_rate: float,
) -> keras.Model:
    """Build subclassed model and compile with Adam."""
    _ = model(tf.convert_to_tensor(sample_x[:1], dtype=tf.float32))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model
