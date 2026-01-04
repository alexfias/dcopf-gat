# dcopf_gat/train.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras

from .data import prepare_dataset
from .model import GraphAttentionNetwork
from .utils import set_global_seed


def build_model_from_meta(meta: Dict[str, Any], lamb: float = 0.001) -> keras.Model:
    num_nodes_orig = meta["num_nodes_orig"]
    num_links = meta["num_links"]
    node_pe_orig = meta["node_pe_orig"]
    link_pe = meta["link_pe"]
    edge_list = meta["edge_list"]
    p_nom_bus = meta["p_nom_bus"]
    demand_max = meta["demand_max"]
    flow_max = meta["flow_max"]
    withd_m = meta["withd_m"]
    injec_m = meta["injec_m"]
    pca = meta["pca"]
    output_weight = meta["output_weight"]

    model = GraphAttentionNetwork(
        num_nodes_orig=num_nodes_orig,
        num_links=num_links,
        node_pe_orig=node_pe_orig,
        link_pe=link_pe,
        edge_list=edge_list,
        g_max=p_nom_bus,
        d_max=demand_max,
        f_max=flow_max,
        withd_m=withd_m,
        injec_m=injec_m,
        pca_obj=pca,
        lamb=lamb,
        output_weight=output_weight,
        hidden_units=64,
        num_heads=3,
        num_layers=3,
    )
    return model


def run_experiment(
    data_dir: str | Path,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 200,
    seed: int = 1234,
):
    set_global_seed(seed)

    train_x, train_y, val_x, val_y, test_x, test_y, meta = prepare_dataset(
        data_dir, pca_flag=False, train_fraction=0.8, seed=seed
    )

    model = build_model_from_meta(meta, lamb=0.001)

    # Build model by calling once
    _ = model(tf.convert_to_tensor(train_x[:1], dtype=tf.float32))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True,
    )

    history = model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_x, val_y),
        callbacks=[early_stop],
        verbose=2,
    )

    test_metrics = model.evaluate(test_x, test_y, verbose=0)
    return model, history, (test_x, test_y), test_metrics
