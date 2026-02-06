# dcopf_gat/train.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import tensorflow as tf
from tensorflow import keras

from .data import prepare_dataset
from .model import GraphAttentionNetwork
from .utils import set_global_seed
from .data_pipeline import make_dataset

import json
import numpy as np
import time

from .windowing import make_windows_concat



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
        link_edges=meta["link_edges"],
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
    window: int = 0,
    use_tfdata: bool = True,
    arch_name: str = "gat_flow_lqat",
    lamb: float = 0.001,
    debug: bool = False,
) -> Tuple[keras.Model, keras.callbacks.History, Tuple, Dict[str, float]]:
    set_global_seed(seed)

    run_dir = Path("runs") / Path(data_dir).name / arch_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, val_x, val_y, test_x, test_y, meta = prepare_dataset(
        data_dir, pca_flag=False, train_fraction=0.8, seed=seed
    )

    if window and window > 1:
        train_x, train_y = make_windows_concat(train_x, train_y, window=window)
        val_x, val_y = make_windows_concat(val_x, val_y, window=window)
        test_x, test_y = make_windows_concat(test_x, test_y, window=window)
    else:
        window == 0 #or 1 => no windowing
        pass

    if debug:
        print("after windowing train_x:", train_x.shape, "val_x:", val_x.shape)

        if use_tfdata:
            xb, yb = next(iter(make_dataset(train_x, train_y, batch_size=32, shuffle=True)))
            print("dataset batch xb:", xb.shape, xb.dtype)
            print("dataset batch yb:", yb.shape, yb.dtype)

    model = build_model_from_meta(meta, lamb=lamb)

    config = {
        "architecture": arch_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "lambda": lamb,
        "use_tfdata": use_tfdata,
        "window": window,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Build model by calling once (subclassed model)
    _ = model(tf.convert_to_tensor(train_x[:1], dtype=tf.float32))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )

    w0 = [w.numpy().copy() for w in model.trainable_weights]

    print("\n[TEST 2]")
    print("Num trainable tensors:", len(w0))
    print("Learning rate:",
          float(tf.keras.backend.get_value(model.optimizer.learning_rate)))
    print("Optimizer iterations (before):",
          int(model.optimizer.iterations.numpy()))

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_R2",
        mode="max",
        patience=50,
        restore_best_weights=True,
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=run_dir / "model_best.weights.h5",
        monitor="val_R2",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    if use_tfdata:
        train_ds = make_dataset(train_x, train_y, batch_size=batch_size, shuffle=True)
        val_ds = make_dataset(val_x, val_y, batch_size=batch_size, shuffle=False)

        if debug:
            history = model.fit(
                train_ds,
                epochs=1,
                steps_per_epoch=5,
                verbose=2,
                callbacks=[],  # important
            )

        else: history = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=[early_stop, checkpoint],
                verbose=2,
            )

    else:
        history = model.fit(
            x=train_x,
            y=train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_x, val_y),
            callbacks=[early_stop, checkpoint],
            verbose=2,
        )

    w1 = [w.numpy().copy() for w in model.trainable_weights]

    deltas = [
        float(np.mean((a - b) ** 2))
        for a, b in zip(w1, w0)
    ]

    print("Mean squared delta per tensor (first 10):", deltas[:10])
    print("Any weight changed?:", any(d > 0 for d in deltas))
    print("Optimizer iterations (after):",
          int(model.optimizer.iterations.numpy()))
    print("==========================================\n")


    np.save(run_dir / "history.npy", history.history)

    # Return dict for easier downstream logging
    test_metrics = model.evaluate(test_x, test_y, verbose=0, return_dict=True)

    with open(run_dir / "metrics_test.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    model.save_weights(run_dir / "model_final.weights.h5")

    return model, history, (test_x, test_y), test_metrics
