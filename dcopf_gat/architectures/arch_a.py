from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .base import Architecture
from .registry import register
from ..model import GraphAttentionNetwork
from ..data_pipeline import make_dataset

def build_gat(meta: Dict[str, Any], lamb: float) -> keras.Model:
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

@register("A")
class ArchA(Architecture):
    def build(self) -> Dict[str, keras.Model]:
        model = build_gat(self.meta, lamb=self.cfg.lamb)
        return {"main": model}

    def fit(self, models, train, val, run_dir) -> Dict[str, Any]:
        model = models["main"]
        (train_x, train_y), (val_x, val_y) = train, val

        _ = model(tf.convert_to_tensor(train_x[:1], dtype=tf.float32))
        model.compile(optimizer=keras.optimizers.Adam(self.cfg.learning_rate))

        train_ds = make_dataset(train_x, train_y, batch_size=self.cfg.batch_size, shuffle=True)
        val_ds   = make_dataset(val_x, val_y, batch_size=self.cfg.batch_size, shuffle=False)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_R2", mode="max", patience=50, restore_best_weights=True
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=run_dir / "model_best.weights.h5",
            monitor="val_R2", mode="max",
            save_best_only=True, save_weights_only=True,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.epochs,
            callbacks=[early_stop, checkpoint],
            verbose=2,
        )
        model.save_weights(run_dir / "model_final.weights.h5")
        return {"history_main": history.history}

    def evaluate(self, models, test) -> Dict[str, float]:
        model = models["main"]
        test_x, test_y = test
        out = model.evaluate(test_x, test_y, verbose=0, return_dict=True)
        return {k: float(v) for k, v in out.items()}
