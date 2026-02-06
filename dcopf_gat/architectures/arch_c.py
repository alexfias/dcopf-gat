from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .base import Architecture
from .registry import register
from .arch_a import build_gat
from ..data_pipeline import make_dataset

def add_soc_as_node_feature(x: np.ndarray, soc: np.ndarray) -> np.ndarray:
    # Broadcast SOC to all nodes (fast first version)
    N, B, F = x.shape
    soc_b = np.repeat(soc[:, None, :], B, axis=1)
    return np.concatenate([x, soc_b], axis=-1)

@register("C")
class ArchC(Architecture):
    def __init__(self, cfg, meta):
        super().__init__(cfg, meta)
        # TODO: define these properly once we confirm target layout:
        self.flow_slice = slice(0, 20)
        self.soc_slice  = slice(20, 34)

    def prepare_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        # no windowing by default; can combine with B later if desired
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    def build(self) -> Dict[str, keras.Model]:
        soc_model = build_gat(self.meta, lamb=self.cfg.lamb)
        flow_model = build_gat(self.meta, lamb=self.cfg.lamb)
        return {"soc": soc_model, "flow": flow_model}

    def fit(self, models, train, val, run_dir) -> Dict[str, Any]:
        (train_x, train_y), (val_x, val_y) = train, val
        y_soc_tr  = train_y[:, self.soc_slice]
        y_flow_tr = train_y[:, self.flow_slice]
        y_soc_va  = val_y[:, self.soc_slice]
        y_flow_va = val_y[:, self.flow_slice]

        # ---- train SOC model ----
        soc_model = models["soc"]
        _ = soc_model(tf.convert_to_tensor(train_x[:1], dtype=tf.float32))
        soc_model.compile(optimizer=keras.optimizers.Adam(self.cfg.learning_rate))

        tr_soc_ds = make_dataset(train_x, y_soc_tr, batch_size=self.cfg.batch_size, shuffle=True)
        va_soc_ds = make_dataset(val_x,   y_soc_va, batch_size=self.cfg.batch_size, shuffle=False)

        soc_ckpt = keras.callbacks.ModelCheckpoint(
            filepath=run_dir / "soc_best.weights.h5",
            monitor="val_R2", mode="max",
            save_best_only=True, save_weights_only=True,
        )

        soc_hist = soc_model.fit(
            tr_soc_ds, validation_data=va_soc_ds,
            epochs=self.cfg.epochs, callbacks=[soc_ckpt],
            verbose=2
        )
        soc_model.save_weights(run_dir / "soc_final.weights.h5")

        # ---- teacher-forced FLOW model (augment with TRUE SOC) ----
        train_x_aug = add_soc_as_node_feature(train_x, y_soc_tr)
        val_x_aug   = add_soc_as_node_feature(val_x,   y_soc_va)

        flow_model = models["flow"]
        _ = flow_model(tf.convert_to_tensor(train_x_aug[:1], dtype=tf.float32))
        flow_model.compile(optimizer=keras.optimizers.Adam(self.cfg.learning_rate))

        tr_flow_ds = make_dataset(train_x_aug, y_flow_tr, batch_size=self.cfg.batch_size, shuffle=True)
        va_flow_ds = make_dataset(val_x_aug,   y_flow_va, batch_size=self.cfg.batch_size, shuffle=False)

        flow_ckpt = keras.callbacks.ModelCheckpoint(
            filepath=run_dir / "flow_best.weights.h5",
            monitor="val_R2", mode="max",
            save_best_only=True, save_weights_only=True,
        )

        flow_hist = flow_model.fit(
            tr_flow_ds, validation_data=va_flow_ds,
            epochs=self.cfg.epochs, callbacks=[flow_ckpt],
            verbose=2
        )
        flow_model.save_weights(run_dir / "flow_final.weights.h5")

        return {"history_soc": soc_hist.history, "history_flow": flow_hist.history}

    def evaluate(self, models, test) -> Dict[str, float]:
        test_x, test_y = test
        y_soc = test_y[:, self.soc_slice]
        y_flow = test_y[:, self.flow_slice]

        soc_model = models["soc"]
        flow_model = models["flow"]

        # SOC metrics
        soc_metrics = soc_model.evaluate(test_x, y_soc, verbose=0, return_dict=True)

        # Flow metrics (inference uses predicted SOC)
        soc_pred = soc_model.predict(test_x, batch_size=self.cfg.batch_size, verbose=0)
        test_x_aug = add_soc_as_node_feature(test_x, soc_pred)
        flow_metrics = flow_model.evaluate(test_x_aug, y_flow, verbose=0, return_dict=True)

        out = {}
        out.update({f"soc_{k}": float(v) for k, v in soc_metrics.items()})
        out.update({f"flow_{k}": float(v) for k, v in flow_metrics.items()})
        return out
