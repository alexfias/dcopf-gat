from __future__ import annotations

from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras

from .base import Architecture
from .registry import register
from ..data_pipeline import make_dataset
from .arch_f_model import build_temporal_block_gat
from ..windowing import make_windows_sequence


@register("F")
class ArchF(Architecture):
    """
    Architecture F — Strong graph-temporal block

    Relative to Architecture A, this is the first model in the ladder that
    changes the backbone itself rather than only changing the input features.

    A is a snapshot model:
      - input: one timestep [N, F]
      - graph attention is applied only on the current network state
      - decoding uses node embeddings from that single snapshot

    F is a graph-temporal extension of A:
      - input: a short sequence [W, N, F]
      - each block first mixes information over time for each node via a
        causal temporal convolution
      - it then mixes information over the graph at each timestep via graph
        attention
      - the final timestep is decoded with the same link-query decoder used
        by A

    In short, A models spatial structure at one time, while F models recent
    temporal evolution together with graph structure. This is mainly intended
    for cases where flows depend on recent system history, such as storage-
    influenced operation.
    """

    def prepare_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        w = self.cfg.window
        if w and w > 1:
            train_x, train_y = make_windows_sequence(train_x, train_y, window=w)
            val_x, val_y = make_windows_sequence(val_x, val_y, window=w)
            test_x, test_y = make_windows_sequence(test_x, test_y, window=w)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    def build(self) -> Dict[str, keras.Model]:
        model = build_temporal_block_gat(self.meta, lamb=self.cfg.lamb)
        return {"main": model}

    def fit(self, models, train, val, run_dir) -> Dict[str, Any]:
        model = models["main"]
        (train_x, train_y), (val_x, val_y) = train, val
        batch_size = min(self.cfg.batch_size, 16)

        _ = model(tf.convert_to_tensor(train_x[:1], dtype=tf.float32))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=min(self.cfg.learning_rate, 3e-4),
                clipnorm=1.0,
            )
        )

        train_ds = make_dataset(train_x, train_y, batch_size=batch_size, shuffle=True)
        val_ds = make_dataset(val_x, val_y, batch_size=batch_size, shuffle=False)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=30, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=run_dir / "model_best.weights.h5",
            monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.epochs,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=2,
        )
        model.save_weights(run_dir / "model_final.weights.h5")
        return {"history_main": history.history}

    def evaluate(self, models, test) -> Dict[str, float]:
        model = models["main"]
        test_x, test_y = test
        out = model.evaluate(test_x, test_y, verbose=0, return_dict=True)
        return {k: float(v) for k, v in out.items()}
