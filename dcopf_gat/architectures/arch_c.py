from __future__ import annotations

from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras

from .base import Architecture
from .registry import register
from .arch_a import build_gat
from ..data_pipeline import make_dataset


@register("C")
class ArchC(Architecture):
    """
    Architecture C — State-conditioned snapshot (explicit SoC)

    With the clean ladder, SoC inclusion is controlled in prepare_dataset(...):
      - A/B: X = [weather..., demand]
      - C/D/E: X = [weather..., soc, demand]   (demand remains last)

    Therefore ArchC does not need to modify x/y at all.
    """

    def build(self) -> Dict[str, keras.Model]:
        model = build_gat(self.meta, lamb=self.cfg.lamb)
        return {"main": model}

    def fit(self, models, train, val, run_dir) -> Dict[str, Any]:
        model = models["main"]
        (train_x, train_y), (val_x, val_y) = train, val

        _ = model(tf.convert_to_tensor(train_x[:1], dtype=tf.float32))
        model.compile(optimizer=keras.optimizers.Adam(self.cfg.learning_rate))

        train_ds = make_dataset(train_x, train_y, batch_size=self.cfg.batch_size, shuffle=True)
        val_ds   = make_dataset(val_x,   val_y,   batch_size=self.cfg.batch_size, shuffle=False)

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
