from __future__ import annotations

from typing import Dict
from tensorflow import keras

from .arch_c import ArchC
from .registry import register
from .arch_e_model import build_gru_gat
from ..windowing import make_windows_sequence


@register("E")
class ArchE(ArchC):
    """
    Architecture E — Sequence-aware temporal encoder (GRU) + GAT

    Compared with D:
      - D uses concatenated windows: [B, N, W*F]
      - E keeps the temporal axis:  [B, W, N, F]

    The GRU encodes each node's feature history over the window into a
    single node embedding, which is then passed through the same general
    graph-attention / link-decoding idea as in the snapshot model.
    """

    def prepare_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        w = self.cfg.window

        if w and w > 1:
            train_x, train_y = make_windows_sequence(train_x, train_y, window=w)
            val_x, val_y = make_windows_sequence(val_x, val_y, window=w)
            test_x, test_y = make_windows_sequence(test_x, test_y, window=w)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    def build(self) -> Dict[str, keras.Model]:
        model = build_gru_gat(self.meta, lamb=self.cfg.lamb)
        return {"main": model}