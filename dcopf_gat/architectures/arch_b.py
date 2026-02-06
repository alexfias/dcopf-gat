from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from .base import Architecture
from .registry import register
from .arch_a import ArchA
from ..windowing import make_windows_concat

@register("B")
class ArchB(ArchA):
    """Same as A but with make_windows_concat preprocessing."""
    def prepare_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        w = self.cfg.window
        if w and w > 1:
            train_x, train_y = make_windows_concat(train_x, train_y, window=w)
            val_x, val_y     = make_windows_concat(val_x, val_y, window=w)
            test_x, test_y   = make_windows_concat(test_x, test_y, window=w)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)
