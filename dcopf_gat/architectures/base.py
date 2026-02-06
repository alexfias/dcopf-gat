from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

@dataclass
class TrainConfig:
    arch: str
    window: int = 0
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 200
    lamb: float = 1e-3
    use_tfdata: bool = True
    seed: int = 1234
    debug: bool = False

class Architecture:
    """Unified interface for architectures A–E."""
    name: str

    def __init__(self, cfg: TrainConfig, meta: Dict[str, Any]):
        self.cfg = cfg
        self.meta = meta

    def prepare_data(
        self,
        train_x: np.ndarray, train_y: np.ndarray,
        val_x: np.ndarray, val_y: np.ndarray,
        test_x: np.ndarray, test_y: np.ndarray,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Override if the architecture needs special preprocessing (windowing, SOC augmentation, target slicing)."""
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    def build(self) -> Dict[str, keras.Model]:
        """Return dict of models (one or multiple). Keys: 'main', 'soc', 'flow', etc."""
        raise NotImplementedError

    def fit(
        self,
        models: Dict[str, keras.Model],
        train: Tuple[np.ndarray, np.ndarray],
        val: Tuple[np.ndarray, np.ndarray],
        run_dir,
    ) -> Dict[str, Any]:
        """Train and return histories/metrics as dict."""
        raise NotImplementedError

    def evaluate(
        self,
        models: Dict[str, keras.Model],
        test: Tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, float]:
        """Return test metrics."""
        raise NotImplementedError
