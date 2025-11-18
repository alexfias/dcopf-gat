# dcopf_gat/utils.py
import numpy as np
import tensorflow as tf
import random
import os


def set_global_seed(seed: int = 1234):
    """Set Python, NumPy, and TF seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def minmax_zero_max(data: np.ndarray, max_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Min-max normalization with lower bound=0 and max = column max or provided max_values.
    Returns: (normalized_data, max_values_used)
    """
    if max_values is None:
        max_values = np.max(data, axis=0)
    max_values = np.where(max_values == 0.0, 1.0, max_values)  # avoid div by zero
    norm = (data / max_values).astype(np.float32)
    norm = np.nan_to_num(norm, nan=0.0)
    return norm, max_values.astype(np.float32)


def maape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Mean Arctangent Absolute Percentage Error.
    Returns MAAPE per sample (axis=1).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + eps
    ratio = np.abs(y_pred - y_true) / denom
    return np.mean(np.arctan(ratio), axis=-1)
