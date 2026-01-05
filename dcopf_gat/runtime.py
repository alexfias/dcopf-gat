# dcopf_gat/runtime.py
from __future__ import annotations

import os
import tensorflow as tf


def configure_runtime(
    device: str = "auto",               # "auto" | "cpu" | "gpu"
    mixed_precision: bool = False,      # enable fp16 on GPU
    xla: bool = False,                  # jit compile
    memory_growth: bool = True,
):
    """
    Configure TF runtime for CPU/GPU with optional mixed precision & XLA.
    Call this ONCE at program start, before building the model.
    """

    device = device.lower()

    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")

    # Force CPU by hiding GPUs from TF
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    gpus = tf.config.list_physical_devices("GPU")

    if device == "gpu" and not gpus:
        raise RuntimeError("device='gpu' requested but no GPU is visible to TensorFlow")

    # Memory growth is generally a good default on workstations
    if memory_growth and gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # XLA
    if xla:
        tf.config.optimizer.set_jit(True)

    # Mixed precision: only makes sense on GPU
    if mixed_precision:
        if not gpus:
            print("mixed_precision requested but no GPU available; ignoring.")
        else:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy("mixed_float16")

    # Print a simple summary (helps debugging)
    print("TF version:", tf.__version__)
    print("Visible GPUs:", gpus)
    print("XLA:", xla, "| Mixed precision:", mixed_precision)
