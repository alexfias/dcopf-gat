# dcopf_gat/train.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np

from .data import prepare_dataset
from .utils import set_global_seed

from .architectures.base import TrainConfig
from .architectures.registry import ARCH_REGISTRY

# side-effect imports to register architectures A–E
from .architectures import arch_a  # noqa: F401
from .architectures import arch_b  # noqa: F401
from .architectures import arch_c  # noqa: F401
# from .architectures import arch_d  # noqa: F401
# from .architectures import arch_e  # noqa: F401


def run_experiment(
    data_dir: str | Path,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 200,
    seed: int = 1234,
    window: int = 0,
    use_tfdata: bool = True,
    arch_name: str = "A",        # <-- architecture key: "A", "B", "C", ...
    lamb: float = 0.001,
    debug: bool = False,
) -> Tuple[dict, dict, Tuple, Dict[str, float]]:
    """
    Orchestrator only:
      - loads dataset
      - instantiates chosen architecture
      - trains/evaluates
      - saves artifacts
    Returns:
      models_dict, histories_dict, (test_x, test_y), test_metrics_dict
    """
    set_global_seed(seed)

    run_dir = Path("runs") / Path(data_dir).name / arch_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset once ----
    # A/B: no SoC in inputs; C/D/E: SoC included
    include_soc_feature = arch_name in {"C", "D", "E"}  # extend if needed

    train_x, train_y, val_x, val_y, test_x, test_y, meta = prepare_dataset(
        data_dir,
        pca_flag=False,
        train_fraction=0.8,
        seed=seed,
        include_soc_feature=include_soc_feature,
    )

    # ---- Build unified config ----
    cfg = TrainConfig(
        arch=arch_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        window=window,
        use_tfdata=use_tfdata,
        lamb=lamb,
        debug=debug,
    )

    # Save config early for reproducibility
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # ---- Choose architecture ----
    if cfg.arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{cfg.arch}'. Available: {sorted(ARCH_REGISTRY)}")

    arch = ARCH_REGISTRY[cfg.arch](cfg, meta)

    # ---- Architecture-specific preprocessing ----
    train, val, test = arch.prepare_data(train_x, train_y, val_x, val_y, test_x, test_y)

    if cfg.debug:
        (tx, ty), (vx, vy), (sx, sy) = train, val, test
        print("[DEBUG] train_x:", tx.shape, "train_y:", ty.shape)
        print("[DEBUG] val_x:", vx.shape, "val_y:", vy.shape)
        print("[DEBUG] test_x:", sx.shape, "test_y:", sy.shape)

    # ---- Build / train / evaluate ----
    models = arch.build()
    histories = arch.fit(models=models, train=train, val=val, run_dir=run_dir)
    test_metrics = arch.evaluate(models=models, test=test)

    # ---- Save artifacts ----
    np.save(run_dir / "history.npy", histories)
    with open(run_dir / "metrics_test.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    return models, histories, test, test_metrics
