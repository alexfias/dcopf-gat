# dcopf_gat/utils_results.py
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def dump_run(
    run_dir,
    rmse_model,
    rmse_baseline,
    rmse_links,
    rmse_base_links,
    config: dict,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # scalar metrics
    metrics = {
        "rmse_model": float(rmse_model),
        "rmse_baseline": float(rmse_baseline),
        "rmse_ratio": float(rmse_model / rmse_baseline),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # per-link breakdown
    df = pd.DataFrame(
        {
            "rmse_model": rmse_links,
            "rmse_baseline": rmse_base_links,
            "ratio": rmse_links / rmse_base_links,
        }
    )
    df.to_csv(run_dir / "per_link_rmse.csv", index_label="link_id")

    # config / hyperparameters
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    print(f"Saved results to {run_dir.resolve()}")
