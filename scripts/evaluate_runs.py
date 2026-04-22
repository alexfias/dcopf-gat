from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from dcopf_gat.architectures.base import TrainConfig
from dcopf_gat.architectures.registry import ARCH_REGISTRY
from dcopf_gat.data import prepare_dataset
from dcopf_gat.utils import maape
from dcopf_gat.utils_results import dump_run

# side-effect imports to register architectures
from dcopf_gat.architectures import arch_a  # noqa: F401
from dcopf_gat.architectures import arch_b  # noqa: F401
from dcopf_gat.architectures import arch_c  # noqa: F401
from dcopf_gat.architectures import arch_d  # noqa: F401
from dcopf_gat.architectures import arch_e  # noqa: F401
from dcopf_gat.architectures import arch_f  # noqa: F401


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return R2 per output dimension."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved dcopf_gat run directory")
    p.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Run directory created by run_experiment, e.g. runs/data_toy_3bus/A",
    )
    p.add_argument(
        "--run_root",
        type=Path,
        default=None,
        help="Directory containing multiple run dirs to compare, e.g. runs/data_toy_3bus",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Dataset directory. Defaults to the run_dir parent name under repo root.",
    )
    p.add_argument(
        "--weights",
        choices=["best", "final"],
        default="best",
        help="Which checkpoint weights to evaluate",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Save per-dimension scatter plots",
    )
    p.add_argument(
        "--compare_plot",
        action="store_true",
        help="Save a flattened true-vs-predicted scatter plot for all selected runs",
    )
    return p.parse_args()


def resolve_run_dirs(run_dir: Path | None, run_root: Path | None) -> list[Path]:
    if run_dir is not None and run_root is not None:
        raise ValueError("Pass either --run_dir or --run_root, not both.")
    if run_dir is not None:
        return [run_dir.resolve()]
    if run_root is not None:
        run_root = run_root.resolve()
        run_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()])
        if not run_dirs:
            raise ValueError(f"No run directories found under {run_root}")
        return run_dirs
    raise ValueError("Pass --run_dir or --run_root.")


def infer_data_dir(run_dir: Path, explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir
    return Path(run_dir.parent.name)


def split_mode_from_cfg(cfg: TrainConfig) -> str:
    return "temporal" if (cfg.arch in {"B", "D", "E", "F"} and cfg.window > 0) else "random"


def load_cfg(run_dir: Path) -> TrainConfig:
    with open(run_dir / "config.json") as f:
        raw = json.load(f)
    return TrainConfig(**raw)


def prepare_from_run(run_dir: Path, data_dir: Path):
    cfg = load_cfg(run_dir)
    include_soc_feature = cfg.arch in {"C", "D", "E", "F"}
    split_mode = split_mode_from_cfg(cfg)

    train_x, train_y, val_x, val_y, test_x, test_y, meta = prepare_dataset(
        data_dir,
        pca_flag=False,
        train_fraction=0.8,
        seed=cfg.seed,
        include_soc_feature=include_soc_feature,
        split_mode=split_mode,
    )

    if cfg.arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{cfg.arch}'. Available: {sorted(ARCH_REGISTRY)}")

    arch = ARCH_REGISTRY[cfg.arch](cfg, meta)
    train, val, test = arch.prepare_data(train_x, train_y, val_x, val_y, test_x, test_y)
    return cfg, arch, train, val, test, meta


def load_models(arch, train, run_dir: Path, weights_name: str):
    models = arch.build()
    sample_x = train[0]
    weights_path = run_dir / f"model_{weights_name}.weights.h5"

    for model in models.values():
        _ = model(tf.convert_to_tensor(sample_x[:1], dtype=tf.float32))
        model.load_weights(weights_path)
    return models


def save_scatter_plots(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    for d in range(y_true.shape[1]):
        plt.figure()
        plt.scatter(y_true[:, d], y_pred[:, d], s=10, alpha=0.7)
        mn = min(y_true[:, d].min(), y_pred[:, d].min())
        mx = max(y_true[:, d].max(), y_pred[:, d].max())
        plt.plot([mn, mx], [mn, mx], linewidth=1.0)
        plt.xlabel("true")
        plt.ylabel("pred")
        plt.title(f"Pred vs True (flow dim {d})")
        plt.savefig(run_dir / f"pred_vs_true_flow_dim{d:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()


def save_flat_scatter(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    lims = [
        min(y_true_f.min(), y_pred_f.min()),
        max(y_true_f.max(), y_pred_f.max()),
    ]

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true_f, y_pred_f, s=5, alpha=0.3)
    plt.plot(lims, lims, "--", linewidth=2)
    plt.xlabel("True line flow")
    plt.ylabel("Predicted line flow")
    plt.title("Predicted vs true line flows (test set)")
    plt.tight_layout()
    out_path = run_dir / "pred_vs_true_scatter_flat.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_residual_hist(run_dir: Path, test_x: np.ndarray, test_y: np.ndarray, y_pred: np.ndarray, meta: dict) -> None:
    import matplotlib.pyplot as plt

    flow_max = np.asarray(meta["flow_max"])
    withd_m = np.asarray(meta["withd_m"])
    injec_m = np.asarray(meta["injec_m"])
    demand_max = np.asarray(meta["demand_max"])
    p_nom_bus = np.asarray(meta["p_nom_bus"])
    num_nodes = int(meta["num_nodes_orig"])

    flows_pred_real = y_pred * flow_max
    withdraw = flows_pred_real @ withd_m
    inject = flows_pred_real @ injec_m
    net_flow = withdraw + inject

    if test_x.ndim == 4:
        demand = test_x[:, -1, :, -1] * demand_max
    else:
        demand = test_x[:, :, -1] * demand_max

    gene_true = test_y[:, :num_nodes] * p_nom_bus
    residual = gene_true - (net_flow + demand)
    residual_f = residual.reshape(-1)

    plt.figure(figsize=(5, 4))
    plt.hist(residual_f, bins=60, density=True)
    plt.xlabel("Nodal power balance residual")
    plt.ylabel("Density")
    plt.title("Nodal power balance residuals (test set)")
    plt.tight_layout()
    out_path = run_dir / "nodal_residual_hist.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_compare_scatter(run_dirs: list[Path], compare_items: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cols = min(3, len(compare_items))
    rows = int(np.ceil(len(compare_items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for ax in axes.flat[len(compare_items):]:
        ax.axis("off")

    for ax, item in zip(axes.flat, compare_items):
        y_true_f = item["y_true"].reshape(-1)
        y_pred_f = item["y_pred"].reshape(-1)
        lims = [
            min(y_true_f.min(), y_pred_f.min()),
            max(y_true_f.max(), y_pred_f.max()),
        ]

        ax.scatter(y_true_f, y_pred_f, s=5, alpha=0.3)
        ax.plot(lims, lims, "--", linewidth=2)
        ax.set_xlabel("True line flow")
        ax.set_ylabel("Predicted line flow")
        ax.set_title(f"{item['label']} | R2={item['r2_mean']:.3f}")

    fig.suptitle("Predicted vs true line flows (test set)", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "pred_vs_true_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved compare scatter plot to {out_path}")


def main():
    args = parse_args()
    run_dirs = resolve_run_dirs(args.run_dir, args.run_root)
    compare_items = []

    for run_dir in run_dirs:
        data_dir = infer_data_dir(run_dir, args.data_dir)

        cfg, arch, train, _val, test, meta = prepare_from_run(run_dir, data_dir)
        models = load_models(arch, train, run_dir, args.weights)

        model = models["main"]
        test_x, test_y = test
        y_pred = model(test_x, training=False).numpy()

        num_links = meta["num_links"]
        y_true = test_y[:, -num_links:]

        err = y_pred - y_true
        rmse_links = np.sqrt(np.mean(err ** 2, axis=0))
        mae_links = np.mean(np.abs(err), axis=0)
        r2_links = r2_score(y_true, y_pred)
        maape_links = maape(y_true, y_pred)

        y_base = np.repeat(np.mean(y_true, axis=0, keepdims=True), y_true.shape[0], axis=0)
        err_base = y_base - y_true
        rmse_base_links = np.sqrt(np.mean(err_base ** 2, axis=0))
        mae_base_links = np.mean(np.abs(err_base), axis=0)
        r2_base_links = r2_score(y_true, y_base)

        np.save(run_dir / "y_true_test.npy", test_y)
        np.save(run_dir / "y_pred_test.npy", y_pred)

        summary = {
            "run_dir": str(run_dir),
            "data_dir": str(data_dir),
            "weights": args.weights,
            "arch": cfg.arch,
            "window": cfg.window,
            "seed": cfg.seed,
            "rmse_model": float(np.sqrt(np.mean(err ** 2))),
            "rmse_baseline": float(np.sqrt(np.mean(err_base ** 2))),
            "mae_model": float(np.mean(np.abs(err))),
            "mae_baseline": float(np.mean(np.abs(err_base))),
            "r2_mean": float(np.mean(r2_links)),
            "r2_baseline_mean": float(np.mean(r2_base_links)),
            "maape_mean": float(np.mean(maape_links)),
            "num_test_samples": int(y_true.shape[0]),
            "num_links": int(num_links),
        }

        dump_run(
            run_dir=run_dir,
            rmse_model=summary["rmse_model"],
            rmse_baseline=summary["rmse_baseline"],
            rmse_links=rmse_links,
            rmse_base_links=rmse_base_links,
            config=summary,
        )

        per_link = pd.DataFrame(
            {
                "rmse_model": rmse_links,
                "rmse_baseline": rmse_base_links,
                "mae_model": mae_links,
                "mae_baseline": mae_base_links,
                "r2_model": r2_links,
                "r2_baseline": r2_base_links,
            }
        )
        per_link.to_csv(run_dir / "per_link_metrics.csv", index_label="link_id")

        if args.plots:
            save_scatter_plots(run_dir, y_true, y_pred)
            save_flat_scatter(run_dir, y_true, y_pred)
            save_residual_hist(run_dir, test_x, test_y, y_pred, meta)

        compare_items.append(
            {
                "label": run_dir.name,
                "y_true": y_true,
                "y_pred": y_pred,
                "r2_mean": summary["r2_mean"],
            }
        )

        print(f"Run dir: {run_dir}")
        print(f"Data dir: {data_dir}")
        print(f"Weights: {args.weights}")
        print(f"Samples: {y_true.shape[0]} | Links: {num_links}")
        print(f"RMSE(model):    {summary['rmse_model']:.6f}")
        print(f"RMSE(baseline): {summary['rmse_baseline']:.6f}")
        print(f"MAE(model):     {summary['mae_model']:.6f}")
        print(f"R2(mean dims):  {summary['r2_mean']:.6f}")
        print(f"MAAPE(mean):    {summary['maape_mean']:.6f}")
        print("")

    if args.compare_plot:
        out_dir = args.run_root.resolve() if args.run_root is not None else run_dirs[0].parent
        save_compare_scatter(run_dirs, compare_items, out_dir)


if __name__ == "__main__":
    main()
