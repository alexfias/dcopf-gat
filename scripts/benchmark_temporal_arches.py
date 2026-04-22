from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from dcopf_gat.utils import set_global_seed
from dcopf_gat.data import prepare_dataset
from dcopf_gat.architectures.base import TrainConfig
from dcopf_gat.architectures.registry import ARCH_REGISTRY
from dcopf_gat.windowing import make_windows_concat, make_windows_sequence

# side-effect imports to register architectures
from dcopf_gat.architectures import arch_a  # noqa: F401
from dcopf_gat.architectures import arch_b  # noqa: F401
from dcopf_gat.architectures import arch_c  # noqa: F401
from dcopf_gat.architectures import arch_d  # noqa: F401
from dcopf_gat.architectures import arch_e  # noqa: F401
from dcopf_gat.architectures import arch_f  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark dcopf_gat architectures across window settings")
    p.add_argument("--data_dir", default="data_ieee14_opf_storage", help="Dataset directory")
    p.add_argument(
        "--arches",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F"],
        help="Architecture keys to benchmark",
    )
    p.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[0, 2, 4, 8],
        help="Window settings to benchmark. For centered mode, each value is a radius.",
    )
    p.add_argument("--epochs", type=int, default=20, help="Training epochs per run")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument(
        "--learning_rates",
        nargs="+",
        type=float,
        default=[1e-3],
        help="One or more learning rates to benchmark",
    )
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument(
        "--window_mode",
        choices=["past", "centered"],
        default="past",
        help=(
            "Window construction mode. "
            "'past' uses [t-window, ..., t]. "
            "'centered' uses [t-window, ..., t, ..., t+window] with window as radius."
        ),
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        default=Path("runs") / "benchmark_temporal_arches.csv",
        help="CSV path for benchmark results",
    )
    p.add_argument(
        "--keep_artifacts",
        action="store_true",
        help="Keep per-run temp directories instead of deleting them",
    )
    p.add_argument(
        "--plot_png",
        type=Path,
        default=Path("runs") / "benchmark_temporal_arches.png",
        help="PNG path for benchmark plots",
    )
    return p.parse_args()


def build_splits(data_dir: str, arch_name: str, seed: int):
    include_soc_feature = arch_name in {"C", "D", "E", "F"}
    return prepare_dataset(
        data_dir,
        pca_flag=False,
        train_fraction=0.8,
        seed=seed,
        include_soc_feature=include_soc_feature,
    )


def make_windows_sequence_centered(x, y, window: int):
    if window <= 0:
        return x, y
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (T,N,F). Got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (T,Y). Got {y.shape}")
    t = x.shape[0]
    if t <= 2 * window:
        raise ValueError(f"Not enough timesteps: T={t} <= 2*window={2 * window}")
    xs = []
    for i in range(window, t - window):
        xs.append(x[i - window : i + window + 1])
    import numpy as np
    xw = np.stack(xs, axis=0)
    yw = y[window : t - window]
    return xw, yw


def make_windows_concat_centered(x, y, window: int):
    if window <= 0:
        return x, y
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (T,N,F). Got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (T,Y). Got {y.shape}")
    t, _, _ = x.shape
    if t <= 2 * window:
        raise ValueError(f"Not enough timesteps: T={t} <= 2*window={2 * window}")
    xs = []
    for i in range(window, t - window):
        xs.append(x[i - window : i + window + 1].reshape(x.shape[1], -1))
    import numpy as np
    xw = np.stack(xs, axis=0)
    yw = y[window : t - window]
    return xw, yw


def apply_windowing(arch_name: str, train_x, train_y, val_x, val_y, test_x, test_y, window: int, window_mode: str):
    if window_mode == "past":
        if arch_name in {"A", "C"} or window <= 0:
            return (train_x, train_y), (val_x, val_y), (test_x, test_y)
        if arch_name in {"B", "D"} and window > 1:
            train_x, train_y = make_windows_concat(train_x, train_y, window=window)
            val_x, val_y = make_windows_concat(val_x, val_y, window=window)
            test_x, test_y = make_windows_concat(test_x, test_y, window=window)
        elif arch_name in {"E", "F"} and window > 1:
            train_x, train_y = make_windows_sequence(train_x, train_y, window=window)
            val_x, val_y = make_windows_sequence(val_x, val_y, window=window)
            test_x, test_y = make_windows_sequence(test_x, test_y, window=window)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    if window_mode == "centered":
        if arch_name in {"A", "C"} or window <= 0:
            return (train_x, train_y), (val_x, val_y), (test_x, test_y)
        if arch_name in {"B", "D"} and window > 0:
            train_x, train_y = make_windows_concat_centered(train_x, train_y, window=window)
            val_x, val_y = make_windows_concat_centered(val_x, val_y, window=window)
            test_x, test_y = make_windows_concat_centered(test_x, test_y, window=window)
        elif arch_name in {"E", "F"} and window > 0:
            train_x, train_y = make_windows_sequence_centered(train_x, train_y, window=window)
            val_x, val_y = make_windows_sequence_centered(val_x, val_y, window=window)
            test_x, test_y = make_windows_sequence_centered(test_x, test_y, window=window)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    raise ValueError(f"Unknown window_mode: {window_mode}")


def validate_combo(arch_name: str, window: int, window_mode: str):
    """
    Return None when the combination is valid, otherwise a human-readable reason.
    """
    if arch_name == "A" and window > 0:
        return "Architecture A is a snapshot model and should only be run with window=0."
    if arch_name == "C" and window > 0:
        return "Architecture C is a snapshot model and should only be run with window=0."
    if arch_name in {"E", "F"} and window_mode == "past" and window <= 1:
        return "E/F require window > 1 for past mode on the current upstream base."
    if arch_name in {"E", "F"} and window_mode == "centered" and window <= 0:
        return "E/F require window > 0 for centered mode."
    return None


def run_one(data_dir: str, arch_name: str, window: int, epochs: int, batch_size: int, learning_rate: float, seed: int, keep_artifacts: bool, window_mode: str):
    set_global_seed(seed)

    train_x, train_y, val_x, val_y, test_x, test_y, meta = build_splits(data_dir, arch_name, seed)
    cfg = TrainConfig(
        arch=arch_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        window=window,
        use_tfdata=True,
        lamb=0.001,
        debug=False,
    )

    arch = ARCH_REGISTRY[arch_name](cfg, meta)
    train, val, test = apply_windowing(
        arch_name,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        window,
        window_mode,
    )
    models = arch.build()

    if keep_artifacts:
        lr_tag = f"{learning_rate:.0e}".replace("+", "")
        run_dir = Path("runs") / "benchmark_tmp" / Path(data_dir).name / window_mode / arch_name / f"window_{window}" / f"lr_{lr_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        histories = arch.fit(models=models, train=train, val=val, run_dir=run_dir)
        metrics = arch.evaluate(models=models, test=test)
        artifact_dir = run_dir
    else:
        with tempfile.TemporaryDirectory(prefix=f"dcopf_{arch_name}_w{window}_lr_") as td:
            run_dir = Path(td)
            histories = arch.fit(models=models, train=train, val=val, run_dir=run_dir)
            metrics = arch.evaluate(models=models, test=test)
        artifact_dir = None

    hist = histories["history_main"]
    return {
        "arch": arch_name,
        "window": window,
        "epochs": epochs,
        "window_mode": window_mode,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "test_R2": float(metrics.get("R2")),
        "test_loss": float(metrics.get("loss")),
        "test_loss1": float(metrics.get("loss1")),
        "test_loss2": float(metrics.get("loss2")),
        "best_val_R2": float(max(hist.get("val_R2", [-999.0]))),
        "best_val_loss": float(min(hist.get("val_loss", [999.0]))),
        "final_val_R2": float(hist.get("val_R2", [float("nan")])[-1]),
        "final_val_loss": float(hist.get("val_loss", [float("nan")])[-1]),
        "artifact_dir": None if artifact_dir is None else str(artifact_dir),
    }


def save_plots(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for arch, sub in df.groupby("arch"):
        sub = sub.sort_values("window")
        axes[0].plot(sub["window"], sub["test_R2"], marker="o", label=arch)
        axes[1].plot(sub["window"], sub["test_loss"], marker="o", label=arch)

    axes[0].set_title("Test R2 vs Window")
    axes[0].set_xlabel("Window")
    axes[0].set_ylabel("Test R2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Test Loss vs Window")
    axes[1].set_xlabel("Window")
    axes[1].set_ylabel("Test Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = []

    for arch_name in args.arches:
        if arch_name not in ARCH_REGISTRY:
            raise ValueError(f"Unknown architecture '{arch_name}'. Available: {sorted(ARCH_REGISTRY)}")
        for window in args.windows:
            for learning_rate in args.learning_rates:
                invalid_reason = validate_combo(arch_name, window, args.window_mode)
                if invalid_reason is not None:
                    row = {
                        "arch": arch_name,
                        "window": window,
                        "epochs": args.epochs,
                        "window_mode": args.window_mode,
                        "batch_size": args.batch_size,
                        "learning_rate": learning_rate,
                        "seed": args.seed,
                        "test_R2": float("nan"),
                        "test_loss": float("nan"),
                        "test_loss1": float("nan"),
                        "test_loss2": float("nan"),
                        "best_val_R2": float("nan"),
                        "best_val_loss": float("nan"),
                        "final_val_R2": float("nan"),
                        "final_val_loss": float("nan"),
                        "artifact_dir": None,
                        "status": "skipped",
                        "skip_reason": invalid_reason,
                    }
                    rows.append(row)
                    print(f"Skipping arch={arch_name} window={window} lr={learning_rate}: {invalid_reason}")
                    continue

                print(f"Running arch={arch_name} window={window} lr={learning_rate} epochs={args.epochs}")
                row = run_one(
                    data_dir=args.data_dir,
                    arch_name=arch_name,
                    window=window,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=learning_rate,
                    seed=args.seed,
                    keep_artifacts=args.keep_artifacts,
                    window_mode=args.window_mode,
                )
                row["status"] = "ok"
                row["skip_reason"] = ""
                rows.append(row)
                print(json.dumps(row, indent=2))

    df = pd.DataFrame(rows).sort_values(["arch", "window", "learning_rate"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("\nBenchmark Summary")
    print(df[["arch", "window", "learning_rate", "window_mode", "test_R2", "test_loss", "best_val_R2", "best_val_loss"]].to_string(index=False))
    print(f"\nSaved CSV to {args.output_csv}")
    save_plots(df, args.plot_png)
    print(f"Saved plot to {args.plot_png}")


if __name__ == "__main__":
    main()
