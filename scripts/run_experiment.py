# scripts/run_experiment.py

import argparse
from pathlib import Path

from dcopf_gat.runtime import configure_runtime
from dcopf_gat.train import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Training entrypoint for dcopf-gat")

    # -------- runtime / hardware --------
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device selection (default: auto)",
    )
    p.add_argument(
        "--mp",
        action="store_true",
        help="Enable mixed precision (GPU only)",
    )
    p.add_argument(
        "--xla",
        action="store_true",
        help="Enable XLA JIT compilation",
    )

    # -------- experiment --------
    p.add_argument(
        "--data_dir",
        type=str,
        default="data_toy_3bus",
        help="Path to dataset directory",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (increase on GPU)",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )

    p.add_argument(
        "--arch",
        type=str,
        default="gat_flow_lqat",
        help="Architecture name (used for logging & comparison)",
    )

    p.add_argument(
        "--window",
        type=int,
        default=0,
        help="Temporal lookback window (0 = static). Concatenates [t-window..t] into features.",
    )

    p.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    return p.parse_args()


def main():
    args = parse_args()

    # -------- runtime configuration --------
    configure_runtime(
        device=args.device,
        mixed_precision=args.mp,
        xla=args.xla,
    )

    # -------- dataset check --------
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir.resolve()}\n"
            f"Tip: generate a dataset or pass --data_dir to an existing one."
        )

    # -------- run experiment --------
    model, history, (test_x, test_y), test_metrics = run_experiment(
        data_dir=str(data_dir),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        arch_name=args.arch,
        use_tfdata=True,   
        window=args.window,
    )

    # -------- logging --------
    print("\n================ MODEL SUMMARY ================\n")
    model.summary()
    print("\n==============================================\n")

    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
