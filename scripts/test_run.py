import argparse
from pathlib import Path

from dcopf_gat.train import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Quick training run for dcopf-gat.")
    p.add_argument(
        "--data_dir",
        type=str,
        default="data_toy_3bus",
        help="Path to dataset directory (default: data_toy_3bus).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir.resolve()}\n"
            f"Tip: run `python scripts/generate_toy_dataset.py` or pass --data_dir to an existing dataset."
        )

    model, history, (test_x, test_y), test_metrics = run_experiment(
        data_dir=str(data_dir),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print("Test metrics:", test_metrics)
