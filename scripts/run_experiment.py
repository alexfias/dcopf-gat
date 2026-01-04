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

    print("test_x type:", type(test_x))
    print("test_y type:", type(test_y))

    try:
        print("test_y shape:", test_y.shape)
    except Exception:
        pass

    y_pred_dbg = model(test_x, training=False)
    print("y_pred type:", type(y_pred_dbg))
    try:
        print("y_pred shape:", y_pred_dbg.shape)
    except Exception:
        pass

    print("Test metrics:", test_metrics)

    from pathlib import Path
    import numpy as np

    out_dir = Path("runs/toy3bus_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predict on the returned test split
    y_pred_test = model(test_x, training=False)

    # Convert to numpy
    y_pred_test = y_pred_test.numpy()
    y_true_test = test_y.numpy() if hasattr(test_y, "numpy") else np.asarray(test_y)

    # Save
    np.save(out_dir / "y_pred_test.npy", y_pred_test)
    np.save(out_dir / "y_true_test.npy", y_true_test)

    print(f"Saved test outputs to {out_dir}")
    print("Shapes:", y_true_test.shape, y_pred_test.shape)



