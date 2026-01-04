from pathlib import Path
import numpy as np
import pandas as pd


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns R² per output dimension (last axis)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    # Where you saved outputs from run_experiment.py
    run_dir = Path("runs/toy3bus_eval")

    # IMPORTANT: point this to the dataset used to generate the saved test outputs
    data_dir = Path("data_toy_3bus")

    # Load full true targets and predictions
    y_true_full = np.load(run_dir / "y_true_test.npy")
    y_pred = np.load(run_dir / "y_pred_test.npy")

    # Ensure 2D
    if y_true_full.ndim == 1:
        y_true_full = y_true_full[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    print("y_true_full shape:", y_true_full.shape, "y_pred shape:", y_pred.shape)

    # Determine number of link-flow targets from links.csv
    links = pd.read_csv(data_dir / "links.csv", index_col=0)
    num_links = links.shape[0]
    print(f"Evaluating flows: using last {num_links} columns of y_true_full")

    # Flows are the last num_links columns in Y = [gen_bus_norm | flow_target]
    y_true = y_true_full[:, -num_links:]

    # Sanity check: model output dims should match number of links
    assert y_true.shape[1] == y_pred.shape[1], (
        f"Target mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}. "
        f"Expected model output dims == num_links ({num_links})."
    )

    print(f"Loaded: y_true {y_true.shape}, y_pred {y_pred.shape}\n")

    # 🔍 DEBUG: inspect first few samples
    print("First 5 rows (y_true vs y_pred):")
    for i in range(min(5, y_true.shape[0])):
        print(f"{i}: true={y_true[i]}  pred={y_pred[i]}")

    # ----- model metrics -----
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    r2 = r2_score(y_true, y_pred)

    print("\nModel metrics (per dim):")
    for i in range(y_true.shape[1]):
        print(f"  dim {i:02d}:  R2={r2[i]: .5f}   RMSE={rmse[i]: .6f}   MAE={mae[i]: .6f}")

    print("\nModel metrics (overall):")
    print(f"  RMSE(all dims) = {np.sqrt(np.mean(err**2)):.6f}")
    print(f"  MAE(all dims)  = {np.mean(np.abs(err)):.6f}")
    print(f"  R2(mean dims)  = {np.mean(r2):.6f}")

    # ----- baseline: predict mean of y_true (per dimension) -----
    y_base = np.mean(y_true, axis=0, keepdims=True) * np.ones_like(y_true)

    err_base = y_base - y_true
    mae_base = np.mean(np.abs(err_base), axis=0)
    rmse_base = np.sqrt(np.mean(err_base ** 2, axis=0))
    r2_base = r2_score(y_true, y_base)

    print("\nBaseline (predict mean of test y_true):")
    for i in range(y_true.shape[1]):
        print(f"  dim {i:02d}:  R2={r2_base[i]: .5f}   RMSE={rmse_base[i]: .6f}   MAE={mae_base[i]: .6f}")

    print("\nModel vs baseline improvement (RMSE):")
    for i in range(y_true.shape[1]):
        ratio = rmse[i] / (rmse_base[i] + 1e-12)
        print(f"  dim {i:02d}:  RMSE_model={rmse[i]:.6f}   RMSE_base={rmse_base[i]:.6f}   ratio={ratio:.3f}")

    # Worst cases
    sample_rmse = np.sqrt(np.mean(err ** 2, axis=1))
    worst_idx = np.argsort(sample_rmse)[-10:][::-1]
    print("\nWorst 10 samples by RMSE:")
    for k, idx in enumerate(worst_idx, 1):
        print(f"  {k:02d}) idx={idx:5d}  sample_RMSE={sample_rmse[idx]:.6f}")

    # ----- plots -----
    import matplotlib.pyplot as plt

    plot_dir = run_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    for d in range(y_true.shape[1]):
        plt.figure()
        plt.scatter(y_true[:, d], y_pred[:, d])
        mn = min(y_true[:, d].min(), y_pred[:, d].min())
        mx = max(y_true[:, d].max(), y_pred[:, d].max())
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("true")
        plt.ylabel("pred")
        plt.title(f"Pred vs True (flow dim {d})")
        plt.savefig(plot_dir / f"pred_vs_true_flow_dim{d:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nSaved scatter plots to {plot_dir}")


if __name__ == "__main__":
    main()
