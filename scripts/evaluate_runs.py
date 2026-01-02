from pathlib import Path
import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Returns R² per output dimension (last axis).
    """
    # ensure 2D: (N, D)
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
    run_dir = Path("runs/toy3bus_eval")
    y_true = np.load(run_dir / "y_true_test.npy")
    y_pred = np.load(run_dir / "y_pred_test.npy")

    # Align if y_true contains extra columns not predicted by the model
    if y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[1] != y_pred.shape[1]:
        if y_true.shape[1] > y_pred.shape[1]:
            print(f"Note: y_true has {y_true.shape[1]} dims but y_pred has {y_pred.shape[1]} dims.")
            print(f"      Evaluating against first {y_pred.shape[1]} columns of y_true.")
            y_true = y_true[:, : y_pred.shape[1]]
        else:
            raise ValueError("y_pred has more dims than y_true — unexpected.")

    # ensure 2D (N, D)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    # ensure 2D (N, D)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    # 🔍 DEBUG: inspect first few samples
    print("\nFirst 5 rows (y_true vs y_pred):")
    for i in range(min(5, y_true.shape[0])):
        print(f"{i}: true={y_true[i]}  pred={y_pred[i]}")

    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    r2 = r2_score(y_true, y_pred)


    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    r2 = r2_score(y_true, y_pred)

    print(f"Loaded: y_true {y_true.shape}, y_pred {y_pred.shape}\n")

    print("Per-dimension metrics:")
    for i in range(y_true.shape[1]):
        print(f"  dim {i:02d}:  R2={r2[i]: .5f}   RMSE={rmse[i]: .6f}   MAE={mae[i]: .6f}")

    print("\nOverall metrics:")
    print(f"  RMSE(all dims) = {np.sqrt(np.mean(err**2)):.6f}")
    print(f"  MAE(all dims)  = {np.mean(np.abs(err)):.6f}")
    print(f"  R2(mean dims)  = {np.mean(r2):.6f}")

    # Worst cases
    sample_rmse = np.sqrt(np.mean(err**2, axis=1))
    worst_idx = np.argsort(sample_rmse)[-10:][::-1]
    print("\nWorst 10 samples by RMSE:")
    for k, idx in enumerate(worst_idx, 1):
        print(f"  {k:02d}) idx={idx:5d}  sample_RMSE={sample_rmse[idx]:.6f}")


if __name__ == "__main__":
    main()
