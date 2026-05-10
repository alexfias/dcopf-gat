from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from dcopf_gat.data_eraa import load_eraa_dataset
import matplotlib.pyplot as plt

def build_model(n_features: int, n_targets: int, hidden: int = 512, dropout: float = 0.1):
    inputs = tf.keras.Input(shape=(None, n_features), name="features")

    x = tf.keras.layers.Dense(hidden, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(hidden, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(hidden // 2, activation="relu")(x)

    outputs = tf.keras.layers.Dense(n_targets, name="flows")(x)

    return tf.keras.Model(inputs, outputs, name="eraa_timedistributed_mlp")


def inverse_y(y_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return y_norm * y_std + y_mean


def compute_metrics(y_true_norm, y_pred_norm, y_mean, y_std):
    y_true = inverse_y(y_true_norm, y_mean, y_std)
    y_pred = inverse_y(y_pred_norm, y_mean, y_std)

    global_std = float(np.std(y_true))
    mse_norm = float(np.mean(((y_true - y_pred) / (global_std + 1e-6)) ** 2))

    mae_mw = float(np.mean(np.abs(y_true - y_pred)))
    rmse_mw = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "mse_norm": mse_norm,
        "mae_mw": mae_mw,
        "rmse_mw": rmse_mw,
        "r2": r2,
    }


def print_metrics(name: str, metrics: dict):
    print(f"{name}:")
    print(f"  normalized MSE: {metrics['mse_norm']:.6f}")
    print(f"  MAE [MW]:       {metrics['mae_mw']:.3f}")
    print(f"  RMSE [MW]:      {metrics['rmse_mw']:.3f}")
    print(f"  R²:             {metrics['r2']:.5f}")


def save_diagnostic_plots(y_true_norm, y_pred_norm, y_mean, y_std, out_dir: Path):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_true = inverse_y(y_true_norm, y_mean, y_std).reshape(-1)
    y_pred = inverse_y(y_pred_norm, y_mean, y_std).reshape(-1)
    err = y_pred - y_true

    # Predicted vs true scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=1, alpha=0.05)
    lim = max(abs(y_true).max(), abs(y_pred).max())
    plt.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
    plt.xlabel("True flow [MW]")
    plt.ylabel("Predicted flow [MW]")
    plt.title("ERAA MLP: Predicted vs true flows")
    plt.tight_layout()
    plt.savefig(fig_dir / "predicted_vs_true_flows.png", dpi=200)
    plt.close()

    # Error histogram
    plt.figure(figsize=(7, 4))
    plt.hist(err, bins=200)
    plt.xlabel("Prediction error [MW]")
    plt.ylabel("Count")
    plt.title("ERAA MLP: Flow error distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "flow_error_histogram.png", dpi=200)
    plt.close()

    # Absolute error versus absolute flow
    plt.figure(figsize=(7, 5))
    plt.scatter(np.abs(y_true), np.abs(err), s=1, alpha=0.05)
    plt.xlabel("|True flow| [MW]")
    plt.ylabel("|Error| [MW]")
    plt.title("ERAA MLP: Error vs flow magnitude")
    plt.tight_layout()
    plt.savefig(fig_dir / "abs_error_vs_abs_flow.png", dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_eraa_ml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="results_eraa_mlp")
    args = parser.parse_args()

    ds = load_eraa_dataset(args.data_dir, max_samples=args.max_samples)

    n_features = ds.X_train.shape[-1]
    n_targets = ds.y_train.shape[-1]

    print("Dataset:")
    print("  X_train:", ds.X_train.shape)
    print("  y_train:", ds.y_train.shape)
    print("  X_val:  ", ds.X_val.shape)
    print("  y_val:  ", ds.y_val.shape)
    print("  X_test: ", ds.X_test.shape)
    print("  y_test: ", ds.y_test.shape)

    model = build_model(
        n_features=n_features,
        n_targets=n_targets,
        hidden=args.hidden,
        dropout=args.dropout,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        ds.X_train,
        ds.y_train,
        validation_data=(ds.X_val, ds.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    y_val_pred = model.predict(ds.X_val, batch_size=args.batch_size, verbose=0)
    y_test_pred = model.predict(ds.X_test, batch_size=args.batch_size, verbose=0)

    val_metrics = compute_metrics(ds.y_val, y_val_pred, ds.y_mean, ds.y_std)
    test_metrics = compute_metrics(ds.y_test, y_test_pred, ds.y_mean, ds.y_std)

    print()
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    save_diagnostic_plots(
        ds.y_test,
        y_test_pred,
        ds.y_mean,
        ds.y_std,
        Path(args.output_dir),
    )
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / "model.keras")

    np.savez_compressed(
        out_dir / "scalers.npz",
        x_mean=ds.x_mean,
        x_std=ds.x_std,
        y_mean=ds.y_mean,
        y_std=ds.y_std,
    )

    np.savez_compressed(
        out_dir / "metrics.npz",
        val_mse_norm=val_metrics["mse_norm"],
        val_mae_mw=val_metrics["mae_mw"],
        val_rmse_mw=val_metrics["rmse_mw"],
        val_r2=val_metrics["r2"],
        test_mse_norm=test_metrics["mse_norm"],
        test_mae_mw=test_metrics["mae_mw"],
        test_rmse_mw=test_metrics["rmse_mw"],
        test_r2=test_metrics["r2"],
    )

    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()