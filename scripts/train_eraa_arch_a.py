from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dcopf_gat.data_eraa import load_eraa_graph_dataset


class SimpleGraphConv(tf.keras.layers.Layer):
    def __init__(self, hidden: int, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.activation = tf.keras.activations.get(activation)
        self.self_dense = tf.keras.layers.Dense(hidden)
        self.neigh_dense = tf.keras.layers.Dense(hidden)

    def call(self, x, edge_index):
        # x: [batch, nodes, features]
        src = edge_index[0]
        dst = edge_index[1]

        src_feat = tf.gather(x, src, axis=1)

        n_nodes = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        hidden_in = tf.shape(x)[2]

        agg = tf.zeros((batch_size, n_nodes, hidden_in), dtype=x.dtype)

        agg = tf.tensor_scatter_nd_add(
            tf.transpose(agg, [1, 0, 2]),
            tf.expand_dims(dst, axis=1),
            tf.transpose(src_feat, [1, 0, 2]),
        )
        agg = tf.transpose(agg, [1, 0, 2])

        deg = tf.math.bincount(dst, minlength=n_nodes, maxlength=n_nodes, dtype=x.dtype)
        deg = tf.reshape(tf.maximum(deg, 1.0), (1, n_nodes, 1))
        agg = agg / deg

        out = self.self_dense(x) + self.neigh_dense(agg)
        return self.activation(out)


class EraaArchA(tf.keras.Model):
    def __init__(self, edge_index: np.ndarray, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.edge_index = tf.constant(edge_index, dtype=tf.int32)

        self.input_proj = tf.keras.layers.Dense(hidden, activation="relu")
        self.gconv1 = SimpleGraphConv(hidden)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.gconv2 = SimpleGraphConv(hidden)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self.edge_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden // 2, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x, training=False):
        # x: [batch, nodes, features]
        h = self.input_proj(x)

        h = self.gconv1(h, self.edge_index)
        h = self.dropout1(h, training=training)

        h = self.gconv2(h, self.edge_index)
        h = self.dropout2(h, training=training)

        src = self.edge_index[0]
        dst = self.edge_index[1]

        h_src = tf.gather(h, src, axis=1)
        h_dst = tf.gather(h, dst, axis=1)

        edge_feat = tf.concat([h_src, h_dst, h_src - h_dst, h_src * h_dst], axis=-1)
        y = self.edge_mlp(edge_feat, training=training)

        return tf.squeeze(y, axis=-1)


def inverse_y(y_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return y_norm * y_std.reshape(1, -1) + y_mean.reshape(1, -1)


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

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=1, alpha=0.05)
    lim = max(abs(y_true).max(), abs(y_pred).max())
    plt.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1)
    plt.xlabel("True flow [MW]")
    plt.ylabel("Predicted flow [MW]")
    plt.title("ERAA Architecture A: Predicted vs true flows")
    plt.tight_layout()
    plt.savefig(fig_dir / "predicted_vs_true_flows.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(err, bins=200)
    plt.xlabel("Prediction error [MW]")
    plt.ylabel("Count")
    plt.title("ERAA Architecture A: Flow error distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "flow_error_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(np.abs(y_true), np.abs(err), s=1, alpha=0.05)
    plt.xlabel("|True flow| [MW]")
    plt.ylabel("|Error| [MW]")
    plt.title("ERAA Architecture A: Error vs flow magnitude")
    plt.tight_layout()
    plt.savefig(fig_dir / "abs_error_vs_abs_flow.png", dpi=200)
    plt.close()


def save_predicted_flows(
    y_pred_norm: np.ndarray,
    y_true_norm: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    out_dir: Path,
    split: str,
    edge_names=None,
):
    import pandas as pd

    flows_pred = inverse_y(y_pred_norm, y_mean, y_std)
    flows_true = inverse_y(y_true_norm, y_mean, y_std)
    flows_error = flows_pred - flows_true

    n_edges = flows_pred.shape[1]

    if edge_names is None:
        edge_names = [f"edge_{i}" for i in range(n_edges)]

    pred_df = pd.DataFrame(flows_pred, columns=edge_names)
    true_df = pd.DataFrame(flows_true, columns=edge_names)
    err_df = pd.DataFrame(flows_error, columns=edge_names)

    pred_df.to_csv(out_dir / f"{split}_predicted_flows_mw.csv", index=False)
    true_df.to_csv(out_dir / f"{split}_true_flows_mw.csv", index=False)
    err_df.to_csv(out_dir / f"{split}_flow_errors_mw.csv", index=False)

    np.savez_compressed(
        out_dir / f"{split}_flows_mw.npz",
        predicted=flows_pred,
        true=flows_true,
        error=flows_error,
        edge_names=np.array(edge_names),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_eraa_ml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="results_eraa_arch_a")
    args = parser.parse_args()

    ds = load_eraa_graph_dataset(args.data_dir, max_samples=args.max_samples)

    print("Dataset:")
    print("  X_nodes_train:", ds.X_nodes_train.shape)
    print("  y_edges_train:", ds.y_edges_train.shape)
    print("  X_nodes_val:  ", ds.X_nodes_val.shape)
    print("  y_edges_val:  ", ds.y_edges_val.shape)
    print("  X_nodes_test: ", ds.X_nodes_test.shape)
    print("  y_edges_test: ", ds.y_edges_test.shape)
    print("  edge_index:   ", ds.edge_index.shape)

    model = EraaArchA(
        edge_index=ds.edge_index,
        hidden=args.hidden,
        dropout=args.dropout,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

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
        ds.X_nodes_train,
        ds.y_edges_train,
        validation_data=(ds.X_nodes_val, ds.y_edges_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    y_val_pred = model.predict(ds.X_nodes_val, batch_size=args.batch_size, verbose=0)
    y_test_pred = model.predict(ds.X_nodes_test, batch_size=args.batch_size, verbose=0)

    val_metrics = compute_metrics(ds.y_edges_val, y_val_pred, ds.y_mean, ds.y_std)
    test_metrics = compute_metrics(ds.y_edges_test, y_test_pred, ds.y_mean, ds.y_std)

    print()
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_names = getattr(ds, "edge_names", None)

    save_predicted_flows(
        y_test_pred,
        ds.y_edges_test,
        ds.y_mean,
        ds.y_std,
        out_dir,
        split="test",
        edge_names=edge_names,
    )

    save_predicted_flows(
        y_val_pred,
        ds.y_edges_val,
        ds.y_mean,
        ds.y_std,
        out_dir,
        split="val",
        edge_names=edge_names,
    )

    save_diagnostic_plots(
        ds.y_edges_test,
        y_test_pred,
        ds.y_mean,
        ds.y_std,
        out_dir,
    )

    model.save_weights(out_dir / "model.weights.h5")

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