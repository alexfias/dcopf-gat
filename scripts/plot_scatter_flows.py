import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

from dcopf_gat.train import build_model_from_meta
from dcopf_gat.data import prepare_dataset


# -------- paths --------
data_dir = "data_ieee14_opf_storage"
run_dir = Path("runs") / "data_ieee14_opf_storage" / "gat_flow_lqat"

# -------- load data --------
train_x, train_y, val_x, val_y, test_x, test_y, meta = prepare_dataset(
    data_dir, pca_flag=False, train_fraction=0.8, seed=1234
)

# -------- rebuild model + load weights --------
model = build_model_from_meta(meta, lamb=0.001)
_ = model(tf.convert_to_tensor(test_x[:1], dtype=tf.float32))  # build
model.load_weights(run_dir / "model_best.weights.h5")

# -------- predict --------
y_pred = model.predict(test_x, batch_size=128)

# only flows (labels already are flows here)
num_nodes = meta["num_nodes_orig"]
y_true = test_y[:, num_nodes:]   # only flows


# flatten everything
y_true_f = y_true.reshape(-1)
y_pred_f = y_pred.reshape(-1)

# -------- scatter plot --------
plt.figure(figsize=(5, 5))
plt.scatter(y_true_f, y_pred_f, s=5, alpha=0.3)
lims = [
    min(y_true_f.min(), y_pred_f.min()),
    max(y_true_f.max(), y_pred_f.max()),
]
plt.plot(lims, lims, "--", linewidth=2)
plt.xlabel("True line flow")
plt.ylabel("Predicted line flow")
plt.title("Predicted vs true line flows (test set)")
plt.tight_layout()
plt.show()


# -------- compute nodal imbalance --------

# predicted flows → real scale
flow_max = meta["flow_max"]
withd_m = meta["withd_m"]
injec_m = meta["injec_m"]
demand_max = meta["demand_max"]

# scale flows back
flows_pred_real = y_pred * flow_max

# nodal injections from flows
withdraw = flows_pred_real @ withd_m
inject = flows_pred_real @ injec_m
net_flow = withdraw + inject

# demand
demand = test_x[:, :, -1] * demand_max

# true generation
num_nodes = meta["num_nodes_orig"]
gene_true = test_y[:, :num_nodes] * meta["p_nom_bus"]

# nodal residual
residual = gene_true - (net_flow + demand)

# flatten
residual_f = residual.reshape(-1)

# -------- histogram --------
plt.figure(figsize=(5, 4))
plt.hist(residual_f, bins=60, density=True)
plt.xlabel("Nodal power balance residual")
plt.ylabel("Density")
plt.title("Nodal power balance residuals (test set)")
plt.tight_layout()
plt.show()
