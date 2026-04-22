from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("runs\data_ieee14_opf_storage")   # change if needed
ARCHS = ["A", "B", "C", "D", "E"]


def find_run_dir(arch_dir: Path) -> Path | None:
    """
    Returns the directory that contains config.json, metrics_test.json, history.npy.
    Works if files are directly in arch_dir or in one subfolder.
    """
    needed = ["config.json", "metrics_test.json", "history.npy"]

    # case 1: directly inside arch_dir
    if all((arch_dir / f).exists() for f in needed):
        return arch_dir

    # case 2: search subdirectories
    candidates = []
    for sub in arch_dir.rglob("*"):
        if sub.is_dir() and all((sub / f).exists() for f in needed):
            candidates.append(sub)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # choose the shortest path / first match
        candidates = sorted(candidates, key=lambda p: len(p.parts))
        return candidates[0]

    return None


def load_json(fp: Path):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history(fp: Path):
    hist = np.load(fp, allow_pickle=True)
    if isinstance(hist, np.ndarray) and hist.shape == ():
        hist = hist.item()
    elif hasattr(hist, "item"):
        hist = hist.item()
    return hist


def safe_last(x):
    if x is None or len(x) == 0:
        return np.nan
    return x[-1]


def safe_best(x, mode="min"):
    if x is None or len(x) == 0:
        return np.nan
    return np.min(x) if mode == "min" else np.max(x)


def safe_best_epoch(x, mode="min"):
    if x is None or len(x) == 0:
        return np.nan
    return int(np.argmin(x)) + 1 if mode == "min" else int(np.argmax(x)) + 1


rows = []
histories = {}

for arch in ARCHS:
    arch_dir = ROOT / arch
    if not arch_dir.exists():
        print(f"[WARN] Missing folder: {arch_dir}")
        continue

    run_dir = find_run_dir(arch_dir)
    if run_dir is None:
        print(f"[WARN] Could not find run files for architecture {arch}")
        continue

    config = load_json(run_dir / "config.json")
    metrics = load_json(run_dir / "metrics_test.json")
    history = load_history(run_dir / "history.npy")

    if isinstance(history, dict) and "history_main" in history:
        history = history["history_main"]

    histories[arch] = history

    val_loss = history.get("val_loss", [])
    val_r2 = history.get("val_R2", history.get("val_r2", []))
    train_loss = history.get("loss", [])
    train_r2 = history.get("R2", history.get("r2", []))

    row = {
        "arch": arch,
        "run_dir": str(run_dir),
        "seed": config.get("seed"),
        "window": config.get("window"),
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),

        "test_loss": metrics.get("loss"),
        "test_R2": metrics.get("R2", metrics.get("r2")),
        "test_loss1": metrics.get("loss1"),
        "test_loss2": metrics.get("loss2"),

        "best_val_loss": safe_best(val_loss, mode="min"),
        "best_val_loss_epoch": safe_best_epoch(val_loss, mode="min"),
        "final_val_loss": safe_last(val_loss),

        "best_val_R2": safe_best(val_r2, mode="max"),
        "best_val_R2_epoch": safe_best_epoch(val_r2, mode="max"),
        "final_val_R2": safe_last(val_r2),

        "final_train_loss": safe_last(train_loss),
        "final_train_R2": safe_last(train_r2),
    }
    rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No valid runs found.")

# sort architectures in A-E order if present
df["arch"] = pd.Categorical(df["arch"], categories=ARCHS, ordered=True)
df = df.sort_values("arch").reset_index(drop=True)

print("\n=== Comparison table ===")
print(df)

# save CSV
csv_path = ROOT / "architecture_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# -------------------------
# Plot 1: summary metrics
# -------------------------
metrics_to_plot = [
    ("test_loss", "Test Loss"),
    ("test_R2", "Test R²"),
    ("best_val_loss", "Best Val Loss"),
    ("best_val_R2", "Best Val R²"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (col, title) in zip(axes, metrics_to_plot):
    ax.bar(df["arch"].astype(str), df[col])
    ax.set_title(title)
    ax.set_xlabel("Architecture")
    ax.set_ylabel(title)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
summary_plot_path = ROOT / "architecture_summary.png"
plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {summary_plot_path}")

print(f"\n--- {arch} ---")
print("History keys:", list(history.keys()))
# -------------------------
# Plot 2: validation loss curves
# -------------------------
plt.figure(figsize=(10, 6))
for arch in ARCHS:
    hist = histories.get(arch)
    if hist is None:
        continue
    y = hist.get("val_loss", [])
    if len(y) == 0:
        continue
    plt.plot(range(1, len(y) + 1), y, label=arch)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss by Architecture")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
val_loss_plot_path = ROOT / "training_curves_val_loss.png"
plt.savefig(val_loss_plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {val_loss_plot_path}")


# -------------------------
# Plot 3: validation R² curves
# -------------------------
plt.figure(figsize=(10, 6))
for arch in ARCHS:
    hist = histories.get(arch)
    if hist is None:
        continue
    y = hist.get("val_R2", hist.get("val_r2", []))
    if len(y) == 0:
        continue
    plt.plot(range(1, len(y) + 1), y, label=arch)

plt.xlabel("Epoch")
plt.ylabel("Validation R²")
plt.title("Validation R² by Architecture")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
val_r2_plot_path = ROOT / "training_curves_val_r2.png"
plt.savefig(val_r2_plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {val_r2_plot_path}")


# -------------------------
# Optional: compact ranking
# -------------------------
rank_cols = ["test_loss", "test_R2", "best_val_loss", "best_val_R2"]
ranking = df[["arch"] + rank_cols].copy()

# lower is better
ranking["rank_test_loss"] = ranking["test_loss"].rank(ascending=True, method="min")
ranking["rank_best_val_loss"] = ranking["best_val_loss"].rank(ascending=True, method="min")

# higher is better
ranking["rank_test_R2"] = ranking["test_R2"].rank(ascending=False, method="min")
ranking["rank_best_val_R2"] = ranking["best_val_R2"].rank(ascending=False, method="min")

ranking["rank_sum"] = (
    ranking["rank_test_loss"]
    + ranking["rank_best_val_loss"]
    + ranking["rank_test_R2"]
    + ranking["rank_best_val_R2"]
)

ranking = ranking.sort_values("rank_sum")
print("\n=== Ranking ===")
print(ranking[[
    "arch",
    "rank_test_loss",
    "rank_best_val_loss",
    "rank_test_R2",
    "rank_best_val_R2",
    "rank_sum"
]])