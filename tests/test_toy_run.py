import numpy as np
from pathlib import Path

from dcopf_gat.train import run_experiment


def test_toy_3bus_runs_end_to_end():
    data_dir = Path("data_toy_3bus")
    assert data_dir.exists(), "Toy dataset not found"

    model, history, (test_x, test_y), test_metrics = run_experiment(
        data_dir=str(data_dir),
        epochs=3,          # keep test FAST
        batch_size=8,
        learning_rate=1e-3,
    )

    # ---- basic sanity checks ----
    y_pred = model(test_x, training=False).numpy()

    assert y_pred.ndim == 2, "Model output must be 2D (N, D)"
    assert y_pred.shape[0] == test_y.shape[0], "Batch size mismatch"

    # model predicts 2 outputs by design
    assert y_pred.shape[1] == 2, "Unexpected number of outputs"

    # no NaNs / infs
    assert np.isfinite(y_pred).all(), "Model output contains NaNs or infs"

    # training history exists
    assert len(history.history) > 0, "No training history recorded"
