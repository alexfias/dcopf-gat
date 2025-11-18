from dcopf_gat.train import run_experiment

if __name__ == "__main__":
    model, history, (test_x, test_y), test_metrics = run_experiment(
        data_dir="data_33nodes_future_wsto_3years",
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
    )
    print("Test metrics:", test_metrics)