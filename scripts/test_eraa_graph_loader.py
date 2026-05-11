from dcopf_gat.data_eraa import load_eraa_graph_dataset


def main():
    ds = load_eraa_graph_dataset("data_eraa_ml", max_samples=10)

    print("X_nodes_train:", ds.X_nodes_train.shape)
    print("y_edges_train:", ds.y_edges_train.shape)
    print("X_nodes_val:  ", ds.X_nodes_val.shape)
    print("y_edges_val:  ", ds.y_edges_val.shape)
    print("X_nodes_test: ", ds.X_nodes_test.shape)
    print("y_edges_test: ", ds.y_edges_test.shape)

    print()
    print("edge_index:", ds.edge_index.shape)
    print("n_buses:", len(ds.bus_names))
    print("n_edges:", len(ds.edge_names))
    print("node features:", ds.node_feature_names)

    print()
    print("First edges:")
    for i in range(min(10, ds.edge_index.shape[1])):
        src = ds.bus_names[ds.edge_index[0, i]]
        dst = ds.bus_names[ds.edge_index[1, i]]
        name = ds.edge_names[i]
        print(f"{i:03d}: {name}: {src} -> {dst}")


if __name__ == "__main__":
    main()