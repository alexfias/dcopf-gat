from dcopf_gat.data_eraa import load_eraa_dataset


def main():
    ds = load_eraa_dataset("data_eraa_ml")

    print("X_train:", ds.X_train.shape)
    print("y_train:", ds.y_train.shape)
    print("X_val:  ", ds.X_val.shape)
    print("y_val:  ", ds.y_val.shape)
    print("X_test: ", ds.X_test.shape)
    print("y_test: ", ds.y_test.shape)

    print()
    print("X mean after normalization:", ds.X_train.mean())
    print("X std after normalization: ", ds.X_train.std())
    print("y mean after normalization:", ds.y_train.mean())
    print("y std after normalization: ", ds.y_train.std())


if __name__ == "__main__":
    main()