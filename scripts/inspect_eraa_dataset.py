from pathlib import Path
import json
import numpy as np


def describe_array(name, x):
    print(f"{name:14s} shape={x.shape}, nan={np.isnan(x).sum()}, "
          f"min={np.nanmin(x):.3f}, max={np.nanmax(x):.3f}, mean={np.nanmean(x):.3f}")


def main():
    data_dir = Path("data_eraa_ml")

    meta = json.loads((data_dir / "metadata.json").read_text())
    print(f"Samples: {meta['n_samples']}")
    print(f"Hours/sample: {meta['hours_per_sample']}")
    print()

    first = data_dir / meta["samples"][0]["file"]
    print(f"Inspecting: {first.name}")

    z = np.load(first)

    for key in z.files:
        describe_array(key, z[key])

    print()
    print("Column counts:")
    for key, cols in meta["columns"].items():
        print(f"{key:14s} {len(cols)}")


if __name__ == "__main__":
    main()