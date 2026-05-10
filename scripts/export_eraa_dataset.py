# scripts/export_eraa_dataset.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


def aggregate_generators_by_bus_carrier(n: pypsa.Network) -> pd.DataFrame:
    """Return dispatch aggregated to bus-carrier time series."""
    p = n.generators_t.p

    meta = n.generators[["bus", "carrier"]].copy()
    cols = pd.MultiIndex.from_frame(meta.loc[p.columns, ["bus", "carrier"]])

    out = p.copy()
    out.columns = cols

    return out.groupby(level=[0, 1], axis=1).sum()


def aggregate_pmax_by_bus_carrier(n: pypsa.Network) -> pd.DataFrame:
    """Return availability/capacity signal aggregated to bus-carrier."""
    p_nom = n.generators.p_nom.reindex(n.generators.index).fillna(0.0)

    if hasattr(n, "generators_t") and "p_max_pu" in n.generators_t:
        pmax_pu = n.generators_t.p_max_pu.reindex(columns=n.generators.index).fillna(1.0)
    else:
        pmax_pu = pd.DataFrame(
            1.0, index=n.snapshots, columns=n.generators.index
        )

    avail = pmax_pu.multiply(p_nom, axis=1)

    meta = n.generators[["bus", "carrier"]].copy()
    cols = pd.MultiIndex.from_frame(meta.loc[avail.columns, ["bus", "carrier"]])
    avail.columns = cols

    return avail.groupby(level=[0, 1], axis=1).sum()


def get_load_by_bus(n: pypsa.Network) -> pd.DataFrame:
    loads = n.loads_t.p_set.reindex(columns=n.loads.index).fillna(0.0)
    bus_map = n.loads.bus
    loads.columns = bus_map.loc[loads.columns].values
    return loads.groupby(axis=1, level=0).sum()


def get_branch_flows(n: pypsa.Network) -> pd.DataFrame:
    """Prefer links if ERAA uses interconnectors as links; include lines if present."""
    parts = []

    if len(n.lines) and hasattr(n, "lines_t") and "p0" in n.lines_t:
        x = n.lines_t.p0.copy()
        x.columns = [f"Line::{c}" for c in x.columns]
        parts.append(x)

    if len(n.links) and hasattr(n, "links_t") and "p0" in n.links_t:
        x = n.links_t.p0.copy()
        x.columns = [f"Link::{c}" for c in x.columns]
        parts.append(x)

    if not parts:
        raise ValueError("No line/link flow time series found.")

    return pd.concat(parts, axis=1)


def get_storage_soc(n: pypsa.Network) -> pd.DataFrame:
    parts = []

    if len(n.storage_units) and hasattr(n, "storage_units_t") and "state_of_charge" in n.storage_units_t:
        x = n.storage_units_t.state_of_charge.copy()
        x.columns = [f"StorageUnit::{c}" for c in x.columns]
        parts.append(x)

    if len(n.stores) and hasattr(n, "stores_t") and "e" in n.stores_t:
        x = n.stores_t.e.copy()
        x.columns = [f"Store::{c}" for c in x.columns]
        parts.append(x)

    if not parts:
        return pd.DataFrame(index=n.snapshots)

    return pd.concat(parts, axis=1)


def topology_signature(n: pypsa.Network) -> dict:
    return {
        "buses": list(n.buses.index),
        "lines": list(n.lines.index),
        "links": list(n.links.index),
        "generators": list(n.generators.index),
        "loads": list(n.loads.index),
        "storage_units": list(n.storage_units.index),
        "stores": list(n.stores.index),
    }


def export_one_network(path: Path, out_dir: Path, reference_signature: dict | None) -> dict:
    print(f"Reading {path}")
    n = pypsa.Network(path)

    sig = topology_signature(n)
    if reference_signature is not None and sig != reference_signature:
        raise ValueError(f"Topology mismatch in {path}")

    snapshots = pd.DatetimeIndex(n.snapshots)

    load = get_load_by_bus(n)
    gen_avail = aggregate_pmax_by_bus_carrier(n)
    gen_dispatch = aggregate_generators_by_bus_carrier(n)
    flows = get_branch_flows(n)
    soc = get_storage_soc(n)

    sample_meta = []

    # Weekly groups, aligned by calendar week
    for week_id, idx in load.groupby([snapshots.year, snapshots.isocalendar().week]).groups.items():
        idx = list(idx)

        if len(idx) != 168:
            continue

        sample_name = f"{path.stem}_y{week_id[0]}_w{int(week_id[1]):02d}"
        sample_path = out_dir / f"{sample_name}.npz"

        load_w = load.loc[idx]
        gen_avail_w = gen_avail.loc[idx]
        gen_dispatch_w = gen_dispatch.loc[idx]
        flows_w = flows.loc[idx]
        soc_w = soc.loc[idx]

        np.savez_compressed(
            sample_path,
            load=load_w.to_numpy(dtype=np.float32),
            gen_avail=gen_avail_w.to_numpy(dtype=np.float32),
            gen_dispatch=gen_dispatch_w.to_numpy(dtype=np.float32),
            flows=flows_w.to_numpy(dtype=np.float32),
            soc=soc_w.to_numpy(dtype=np.float32),
        )

        sample_meta.append(
            {
                "sample": sample_name,
                "source_file": path.name,
                "start": str(load_w.index[0]),
                "end": str(load_w.index[-1]),
                "n_hours": len(idx),
                "file": sample_path.name,
            }
        )

    return {
        "signature": sig,
        "samples": sample_meta,
        "columns": {
            "load": list(load.columns),
            "gen_avail": [list(c) for c in gen_avail.columns],
            "gen_dispatch": [list(c) for c in gen_dispatch.columns],
            "flows": list(flows.columns),
            "soc": list(soc.columns),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with solved yearly PyPSA .nc files")
    parser.add_argument("--output_dir", required=True, help="Output dataset folder")
    parser.add_argument("--pattern", default="*.nc")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} with pattern {args.pattern}")

    reference_signature = None
    all_samples = []
    columns = None

    for i, path in enumerate(files):
        result = export_one_network(path, out_dir, reference_signature)

        if i == 0:
            reference_signature = result["signature"]
            columns = result["columns"]

        all_samples.extend(result["samples"])

    metadata = {
        "n_source_networks": len(files),
        "n_samples": len(all_samples),
        "hours_per_sample": 168,
        "samples": all_samples,
        "columns": columns,
        "topology_signature": reference_signature,
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {len(all_samples)} weekly samples to {out_dir}")


if __name__ == "__main__":
    main()