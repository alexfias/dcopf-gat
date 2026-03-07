from __future__ import annotations

"""
Utilities to export a solved PyPSA network into the CSV layout currently
expected by dcopf_gat.data.load_raw_data().

This exporter writes the following files into an output directory:

    - buses.csv
    - nodes_orig.csv
    - links.csv
    - generators.csv
    - generators_t_p.csv
    - p_max_pu.csv
    - loads-p_set.csv
    - linkf.csv
    - stores_t_e.csv

Design notes
------------
1. The current ML pipeline expects a "link-style" network description
   (bus0, bus1, p_nom, efficiency) and a corresponding flow time series
   in `linkf.csv`. Since PyPSA networks may contain both `links` and `lines`,
   this exporter maps both components into a single unified "links.csv".

2. Generators are exported at generator level, not aggregated by bus.
   Bus aggregation is already handled later by `dcopf_gat.data.prepare_dataset()`.

3. Storage information is exported to `stores_t_e.csv` using both:
      - `stores_t.e` (for Store components)
      - `storage_units_t.state_of_charge` (for StorageUnit components)
   Columns are named `<component_name>@<bus>` so they can be mapped back
   to nodes by the downstream loader.

4. For an ERAA-like zonal model, the typical usage is:
      - one bus per country
      - `nodes_orig.csv` == all buses
      - interconnectors exported as unified links
      - hydro storage exported via stores/storage_units

This module does NOT solve the network. It assumes the network is already built,
and ideally already solved if you want dispatch and flow outputs.
"""

from pathlib import Path
from typing import Iterable, Sequence
import json

import numpy as np
import pandas as pd


def _copy_df(df: pd.DataFrame | None, index_name: str | None = "name") -> pd.DataFrame:
    if df is None:
        out = pd.DataFrame()
    else:
        out = df.copy()
    if index_name is not None:
        out.index.name = index_name
    return out


def _as_datetime_index(index: pd.Index) -> pd.Index:
    try:
        return pd.DatetimeIndex(index)
    except Exception:
        return index


def _normalize_time_index(df: pd.DataFrame, snapshots: pd.Index) -> pd.DataFrame:
    out = df.copy()
    out.index = _as_datetime_index(snapshots)
    return out


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _empty_timeframe(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=_as_datetime_index(index))


def _generator_availability_table(n) -> pd.DataFrame:
    """
    Build p_max_pu.csv with one column per generator.

    Preference order:
      1. n.generators_t.p_max_pu if present
      2. static n.generators.p_max_pu broadcast in time
      3. all ones
    """
    snapshots = _as_datetime_index(n.snapshots)
    generators = list(n.generators.index)

    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p_max_pu"):
        df = n.generators_t.p_max_pu.copy()
        df = df.reindex(index=snapshots, columns=generators)
        return df.astype(float).fillna(1.0)

    if "p_max_pu" in n.generators.columns:
        vals = n.generators["p_max_pu"].astype(float).reindex(generators).fillna(1.0).to_numpy()
        arr = np.repeat(vals[None, :], len(snapshots), axis=0)
        return pd.DataFrame(arr, index=snapshots, columns=generators)

    arr = np.ones((len(snapshots), len(generators)), dtype=float)
    return pd.DataFrame(arr, index=snapshots, columns=generators)


def _generator_dispatch_table(n) -> pd.DataFrame:
    """
    Build generators_t_p.csv with one column per generator.
    """
    snapshots = _as_datetime_index(n.snapshots)
    generators = list(n.generators.index)

    if not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        raise ValueError("Network does not contain generators_t.p. Solve the network first.")

    df = n.generators_t.p.copy()
    df = df.reindex(index=snapshots, columns=generators)
    return df.astype(float).fillna(0.0)


def _load_table(n) -> pd.DataFrame:
    """
    Build loads-p_set.csv with columns `<bus> total_demand`.
    Loads connected to the same bus are summed.
    """
    snapshots = _as_datetime_index(n.snapshots)
    buses = list(n.buses.index)

    if len(n.loads.index) == 0:
        cols = [f"{bus} total_demand" for bus in buses]
        return pd.DataFrame(0.0, index=snapshots, columns=cols)

    if hasattr(n, "loads_t") and hasattr(n.loads_t, "p_set") and not n.loads_t.p_set.empty:
        loads_t = n.loads_t.p_set.copy()
        loads_t = loads_t.reindex(index=snapshots, columns=n.loads.index).fillna(0.0)
    else:
        static = n.loads.get("p_set", pd.Series(0.0, index=n.loads.index)).astype(float)
        arr = np.repeat(static.to_numpy()[None, :], len(snapshots), axis=0)
        loads_t = pd.DataFrame(arr, index=snapshots, columns=n.loads.index)

    bus_map = n.loads["bus"].copy()
    loads_by_bus = loads_t.T
    loads_by_bus["bus"] = bus_map
    loads_by_bus = loads_by_bus.groupby("bus").sum().T
    loads_by_bus = loads_by_bus.reindex(columns=buses, fill_value=0.0)

    loads_by_bus.columns = [f"{bus} total_demand" for bus in loads_by_bus.columns]
    return loads_by_bus.astype(float)


def _lines_to_link_table(n) -> pd.DataFrame:
    if len(n.lines.index) == 0:
        return pd.DataFrame(columns=["bus0", "bus1", "p_nom", "efficiency", "component", "original_name"])

    lines = n.lines.copy()
    _require_columns(lines, ["bus0", "bus1"], "n.lines")

    if "s_nom" in lines.columns:
        p_nom = lines["s_nom"].astype(float)
    elif "p_nom" in lines.columns:
        p_nom = lines["p_nom"].astype(float)
    else:
        p_nom = pd.Series(1.0, index=lines.index, dtype=float)

    out = pd.DataFrame(
        {
            "bus0": lines["bus0"],
            "bus1": lines["bus1"],
            "p_nom": p_nom,
            "efficiency": 1.0,
            "component": "Line",
            "original_name": lines.index,
        },
        index=[f"line::{name}" for name in lines.index],
    )
    out.index.name = "name"
    return out


def _links_to_link_table(n) -> pd.DataFrame:
    if len(n.links.index) == 0:
        return pd.DataFrame(columns=["bus0", "bus1", "p_nom", "efficiency", "component", "original_name"])

    links = n.links.copy()
    _require_columns(links, ["bus0", "bus1"], "n.links")

    if "p_nom" in links.columns:
        p_nom = links["p_nom"].astype(float)
    else:
        p_nom = pd.Series(1.0, index=links.index, dtype=float)

    efficiency = links["efficiency"].astype(float) if "efficiency" in links.columns else 1.0

    out = pd.DataFrame(
        {
            "bus0": links["bus0"],
            "bus1": links["bus1"],
            "p_nom": p_nom,
            "efficiency": efficiency,
            "component": "Link",
            "original_name": links.index,
        },
        index=[f"link::{name}" for name in links.index],
    )
    out.index.name = "name"
    return out


def _combined_link_table(n) -> pd.DataFrame:
    tables = [_lines_to_link_table(n), _links_to_link_table(n)]
    tables = [t for t in tables if not t.empty]
    if not tables:
        out = pd.DataFrame(columns=["bus0", "bus1", "p_nom", "efficiency", "component", "original_name"])
        out.index.name = "name"
        return out

    out = pd.concat(tables, axis=0)
    out.index.name = "name"
    return out


def _line_flow_table(n) -> pd.DataFrame:
    snapshots = _as_datetime_index(n.snapshots)

    if len(n.lines.index) == 0:
        return _empty_timeframe(snapshots)

    if not hasattr(n, "lines_t") or not hasattr(n.lines_t, "p0"):
        raise ValueError("Network contains lines but not lines_t.p0. Solve the network first.")

    df = n.lines_t.p0.copy()
    df = df.reindex(index=snapshots, columns=n.lines.index).fillna(0.0)
    df.columns = [f"line::{name}" for name in df.columns]
    return df.astype(float)


def _link_flow_table(n) -> pd.DataFrame:
    snapshots = _as_datetime_index(n.snapshots)

    if len(n.links.index) == 0:
        return _empty_timeframe(snapshots)

    if not hasattr(n, "links_t") or not hasattr(n.links_t, "p0"):
        raise ValueError("Network contains links but not links_t.p0. Solve the network first.")

    df = n.links_t.p0.copy()
    df = df.reindex(index=snapshots, columns=n.links.index).fillna(0.0)
    df.columns = [f"link::{name}" for name in df.columns]
    return df.astype(float)


def _combined_flow_table(n, link_index: pd.Index) -> pd.DataFrame:
    """
    Build linkf.csv matching the row order of the unified links.csv table.
    Positive sign follows the PyPSA convention of p0:
      power withdrawn at bus0.
    This is consistent with the downstream incidence-matrix logic.
    """
    snapshots = _as_datetime_index(n.snapshots)
    parts = [_line_flow_table(n), _link_flow_table(n)]
    parts = [p for p in parts if not p.empty]

    if not parts:
        return pd.DataFrame(index=snapshots, columns=link_index, dtype=float).fillna(0.0)

    out = pd.concat(parts, axis=1)
    out = out.reindex(index=snapshots, columns=link_index).fillna(0.0)
    return out.astype(float)


def _storage_energy_table(n) -> pd.DataFrame:
    """
    Build stores_t_e.csv from both Store and StorageUnit components.

    Output columns are named `<name>@<bus>`.
    """
    snapshots = _as_datetime_index(n.snapshots)
    out = pd.DataFrame(index=snapshots)

    if len(n.stores.index) > 0:
        if not hasattr(n, "stores_t") or not hasattr(n.stores_t, "e"):
            raise ValueError("Network contains stores but not stores_t.e.")
        e = n.stores_t.e.copy().reindex(index=snapshots, columns=n.stores.index).fillna(0.0)
        e.columns = [f"{name}@{n.stores.loc[name, 'bus']}" for name in e.columns]
        out = pd.concat([out, e.astype(float)], axis=1)

    if len(n.storage_units.index) > 0:
        if not hasattr(n, "storage_units_t") or not hasattr(n.storage_units_t, "state_of_charge"):
            raise ValueError("Network contains storage_units but not storage_units_t.state_of_charge.")
        soc = n.storage_units_t.state_of_charge.copy()
        soc = soc.reindex(index=snapshots, columns=n.storage_units.index).fillna(0.0)
        soc.columns = [f"{name}@{n.storage_units.loc[name, 'bus']}" for name in soc.columns]
        out = pd.concat([out, soc.astype(float)], axis=1)

    return out


def export_pypsa_csv_bundle(
    n,
    out_dir: str | Path,
    *,
    nodes_orig: Iterable[str] | None = None,
    include_lines: bool = True,
    include_links: bool = True,
    write_metadata: bool = True,
) -> dict[str, Path]:
    """
    Export a solved PyPSA network into the CSV bundle expected by the current
    dcopf_gat data loader.

    Parameters
    ----------
    n :
        A PyPSA Network object.
    out_dir :
        Output directory.
    nodes_orig :
        Optional iterable specifying the bus order to use in nodes_orig.csv.
        By default all buses are exported in the existing network order.
    include_lines :
        If True, map `n.lines` into the unified links.csv / linkf.csv.
    include_links :
        If True, map `n.links` into the unified links.csv / linkf.csv.
    write_metadata :
        If True, also write export_metadata.json.

    Returns
    -------
    dict[str, Path]
        Mapping from logical artifact name to written file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buses = _copy_df(n.buses)
    if nodes_orig is None:
        node_names = list(buses.index)
    else:
        node_names = list(nodes_orig)

    missing_nodes = sorted(set(node_names) - set(buses.index))
    if missing_nodes:
        raise ValueError(f"nodes_orig contains buses not present in n.buses: {missing_nodes}")

    buses = buses.reindex(index=node_names)
    nodes_orig_df = pd.DataFrame({"name": node_names})

    generators = _copy_df(n.generators)
    _require_columns(generators, ["bus", "p_nom"], "n.generators")

    generators_t_p = _generator_dispatch_table(n)
    p_max_pu = _generator_availability_table(n)
    loads_p_set = _load_table(n)

    # Unified link/flow export
    line_table = _lines_to_link_table(n) if include_lines else pd.DataFrame()
    link_table = _links_to_link_table(n) if include_links else pd.DataFrame()
    tables = [t for t in [line_table, link_table] if not t.empty]

    if tables:
        links = pd.concat(tables, axis=0)
        links.index.name = "name"
    else:
        links = pd.DataFrame(columns=["bus0", "bus1", "p_nom", "efficiency", "component", "original_name"])
        links.index.name = "name"

    # Keep only the columns used by the current downstream pipeline in links.csv.
    links_csv = links.reindex(columns=["bus0", "bus1", "p_nom", "efficiency"]).copy()

    flow_parts = []
    if include_lines:
        flow_parts.append(_line_flow_table(n))
    if include_links:
        flow_parts.append(_link_flow_table(n))

    if flow_parts:
        linkf = pd.concat(flow_parts, axis=1).reindex(index=_as_datetime_index(n.snapshots), columns=links.index).fillna(0.0)
    else:
        linkf = pd.DataFrame(index=_as_datetime_index(n.snapshots), columns=links.index, dtype=float).fillna(0.0)

    stores_t_e = _storage_energy_table(n)

    files = {
        "buses": out_dir / "buses.csv",
        "nodes_orig": out_dir / "nodes_orig.csv",
        "links": out_dir / "links.csv",
        "generators": out_dir / "generators.csv",
        "generators_t_p": out_dir / "generators_t_p.csv",
        "p_max_pu": out_dir / "p_max_pu.csv",
        "loads_p_set": out_dir / "loads-p_set.csv",
        "linkf": out_dir / "linkf.csv",
        "stores_t_e": out_dir / "stores_t_e.csv",
    }

    buses.to_csv(files["buses"])
    nodes_orig_df.to_csv(files["nodes_orig"], index=False)
    links_csv.to_csv(files["links"])
    generators.to_csv(files["generators"])
    generators_t_p.to_csv(files["generators_t_p"])
    p_max_pu.to_csv(files["p_max_pu"])
    loads_p_set.to_csv(files["loads_p_set"])
    linkf.to_csv(files["linkf"])
    stores_t_e.to_csv(files["stores_t_e"])

    if write_metadata:
        meta = {
            "num_snapshots": int(len(n.snapshots)),
            "num_buses": int(len(buses.index)),
            "num_generators": int(len(generators.index)),
            "num_unified_links": int(len(links_csv.index)),
            "num_stores_columns": int(stores_t_e.shape[1]),
            "nodes_orig": node_names,
            "included_components": {
                "lines": bool(include_lines),
                "links": bool(include_links),
                "stores": bool(len(n.stores.index) > 0),
                "storage_units": bool(len(n.storage_units.index) > 0),
            },
        }
        meta_path = out_dir / "export_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        files["metadata"] = meta_path

    return files


def summarize_exportable_network(n) -> dict[str, int]:
    """
    Small helper for quick inspection before export.
    """
    return {
        "snapshots": int(len(n.snapshots)),
        "buses": int(len(n.buses.index)),
        "generators": int(len(n.generators.index)),
        "loads": int(len(n.loads.index)),
        "lines": int(len(n.lines.index)),
        "links": int(len(n.links.index)),
        "stores": int(len(n.stores.index)),
        "storage_units": int(len(n.storage_units.index)),
    }


__all__ = [
    "export_pypsa_csv_bundle",
    "summarize_exportable_network",
]
