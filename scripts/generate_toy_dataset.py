import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # Where to store the synthetic data
    out_dir = Path("data_toy_3bus")
    out_dir.mkdir(exist_ok=True)

    # Basic dimensions
    T = 100          # time steps
    buses = ["Bus0", "Bus1", "Bus2"]
    N = len(buses)
    links = ["L0", "L1"]
    L = len(links)

    # --- BUSES ---
    # Minimal buses.csv: we only really use the index (bus names)
    buses_df = pd.DataFrame(index=buses)
    buses_df.index.name = "name"
    buses_df.to_csv(out_dir / "buses.csv")

    # --- NODES_ORIG ---
    # Just take all buses as "nodes_orig"
    nodes_orig_df = pd.DataFrame({"name": buses})
    nodes_orig_df.to_csv(out_dir / "nodes_orig.csv", index=False)

    # --- LINKS ---
    # Chain: Bus0 -- Bus1 -- Bus2
    links_df = pd.DataFrame(
        {
            "bus0": ["Bus0", "Bus1"],
            "bus1": ["Bus1", "Bus2"],
            "p_nom": [100.0, 100.0],
            "efficiency": [1.0, 1.0],
        },
        index=links,
    )
    links_df.index.name = "name"
    links_df.to_csv(out_dir / "links.csv")

    # --- GENERATORS ---
    # One generator per bus
    gen_names = ["G0", "G1", "G2"]
    gen_df = pd.DataFrame(
        {
            "bus": buses,
            "p_nom": [100.0, 100.0, 100.0],
            "carrier": ["wind", "wind", "wind"],
        },
        index=gen_names,
    )
    gen_df.index.name = "name"
    gen_df.to_csv(out_dir / "generators.csv")

    # --- TIME INDEX ---
    time_index = pd.date_range("2020-01-01", periods=T, freq="H")

    # --- DEMAND ---
    # loads-p_set.csv: columns like "BusX total_demand"
    rng = np.random.default_rng(42)
    demand = rng.uniform(30, 60, size=(T, N))  # MW-ish
    demand_cols = [f"{b} total_demand" for b in buses]
    demand_df = pd.DataFrame(demand, index=time_index, columns=demand_cols)
    demand_df.to_csv(out_dir / "loads-p_set.csv")

    # --- FLOWS + GENERATION (consistent with nodal balance) ---
    # Orientation:
    #   L0: Bus0 (withdraw) -> Bus1 (inject)
    #   L1: Bus1 (withdraw) -> Bus2 (inject)
    #
    # For each t:
    #   gen0 = demand0 + f0
    #   gen1 = demand1 + f1 - f0
    #   gen2 = demand2 - f1
    #
    # So if we choose flows and demand, we can compute generation.

    flows = np.zeros((T, L))
    gen = np.zeros((T, N))

    for t in range(T):
        d0, d1, d2 = demand[t]

        # small-ish flows relative to demand, so generation stays positive
        f0 = rng.uniform(-10, 10)
        f1 = rng.uniform(-10, 10)

        g0 = d0 + f0
        g1 = d1 + f1 - f0
        g2 = d2 - f1

        # simple clipping: if any generator goes negative, just zero flows
        if g0 < 0 or g1 < 0 or g2 < 0 or g0 > 100 or g1 > 100 or g2 > 100:
            f0 = 0.0
            f1 = 0.0
            g0 = d0
            g1 = d1
            g2 = d2

        flows[t, :] = [f0, f1]
        gen[t, :] = [g0, g1, g2]

    # generators_t_p.csv: time series per generator
    gen_ts = pd.DataFrame(gen, index=time_index, columns=gen_names)
    gen_ts.to_csv(out_dir / "generators_t_p.csv")

    # linkf.csv: time series per link
    flow_ts = pd.DataFrame(flows, index=time_index, columns=links)
    flow_ts.to_csv(out_dir / "linkf.csv")

    # --- WEATHER / p_max_pu ---
    # Very simple: capacity factors in [0, 1] for each generator
    cf = rng.uniform(0.2, 1.0, size=(T, len(gen_names)))
    cf_df = pd.DataFrame(cf, index=time_index, columns=gen_names)
    cf_df.to_csv(out_dir / "p_max_pu.csv")

    # --- STORES (dummy) ---
    # stores_t_e.csv: not used by our pipeline, but we create an empty / zero one
    stores_df = pd.DataFrame(
        np.zeros((T, 1)),
        index=time_index,
        columns=["Store0"],
    )
    stores_df.to_csv(out_dir / "stores_t_e.csv")

    print(f"Toy dataset written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
