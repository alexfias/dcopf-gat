# make_toy_ieee14.py
import numpy as np
import pandas as pd
from pathlib import Path
from dcopf_gat.dcopf import DcOpfTopology, solve_dcopf_ptdf




def build_bbus(num_buses: int, branches, base_mva: float = 100.0) -> np.ndarray:
    """
    Build DC power flow Bbus matrix:
      P = Bbus * theta
    where P in MW and theta in radians (scaled by base_mva/x).
    branches: list of (bus_i, bus_j, x_pu)
    buses are 1-indexed in branches.
    """
    B = np.zeros((num_buses, num_buses), dtype=float)
    for (i, j, x) in branches:
        if x <= 0:
            raise ValueError(f"Invalid branch reactance x={x} for ({i},{j}).")
        b = base_mva / x
        ii = i - 1
        jj = j - 1
        B[ii, ii] += b
        B[jj, jj] += b
        B[ii, jj] -= b
        B[jj, ii] -= b
    return B


def build_incidence(num_buses: int, branches) -> np.ndarray:
    """
    Incidence matrix A (n_l, n_b): each row has +1 at bus0 and -1 at bus1
    consistent with your compute_dc_pf_flows sign convention (bus0 -> bus1).
    branches are 1-indexed.
    """
    L = len(branches)
    A = np.zeros((L, num_buses), dtype=float)
    for ell, (i, j, x) in enumerate(branches):
        A[ell, i - 1] = 1.0
        A[ell, j - 1] = -1.0
    return A


def build_gen_map(num_buses: int) -> np.ndarray:
    """
    One generator per bus => G is identity (n_b, n_g) with n_g = n_b.
    If later you have multiple gens per bus, change accordingly.
    """
    return np.eye(num_buses, dtype=float)



def compute_dc_pf_flows(theta: np.ndarray, branches, base_mva: float = 100.0) -> np.ndarray:
    """
    Compute branch flows f_ij = base_mva/x * (theta_i - theta_j) in MW,
    with sign convention from bus0 -> bus1 as given in branches list.
    """
    flows = []
    for (i, j, x) in branches:
        b = base_mva / x
        fi = b * (theta[i - 1] - theta[j - 1])
        flows.append(fi)
    return np.asarray(flows, dtype=float)


def main():
    out_dir = Path("data_ieee14")
    out_dir.mkdir(exist_ok=True)

    mode = "both"   # "pf" | "opf" | "both"

    out_dir = Path("data_ieee14")
    if mode == "pf":
        out_dir = Path("data_ieee14_pf")
    elif mode == "opf":
        out_dir = Path("data_ieee14_opf")
    elif mode == "both":
        out_dir = Path("data_ieee14_both")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_dir.mkdir(exist_ok=True)

    # ----------------------------
    # Basic dimensions
    # ----------------------------
    T = 5000
    base_mva = 100.0

    buses = [f"Bus{i}" for i in range(1, 15)]
    N = len(buses)

    # IEEE-14 standard branch reactances (x) in p.u. (commonly used set)
    # Each tuple: (bus0, bus1, x_pu)
    branches = [
        (1, 2, 0.05917),
        (1, 5, 0.22304),
        (2, 3, 0.19797),
        (2, 4, 0.17632),
        (2, 5, 0.17388),
        (3, 4, 0.17103),
        (4, 5, 0.04211),
        (4, 7, 0.20912),
        (4, 9, 0.55618),
        (5, 6, 0.25202),
        (6, 11, 0.19890),
        (6, 12, 0.25581),
        (6, 13, 0.13027),
        (7, 8, 0.17615),
        (7, 9, 0.11001),
        (9, 10, 0.08450),
        (9, 14, 0.27038),
        (10, 11, 0.19207),
        (12, 13, 0.19988),
        (13, 14, 0.34802),
    ]
    L = len(branches)
    link_names = [f"L{k}" for k in range(L)]

    # ----------------------------
    # Write buses.csv
    # ----------------------------
    buses_df = pd.DataFrame(index=buses)
    buses_df.index.name = "name"
    buses_df.to_csv(out_dir / "buses.csv")

    # ----------------------------
    # nodes_orig.csv (no index, column "name")
    # ----------------------------
    nodes_orig_df = pd.DataFrame({"name": buses})
    nodes_orig_df.to_csv(out_dir / "nodes_orig.csv", index=False)

    # ----------------------------
    # links.csv
    # ----------------------------
    # We set p_nom to a consistent thermal limit for normalization
    # If you want tighter constraints, reduce p_nom (e.g. 50).
    p_nom_link = 50.0
    links_df = pd.DataFrame(
        {
            "bus0": [f"Bus{i}" for (i, j, x) in branches],
            "bus1": [f"Bus{j}" for (i, j, x) in branches],
            "p_nom": [p_nom_link] * L,
            "efficiency": [1.0] * L,
        },
        index=link_names,
    )
    links_df.index.name = "name"
    links_df.to_csv(out_dir / "links.csv")

    # ----------------------------
    # generators.csv: one generator per bus
    # ----------------------------
    gen_names = [f"G{i}" for i in range(1, 15)]
    # keep same p_nom everywhere for toy; can vary later
    p_nom_gen = np.full(N, 100.0, dtype=float)

    gen_df = pd.DataFrame(
        {
            "bus": buses,
            "p_nom": p_nom_gen,
            "carrier": ["wind"] * N,
        },
        index=gen_names,
    )
    gen_df.index.name = "name"
    gen_df.to_csv(out_dir / "generators.csv")

    # ----------------------------
    # time index
    # ----------------------------
    time_index = pd.date_range("2020-01-01", periods=T, freq="h")

    # ----------------------------
    # demand: loads-p_set.csv columns must be "{bus} total_demand"
    # ----------------------------
    rng = np.random.default_rng(42)
    # slightly wider demand range than 3-bus toy
    demand = rng.uniform(10, 60, size=(T, N)).astype(float)
    demand_cols = [f"{b} total_demand" for b in buses]
    demand_df = pd.DataFrame(demand, index=time_index, columns=demand_cols)
    demand_df.to_csv(out_dir / "loads-p_set.csv")

    # ----------------------------
    # weather: p_max_pu.csv (capacity factor per generator)
    # IMPORTANT: your pipeline treats this as global features (broadcast to nodes),
    # but it still works and provides learnable signal.
    # ----------------------------
    cf = rng.uniform(0.2, 1.0, size=(T, N)).astype(float)
    cf_df = pd.DataFrame(cf, index=time_index, columns=gen_names)
    cf_df.to_csv(out_dir / "p_max_pu.csv")

    # ----------------------------
    # Build DC PF matrices
    # ----------------------------
    Bbus = build_bbus(N, branches, base_mva=base_mva)
    slack = 0  # Bus1 is slack (0-index)

    # reduced system (remove slack)
    mask = np.ones(N, dtype=bool)
    mask[slack] = False
    Bred = Bbus[np.ix_(mask, mask)]

    # ----------------------------
    # Build DC OPF (PTDF) matrices (fixed topology)
    # ----------------------------
    A = build_incidence(N, branches)  # (L, N)
    b = np.array([base_mva / x for (_, _, x) in branches], dtype=float)  # (L,)
    f_max = np.full(L, p_nom_link, dtype=float)  # (L,)

    topo = DcOpfTopology(n_b=N, n_l=L, slack=slack, A=A, b=b, f_max=f_max)
    PTDF = topo.ptdf()

    # one generator per bus in your toy data
    G = build_gen_map(N)  # (N, N)

    # DC-OPF costs and pmin
    pmin = np.zeros(N, dtype=float)
    c = np.linspace(10.0, 50.0, N).astype(float)  # simple fixed merit order



    # ----------------------------
    # Generate samples
    # ----------------------------
    if mode in ("pf", "both"):
        gen_pf = np.zeros((T, N), dtype=float)
        flows_pf = np.zeros((T, L), dtype=float)

    if mode in ("opf", "both"):
        gen_opf = np.zeros((T, N), dtype=float)
        flows_opf = np.zeros((T, L), dtype=float)

    gmax = p_nom_gen

    for t in range(T):
        d = demand[t].copy()

        # ===== PF generation (your existing logic) =====
        if mode in ("pf", "both"):
            for _ in range(50):
                g = np.zeros(N, dtype=float)
                beta = rng.uniform(0.3, 0.95, size=N)

                for i in range(1, N):
                    g[i] = beta[i] * cf[t, i] * gmax[i]

                g[slack] = float(np.sum(d) - np.sum(g[1:]))

                if not (0.0 <= g[slack] <= gmax[slack]):
                    continue

                P = g - d
                Pred = P[mask]
                try:
                    thetared = np.linalg.solve(Bred, Pred)
                except np.linalg.LinAlgError:
                    continue

                theta = np.zeros(N, dtype=float)
                theta[mask] = thetared
                theta[slack] = 0.0

                f = compute_dc_pf_flows(theta, branches, base_mva=base_mva)

                if np.max(np.abs(f)) > p_nom_link:
                    continue

                gen_pf[t, :] = g
                flows_pf[t, :] = f
                break
            else:
                gen_pf[t, :] = 0.0
                gen_pf[t, slack] = float(np.sum(d))
                flows_pf[t, :] = 0.0

        # ===== OPF generation =====
        if mode in ("opf", "both"):
            # availability-limited max dispatch from cf * p_nom
            pmax_t = cf[t, :] * p_nom_gen

            try:
                pg, pinj, f = solve_dcopf_ptdf(
                    topo=topo,
                    PTDF=PTDF,
                    d=d,
                    G=G,
                    c=c,
                    pmin=pmin,
                    pmax=pmax_t,
                    include_line_limits=True,
                )

                # ---------- QUICK SANITY CHECK (run once) ----------
                if t == 0:
                    print("=== DC-OPF sanity check (t=0) ===")
                    print("Balance residual (sum pinj):", pinj.sum())
                    print("Total load:", d.sum(), "Total generation:", pg.sum())
                    print("Max line violation:", np.max(np.abs(f) - topo.f_max))
                    print("Max |flow|:", np.max(np.abs(f)))
                    print("================================")
                # --------------------------------------------------

                gen_opf[t, :] = pg
                flows_opf[t, :] = f
            except Exception:
                gen_opf[t, :] = 0.0
                gen_opf[t, slack] = float(np.sum(d))
                flows_opf[t, :] = 0.0


    # ----------------------------
    # Save outputs
    # ----------------------------
    if mode == "pf":
        pd.DataFrame(gen_pf, index=time_index, columns=gen_names).to_csv(out_dir / "generators_t_p.csv")
        pd.DataFrame(flows_pf, index=time_index, columns=link_names).to_csv(out_dir / "linkf.csv")

    elif mode == "opf":
        pd.DataFrame(gen_opf, index=time_index, columns=gen_names).to_csv(out_dir / "generators_t_p.csv")
        pd.DataFrame(flows_opf, index=time_index, columns=link_names).to_csv(out_dir / "linkf.csv")

    elif mode == "both":
        # keep the same topology files in this folder, but store time series separately
        pf_dir = out_dir / "pf"
        opf_dir = out_dir / "opf"
        pf_dir.mkdir(exist_ok=True)
        opf_dir.mkdir(exist_ok=True)

        pd.DataFrame(gen_pf, index=time_index, columns=gen_names).to_csv(pf_dir / "generators_t_p.csv")
        pd.DataFrame(flows_pf, index=time_index, columns=link_names).to_csv(pf_dir / "linkf.csv")

        pd.DataFrame(gen_opf, index=time_index, columns=gen_names).to_csv(opf_dir / "generators_t_p.csv")
        pd.DataFrame(flows_opf, index=time_index, columns=link_names).to_csv(opf_dir / "linkf.csv")


    # ----------------------------
    # stores_t_e.csv (dummy)
    # ----------------------------
    stores_df = pd.DataFrame(np.zeros((T, 1)), index=time_index, columns=["Store0"])
    stores_df.to_csv(out_dir / "stores_t_e.csv")

    print(f"IEEE14 toy dataset written to: {out_dir.resolve()}")
    print(f"T = {T}, N = {N}, L = {L}")


if __name__ == "__main__":
    main()
