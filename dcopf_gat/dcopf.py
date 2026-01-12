import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional

def solve_dcopf_storage_ptdf(
    topo,
    PTDF: np.ndarray,              # (n_l, n_b)
    d: np.ndarray,                 # (H, n_b)
    G: np.ndarray,                 # (n_b, n_g)
    c_gen: np.ndarray,             # (n_g,)
    pmin: np.ndarray,              # (n_g,)
    pmax: np.ndarray,              # (H, n_g)
    store_bus: int,                # bus index [0..n_b-1]
    E_max: float,
    P_ch_max: float,
    P_dis_max: float,
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
    e_init: float = 0.5,
    dt: float = 1.0,
    terminal_equal_init: bool = True,
    storage_cycle_cost: float = 0.0,  # small eps like 1e-4 if you want
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-period DC-OPF with 1 storage unit (battery) using PTDF line constraints.
    Returns:
      pg   (H, n_g)
      pinj (H, n_b)
      f    (H, n_l)
      e    (H, 1)    SoC
      p_st (H, 1)    net storage power = p_dis - p_ch
    """
    d = np.asarray(d, dtype=float)
    assert d.ndim == 2
    H, n_b = d.shape
    n_l = topo.n_l
    n_g = G.shape[1]
    assert PTDF.shape == (n_l, n_b)
    assert pmax.shape == (H, n_g)
    assert pmin.shape == (n_g,)
    assert c_gen.shape == (n_g,)

    # Mapping vector for storage injection at its bus
    s_bus = np.zeros(n_b, dtype=float)
    s_bus[store_bus] = 1.0
    k = PTDF @ s_bus            # (n_l,)
    Mgen = PTDF @ G             # (n_l, n_g)

    # Decision vector x stacks:
    # pg[t,g] for all t,g
    # pch[t]  for all t
    # pdis[t] for all t
    # e[t]    for all t
    n_pg = H * n_g
    n_pch = H
    n_pdis = H
    n_e = H
    n_x = n_pg + n_pch + n_pdis + n_e

    def idx_pg(t, g):   return t * n_g + g
    def idx_pch(t):     return n_pg + t
    def idx_pdis(t):    return n_pg + n_pch + t
    def idx_e(t):       return n_pg + n_pch + n_pdis + t

    # Objective: sum_t c_gen^T pg[t] + eps*(pch+pdis)
    c = np.zeros(n_x, dtype=float)
    for t in range(H):
        for g in range(n_g):
            c[idx_pg(t, g)] = c_gen[g]
        if storage_cycle_cost != 0.0:
            c[idx_pch(t)] = storage_cycle_cost
            c[idx_pdis(t)] = storage_cycle_cost

    # Equality constraints:
    Aeq = []
    beq = []

    # (1) System balance each time: sum(pg) + pdis - pch = sum(demand)
    for t in range(H):
        row = np.zeros(n_x, dtype=float)
        # sum over gens
        for g in range(n_g):
            row[idx_pg(t, g)] = 1.0
        row[idx_pdis(t)] = 1.0
        row[idx_pch(t)] = -1.0
        Aeq.append(row)
        beq.append(float(d[t].sum()))

    # (2) Storage dynamics:
    # e[t+1] - e[t] - eta_ch*pch[t]*dt + (1/eta_dis)*pdis[t]*dt = 0
    # Fix initial SoC: e[0] = e_init
    row = np.zeros(n_x, dtype=float)
    row[idx_e(0)] = 1.0
    Aeq.append(row)
    beq.append(float(e_init))

    for t in range(H - 1):
        row = np.zeros(n_x, dtype=float)
        row[idx_e(t + 1)] = 1.0
        row[idx_e(t)] = -1.0
        row[idx_pch(t)] = -eta_ch * dt
        row[idx_pdis(t)] = (1.0 / eta_dis) * dt
        Aeq.append(row)
        beq.append(0.0)

    # Optional terminal equality e[H-1] = e_init (prevents end-of-horizon dumping)
    if terminal_equal_init:
        row = np.zeros(n_x, dtype=float)
        row[idx_e(H - 1)] = 1.0
        Aeq.append(row)
        beq.append(float(e_init))

    Aeq = np.vstack(Aeq)
    beq = np.asarray(beq, dtype=float)

    # Inequality constraints (line limits) per time t:
    A_ub = []
    b_ub = []

    for t in range(H):
        h = PTDF @ d[t]  # (n_l,)

        # f = Mgen*pg + k*pdis - k*pch - h
        # +f <= fmax  ->  Mgen*pg - k*pch + k*pdis <= fmax + h
        # -f <= fmax  -> -Mgen*pg + k*pch - k*pdis <= fmax - h
        for sign in (+1, -1):
            # sign=+1 => +f<=fmax ; sign=-1 => -f<=fmax
            for ell in range(n_l):
                row = np.zeros(n_x, dtype=float)

                # Generator part: sign * Mgen[ell,:]
                for g in range(n_g):
                    row[idx_pg(t, g)] = sign * Mgen[ell, g]

                # Storage parts
                row[idx_pch(t)] = sign * (-k[ell])
                row[idx_pdis(t)] = sign * (k[ell])

                # RHS
                if sign == +1:
                    rhs = topo.f_max[ell] + h[ell]
                else:
                    rhs = topo.f_max[ell] - h[ell]

                A_ub.append(row)
                b_ub.append(float(rhs))

    A_ub = np.vstack(A_ub)
    b_ub = np.asarray(b_ub, dtype=float)

    # Bounds
    bounds = [None] * n_x

    for t in range(H):
        for g in range(n_g):
            bounds[idx_pg(t, g)] = (float(pmin[g]), float(pmax[t, g]))
        bounds[idx_pch(t)] = (0.0, float(P_ch_max))
        bounds[idx_pdis(t)] = (0.0, float(P_dis_max))
        bounds[idx_e(t)] = (0.0, float(E_max))

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"DCOPF+storage failed: {res.message}")

    x = res.x

    # Unpack
    pg = np.zeros((H, n_g), dtype=float)
    pch = np.zeros((H, 1), dtype=float)
    pdis = np.zeros((H, 1), dtype=float)
    e = np.zeros((H, 1), dtype=float)

    for t in range(H):
        for g in range(n_g):
            pg[t, g] = x[idx_pg(t, g)]
        pch[t, 0] = x[idx_pch(t)]
        pdis[t, 0] = x[idx_pdis(t)]
        e[t, 0] = x[idx_e(t)]

    p_st = pdis - pch  # net injection from storage at its bus

    # Compute injections and flows per time
    pinj = (pg @ G.T).astype(float)  # careful: pg is (H,n_g), G is (n_b,n_g)
    # pg @ G.T gives (H,n_b) if G is one-hot mapping (identity in your case)
    # General: pinj = (G @ pg[t]) - d[t] + s_bus*p_st[t]
    pinj = pinj - d
    pinj[:, store_bus] += p_st[:, 0]

    f = (PTDF @ pinj.T).T  # (H,n_l)

    return pg, pinj, f, e, p_st





@dataclass
class DcOpfTopology:
    n_b: int                 # number of buses
    n_l: int                 # number of lines
    slack: int               # slack bus index [0..n_b-1]
    A: np.ndarray            # incidence matrix (n_l, n_b) with +1/-1
    b: np.ndarray            # line susceptances 1/x (n_l,)
    f_max: np.ndarray        # thermal limits (n_l,)

    def ptdf(self) -> np.ndarray:
        """
        Returns PTDF of shape (n_l, n_b). Maps bus injections (sum=0)
        to line flows. Slack bus handled internally.
        """
        n_b = self.n_b
        slack = self.slack

        # Build Bbus = A^T * diag(b) * A
        Bbus = self.A.T @ (self.b[:, None] * self.A)

        # Remove slack row/col to make invertible
        keep = [i for i in range(n_b) if i != slack]
        Bred = Bbus[np.ix_(keep, keep)]

        # Invert reduced
        Bred_inv = np.linalg.inv(Bred)

        # Build full inverse-like mapping: theta_keep = Bred_inv * p_keep
        # where p_keep excludes slack injection (determined by balance)
        # Then theta_slack = 0
        # Finally f = diag(b) * A * theta
        # We want PTDF s.t. f = PTDF * p  (p includes slack component but sum=0)
        PTDF = np.zeros((self.n_l, n_b), dtype=float)

        # For each bus k, inject +1 at k and -1 at slack (so sum=0)
        for k in range(n_b):
            if k == slack:
                continue
            p = np.zeros(n_b)
            p[k] = 1.0
            p[slack] = -1.0

            # reduced vector
            p_keep = p[keep]
            theta_keep = Bred_inv @ p_keep

            theta = np.zeros(n_b)
            theta[keep] = theta_keep
            theta[slack] = 0.0

            f = self.b * (self.A @ theta)
            PTDF[:, k] = f  # since injection is defined vs slack

        # Column for slack bus: injection at slack vs slack is zero in this convention
        PTDF[:, slack] = 0.0
        return PTDF


from scipy.optimize import linprog

def solve_dcopf_ptdf(
    topo: DcOpfTopology,
    PTDF: np.ndarray,
    d: np.ndarray,                 # (n_b,)
    G: np.ndarray,                 # (n_b, n_g) generator-bus map
    c: np.ndarray,                 # (n_g,) linear cost
    pmin: np.ndarray,              # (n_g,)
    pmax: np.ndarray,              # (n_g,)
    include_line_limits: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pg, pinj, f)
      pg: generator dispatch (n_g,)
      pinj: bus injections (n_b,) = G@pg - d, sums to 0
      f: line flows (n_l,)
    """
    n_b = topo.n_b
    n_l = topo.n_l
    n_g = G.shape[1]

    d = d.astype(float).reshape(-1)
    assert d.shape == (n_b,)
    assert c.shape == (n_g,)
    assert pmin.shape == (n_g,)
    assert pmax.shape == (n_g,)

    # Equality: sum(pg) = sum(d)
    Aeq = np.ones((1, n_g))
    beq = np.array([d.sum()])

    A_ub = []
    b_ub = []

    if include_line_limits:
        # f = PTDF * (G@pg - d) = (PTDF@G) pg - PTDF@d
        M = PTDF @ G               # (n_l, n_g)
        h = PTDF @ d               # (n_l,)

        # +f <= fmax  =>  M pg - h <= fmax
        A_ub.append(M)
        b_ub.append(topo.f_max + h)

        # -f <= fmax  => -M pg + h <= fmax
        A_ub.append(-M)
        b_ub.append(topo.f_max - h)

    if A_ub:
        A_ub = np.vstack(A_ub)
        b_ub = np.concatenate(b_ub)
    else:
        A_ub = None
        b_ub = None

    bounds = list(zip(pmin.tolist(), pmax.tolist()))

    res = linprog(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=Aeq, b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"DCOPF failed: {res.message}")

    pg = res.x
    pinj = G @ pg - d
    # Ensure balance numerically by putting residual on slack if you want:
    # pinj[topo.slack] -= pinj.sum()

    f = PTDF @ pinj
    return pg, pinj, f
