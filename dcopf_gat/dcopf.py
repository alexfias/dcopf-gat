import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

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
