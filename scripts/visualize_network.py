# scripts/visualize_network.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def main(data_dir: Path, out: Path | None):
    flow = pd.read_csv(data_dir / "linkf.csv", index_col=0)
    links = pd.read_csv(data_dir / "links.csv", index_col=0)
    buses = pd.read_csv(data_dir / "buses.csv", index_col=0)

    fmax = links["p_nom"].values
    util_mean = np.abs(flow.values).mean(axis=0) / fmax
    util_near = (np.abs(flow.values) >= 0.95 * fmax).mean(axis=0)

    links = links.copy()
    links["util_mean"] = util_mean
    links["util_near"] = util_near

    # build graph
    G = nx.Graph()
    for bus in buses.index:
        G.add_node(bus)

    for _, row in links.iterrows():
        G.add_edge(
            row["bus0"],
            row["bus1"],
            util=row["util_mean"],
            near=row["util_near"],
        )

    pos = nx.spring_layout(G, seed=42)

    edges = G.edges(data=True)
    edge_colors = [e[2]["util"] for e in edges]
    edge_widths = [1 + 6 * e[2]["util"] for e in edges]

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="lightgray", ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Reds,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    # colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Reds,
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])  # <- important for matplotlib
    fig.colorbar(sm, ax=ax, label="Mean line utilization")

    plt.title("IEEE-14 network – congestion & storage")
    plt.axis("off")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.data_dir, args.out)
