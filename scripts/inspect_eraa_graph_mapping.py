from __future__ import annotations

import json
from pathlib import Path

import pypsa


def main():
    data_dir = Path("data_eraa_ml")
    meta = json.loads((data_dir / "metadata.json").read_text())

    source_file = meta["samples"][0]["source_file"]
    network_path = data_dir / "solved_networks" / source_file

    print(f"Reading network: {network_path}")
    n = pypsa.Network(network_path)

    flow_cols = meta["columns"]["flows"]

    print()
    print("Basic model structure")
    print("---------------------")
    print("buses:      ", len(n.buses))
    print("lines:      ", len(n.lines))
    print("links:      ", len(n.links))
    print("stores:     ", len(n.stores))
    print("generators: ", len(n.generators))
    print("flow cols:  ", len(flow_cols))

    print()
    print("Flow column examples")
    print("--------------------")
    for c in flow_cols[:20]:
        print(c)

    mapped = []
    missing = []

    for c in flow_cols:
        kind, name = c.split("::", 1)

        if kind == "Line":
            if name in n.lines.index:
                row = n.lines.loc[name]
                mapped.append((c, row.bus0, row.bus1))
            else:
                missing.append(c)

        elif kind == "Link":
            if name in n.links.index:
                row = n.links.loc[name]
                mapped.append((c, row.bus0, row.bus1))
            else:
                missing.append(c)

        else:
            missing.append(c)

    print()
    print("Mapping check")
    print("-------------")
    print("mapped: ", len(mapped))
    print("missing:", len(missing))

    if missing:
        print()
        print("Missing examples:")
        for c in missing[:20]:
            print(c)

    print()
    print("Mapped examples")
    print("---------------")
    for c, bus0, bus1 in mapped[:20]:
        print(f"{c}: {bus0} -> {bus1}")

    print()
    print("Generator carriers")
    print("------------------")
    print(sorted(n.generators.carrier.dropna().unique()))

    print()
    print("Store carriers")
    print("--------------")
    print(sorted(n.stores.carrier.dropna().unique()))


if __name__ == "__main__":
    main()