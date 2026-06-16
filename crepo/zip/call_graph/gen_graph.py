import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network

if not hasattr(sys.stderr, "isatty"):
    sys.stderr.isatty = lambda: False
import json

import numpy as np

from state.state import State

BLOCKS = r"\[([^\[\]]*)\]"


def process_first_block(first_block: str) -> tuple[str, int] | int | str:
    """The format is [file_name:line_number]
    - Possibilities:
    1. [file_name]
    2. [line_numnber]
    3. [file_name:line_number]
    One of them will always be there.."""
    if ":" in first_block:
        # means both are present
        parts = first_block.split(":")
        file_name = parts[0]
        line_number = int(parts[1])
        return (file_name, line_number)
    else:
        # either one is present...
        try:
            int(first_block)
            return int(first_block)
        except Exception as e:
            return first_block


def process_second_block(second_block: str) -> None | tuple[int, int]:
    """
    Only possibilities are:
    - Either no second_block
    - Or [start_line: end_line]
    """
    if not second_block:
        return second_block
    parts = second_block.split(":")
    return (int(parts[0]), int(parts[1]))


def process_whole_string(node_str: str) -> tuple[any, any, str]:
    """
    Docstring for process_whole_string
    returns the first and second `[]` for that node.
    """
    node_string = node_str
    function_name = re.sub(BLOCKS, "", node_string)
    node_string = node_str
    matches = re.finditer(BLOCKS, node_string)
    first_block = None
    second_block = None
    for match in matches:
        if not first_block:
            first_block = match.group(1)
        elif not second_block:
            second_block = match.group(1)
    return (first_block, second_block, function_name)


def graph_to_mermaid(
    graph: dict[str, set[tuple[str, str, int]]],
    direction: str = "TD",
    title: str | None = None,
) -> str:
    """
    Convert an adjacency list to a Mermaid flowchart string.
    Uses numeric IDs internally, preserves original names as labels.
    """
    lines = []

    if title:
        lines.append("---")
        lines.append(f"title: {title}")
        lines.append("---")

    lines.append(f"flowchart {direction}")
    # pprint(graph)
    # Collect all nodes
    all_nodes: set[str] = set(graph.keys())
    # pprint(all_nodes  )
    for neighbors in graph.values():
        for neighbor, _, _ in neighbors:
            all_nodes.add(f"{neighbor}")
    # pprint(all_nodes)
    # Assign a clean numeric ID to each node
    node_id_map: dict[str, str] = {}
    for i, node in enumerate(sorted(all_nodes, key=lambda x: str(x))):
        node_id_map[f"{node}"] = f"n{i}"

    # Declare nodes: numeric ID with original name as label
    for node in sorted(all_nodes, key=lambda x: str(x)):
        lines.append(f"    {node_id_map[node]}[{node}]")

    lines.append("")

    # Declare edges with weights
    for src, neighborss in graph.items():
        for dst, weight, level in sorted(neighborss, key=lambda x: str(x[0])):
            lines.append(
                f"    {node_id_map[f'{src}']} -->|{weight}| {node_id_map[f'{dst}']}"
            )

    return "\n".join(lines)
    # return "\n".join(lines)


def visualize_large_graph(
    adj_list: dict[str, set[tuple[str, str, int]]],
    destination_name: str,
    funcs_to_files: dict[str, tuple[str, int, int]],
):
    net = Network(
        height="100vh",
        width="100vw",
        font_color="black",
        directed=True,
        select_menu=True,
        cdn_resources="remote",
    )

    # ── How many individual edges per (src,dst) before we aggregate ──
    MULTI_EDGE_FAN_LIMIT = 6

    # ================================================================
    # OPTIONS — hierarchical layout keeps nodes vertically ordered
    # ================================================================
    options = {
        "nodes": {
            "shape": "box",
            "margin": 10,
            "font": {
                "multi": "html",
                "size": 16,
                "face": "monospace",
                "align": "center",
            },
            "borderWidth": 2,
        },
        "edges": {
            "font": {"multi": "html", "size": 10, "align": "top"},
            # ── global default: nice vertical curves for single edges ──
            "smooth": {
                "enabled": True,
                "type": "cubicBezier",
                "forceDirection": "vertical",
                "roundness": 0.5,
            },
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
        },
        # ── THIS is what keeps nodes in vertical tiers ──
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",  # Up → Down
                "sortMethod": "directed",  # follow edge direction
                "levelSeparation": 200,
                "nodeSpacing": 200,
                "treeSpacing": 300,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            },
        },
        "physics": {
            "enabled": True,
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 150,
                "springConstant": 0.01,
                "nodeDistance": 300,
                "damping": 0.09,
            },
            "solver": "hierarchicalRepulsion",
            "stabilization": {
                "enabled": True,
                "iterations": 2000,
                "updateInterval": 25,
                "onlyDynamicEdges": False,
                "fit": True,
            },
            "adaptiveTimestep": True,
        },
        "interaction": {
            "dragNodes": True,
            "hover": True,
        },
    }

    net.set_options(json.dumps(options))
    destination = destination_name

    # ================================================================
    # Helpers
    # ================================================================
    def _node_label(name: str) -> str:
        f = funcs_to_files.get(name, "")
        if f != "":
            return f"<b>[{f[0]}]</b>\n<b>{name}[{f[1]}:{f[2]}]</b>"
        return f"<b>{name}</b>"

    added_nodes: set[str] = set()

    def _add_node(name: str, level: int = 1):
        if name in added_nodes:
            return
        added_nodes.add(name)

        if name == "main":
            color, size = "#ff0000", 50
        elif name == destination:
            color, size = "#00ff1e", 50
        else:
            color, size = "#f69e05", 30

        net.add_node(
            name,
            size=size,
            level=level,
            label=_node_label(name),
            color=color,
        )

    # ================================================================
    # Group edges by (source, dest)
    # ================================================================
    pair_edges: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)

    for source, edges in adj_list.items():
        if not edges:
            continue
        for dest, weight, level in edges:
            pair_edges[(source, dest)].append((weight, level))

    # ================================================================
    # Add nodes + edges
    # ================================================================
    for (source, dest), edge_group in pair_edges.items():
        # --- nodes ---
        _add_node(source, level=1)
        # use the max level among edges for the destination node
        dest_level = max(lv for _, lv in edge_group)
        _add_node(dest, level=dest_level)

        n = len(edge_group)

        # ==============================================================
        # CASE 1 — single edge: use global default smooth (do nothing)
        # ==============================================================
        if n == 1:
            weight, level = edge_group[0]
            net.add_edge(
                source,
                dest,
                label=f"<b>{weight}</b>",
                title=f"Weight: {weight}",
                # no per-edge smooth → inherits the global cubicBezier
            )

        # ==============================================================
        # CASE 2 — few edges: fan out with CW / CCW on both sides
        # ==============================================================
        elif n <= MULTI_EDGE_FAN_LIMIT:
            for idx, (weight, level) in enumerate(edge_group):
                # Alternate sides: even→CW, odd→CCW
                if idx % 2 == 0:
                    curve_type = "curvedCW"
                else:
                    curve_type = "curvedCCW"

                # Pair index: edges 0,1→pair 0 | 2,3→pair 1 | …
                pair_idx = idx // 2
                # Start at 0.2, step by 0.2 → 0.2, 0.4, 0.6 per side
                roundness = 0.2 + 0.2 * pair_idx

                net.add_edge(
                    source,
                    dest,
                    label=f"<b>{weight}</b>",
                    title=f"Weight: {weight}",
                    smooth={
                        "enabled": True,
                        "type": curve_type,
                        "roundness": roundness,
                    },
                )

        # ==============================================================
        # CASE 3 — many edges: AGGREGATE into one bundled edge
        # ==============================================================
        else:
            weights = [w for w, _ in edge_group]
            count = len(weights)

            # ── numeric weights? compute stats ──
            try:
                numeric = [
                    int(w) if isinstance(w, str) and w.isdigit() else w for w in weights
                ]
                total = sum(numeric)
                mn, mx = min(numeric), max(numeric)
                label = f"<b>{count} calls</b>\n" f"<b>Σ={total} [{mn}–{mx}]</b>"
            except TypeError:
                label = f"<b>{count} calls</b>"

            # ── tooltip: list individual weights (truncated) ──
            sorted_w = sorted(str(w) for w in weights)
            if count > 30:
                detail = ", ".join(sorted_w[:30]) + f"\n… +{count - 30} more"
            else:
                detail = ", ".join(sorted_w)
            title = f"{count} edges between {source} → {dest}\n{detail}"

            # ── edge width scales with count (log so it doesn't explode) ──
            width = 1.5 + 2.0 * math.log2(count)

            net.add_edge(
                source,
                dest,
                label=label,
                title=title,
                width=min(width, 15),
                color={"color": "#e04141", "highlight": "#ff0000"},
                # use the global vertical cubicBezier
            )

    # ================================================================
    # Save
    # ================================================================
    results_path = Path(
        f"/home/seigyo/c_repo/c_repo/results/csv_results/graphs/"
        f'{State().get("PROJECT_NAME", default="DBG")}/{destination_name}'
    )
    results_path.mkdir(parents=True, exist_ok=True)
    net.show(str(results_path / f"graph_{destination_name}.html"), notebook=False)
    # return net


def make_graph(
    paths: list[list[str]],
) -> None:  # adjacency list for the graph.. #adjacency list for the graph..
    # block_regex = r'\[([^\[\]]*)\]'
    adjacency_dict: dict[str, set[tuple[str, str, int]]] = defaultdict(
        set
    )  # file_name,file_defined, set of [function_name, line_used.]
    function_to_file: dict[str, tuple[str, int, int]] = {}
    # level_info: dict[str,int]
    destination_name = None
    for path in paths:
        # pa list[str]\
        last_function = None  # name of the last_function used.
        level = 1
        for node_number, node in enumerate(path):
            first_block, second_block, function_name = process_whole_string(
                node_str=node
            )
            # if function_name==last_function: continue
            if node_number == len(path) - 1:
                destination_name = function_name
            first_block_result = process_first_block(first_block=first_block)
            second_block_result = process_second_block(second_block=second_block)
            if isinstance(
                first_block_result, tuple
            ):  # both filename and line_number used are there.
                file_name_defined = first_block_result[0]  # doesn't matter.
                line_number_used = first_block_result[1]
                if function_name not in adjacency_dict:
                    adjacency_dict[function_name] = set()
                # this node will have an egdge between it and the last node and edge weight is the linue_number_used.
                # get the last_node's name.
                adjacency_dict[last_function].add(
                    (function_name, str(line_number_used), level)
                )
                if function_name not in function_to_file:
                    function_to_file[function_name] = (
                        file_name_defined,
                        second_block_result[0],
                        second_block_result[1],
                    )
                level += 1
                # last_function = function_name
            elif isinstance(first_block_result, int):
                # only line number is there (line number used.)
                if (
                    " (macro expansion)-> " in function_name
                    or " (accepts callback)-> " in function_name
                ):
                    macro_or_callback_function = function_name.split(
                        " (macro expansion)-> "
                    )[0].split(" (accepts callback)-> ")[0]
                    expanded_or_event_function = (
                        function_name.split(" (macro expansion)-> ")[1]
                        if " (macro expansion)-> " in function_name
                        else function_name.split(" (accepts callback)-> ")[1]
                    )
                    line_number_used = first_block_result
                    adjacency_dict[last_function].add(
                        (macro_or_callback_function, line_number_used, level)
                    )
                    # level+=1
                    # adjacency_dict[macro_or_callback_function].add((expanded_or_event_function,'callback' if 'callback' in function_name else 'macro_expansion',level))
                    # last_function =
                else:
                    line_number_used = first_block_result
                    # function_to_file[function_name]=file_name_defined

                    adjacency_dict[last_function].add(
                        (function_name, str(line_number_used), level)
                    )
                level += 1
            elif isinstance(first_block_result, str):
                # this is only for the first node. In this case just make a node for this function.
                adjacency_dict[function_name] = (
                    set()
                    if function_name not in adjacency_dict
                    else adjacency_dict[function_name]
                )
                if function_name not in function_to_file:
                    function_to_file[function_name] = (
                        first_block_result,
                        second_block_result[0],
                        second_block_result[1],
                    )
                # function_to_file[function_name] = first_block_result if function_name not in function_to_file else function_to_file[function_name]
                # last_funcion = function_name
                level += 1

            if not any(
                word in function_name
                for word in [" (macro expansion)-> ", " (accepts callback)-> "]
            ):
                last_function = function_name
            else:
                last_function = function_name.split(" (macro expansion)-> ")[0].split(
                    " (accepts callback)-> "
                )[0]

            # level+=1

    # pprint(adjacency_dict)
    # pprint(function_to_file)
    # print(file_name_defined)
    # mermaid_diag = graph_to_mermaid(graph = adjacency_dict)
    # print(mermaid_diag)
    # pprint(adjacency_dict)
    mermaid_diag = graph_to_mermaid(graph=adjacency_dict)
    results_path = Path(
        f"/home/seigyo/c_repo/c_repo/results/csv_results/graphs/{State().get('PROJECT_NAME',default='DBG')}/{destination_name}"
    )
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"graph_{destination_name}.md", "w") as f:
        f.write(f"DIAGRAM FOR {destination_name}\n```mermaid\n{mermaid_diag}\n```")

    # print(mermaid_diag)
    visualize_large_graph(
        adjacency_dict,
        destination_name=destination_name,
        funcs_to_files=function_to_file,
    )
    # html_path = results_path/f'graph_{destination_name}.html'
    # net.show(str(html_path),notebook=False)


if __name__ == "__main__":
    import pickle

    with open(Path("/home/seigyo/c_repo/c_repo/dio000_path_str.pkl"), "rb") as f:
        path_str = pickle.load(f)

    # pprint(path_str)
    make_graph(paths=path_str)
