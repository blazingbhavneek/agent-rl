from __future__ import annotations

import sys
from pathlib import Path

from stub.stub import ProjectAnalyzer  # type: ignore

from .config import PipelineConfig
from .common import load_json, write_json


def run_or_load_analysis(cfg: PipelineConfig, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return load_json(out_path)
        except Exception as e:
            print(f"[pipeline] corrupt analysis.json: {e}, re-running analyzer",
                file=sys.stderr)

    print(f"[pipeline] running analyzer on {cfg.source_dir}", file=sys.stderr)
    analyzer = ProjectAnalyzer(
        project_root=cfg.source_dir,
        system_json_path=cfg.system_json,
        discover_system_headers=False,
    )
    result = analyzer.analyze()
    data = result.model_dump()
    write_json(out_path, data)
    print(f"[pipeline] analysis done: functions={result.total_project_functions}",
        file=sys.stderr)
    return data


def functions_leaf_first(analysis: dict) -> list[dict]:
    """
    Order functions leaf -> root.
    function_levels maps depth-string -> list of function ids.
    Leaves have the highest depth.
    """
    levels: dict[int, list[str]] = {}
    for k, v in (analysis.get("function_levels") or {}).items():
        try:
            levels[int(k)] = list(v)
        except Exception:
            continue

    index: dict[str, dict] = {}
    for f in analysis.get("functions", []) or []:
        index[f["id"]] = f

    ordered: list[dict] = []
    for depth in sorted(levels.keys(), reverse=True):
        for fid in levels[depth]:
            if fid in index:
                ordered.append(index[fid])
    return ordered


def collect_stub_candidates(analysis: dict) -> list[str]:
    names: set[str] = set()
    for funcs in (analysis.get("stub_candidates") or {}).values():
        for n in funcs or []:
            names.add(n)
    return sorted(names)
