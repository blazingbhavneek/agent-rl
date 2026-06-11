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
    Leaves have the highest depth. Keep duplicate function ids as separate
    records because multi-C projects can have same-named static functions.
    """
    funcs = list(analysis.get("functions", []) or [])
    def _depth(func: dict) -> int:
        try:
            return int(func.get("depth", 0))
        except Exception:
            return 0

    return sorted(funcs, key=_depth, reverse=True)


def collect_stub_candidates(analysis: dict) -> list[str]:
    names: set[str] = set()
    for funcs in (analysis.get("stub_candidates") or {}).values():
        for n in funcs or []:
            names.add(n)
    return sorted(names)
