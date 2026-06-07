from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    source_dir: Path
    agent_js: Path
    system_json: Optional[Path] = None
    func_docs_dir: Path = Path("/home/seigyo/rl/moove_docs/func")
    agent_timeout_sec: int = 1800
    max_agent_iterations: int = 25
    max_compile_fix_attempts: int = 5
    max_test_attempts: int = 4
    # Skip a function if it already has line coverage >= this value.
    coverage_threshold: float = 80.0
    # Stubs are integrated one-at-a-time; kept for forward-compat.
    stub_batch_size: int = 8
    only_function: Optional[str] = None
    only_level: Optional[int] = None
    max_functions: Optional[int] = None
    dry_run: bool = False
    python_bin: str = sys.executable
    max_stub_gen_retries: int = 3
    max_stub_integrate_retries: int = 3
    max_minimal_test_attempts: int = 5
    semantic_judge_min_score: int = 75
    max_unit_test_workers: int = 4
    max_fix_attempts: int = 20


def derive_test_dir(src_dir: Path) -> Path:
    src_dir = src_dir.resolve()
    if src_dir.parent.name != "src":
        raise ValueError(
            f"Source folder parent must be 'src'. Got: {src_dir.parent}"
        )
    return src_dir.parent.parent / "tests" / src_dir.name


def derive_paths(cfg: PipelineConfig):
    test_dir = derive_test_dir(cfg.source_dir)
    process_name = cfg.source_dir.name
    test_file = test_dir / f"test_{process_name}.c"
    makefile = test_dir / "Makefile"
    history_dir = test_dir / "agent_history"
    analysis_path = test_dir / "analysis.json"
    report_file = test_dir / f"test_{process_name}_report.txt"
    log_file = test_dir / f"test_{process_name}_logs.txt"
    return {
        "test_dir": test_dir,
        "process_name": process_name,
        "test_file": test_file,
        "makefile": makefile,
        "history_dir": history_dir,
        "analysis_path": analysis_path,
        "report_file": report_file,
        "log_file": log_file,
    }
