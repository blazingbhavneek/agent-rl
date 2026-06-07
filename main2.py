from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import PipelineConfig, derive_paths
from pipeline.analysis import run_or_load_analysis
from pipeline.stage1_scaffold import ensure_test_file
from pipeline.stage2_makefile import build_annotated_makefile, parse_source_makefile_flags
from pipeline.stage3_stubs import handle_stubs
from pipeline.stage4_minimal import ensure_minimal_test_runs
from pipeline.stage5_unit_tests import parallel_generate_unit_tests
from pipeline.stage6_integrate import integrate_all_unit_tests_sequential


def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(description="CUnit test generation pipeline")
    p.add_argument("source_dir", type=Path)
    p.add_argument("process_name")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--api-url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--coverage-threshold", type=float, default=70.0)
    p.add_argument("--max-functions", type=int, default=None)
    p.add_argument("--max-unit-test-attempts", type=int, default=4)
    p.add_argument("--max-unit-test-workers", type=int, default=4)
    p.add_argument("--only-function", default=None)
    p.add_argument("--only-level", type=int, default=None)
    p.add_argument("--skip-stubs", action="store_true")
    p.add_argument("--skip-minimal", action="store_true")
    p.add_argument("--skip-unit-tests", action="store_true")
    p.add_argument("--skip-integrate", action="store_true")
    p.add_argument("--agent-path", type=Path, default=None)
    p.add_argument("--agent-timeout", type=int, default=300)
    p.add_argument("--semantic-judge-threshold", type=float, default=75.0)
    a = p.parse_args()
    return PipelineConfig(
        source_dir=a.source_dir.resolve(),
        process_name=a.process_name,
        output_dir=a.output_dir,
        api_url=a.api_url,
        model=a.model,
        coverage_threshold=a.coverage_threshold,
        max_functions=a.max_functions,
        max_unit_test_attempts=a.max_unit_test_attempts,
        max_unit_test_workers=a.max_unit_test_workers,
        only_function=a.only_function,
        only_level=a.only_level,
        skip_stubs=a.skip_stubs,
        skip_minimal=a.skip_minimal,
        skip_unit_tests=a.skip_unit_tests,
        skip_integrate=a.skip_integrate,
        agent_path=a.agent_path,
        agent_timeout=a.agent_timeout,
        semantic_judge_threshold=a.semantic_judge_threshold,
    )


def run(cfg: PipelineConfig) -> None:
    paths = derive_paths(cfg)

    # Stage 1: ensure test scaffold exists
    ensure_test_file(cfg, paths)

    # Stage 2: build annotated Makefile with coverage + wrap flags
    flags = parse_source_makefile_flags(cfg, paths)
    build_annotated_makefile(cfg, paths, flags)

    # Stage 3: stub generation and validation
    if not cfg.skip_stubs:
        analysis = run_or_load_analysis(cfg, paths)
        handle_stubs(cfg, paths, analysis, flags)
    else:
        analysis = run_or_load_analysis(cfg, paths)

    # Stage 4: ensure minimal test compiles and runs
    if not cfg.skip_minimal:
        ensure_minimal_test_runs(cfg, paths, flags)

    # Stage 5: per-function unit test generation
    if not cfg.skip_unit_tests:
        parallel_generate_unit_tests(cfg, paths, analysis, flags)

    # Stage 6: integrate passing unit tests into master test file
    if not cfg.skip_integrate:
        integrate_all_unit_tests_sequential(cfg, paths)


def main() -> None:
    cfg = parse_args()
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
