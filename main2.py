from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pipeline.config import PipelineConfig, derive_paths
from pipeline.analysis import run_or_load_analysis
from pipeline.common import (
    build_output_with_runtime_diagnostics,
    load_json,
    run_agent,
    run_make_test,
)
from pipeline.stage1_scaffold import ensure_test_file
from pipeline.stage2_makefile import (
    build_annotated_makefile,
    parse_source_makefile_flags,
)
from pipeline.stage3_stubs import handle_stubs
from pipeline.stage4_minimal import ensure_minimal_test_runs
from pipeline.stage5_unit_tests import parallel_generate_unit_tests
from pipeline.stage6_integrate import (
    _existing_wrap_symbols,
    _source_files_json_for_prompt,
    integrate_all_unit_tests_sequential,
    prompt_for_master_test_fix,
    sync_wrap_flags,
)


def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(description="CUnit test generation pipeline")
    p.add_argument("source_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--coverage-threshold", type=float, default=70.0)
    p.add_argument("--max-functions", type=int, default=None)
    p.add_argument("--max-test-attempts", type=int, default=4)
    p.add_argument("--max-unit-test-workers", type=int, default=4)
    p.add_argument("--only-function", default=None)
    p.add_argument("--only-level", type=int, default=None)
    p.add_argument(
        "--agent-js",
        type=Path,
        default=Path("/home/seigyo/rl/agent.js"),
    )
    p.add_argument(
        "--system-json",
        type=Path,
        default=Path("/home/seigyo/rl/system_functions.json"),
    )
    p.add_argument("--agent-timeout-sec", type=int, default=1800)
    p.add_argument("--semantic-judge-min-score", type=int, default=75)
    a = p.parse_args()
    return PipelineConfig(
        source_dir=a.source_dir.resolve(),
        coverage_threshold=a.coverage_threshold,
        max_functions=a.max_functions,
        max_test_attempts=a.max_test_attempts,
        max_unit_test_workers=a.max_unit_test_workers,
        only_function=a.only_function,
        only_level=a.only_level,
        agent_js=a.agent_js,
        system_json=a.system_json,
        agent_timeout_sec=a.agent_timeout_sec,
        semantic_judge_min_score=a.semantic_judge_min_score,
    )


def run(cfg: PipelineConfig) -> None:
    paths = derive_paths(cfg)

    # Check if unit tests are already completed to skip previous stages
    context_file = Path(paths["test_dir"]) / "_pipeline_context.json"
    if context_file.exists():
        ctx = load_json(context_file)
        if ctx.get("unit_tests_completed"):
            print(
                "[pipeline] Unit tests already completed. Skipping to integration.",
                file=sys.stderr,
            )
            flags = ctx.get("flags", {})
            unit_results = ctx.get("unit_test_results", {})
            analysis = run_or_load_analysis(cfg, paths["analysis_path"])
            pre = run_make_test(paths["test_dir"])
            if not pre.get("ok"):
                print(
                    "[pipeline] WARN: existing master test suite does not compile before integration; attempting repairs",
                    file=sys.stderr,
                )
                test_file = Path(paths["test_file"])
                makefile = Path(paths["makefile"])
                for attempt in range(1, 6):
                    diag = build_output_with_runtime_diagnostics(
                        paths["test_dir"],
                        test_file,
                        pre,
                    )
                    prompt = prompt_for_master_test_fix(
                        process_name=paths["process_name"],
                        master_test_file=str(test_file),
                        master_makefile=str(makefile),
                        unit_test_file="",
                        unit_makefile="",
                        source_file_abs="",
                        func={
                            "id": "pre_integration_check",
                            "name": "pre_integration_check",
                        },
                        unit_coverage_pct="unknown",
                        master_coverage_pct="unknown",
                        actual_source_files=_source_files_json_for_prompt(cfg),
                        source_makefiles=" - none found",
                        existing_wraps=_existing_wrap_symbols(test_file),
                        build_output=diag,
                    )
                    run_agent(
                        cfg,
                        paths["test_dir"],
                        prompt,
                        f"_pre_integration_fix_{int(time.time())}.json",
                        folder=cfg.source_dir.parent.parent.resolve(),
                    )
                    sync_wrap_flags(test_file, makefile)
                    pre = run_make_test(paths["test_dir"])
                    if pre.get("ok"):
                        print(
                            f"[pipeline] pre-integration master build passed on attempt {attempt}",
                            file=sys.stderr,
                        )
                        break
                    if attempt >= 5:
                        print(
                            "[pipeline] ERROR: pre-integration build still failing after 5 attempts; aborting",
                            file=sys.stderr,
                        )
                        return

            integrate_all_unit_tests_sequential(
                cfg,
                paths,
                analysis,
                unit_results,
                flags,
            )
            return

    # Stage 1: ensure test scaffold exists
    ensure_test_file(cfg, paths)

    # Stage 2: build annotated Makefile with coverage + wrap flags
    flags = parse_source_makefile_flags(cfg.source_dir / "Makefile")
    flags = build_annotated_makefile(cfg, paths)

    # Stage 3: stub generation and validation
    analysis = run_or_load_analysis(cfg, paths["analysis_path"])
    handle_stubs(cfg, paths, analysis)

    # Stage 4: ensure minimal test compiles and runs
    ensure_minimal_test_runs(cfg, paths)

    # Stage 5: per-function unit test generation
    unit_results = parallel_generate_unit_tests(cfg, paths, analysis, flags)

    # Stage 6: integrate passing unit tests into master test file
    integrate_all_unit_tests_sequential(cfg, paths, analysis, unit_results, flags)


def main() -> None:
    cfg = parse_args()
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
