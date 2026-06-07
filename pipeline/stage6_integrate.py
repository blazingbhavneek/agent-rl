from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

from .config import PipelineConfig
from .analysis import functions_leaf_first
from .common import (
    _project_source_files,
    _safe_filename,
    build_output_with_runtime_diagnostics,
    prompt_for_compile_fix,
    read_text,
    run_agent,
    run_make_test,
    sync_wrap_flags,
    write_text,
)


def extract_test_additions(unit_test_file: Path) -> tuple[str, list[str]]:
    """Extract test case functions and CU_add_test calls from a unit test file."""
    text = read_text(unit_test_file)

    cases_start = text.find("/* === Test Cases === */")
    reg_start = text.find("/* === Test Registration === */")
    if cases_start == -1:
        return "", []

    end = reg_start if reg_start != -1 else len(text)
    test_cases = text[cases_start + len("/* === Test Cases === */"):end].strip()

    reg_calls = re.findall(r'CU_add_test\s*\([^;]+\)\s*;', text)
    return test_cases, reg_calls


def integrate_all_unit_tests_sequential(
    cfg: PipelineConfig,
    paths: dict,
    analysis: dict,
    unit_test_results: dict[str, dict],
    flags: dict,
) -> bool:
    """
    Stage 5: integrate passed unit tests one-by-one into master test file.
    After each: gcc -fsyntax-only check. Loop compile-fix on failure.
    """
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    test_dir: Path = paths["test_dir"]
    source_dir = cfg.source_dir.resolve()
    repo_root = cfg.source_dir.parent.parent.resolve()

    cflags_str = " ".join(filter(None, [
        flags.get("CFLAGS", ""), flags.get("CFLAGS_LINUX", ""),
        flags.get("CPPFLAGS", ""), flags.get("INCLUDE", ""),
    ]))

    for func in functions_leaf_first(analysis):
        func_id = func["id"]
        safe_id = _safe_filename(func_id)
        result = unit_test_results.get(func_id, {})
        if not result.get("passed"):
            continue

        if f"/* --- unit: {func_id} --- */" in read_text(test_file):
            print(f"[pipeline] already integrated: {func_id}", file=sys.stderr)
            continue

        unit_dir = Path(result.get("unit_dir", ""))
        unit_test_file = unit_dir / f"test_{safe_id}.c"
        if not unit_test_file.exists():
            continue

        test_cases, reg_calls = extract_test_additions(unit_test_file)
        if not test_cases.strip() and not reg_calls:
            print(f"[pipeline] no additions to integrate for {func_id}", file=sys.stderr)
            continue

        print(f"[pipeline] Stage 5: integrating {func_id}", file=sys.stderr)
        current = read_text(test_file)

        cases_marker = "/* === Test Cases === */"
        reg_marker = "CU_basic_set_mode"

        if cases_marker in current and test_cases.strip():
            insertion = f"\n\n/* --- unit: {func_id} --- */\n{test_cases}\n"
            current = current.replace(cases_marker, cases_marker + insertion, 1)

        if reg_calls and reg_marker in current:
            reg_insertion = "    " + "\n    ".join(reg_calls) + "\n    "
            current = current.replace(reg_marker, reg_insertion + reg_marker, 1)

        write_text(test_file, current)
        sync_wrap_flags(test_file, makefile)

        # Syntax-check the master test file after each merge.
        # gcc -fsyntax-only is much faster than a full build + link and is
        # enough to catch declaration conflicts or missing types introduced
        # by the newly pasted test code.
        attempt = 1
        while True:
            chk = subprocess.run(
                f"gcc -fsyntax-only {cflags_str} {test_file}",
                shell=True, cwd=str(test_dir),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=60,
            )
            if chk.returncode == 0:
                print(f"[pipeline] master syntax OK after {func_id}", file=sys.stderr)
                break
            err = (chk.stderr + chk.stdout)[:6000]
            print(f"[pipeline] syntax error after {func_id} attempt {attempt}, fixing", file=sys.stderr)
            run_agent(
                cfg, test_dir,
                prompt_for_compile_fix(
                    str(makefile), str(test_file), err,
                    source_dir=str(source_dir),
                    source_makefile=str(source_dir / "Makefile"),
                    actual_source_files=[str(p) for p in _project_source_files(cfg)],
                ),
                f"_unit_integrate_fix_{safe_id}_{int(time.time())}.json",
                folder=repo_root,
            )
            attempt += 1

    print("[pipeline] Stage 5: running final make test on master suite", file=sys.stderr)
    final = run_make_test(test_dir)
    if not final["ok"]:
        print("[pipeline] WARN: final master make test failed after unit test integration", file=sys.stderr)
        source_dir_f = cfg.source_dir.resolve()
        repo_root_f = cfg.source_dir.parent.parent.resolve()
        diag = build_output_with_runtime_diagnostics(test_dir, test_file, final)
        attempt = 1
        while True:
            run_agent(
                cfg, test_dir,
                prompt_for_compile_fix(
                    str(makefile), str(test_file), diag,
                    source_dir=str(source_dir_f),
                    source_makefile=str(source_dir_f / "Makefile"),
                    actual_source_files=[str(p) for p in _project_source_files(cfg)],
                ),
                f"_final_master_fix_{int(time.time())}.json",
                folder=repo_root_f,
            )
            sync_wrap_flags(test_file, makefile)
            final = run_make_test(test_dir)
            if final["ok"]:
                print(f"[pipeline] final master make test passed on attempt {attempt}", file=sys.stderr)
                break
            diag = build_output_with_runtime_diagnostics(test_dir, test_file, final)
            attempt += 1
    return final["ok"]
