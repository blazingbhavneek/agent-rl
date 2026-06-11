from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from .config import PipelineConfig
from .analysis import functions_leaf_first
from .common import (
    _function_artifact_key,
    _safe_c_identifier,
    _resolve_source_file,
    _source_files_json_for_prompt,
    build_output_with_runtime_diagnostics,
    check_function_coverage,
    read_text,
    run_agent,
    run_make_test,
    sync_wrap_flags,
    write_text,
)
from .execution import run_command


def _extract_func_body(text: str, func_name: str) -> str:
    """Extract a complete void function definition by brace counting."""
    m = re.search(rf'(?:static\s+)?void\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{', text)
    if not m:
        return ""
    start = m.start()
    depth = 1
    i = m.end()
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i]


def extract_test_additions(unit_test_file: Path, *, name_prefix: str = "") -> tuple[str, list[str]]:
    """Extract test functions and CU_add_test calls from a standalone unit test file.

    Unit test files are complete CUnit files (not marker-based fragments).
    We identify test functions from CU_add_test() calls, extract their bodies
    by brace counting, and strip the production-source #include block (already
    present in the master test).
    """
    text = read_text(unit_test_file)

    raw_reg_calls = re.findall(r'CU_add_test\s*\([^;]+\)\s*;', text)
    if not raw_reg_calls:
        return "", []

    # Collect test function names from CU_add_test(suite, "label", func_name)
    func_names: list[str] = []
    for call in raw_reg_calls:
        m = re.search(r'CU_add_test\s*\(\s*\w+\s*,\s*"[^"]*"\s*,\s*(\w+)\s*\)', call)
        if m:
            func_names.append(m.group(1))

    # Normalize suite variable to 'suite' so calls work in the master test scope.
    reg_calls = [
        re.sub(r'(CU_add_test\s*\()\s*\w+\s*,', r'\1suite,', call, count=1)
        for call in raw_reg_calls
    ]

    # Extract each test function body.
    bodies: list[str] = []
    rename_map: dict[str, str] = {}
    for name in func_names:
        body = _extract_func_body(text, name)
        if body:
            if name_prefix:
                new_name = _safe_c_identifier(f"{name_prefix}_{name}")
                body = re.sub(
                    rf"((?:static\s+)?void\s+){re.escape(name)}\b",
                    rf"\1{new_name}",
                    body,
                    count=1,
                )
                rename_map[name] = new_name
            bodies.append(body)

    if rename_map:
        renamed_calls: list[str] = []
        for call in reg_calls:
            new_call = call
            for old, new in rename_map.items():
                new_call = re.sub(rf"\b{re.escape(old)}\b", new, new_call)
            renamed_calls.append(new_call)
        reg_calls = renamed_calls

    test_cases = "\n\n".join(bodies)
    return test_cases, reg_calls


def _existing_wrap_symbols(test_file: Path) -> str:
    text = read_text(test_file)
    names = sorted(set(re.findall(r"__wrap_(\w+)\b", text)))
    return "\n".join(f" - {name}" for name in names) if names else " - none"


def _source_makefile_text(existing_source_makefiles: list[Path]) -> str:
    return (
        "\n".join(f" - {p}" for p in existing_source_makefiles)
        if existing_source_makefiles
        else " - none found"
    )


def _coverage_pct(cov_obj: dict | None) -> float | None:
    if not isinstance(cov_obj, dict):
        return None

    pct_val = (cov_obj.get("summary", {}) or {}).get("coverage_percent")
    if pct_val is None:
        pct_val = cov_obj.get("coverage_percent")
    if pct_val is None:
        pct_val = cov_obj.get("percent")
    if pct_val is None:
        pct_val = cov_obj.get("coverage")
    if pct_val is None:
        pct_val = cov_obj.get("pct")

    if pct_val is None:
        return None

    try:
        return float(pct_val)
    except Exception:
        return None


def _coverage_matches_target(
    master_pct: float | None,
    unit_pct: float | None,
    threshold: float,
    tolerance: float = 5.0,
) -> bool:
    if master_pct is None:
        return False
    if unit_pct is None:
        return master_pct >= threshold
    return master_pct >= max(threshold, unit_pct - tolerance)


def _current_function_coverage(
    cfg: PipelineConfig,
    test_dir: Path,
    source_file_abs: Path,
    func: dict,
) -> tuple[dict | None, float | None]:
    cov_obj = check_function_coverage(
        test_dir,
        source_file_abs,
        func["start_line"],
        func["end_line"],
        source_root=cfg.source_dir.resolve(),
    )
    return cov_obj, _coverage_pct(cov_obj)


def prompt_for_master_test_fix(
    *,
    process_name: str,
    master_test_file: str,
    master_makefile: str,
    unit_test_file: str,
    unit_makefile: str,
    source_file_abs: str,
    func: dict,
    unit_coverage_pct: str,
    master_coverage_pct: str,
    actual_source_files: str,
    source_makefiles: str,
    existing_wraps: str,
    build_output: str,
) -> str:
    fid = func.get("id") or func.get("name") or "unknown_function"

    return f"""
You are fixing an integrated master CUnit test file after merging one standalone
unit test back into the master suite.

Process:
{process_name}

Target function:
{fid}

Target function metadata:
{json.dumps(func, indent=2, default=str)}

Master integrated test file:
{master_test_file}

Master test Makefile:
{master_makefile}

Standalone unit test file being merged:
{unit_test_file}

Standalone unit Makefile:
{unit_makefile}

Production source file:
{source_file_abs}

Standalone unit coverage target:
{unit_coverage_pct}

Current master coverage for this function:
{master_coverage_pct}

Actual project .c source files discovered:
{actual_source_files}

Source Makefile candidates:
{source_makefiles}

Wrapper symbols already present in the master file:
{existing_wraps}

FILES YOU MAY EDIT
-------------
- Master integrated test file:
  `{master_test_file}`
- Master test Makefile:
  `{master_makefile}`

FILES YOU MUST READ
-------------
- Master integrated test file:
  `{master_test_file}`
- Standalone unit test file:
  `{unit_test_file}`
- Standalone unit Makefile:
  `{unit_makefile}`
- Production source:
  `{source_file_abs}`
- Actual source files:
{actual_source_files}
- Source Makefile candidates:
{source_makefiles}

BUILD / CONTENT RULES
1. Read the master test file first. A lot of wrapper stubs/helpers are already
   present, so do not duplicate them.
2. Read the standalone unit test file before editing the master file.
3. Normalize the master file before adding new content:
   - remove repeated `#include` lines if the same header already exists
   - remove duplicate wrapper/helper definitions
   - keep wrappers in the wrapper section, helpers in the helper section, and
     tests in the test section
   - keep registration lines together and aligned
4. Keep the merged master file organized, compact, and easy to read.
5. Preserve existing section markers and grouping.
6. If a wrapper/helper already exists in the master file, reuse it instead of
   copying another copy.
7. If the unit test introduces a genuinely new helper or wrapper, add only the
   missing piece and keep it in the right section.
8. Do not copy the unit-test `main()` into the master file.
9. Do not add the production .c as a separate compiled object.
10. Do not redefine existing Makefile variables; preserve existing `WRAP_FUNCS`
    / `WRAP_FLAGS` entries and append only the missing wrap flags.
11. If the master file already contains this function's merged block, do not
    paste it again. Repair the existing block so the master coverage matches the
    standalone unit-test coverage instead.
12. If the master file is messy, rewrite the affected sections so the final
    file stays clean and readable rather than layering more clutter on top.
13. If compilation fails, fix include paths, macros, or wrap flags first rather
    than deleting the merged tests.
14. After any edit, run:
    `make test`
    Then inspect the generated report and coverage output:
    `ls -lt *_report.txt`
    `cat *_report.txt`
    `gcov -p -b -c *.gcno`
15. Keep the merged file readable: group tests, helpers, wrappers, and
    registrations cleanly instead of scattering additions around the file.

FAILING BUILD / TEST OUTPUT
{build_output}

STRICT RULES
-------------
- You may only edit the master integrated test file and its Makefile.
- You must not edit production source files or headers.
- Do not remove existing working wrappers just because they look unrelated.
- Do not weaken assertions just to make the build pass.
- If the master file already has the needed wrappers, leave them alone.

"""


def integrate_all_unit_tests_sequential(
    cfg: PipelineConfig,
    paths: dict,
    analysis: dict,
    unit_test_results: dict[str, dict],
    flags: dict,
) -> bool:
    """
    Stage 6: integrate passed unit tests one-by-one into master test file.
    After each: gcc -fsyntax-only check. Loop compile-fix on failure.
    """
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    test_dir: Path = paths["test_dir"]
    process_name: str = paths["process_name"]
    repo_root = cfg.source_dir.parent.parent.resolve()

    cflags_str = " ".join(filter(None, [
        flags.get("CFLAGS", ""), flags.get("CFLAGS_LINUX", ""),
        flags.get("CPPFLAGS", ""), flags.get("INCLUDE", ""),
    ]))

    last_merge_context: dict | None = None
    fix_limit = max(1, int(getattr(cfg, "max_fix_attempts", 5)))
    coverage_threshold = float(getattr(cfg, "coverage_threshold", 100.0))
    all_ok = True

    pre_source_makefiles = _source_makefile_text([
        cfg.source_dir.resolve() / "Makefile",
        cfg.source_dir.parent.parent.resolve() / "Makefile",
    ])
    pre = run_make_test(cfg, test_dir)
    if not pre.get("ok"):
        print("[pipeline] WARN: master test suite does not compile before integration; attempting repairs", file=sys.stderr)
        for attempt in range(1, fix_limit + 1):
            diag = build_output_with_runtime_diagnostics(cfg, test_dir, test_file, pre)
            prompt = prompt_for_master_test_fix(
                process_name=process_name,
                master_test_file=str(test_file),
                master_makefile=str(makefile),
                unit_test_file="",
                unit_makefile="",
                source_file_abs="",
                func={"id": "pre_integration_check", "name": "pre_integration_check"},
                unit_coverage_pct="unknown",
                master_coverage_pct="unknown",
                actual_source_files=_source_files_json_for_prompt(cfg),
                source_makefiles=pre_source_makefiles,
                existing_wraps=_existing_wrap_symbols(test_file),
                build_output=diag,
            )
            run_agent(
                cfg,
                test_dir,
                prompt,
                f"_pre_integration_fix_{int(time.time())}.json",
                folder=repo_root,
            )
            sync_wrap_flags(test_file, makefile)
            pre = run_make_test(cfg, test_dir)
            if pre.get("ok"):
                print(f"[pipeline] pre-integration master build passed on attempt {attempt}", file=sys.stderr)
                break
            if attempt >= fix_limit:
                print("[pipeline] ERROR: pre-integration build still failing; continuing with integration anyway", file=sys.stderr)

    for func in functions_leaf_first(analysis):
        func_id = func["id"]
        function_key = _function_artifact_key(cfg, func)
        safe_id = function_key
        result = unit_test_results.get(function_key) or unit_test_results.get(func_id, {})
        if not result.get("passed"):
            continue

        unit_marker = f"/* --- unit: {function_key} --- */"
        if unit_marker in read_text(test_file):
            print(f"[pipeline] already integrated: {func_id}", file=sys.stderr)
            continue

        unit_dir = Path(result.get("unit_dir", ""))
        unit_test_file = unit_dir / f"test_{safe_id}.c"
        unit_makefile = unit_dir / "Makefile"
        if not unit_test_file.exists():
            continue

        test_cases, reg_calls = extract_test_additions(
            unit_test_file,
            name_prefix=_safe_c_identifier(function_key),
        )
        if not test_cases.strip() and not reg_calls:
            print(f"[pipeline] no additions to integrate for {func_id}", file=sys.stderr)
            continue

        source_file_abs = _resolve_source_file(cfg, func["source_file"])
        unit_cov_raw = result.get("coverage_pct")
        unit_cov_pct = float(unit_cov_raw) if isinstance(unit_cov_raw, (int, float)) else None
        current_cov_obj, current_cov_pct = _current_function_coverage(
            cfg, test_dir, source_file_abs, func
        )
        current_has_block = unit_marker in read_text(test_file)

        if current_has_block and _coverage_matches_target(
            current_cov_pct,
            unit_cov_pct,
            coverage_threshold,
        ):
            print(f"[pipeline] already integrated: {func_id}", file=sys.stderr)
            continue

        print(f"[pipeline] Stage 6: integrating {func_id}", file=sys.stderr)
        current = read_text(test_file)

        cases_marker = "/* === Test Cases === */"
        reg_marker = "CU_basic_set_mode"

        if not current_has_block and cases_marker in current and test_cases.strip():
            insertion = f"\n\n{unit_marker}\n/* function id: {func_id} */\n{test_cases}\n"
            current = current.replace(cases_marker, cases_marker + insertion, 1)

        if not current_has_block and reg_calls and reg_marker in current:
            reg_insertion = "    " + "\n    ".join(reg_calls) + "\n    "
            current = current.replace(reg_marker, reg_insertion + reg_marker, 1)

        write_text(test_file, current)
        sync_wrap_flags(test_file, makefile)

        source_makefile_candidates: list[Path] = []
        try:
            resolved_source_dir = cfg.source_dir.resolve()
            if resolved_source_dir.is_dir():
                source_makefile_candidates.append(resolved_source_dir / "Makefile")
            source_makefile_candidates.append(source_file_abs.parent / "Makefile")
            source_makefile_candidates.append(source_file_abs.parent.parent / "Makefile")
            source_makefile_candidates.append(repo_root / "Makefile")
        except Exception:
            pass

        source_makefile_candidates = list(dict.fromkeys(source_makefile_candidates))
        existing_source_makefiles = [p for p in source_makefile_candidates if p.exists()]
        prompt = prompt_for_master_test_fix(
            process_name=process_name,
            master_test_file=str(test_file),
            master_makefile=str(makefile),
            unit_test_file=str(unit_test_file),
            unit_makefile=str(unit_makefile),
            source_file_abs=str(source_file_abs),
            func={**func, "source_file": str(source_file_abs)},
            unit_coverage_pct=str(unit_cov_pct if unit_cov_pct is not None else "unknown"),
            master_coverage_pct=str(current_cov_pct if current_cov_pct is not None else "unknown"),
            actual_source_files=_source_files_json_for_prompt(cfg),
            source_makefiles=_source_makefile_text(existing_source_makefiles),
            existing_wraps=_existing_wrap_symbols(test_file),
            build_output=(
                "Merged or repaired unit test additions in the master test file.\n"
                f"Unit test file: {unit_test_file}\n"
                f"Extracted CU_add_test registrations: {len(reg_calls)}\n"
                f"Extracted test bodies: {len(test_cases.splitlines()) if test_cases else 0} lines"
            ),
        )

        run_agent(
            cfg,
            test_dir,
            prompt,
            f"_unit_integrate_cleanup_{safe_id}_{int(time.time())}.json",
            folder=repo_root,
        )
        sync_wrap_flags(test_file, makefile)

        attempt = 1
        final = None
        final_cov_pct = current_cov_pct
        while attempt <= fix_limit:
            chk = run_command(
                cfg,
                f"gcc -fsyntax-only {cflags_str} {test_file}",
                shell=True,
                cwd=test_dir,
                timeout=60,
            )
            if chk.returncode != 0:
                err = (chk.stderr + chk.stdout)
                print(f"[pipeline] syntax error after {func_id} attempt {attempt}, fixing", file=sys.stderr)
                run_agent(
                    cfg,
                    test_dir,
                    prompt_for_master_test_fix(
                        process_name=process_name,
                        master_test_file=str(test_file),
                        master_makefile=str(makefile),
                        unit_test_file=str(unit_test_file),
                        unit_makefile=str(unit_makefile),
                        source_file_abs=str(source_file_abs),
                        func={**func, "source_file": str(source_file_abs)},
                        unit_coverage_pct=str(unit_cov_pct if unit_cov_pct is not None else "unknown"),
                        master_coverage_pct=str(final_cov_pct if final_cov_pct is not None else "unknown"),
                        actual_source_files=_source_files_json_for_prompt(cfg),
                        source_makefiles=_source_makefile_text(existing_source_makefiles),
                        existing_wraps=_existing_wrap_symbols(test_file),
                        build_output=err,
                    ),
                    f"_unit_integrate_fix_{safe_id}_{int(time.time())}.json",
                    folder=repo_root,
                )
                sync_wrap_flags(test_file, makefile)
                attempt += 1
                continue

            final = run_make_test(cfg, test_dir)
            current_cov_obj, final_cov_pct = _current_function_coverage(
                cfg, test_dir, source_file_abs, func
            )
            if final["ok"] and _coverage_matches_target(
                final_cov_pct,
                unit_cov_pct,
                coverage_threshold,
            ):
                print(
                    f"[pipeline] master OK after {func_id} coverage={final_cov_pct}%",
                    file=sys.stderr,
                )
                break

            diag = build_output_with_runtime_diagnostics(cfg, test_dir, test_file, final)
            diag += (
                "\n\n==================== FUNCTION COVERAGE CHECK ====================\n"
                f"Target function: {func_id}\n"
                f"Unit coverage: {unit_cov_pct}\n"
                f"Master coverage: {final_cov_pct}\n"
                f"Coverage threshold: {coverage_threshold}\n"
                "The master file is not considered fully integrated until the "
                "function's coverage is close to the standalone unit-test coverage.\n"
                "Fix the merged test so it preserves the unit behavior and "
                "coverage, not just compilation.\n"
            )
            print(
                f"[pipeline] coverage mismatch after {func_id} attempt {attempt}, fixing",
                file=sys.stderr,
            )
            run_agent(
                cfg,
                test_dir,
                prompt_for_master_test_fix(
                    process_name=process_name,
                    master_test_file=str(test_file),
                    master_makefile=str(makefile),
                    unit_test_file=str(unit_test_file),
                    unit_makefile=str(unit_makefile),
                    source_file_abs=str(source_file_abs),
                    func={**func, "source_file": str(source_file_abs)},
                    unit_coverage_pct=str(unit_cov_pct if unit_cov_pct is not None else "unknown"),
                    master_coverage_pct=str(final_cov_pct if final_cov_pct is not None else "unknown"),
                    actual_source_files=_source_files_json_for_prompt(cfg),
                    source_makefiles=_source_makefile_text(existing_source_makefiles),
                    existing_wraps=_existing_wrap_symbols(test_file),
                    build_output=diag,
                ),
                f"_unit_integrate_fix_{safe_id}_{int(time.time())}.json",
                folder=repo_root,
            )
            sync_wrap_flags(test_file, makefile)
            attempt += 1

        last_merge_context = {
            "func": {**func, "source_file": str(source_file_abs)},
            "unit_test_file": str(unit_test_file),
            "unit_makefile": str(unit_makefile),
            "source_file_abs": str(source_file_abs),
            "source_makefiles": _source_makefile_text(existing_source_makefiles),
            "existing_wraps": _existing_wrap_symbols(test_file),
            "unit_coverage_pct": str(unit_cov_pct if unit_cov_pct is not None else "unknown"),
            "master_coverage_pct": str(final_cov_pct if final_cov_pct is not None else "unknown"),
        }
        if final is None or not final.get("ok") or not _coverage_matches_target(
            final_cov_pct,
            unit_cov_pct,
            coverage_threshold,
        ):
            all_ok = False

    print("[pipeline] Stage 6: running final make test on master suite", file=sys.stderr)
    final = run_make_test(cfg, test_dir)
    if not final["ok"]:
        print("[pipeline] WARN: final master make test failed after unit test integration", file=sys.stderr)
        repo_root_f = cfg.source_dir.parent.parent.resolve()
        diag = build_output_with_runtime_diagnostics(cfg, test_dir, test_file, final)
        attempt = 1
        while attempt <= fix_limit:
            ctx = last_merge_context or {}
            run_agent(
                cfg, test_dir,
                prompt_for_master_test_fix(
                    process_name=process_name,
                    master_test_file=str(test_file),
                    master_makefile=str(makefile),
                    unit_test_file=str(ctx.get("unit_test_file", "")),
                    unit_makefile=str(ctx.get("unit_makefile", "")),
                    source_file_abs=str(ctx.get("source_file_abs", "")),
                    func=ctx.get("func", {"id": "unknown_function"}),
                    unit_coverage_pct=str(ctx.get("unit_coverage_pct", "unknown")),
                    master_coverage_pct=str(ctx.get("master_coverage_pct", "unknown")),
                    actual_source_files=_source_files_json_for_prompt(cfg),
                    source_makefiles=str(ctx.get("source_makefiles", " - none found")),
                    existing_wraps=str(ctx.get("existing_wraps", " - none")),
                    build_output=diag,
                ),
                f"_final_master_fix_{int(time.time())}.json",
                folder=repo_root_f,
            )
            sync_wrap_flags(test_file, makefile)
            final = run_make_test(cfg, test_dir)
            if final["ok"]:
                print(f"[pipeline] final master make test passed on attempt {attempt}", file=sys.stderr)
                break
            diag = build_output_with_runtime_diagnostics(cfg, test_dir, test_file, final)
            attempt += 1
    return bool(final["ok"] and all_ok)
