from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .analysis import functions_leaf_first
from .common import (
    _project_source_files,
    _resolve_source_file,
    _safe_filename,
    _source_files_json_for_prompt,
    build_output_with_runtime_diagnostics,
    check_function_coverage,
    load_json,
    prompt_for_compile_fix,
    read_text,
    run_agent,
    run_make_test,
    sync_wrap_flags,
    write_json,
    write_text,
)
from .semantic import (
    _append_semantic_context,
    _load_semantic_context,
    backup_good_cunit_if_best,
    prompt_for_function_test_with_semantic_context,
    prompt_for_semantic_test_repair,
    run_semantic_test_judge,
)


# region Stub sync helpers (used only by this stage)

# get list of all stub dirs which have validated stubs
def _stub_srcs_absolute(test_dir: Path) -> list[tuple[str, str]]:
    """Return [(abs_stub_path, func_name)] for all validated stubs."""
    stubs_dir = test_dir / "_stub_gen"
    result: list[tuple[str, str]] = []
    if not stubs_dir.exists():
        return result
    for result_file in sorted(stubs_dir.glob("*/result.json")):
        try:
            data = load_json(result_file)
            if not data.get("validated"):
                continue
            func_name = data.get("func_name", result_file.parent.name)
            stub_c = result_file.parent / "stub.c"
            if stub_c.exists():
                result.append((str(stub_c.resolve()), func_name))
        except Exception:
            pass
    return result

# Deduplication of __wrap_X functions if defined inside the unit test file and stub also included
# TODO: Remove this, dont link stub to it, rather tell agent to read from it
def _sync_stub_srcs(unit_test_file: Path, unit_makefile: Path, test_dir: Path) -> None:
    """
    Update STUB_SRCS in unit Makefile to exclude stubs whose __wrap_ function
    is locally DEFINED (not just called) in the unit test file.
    Prevents duplicate-symbol link errors when the test overrides a stub.
    """
    text = read_text(unit_test_file)
    local_overrides = set(re.findall(r'__wrap_(\w+)\s*\([^)]*\)\s*\{', text))
    filtered = [
        abs_path
        for abs_path, fname in _stub_srcs_absolute(test_dir)
        if fname not in local_overrides
    ]
    stub_srcs_line = "STUB_SRCS = " + " ".join(filtered)
    mk_text = read_text(unit_makefile)
    new_mk = re.sub(r'^STUB_SRCS\s*=.*$', stub_srcs_line, mk_text, flags=re.MULTILINE)
    if new_mk != mk_text:
        write_text(unit_makefile, new_mk)

# endregion Stub sync helpers


# region Scaffold

def _scaffold_unit_test_dir(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    *,
    unit_dir_override: Optional[Path] = None,
) -> Path:
    """Create _unit_tests/<func_id>/ with skeleton test file + generated Makefile."""
    test_dir: Path = paths["test_dir"]
    process_name: str = paths["process_name"]
    func_id = func["id"]
    safe_id = _safe_filename(func_id)
    test_program = f"test_{safe_id}"
    test_src = f"{test_program}.c"

    unit_dir = unit_dir_override or (test_dir / "_unit_tests" / safe_id)
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "agent_history").mkdir(exist_ok=True)

    unit_test_file = unit_dir / f"test_{safe_id}.c"
    unit_makefile = unit_dir / "Makefile"

    if not unit_test_file.exists():
        skeleton = f"""/*
     * PLACEHOLDER ONLY.
     *
     * Target:
     *   {func_id}
     *
     * This file must be completely rewritten by the unit-test agent.
     *
     * Required final file:
     *   - complete CUnit test file
     *   - at least one CU_add_test()
     *   - real test function(s)
     *   - calls target function directly or through a real caller
     *   - uses wrappers/stubs from the master test or _stub_gen as needed
     */
     """
        write_text(unit_test_file, skeleton)

    if not unit_makefile.exists() or not _unit_makefile_matches_contract(unit_makefile, test_program, test_src):
        _generate_unit_test_makefile(cfg, paths, func, safe_id, unit_dir, unit_test_file)

    return unit_dir

# Detect old/generated Makefiles that belong to some other test and regenerate them instead of reusing them
# TODO: Why we needed this in first place?
def _unit_makefile_matches_contract(unit_makefile: Path, test_program: str, test_src: str) -> bool:
    """Keep stale per-function Makefiles from drifting off the current file-name contract."""
    text = read_text(unit_makefile)
    if not text:
        return False

    program_ok = re.search(
        rf"(?m)^TEST_PROGRAM\s*=\s*{re.escape(test_program)}\s*$",
        text,
    ) is not None
    src_ok = re.search(
        rf"(?m)^TEST_SRCS\s*=\s*{re.escape(test_src)}\s*$",
        text,
    ) is not None
    return program_ok and src_ok


def _generate_unit_test_makefile(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    safe_id: str,
    unit_dir: Path,
    unit_test_file: Path,
) -> None:
    """Generate Makefile for a per-function unit test directory.

    The production source is always #included directly in the test .c file.
    The Makefile never compiles a separate prod_under_test.o — doing so while
    also #including the same .c would duplicate every symbol and cause linker
    errors.

    All paths written into the Makefile are absolute so the file works
    regardless of how deep the unit test directory is in the tree.
    """
    context_file = paths["test_dir"] / "_pipeline_context.json"
    flags: dict = {}
    if context_file.exists():
        try:
            flags = load_json(context_file).get("flags", {})
        except Exception:
            pass

    func_id = func["id"]
    process_name: str = paths["process_name"]

    # Absolute paths — no relative-depth ambiguity regardless of directory depth.
    master_makefile_abs = str(paths["makefile"].resolve())
    source_file_abs = _resolve_source_file(cfg, func["source_file"])
    prod_src_abs = str(source_file_abs.resolve())

    test_program = f"test_{safe_id}"
    test_src = unit_test_file.name

    stub_srcs_str = ""

    # Production source always #included in test .c — never compile separate PROD_OBJ.

    content = f"""# Unit test Makefile for {func_id}

# Pull in the same include paths, libraries, architecture flags, and wrap flags
# as the master test Makefile. Unit-specific rules below override only what
# must differ for this per-function test.
MASTER_MAKEFILE = {master_makefile_abs}
include $(MASTER_MAKEFILE)

.DEFAULT_GOAL := test

CC ?= gcc

# Fallbacks from _pipeline_context.json, used only if the master Makefile did
# not define them.
CFLAGS ?= {flags.get('CFLAGS', '')}
CFLAGS_LINUX ?= {flags.get('CFLAGS_LINUX', '')}
CPPFLAGS ?= {flags.get('CPPFLAGS', '')}
INCLUDE ?= {flags.get('INCLUDE', '')}
LDFLAGS ?= {flags.get('LDFLAGS', '')}
LDLIBS ?= {flags.get('LDLIBS', '')}
LIBS ?= {flags.get('LIBS', '')}

# Unit-specific files. Absolute paths — no depth guessing needed.
TEST_PROGRAM = {test_program}
TEST_SRCS = {test_src}
# PROD_SRC is #included in TEST_SRCS — do NOT add it to the link command.
# Compiling it separately while #including it causes duplicate symbol errors.
PROD_SRC = {prod_src_abs}

# Keep only the production gcov output.
TARGET_GCOV = $(notdir $(PROD_SRC)).gcov

# Test object gcno — only TEST_SRCS produce gcno because PROD_SRC is #included.
TEST_GCNO = $(TEST_SRCS:.c=.gcno)

# Validated stub bodies (absolute paths).
STUB_SRCS = {stub_srcs_str}

TEST_LIBS += -lcunit
TEST_REPORT_FILE = {test_program}_report.txt
TEST_LOG_FILE = {test_program}_log.txt

# Legacy parent Makefile compatibility. Keep the real suffixed unit test file,
# but let older targets resolve without needing a shim source file.
LEGACY_TEST_SRCS = test_{process_name}.c
LEGACY_TEST_PROGRAM = unit_test_{process_name}

COVERAGE_FLAGS += --coverage -ffunction-sections -fdata-sections
WRAP_FLAGS = $(WRAP_FUNCS)

.PHONY: test clean-test coverage-test

test: clean-test $(TEST_PROGRAM)
\t@set +e; \
\t./$(TEST_PROGRAM) > $(TEST_REPORT_FILE) 2>$(TEST_LOG_FILE); \
\tstatus=$$?; \
\t$(MAKE) coverage-test; \
\texit $$status

$(TEST_PROGRAM): $(TEST_SRCS) $(STUB_SRCS)
\t$(CC) $(CPPFLAGS) $(CFLAGS_LINUX) $(CFLAGS) $(INCLUDE) $(COVERAGE_FLAGS) $(TEST_SRCS) $(STUB_SRCS) \
\t-o $(TEST_PROGRAM) \
\t$(TEST_LIBS) $(LDFLAGS) $(LDLIBS) $(LIBS) \
\t-Wl,--gc-sections \
\t$(WRAP_FLAGS)

coverage-test:
\t@echo "=== coverage-test ===" >> $(TEST_REPORT_FILE)
\t@echo "PWD=$$(pwd)" >> $(TEST_REPORT_FILE)
\t@echo "Target production gcov file: $(TARGET_GCOV)" >> $(TEST_REPORT_FILE)
\t@echo "Test gcno file: $(TEST_GCNO)" >> $(TEST_REPORT_FILE)
\t@echo "Files before gcov:" >> $(TEST_REPORT_FILE)
\t@ls -la >> $(TEST_REPORT_FILE) 2>&1 || true
\t@found=0; \
\tfor f in $(TEST_GCNO); do \
\t\t[ -e "$$f" ] || continue; \
\t\tfound=1; \
\t\techo "Running gcov -b -c $$f" >> $(TEST_REPORT_FILE); \
\t\tgcov -b -c "$$f" >> $(TEST_REPORT_FILE) 2>&1 || true; \
\tdone; \
\tif [ "$$found" -eq 0 ]; then \
\t\techo "No .gcno files found; coverage cannot be generated." >> $(TEST_REPORT_FILE); \
\tfi
\t@echo "Filtering .gcov files. Keeping only: $(TARGET_GCOV)" >> $(TEST_REPORT_FILE)
\t@find . -maxdepth 1 -name '*.gcov' ! -name '$(TARGET_GCOV)' -delete
\t@if [ ! -f "$(TARGET_GCOV)" ]; then \
\t\techo "WARNING: target production gcov file was not generated: $(TARGET_GCOV)" >> $(TEST_REPORT_FILE); \
\tfi
	@echo "Files after gcov filtering:" >> $(TEST_REPORT_FILE)
	@ls -la >> $(TEST_REPORT_FILE) 2>&1 || true
	@echo "=== COVERAGE REPORT ==="
	@cat $(TEST_REPORT_FILE)

clean-test:
\trm -f $(TEST_PROGRAM) $(TEST_REPORT_FILE) $(TEST_LOG_FILE) *.gcda *.gcno *.gcov *.o
$(LEGACY_TEST_SRCS): $(TEST_SRCS)
\t@:

$(LEGACY_TEST_PROGRAM): $(TEST_PROGRAM)
\t@:
"""
    write_text(unit_dir / "Makefile", content)

# endregion Scaffold


# region Per-function unit test generation

def _generate_unit_test_for_func(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    flags: dict,
    semantic_context_snapshot: dict,
    *,
    unit_dir_override: Optional[Path] = None,
) -> tuple[str, dict]:
    """
    Generate, compile/run, coverage-check, semantic-judge, and repair one unit test.

    Updated behavior:
      - if an existing test already produces valid gcov above threshold, start from judge
      - judge runs when coverage is above threshold, even if make/test returned non-zero
      - judge JSON parse failure raises and does NOT regenerate tests
      - backs up best CUnit test + Makefile by coverage >= threshold and best judge score
      - semantic judge JSON is simple: {"score": int, "reason": str}
    """
    func_id = func["id"]
    safe_id = _safe_filename(func_id)
    test_dir: Path = paths["test_dir"]
    process_name: str = paths["process_name"]

    repo_root = cfg.source_dir.parent.parent.resolve()
    unit_dir = unit_dir_override or (test_dir / "_unit_tests" / safe_id)
    unit_test_file = unit_dir / f"test_{safe_id}.c"
    unit_makefile = unit_dir / "Makefile"
    judge_verdict_file = unit_dir / "judge_verdict.json"
    coverage_file = unit_dir / "coverage.json"
    source_file_abs = _resolve_source_file(cfg, func["source_file"])

    master_test_file = test_dir / f"test_{process_name}.c"
    master_makefile = paths["makefile"]
    stub_gen_dir = test_dir / "_stub_gen"

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
    source_makefile_for_compile_fix = (
        str(existing_source_makefiles[0])
        if existing_source_makefiles
        else str(cfg.source_dir.resolve() / "Makefile")
    )

    actual_source_files_text = _source_files_json_for_prompt(cfg)

    def _strip_c_comments_and_strings_for_test_check(text: str) -> str:
        try:
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
            text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
            text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
            return text
        except Exception:
            return text

    def _has_real_cu_add_test(path: Path) -> bool:
        try:
            txt = read_text(path)
        except Exception:
            return False
        cleaned = _strip_c_comments_and_strings_for_test_check(txt)
        return re.search(r"\bCU_add_test\s*\(", cleaned) is not None

    def _coverage_pct(cov_obj: Optional[dict]) -> Optional[float]:
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

    last_make_ok = True
    # coverage_result: raw gcov dict returned by check_function_coverage().
    # coverage_pct:   line-coverage % for the target function's line range.
    coverage_result: Optional[dict] = None
    coverage_pct: Optional[float] = None
    last_build_diag: Optional[str] = None
    last_judge: Optional[dict] = None

    # If the per-function test already exists and coverage is already good
    # enough, validate it first and skip any scaffold generation.
    if unit_test_file.exists() and unit_makefile.exists() and _has_real_cu_add_test(unit_test_file):
        try:
            _sync_stub_srcs(unit_test_file, unit_makefile, test_dir)
            sync_wrap_flags(unit_test_file, unit_makefile)

            existing_make_res = run_make_test(cfg, unit_dir)
            last_make_ok = bool(existing_make_res.get("ok"))

            coverage_result = check_function_coverage(
                unit_dir,
                source_file_abs,
                func["start_line"],
                func["end_line"],
                source_root=cfg.source_dir.resolve(),
            )
            coverage_pct = _coverage_pct(coverage_result)

            if coverage_pct is not None and coverage_pct >= float(cfg.coverage_threshold):
                print(
                    f"[pipeline] existing test has coverage={coverage_pct}%, starting from semantic judge: {func_id}",
                    file=sys.stderr,
                )

                func_for_prompt = {**func, "source_file": str(source_file_abs)}

                judge = run_semantic_test_judge(
                    cfg,
                    test_dir=unit_dir,
                    repo_root=repo_root,
                    process_name=process_name,
                    test_file=unit_test_file,
                    func=func_for_prompt,
                    coverage=coverage_result or {},
                    make_result=existing_make_res or {},
                )

                backup_good_cunit_if_best(
                    unit_dir=unit_dir,
                    unit_test_file=unit_test_file,
                    unit_makefile=unit_makefile,
                    func=func_for_prompt,
                    coverage_pct=coverage_pct,
                    coverage=coverage_result or {},
                    make_result=existing_make_res or {},
                    judge_verdict=judge,
                    cfg=cfg,
                )

                write_json(coverage_file, coverage_result or {})
                write_json(judge_verdict_file, judge)
                last_judge = judge

                if judge.get("passed"):
                    print(
                        f"[pipeline] unit test PASSED from existing gcov: {func_id} score={judge.get('score')}",
                        file=sys.stderr,
                    )
                    return func_id, {
                        "passed": True,
                        "coverage_pct": coverage_pct,
                        "semantic_score": judge.get("score"),
                        "verdict": judge,
                        "unit_dir": str(unit_dir),
                    }

                last_build_diag = (
                    "Existing test reached coverage threshold, but semantic judge failed.\n"
                    + json.dumps(judge, ensure_ascii=False, indent=2, default=str)
                )
            else:
                print(
                    f"[pipeline] existing test judge FAILED: {func_id} score={last_judge.get('score') if last_judge else 'n/a'}",
                    file=sys.stderr,
                )

        except RuntimeError:
            raise
        except Exception as e:
            print(
                f"[pipeline] existing-test judge fast path skipped for {func_id}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    def _safe_build_diag(make_res: dict) -> str:
        parts: list[str] = []

        try:
            diag = build_output_with_runtime_diagnostics(
                cfg,
                unit_dir,
                unit_test_file,
                make_res,
            )
            if diag:
                parts.append(str(diag))
        except Exception as e:
            parts.append(
                "[pipeline diagnostic fallback]\n"
                f"build_output_with_runtime_diagnostics failed: {type(e).__name__}: {e}\n"
            )

        try:
            parts.append(
                "\n===== make_res object =====\n"
                + json.dumps(make_res, ensure_ascii=False, indent=2, default=str)
            )
        except Exception:
            parts.append(f"\n===== make_res repr =====\n{make_res!r}")

        possible_logs = [
            unit_dir / f"test_{safe_id}_report.txt",
            unit_dir / f"test_{safe_id}_log.txt",
            unit_dir / "make.log",
            unit_dir / "build.log",
        ]

        try:
            possible_logs.extend(sorted(unit_dir.glob("*_report.txt")))
            possible_logs.extend(sorted(unit_dir.glob("*_log.txt")))
            possible_logs.extend(sorted(unit_dir.glob("*.log")))
        except Exception:
            pass

        seen_logs: set[Path] = set()
        for log_path in possible_logs:
            try:
                log_path = log_path.resolve()
            except Exception:
                pass

            if log_path in seen_logs:
                continue
            seen_logs.add(log_path)

            try:
                if log_path.exists() and log_path.is_file():
                    txt = read_text(log_path)
                    if len(txt) > 40000:
                        txt = "[truncated to last 40000 chars]\n" + txt[-40000:]
                    parts.append(f"\n===== log file: {log_path} =====\n{txt}")
            except Exception as e:
                parts.append(
                    f"\n===== log file read failed: {log_path} =====\n"
                    f"{type(e).__name__}: {e}"
                )

        if not parts:
            return (
                "No build diagnostics were available. "
                "make_res contained no readable output and no log files were found."
            )

        diag = "\n".join(parts)
        failure_hint = _repetitive_failure_hint(diag)
        if failure_hint:
            diag += "\n\n===== repetitive-failure hint =====\n" + failure_hint
        return diag

    def _repetitive_failure_hint(text: str) -> str:
        lowered = text.lower()
        if "no rule to make target" in lowered:
            return (
                "The Makefile/test-file contract is likely stale. "
                "Re-check TEST_PROGRAM and TEST_SRCS first. If the old build graph still asks for the unsuffixed names, use compatibility targets in the unit Makefile instead of patching wrappers."
            )
        if "overriding recipe for target" in lowered:
            return (
                "The generated Makefile is being overridden instead of extended cleanly. "
                "Regenerate the unit Makefile rather than layering more conflicting targets."
            )
        if "multiple definition" in lowered or "duplicate symbol" in lowered:
            return (
                "A wrapper/stub ownership conflict is present. "
                "Keep exactly one owner per symbol and remove the duplicate from STUB_SRCS or local __wrap_*."
            )
        if "undefined reference to '__wrap_" in lowered or "undefined reference to \"__wrap_" in lowered:
            return (
                "The wrap flag probably does not match the real link-time symbol. "
                "Verify the exact symbol name before adding more wrappers."
            )
        if "heap out of memory" in lowered or "invalid table size" in lowered:
            return (
                "This looks like an agent/runtime failure rather than a C test failure. "
                "Simplify the workflow or retry the agent, do not keep changing wrappers blindly."
            )
        return ""

    def _validated_stub_text() -> str:
        try:
            lines = [
                f" - func={fname} abs={abs_path}"
                for abs_path, fname in _stub_srcs_absolute(test_dir)
            ]
            return "\n".join(lines) if lines else " - none"
        except Exception as e:
            return f" - could not list validated stubs: {type(e).__name__}: {e}"

    def _source_makefile_text() -> str:
        return (
            "\n".join(f" - {p}" for p in existing_source_makefiles)
            if existing_source_makefiles
            else " - none found"
        )

    def _run_current_coverage() -> tuple[Optional[dict], Optional[float]]:
        current_source = _resolve_source_file(cfg, func["source_file"])
        cov_obj = check_function_coverage(
            unit_dir,
            current_source,
            func["start_line"],
            func["end_line"],
            source_root=cfg.source_dir.resolve(),
        )
        return cov_obj, _coverage_pct(cov_obj)

    def _location_context(
        func_for_prompt: dict,
        last_feedback: Optional[str],
    ) -> str:
        previous_diag_block = ""
        if last_feedback:
            previous_diag_block = f"""
================================================================================
PREVIOUS ATTEMPT FAILURE / DIAGNOSTICS
================================================================================

The previous attempt failed or did not satisfy requirements.
Read this before editing so you do not repeat the same mistake.

{last_feedback}

========================
END PREVIOUS ATTEMPT FAILURE / DIAGNOSTICS
========================

"""

        return f"""
========================
UNIT TEST GENERATION FILESYSTEM CONTEXT
========================

You are running inside an agent with repository folder:

 {repo_root}

The current unit test file is only a placeholder or previous failed attempt.
Rewrite it from scratch if needed.

TARGET FUNCTION
-------------
id:
 {func_id}

function object:
 {json.dumps(func_for_prompt, ensure_ascii=False, indent=2)}

production source file:
 {source_file_abs}

source line range:
 {func.get("start_line")} - {func.get("end_line")}

This is a test-writing task, not a compile-fix task.
Prefer a fresh, meaningful unit test over patching around old broken attempts.
Do not remove production-source wiring or replace it with an empty compile-only harness just to make the build pass.

FILES YOU MAY EDIT
-------------
Unit test file to write/repair:
 {unit_test_file}

Unit Makefile to edit only if required for this one unit build:
 {unit_makefile}

FILES TO READ
-------------
Production source:
 {source_file_abs}

MUST include the production source in the test like this:

#define main {process_name}_entry_main
#include "{source_file_abs}"
#undef main

Master integrated test file with working wrappers/stubs/helpers:
 {master_test_file}

Master test Makefile:
 {master_makefile}

Source Makefile candidates:
{_source_makefile_text()}

Generated stub directory:
 {stub_gen_dir}

Validated generated stubs currently linked by unit Makefile:
{_validated_stub_text()}

Actual project .c source files discovered:
{actual_source_files_text}

BUILD / CONTENT RULES
1. Keep the unit test focused on the target function and its real observable behavior.
2. For static targets, execute through a real caller or include the production .c in the test harness when needed.
3. If compilation fails, fix the Makefile/include-path issue first instead of stripping source inclusion.
4. Use the master test file only as a narrow reference for wrappers/helpers when needed.
5. Do not preserve old failed attempts unless they are still useful.
6. Do not add coverage-only tests or empty smoke tests.
7. Do not create fake project headers/types/macros/functions.
8. The final file must define real tests and real `CU_add_test(...)` registrations.
9. A compiling test is not enough. The generated CUnit executable must actually run the registered tests and execute the real production target function.
   Always call CU_initialize_registry() before CU_add_suite(), register tests with
   CU_add_test(), call CU_basic_run_tests(), and verify the target function is
   called at least once in gcov. If gcov says "function <target> called 0" or the
   target lines are still "#####", the test is invalid even if make succeeded.
10. If testing a static function by including the production .c file, make sure
    the Makefile does not also compile/link that same production .c separately.
    Generate gcov from the object that contains the included production source.
11. If the target calls a terminating function such as pmf_exit(), exit(), or abort(), add linker wrappers and safe __wrap_* functions so the test process continues long enough to write coverage data.
12. If the build graph still mentions the legacy unsuffixed names `test_{process_name}.c` or `unit_test_{process_name}`, do not create a temporary shim source file. Keep the real suffixed unit test file and satisfy the old names with compatibility targets in the unit Makefile.

{previous_diag_block}

END UNIT TEST GENERATION FILESYSTEM CONTEXT

Use the provided source snippet, current unit test, Makefile, and build log as the primary evidence.

Infer as much as possible from the given code before using tools.

Do not search broadly through the repository unless the build log identifies a specific missing symbol, type, macro, field, or header that cannot be resolved from the provided context.

If searching is necessary:
- Search only production source/include directories.
- Try to infer as much as you can from the source itself, prefer hit and trial in the given allowed code locations.
- Do not search tests/, _unit_tests/, backup directories, or agent_history/.
- Do not inspect previous agent transcripts.
- Do not run broad commands like:
    find / -...
    find <repo-root> -name ...
    ls -R <large-dir>
    grep -r <repo-root> ...
- Prefer one narrow query for the exact symbol/header.
- Stop after 2 failed searches and make the smallest reasonable inference.

Dont exit until you have written the test code in target test file and made sure it compiles and generates a .gcov file. Read the makefile to know what you can do.
"""

    # Continuation: already passed.
    if judge_verdict_file.exists():
        try:
            v = load_json(judge_verdict_file)
            if v.get("passed"):
                cached_coverage = None
                if coverage_file.exists():
                    try:
                        cached_coverage = load_json(coverage_file)
                    except Exception:
                        cached_coverage = None

                cached_pct = _coverage_pct(cached_coverage)

                print(f"[pipeline] unit test already done: {func_id}", file=sys.stderr)
                return func_id, {
                    "passed": True,
                    "coverage_pct": cached_pct,
                    "semantic_score": v.get("score"),
                    "verdict": v,
                    "unit_dir": str(unit_dir),
                }
        except Exception:
            pass

    max_attempts = int(cfg.max_test_attempts or 4)

    unit_dir = _scaffold_unit_test_dir(
        cfg,
        paths,
        func,
        unit_dir_override=unit_dir_override,
    )
    unit_test_file = unit_dir / f"test_{safe_id}.c"
    unit_makefile = unit_dir / "Makefile"
    judge_verdict_file = unit_dir / "judge_verdict.json"
    coverage_file = unit_dir / "coverage.json"

    attempt = 1

    # === Main generation loop ===
    # Each attempt: run agent → verify CU_add_test exists → build → check coverage → judge.
    # If coverage meets threshold: run semantic judge; pass on score >= min_score.
    # If build fails: run compile-fix agent, then re-check coverage.
    # If coverage stays below threshold: feed diagnostics into next attempt.
    while attempt <= max_attempts:
        source_file_abs = _resolve_source_file(cfg, func["source_file"])
        func_for_prompt = {**func, "source_file": str(source_file_abs)}
        semantic_context = dict(semantic_context_snapshot)

        unit_agent_location_context = _location_context(
            func_for_prompt=func_for_prompt,
            last_feedback=last_build_diag,
        )

        if last_judge and not last_judge.get("passed"):
            prompt = prompt_for_semantic_test_repair(
                process_name=process_name,
                test_file=str(unit_test_file),
                source_file_abs=str(source_file_abs.resolve()),
                func=func_for_prompt,
                coverage=coverage_result or {},
                judge_verdict=last_judge,
                semantic_context=semantic_context,
            )
        else:
            prompt = prompt_for_function_test_with_semantic_context(
                func=func_for_prompt,
                coverage=coverage_result or {},
                test_file=str(unit_test_file),
                source_file_abs=str(source_file_abs.resolve()),
                process_name=process_name,
                attempt=attempt,
                max_attempts=max_attempts,
                make_ok=last_make_ok,
                semantic_context=semantic_context,
                last_judge_verdict=last_judge,
            )

        prompt = f"""{prompt}

{unit_agent_location_context}

IMPORTANT BASELINE TEMPLATE RULE
--------------------------------
There is already a working parent minimal CUnit test for this process:

  {master_test_file}

This parent test is known to:
- compile
- link
- run CUnit correctly
- include the real production source correctly
- provide working wrappers/stubs/helpers
- generate non-zero coverage for the real src/ production file

Before writing this per-function unit test, read the parent minimal test file.

Use the parent minimal test as the base template.

Do NOT #include the parent test .c file directly.

Instead:
1. Copy/reuse its #include list.
2. Copy/reuse its production source include block:
     #define main {process_name}_entry_main
     #include "{source_file_abs}"
     #undef main
3. Copy/reuse its working wrapper functions.
4. Copy/reuse its mock globals and reset_mocks() helper if present.
5. Copy/reuse its CUnit main structure.
6. Replace or extend only the CUnit test cases for this target function.

Do not remove wrappers from the parent template just because they do not look
directly related to the target function. Since the production .c file is included
directly, those wrappers may still be needed for linking.

Prefer local wrappers copied from the parent minimal test over linking broad
generated _stub_gen/stub.c files.

If a generated _stub_gen/stub.c causes duplicate symbols such as:

  multiple definition of `err_code'

then do NOT use that stub.c. Remove it from STUB_SRCS or leave STUB_SRCS empty
and implement only the required __wrap_* function locally in this unit test file.

If the link error says:

  undefined reference to `__wrap_xxx'

then add/copy a local wrapper:

  __wrap_xxx(...)

with the real signature if known, or the smallest compatible signature inferred
from the production/header files.

Do not add global variables to stubs unless they are declared extern and already
owned by production. Avoid defining project globals in generated stubs.

FINAL TASK:
- Rewrite or repair this file with real CUnit tests:
    {unit_test_file}

FINAL REQUIREMENTS:
- Add at least one real CU_add_test().
- Execute target range {func.get("start_line")}-{func.get("end_line")} directly or through a real caller.
- Read the production source:
    {source_file_abs}
- Read the master integrated test as wrapper/helper reference:
    {master_test_file}
- Read/use generated stubs as needed:
    {stub_gen_dir}
- Edit the unit Makefile only if required:
    {unit_makefile}
- Do not create fake project headers, fake replacement headers, spoof typedefs,
  spoof structs, spoof enums, spoof macros, or fake production functions.
- If compilation fails due to missing headers/types/macros, fix the unit
  Makefile include paths/flags using the real source Makefile and real headers.
- Prefer real project definitions from production headers over local test declarations.

CRITICAL WRAP FLAG RULE
-----------------------
If you define any local wrapper function in the test file, such as:

  __wrap_clg_logoutput
  __wrap_pmf_exit
  __wrap_NcmConnect

then the unit Makefile must contain the matching linker wrap flag:

  -Wl,--wrap=clg_logoutput
  -Wl,--wrap=pmf_exit
  -Wl,--wrap=NcmConnect

A local __wrap_xxx function is not enough by itself.

Before running make test:
1. Inspect the unit test file for every __wrap_xxx function you defined.
2. Inspect the unit Makefile.
3. Ensure WRAP_FLAGS or WRAP_FUNCS contains matching:
     -Wl,--wrap=xxx
4. If the flag is missing, edit the unit Makefile and add it.

Example:

  WRAP_FUNCS += -Wl,--wrap=clg_logoutput
  WRAP_FUNCS += -Wl,--wrap=pmf_exit
  WRAP_FUNCS += -Wl,--wrap=NcmConnect

or:

  WRAP_FLAGS += -Wl,--wrap=clg_logoutput

Do not finish until every __wrap_xxx has a matching -Wl,--wrap=xxx flag.

CRITICAL EXECUTION / COVERAGE REQUIREMENTS
----------------------------------------
A unit test is valid only if it compiles, runs, and executes the real production
source code for the target function.

You must ensure all of the following:

1. CUnit must be initialized correctly:
   - Call CU_initialize_registry() before CU_add_suite().
   - Check CU_add_suite() and CU_add_test() return values.
   - Call CU_basic_run_tests().
   - Do not exit before CU_basic_run_tests().

2. Register real tests:
   - The file must contain at least one real CU_add_test(...).
   - The registered test function must execute the target function or a real caller
     that reaches the target function.

3. Execute the real production code:
   - Do not create a fake implementation of the target function.
   - Do not replace the target function with a stub.
   - Do not only test mocks or wrappers.
   - The test must cause the real source lines in the target range to run.

4. For static target functions:
   - Prefer including the production .c file directly in the test file:
     #define main <renamed_main>
     #include "path/to/production.c"
     #undef main
   - Then call the static target function directly from the test.
   - If the production .c file is included in the test file, do NOT also compile
     or link that same production .c separately, because that can cause duplicate
     definitions or coverage mismatch.

5. For non-static target functions:
   - Either link the real production object or include the production .c.
   - Do not declare or define a fake local copy of the function.

6. Prevent premature process termination:
   - If the target calls exit(), pmf_exit(), abort(), longjmp-like framework exits,
     or similar terminating functions, add proper linker wrappers such as:
     -Wl,--wrap=pmf_exit
     -Wl,--wrap=exit
   - Implement __wrap_<symbol>() in the test file so the test can continue.
   - The wrapper should record arguments and return safely if possible.

7. Wrappers must match actual called symbols:
   - If production calls pmf_exit(), implement __wrap_pmf_exit().
   - If production calls pmf_rmtimer(), implement __wrap_pmf_rmtimer().
   - Make sure the Makefile contains the matching -Wl,--wrap=<symbol> flag.
   - If a function is a macro that expands to another symbol, wrap or stub the expanded symbol, not only the macro name.

8. Coverage must be generated from the object that contains the production code:
   - Compile with --coverage or equivalent gcov flags.
   - Run the test binary so .gcda files are produced.
   - Run gcov on the .gcno file for the object that actually contains the included
     or linked production source.
   - The resulting .gcov must correspond to the real production source file.

9. The generated gcov must prove execution:
   - The target function must not show:
     function <target> called 0
     #####: lines in the target range
   - It should show:
     function <target> called 1 or more
     numeric execution counts on the target lines

10. Do not mistake successful build for successful coverage:
   - A test that compiles but never calls the target is invalid.
   - A test that initializes CUnit incorrectly and exits before running tests is invalid.
   - A test that only verifies stub counters without reaching the target source is invalid.

CUNIT API REFERENCE (use only these — do not invent variants)
------------------------------------------------------------------
Every macro below has a _FATAL version (stops test on failure) EXCEPT CU_PASS:
  CU_ASSERT(expr)              CU_ASSERT_FATAL(expr)
  CU_TEST(expr)                CU_TEST_FATAL(expr)
  CU_ASSERT_TRUE(v)            CU_ASSERT_TRUE_FATAL(v)
  CU_ASSERT_FALSE(v)           CU_ASSERT_FALSE_FATAL(v)
  CU_ASSERT_EQUAL(a, e)        CU_ASSERT_EQUAL_FATAL(a, e)
  CU_ASSERT_NOT_EQUAL(a, e)    CU_ASSERT_NOT_EQUAL_FATAL(a, e)
  CU_ASSERT_PTR_EQUAL(a, b)    CU_ASSERT_PTR_EQUAL_FATAL(a, b)
  CU_ASSERT_PTR_NOT_EQUAL(a,b) CU_ASSERT_PTR_NOT_EQUAL_FATAL(a,b)
  CU_ASSERT_PTR_NULL(p)        CU_ASSERT_PTR_NULL_FATAL(p)
  CU_ASSERT_PTR_NOT_NULL(p)    CU_ASSERT_PTR_NOT_NULL_FATAL(p)
  CU_ASSERT_STRING_EQUAL(a,b)  CU_ASSERT_STRING_EQUAL_FATAL(a,b)
  CU_ASSERT_STRING_NOT_EQUAL(a,b)  CU_ASSERT_STRING_NOT_EQUAL_FATAL(a,b)
  CU_ASSERT_NSTRING_EQUAL(a,b,n)   CU_ASSERT_NSTRING_EQUAL_FATAL(a,b,n)
  CU_ASSERT_NSTRING_NOT_EQUAL(a,b,n) CU_ASSERT_NSTRING_NOT_EQUAL_FATAL(a,b,n)
  CU_ASSERT_DOUBLE_EQUAL(a,e,g)    CU_ASSERT_DOUBLE_EQUAL_FATAL(a,e,g)
  CU_ASSERT_DOUBLE_NOT_EQUAL(a,e,g) CU_ASSERT_DOUBLE_NOT_EQUAL_FATAL(a,e,g)
  CU_PASS(msg)   [no fatal variant]
  CU_FAIL(msg)                 CU_FAIL_FATAL(msg)

Do NOT use: CU_ASSERT_INT, CU_ASSERT_INT_EQUAL, CU_ASSERT_NULL,
            CU_ASSERT_NOT_NULL, CU_ASSERT_ZERO, CU_PASS_FATAL,
            or any other variant not listed above — they do not exist.

Setup boilerplate:
  CU_initialize_registry();
  CU_pSuite s = CU_add_suite("name", NULL, NULL);
  CU_add_test(s, "test_name", test_func);
  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  unsigned failures = CU_get_number_of_failures();
  CU_cleanup_registry();
  return failures == 0 ? 0 : 1;
------------------------------------------------------------------

Before finishing, run:
  make test

Then verify:
 - test binary executed
 - CUnit tests ran
 - .gcno exists
 - .gcda exists after running the test
 - .gcov exists
 - target function has called count >= 1 in gcov
"""

        run_agent(
            cfg,
            unit_dir,
            prompt,
            f"{safe_id}_test_{int(time.time())}.json",
            folder=repo_root,
            history_dir=unit_dir / "agent_history",
        )

        if not _has_real_cu_add_test(unit_test_file):
            write_text(unit_test_file, f"/* PLACEHOLDER — agent wrote no CU_add_test, rewriting. Target: {func_id} */\n")
            last_build_diag = (
                "Main agent produced no real CU_add_test(). "
                "File reset to placeholder — rewrite from scratch on next attempt."
            )
            print(
                f"[pipeline] {func_id} attempt {attempt}: no CU_add_test, reset to placeholder",
                file=sys.stderr,
            )
            attempt += 1
            continue

        _sync_stub_srcs(unit_test_file, unit_makefile, test_dir)
        sync_wrap_flags(unit_test_file, unit_makefile)

        make_res = run_make_test(cfg, unit_dir)
        last_make_ok = bool(make_res.get("ok"))

        coverage_result, coverage_pct = _run_current_coverage()

        if coverage_pct is not None and coverage_pct >= float(cfg.coverage_threshold):
            judge = run_semantic_test_judge(
                cfg,
                test_dir=unit_dir,
                repo_root=repo_root,
                process_name=process_name,
                test_file=unit_test_file,
                func=func_for_prompt,
                coverage=coverage_result or {},
                make_result=make_res or {},
            )

            backup_good_cunit_if_best(
                unit_dir=unit_dir,
                unit_test_file=unit_test_file,
                unit_makefile=unit_makefile,
                func=func_for_prompt,
                coverage_pct=coverage_pct,
                coverage=coverage_result or {},
                make_result=make_res or {},
                judge_verdict=judge,
                cfg=cfg,
            )

            write_json(coverage_file, coverage_result or {})
            write_json(judge_verdict_file, judge)
            last_judge = judge

            if judge.get("passed"):
                print(
                    f"[pipeline] unit test PASSED: {func_id} score={judge.get('score')}",
                    file=sys.stderr,
                )
                return func_id, {
                    "passed": True,
                    "coverage_pct": coverage_pct,
                    "semantic_score": judge.get("score"),
                    "verdict": judge,
                    "unit_dir": str(unit_dir),
                }

            print(
                f"[pipeline] judge FAILED: {func_id} score={judge.get('score')}",
                file=sys.stderr,
            )

            try:
                last_build_diag = (
                    "Semantic judge failed even though target coverage met threshold.\n"
                    + json.dumps(judge, ensure_ascii=False, indent=2, default=str)
                )
            except Exception:
                last_build_diag = f"Semantic judge failed: {judge!r}"

            attempt += 1
            continue

        if not make_res["ok"]:
            diag = _safe_build_diag(make_res)
            last_build_diag = diag

            compile_fix_prompt = prompt_for_compile_fix(
                str(unit_makefile),
                str(unit_test_file),
                diag,
                source_dir=str(cfg.source_dir.resolve()),
                source_makefile=source_makefile_for_compile_fix,
                actual_source_files=[str(p) for p in _project_source_files(cfg)],
            )

            compile_fix_prompt = f"""{compile_fix_prompt}

{_location_context(func_for_prompt=func_for_prompt, last_feedback=diag)}

COMPILE/RUNTIME FIX INSTRUCTIONS:
- You may edit:
    {unit_test_file}
    {unit_makefile}

- Use this master integrated test as reference for working wrappers/helpers:
    {master_test_file}

- Use this production source:
    {source_file_abs}

- Use generated stubs from:
    {stub_gen_dir}

- Use the real source Makefile/header layout before changing declarations:
    {source_makefile_for_compile_fix}

- Fix the compile/link/runtime issue with minimal changes.
- Preserve real CU_add_test registrations.
- Do not revert to an empty scaffold.
- If compile/runtime logs are missing, use the make_res object and current files
to infer the failure.

STRICT COMPILE-FIX RULES:
- Do not create fake project headers.
- Do not create spoof replacement headers with names matching real project headers.
- Do not invent typedefs, structs, enums, macros, globals, or prototypes that
  already exist in the real project.
- Do not define fake production functions to satisfy the linker.
- Do not bypass the real production source under test.
- If a header is missing, fix INCLUDE/CPPFLAGS using the original Makefile or
  real project include directories.
- If a macro is missing, find where the real build defines it and add the same
  macro to this unit Makefile.
- If a type is missing, include the real header that defines it.
- If a symbol is unresolved, prefer adding the real source object/library or a
  proper __wrap_<symbol>() wrapper over fake implementations.
- Only add a local extern declaration when no usable real header exists and the
  declaration matches the production source exactly.
"""

            run_agent(
                cfg,
                unit_dir,
                compile_fix_prompt,
                f"{safe_id}_compile_fix_{int(time.time())}.json",
                folder=repo_root,
                history_dir=unit_dir / "agent_history",
            )

            if not _has_real_cu_add_test(unit_test_file):
                write_text(unit_test_file, f"/* PLACEHOLDER — compile-fix removed CU_add_test, rewriting. Target: {func_id} */\n")
                last_build_diag = (
                    "Compile-fix agent removed all CU_add_test() registrations. "
                    "File reset to placeholder — rewrite from scratch."
                )
                print(
                    f"[pipeline] {func_id} attempt {attempt}: compile_fix ate CU_add_test, reset to placeholder",
                    file=sys.stderr,
                )
                attempt += 1
                continue

            _sync_stub_srcs(unit_test_file, unit_makefile, test_dir)
            sync_wrap_flags(unit_test_file, unit_makefile)

            make_res = run_make_test(cfg, unit_dir)
            last_make_ok = bool(make_res.get("ok"))

            coverage_result, coverage_pct = _run_current_coverage()

            if coverage_pct is not None and coverage_pct >= float(cfg.coverage_threshold):
                judge = run_semantic_test_judge(
                    cfg,
                    test_dir=unit_dir,
                    repo_root=repo_root,
                    process_name=process_name,
                    test_file=unit_test_file,
                    func=func_for_prompt,
                    coverage=coverage_result or {},
                    make_result=make_res or {},
                )

                backup_good_cunit_if_best(
                    unit_dir=unit_dir,
                    unit_test_file=unit_test_file,
                    unit_makefile=unit_makefile,
                    func=func_for_prompt,
                    coverage_pct=coverage_pct,
                    coverage=coverage_result or {},
                    make_result=make_res or {},
                    judge_verdict=judge,
                    cfg=cfg,
                )

                write_json(coverage_file, coverage_result or {})
                write_json(judge_verdict_file, judge)
                last_judge = judge

                if judge.get("passed"):
                    print(
                        f"[pipeline] unit test PASSED after compile_fix: {func_id} score={judge.get('score')}",
                        file=sys.stderr,
                    )
                    return func_id, {
                        "passed": True,
                        "coverage_pct": coverage_pct,
                        "semantic_score": judge.get("score"),
                        "verdict": judge,
                        "unit_dir": str(unit_dir),
                    }

                print(
                    f"[pipeline] judge FAILED after compile_fix: {func_id} score={judge.get('score')}",
                    file=sys.stderr,
                )

                last_build_diag = (
                    "Semantic judge failed after compile/runtime fix even though coverage met threshold.\n"
                    + json.dumps(judge, ensure_ascii=False, indent=2, default=str)
                )
            else:
                if not make_res["ok"]:
                    last_build_diag = _safe_build_diag(make_res)
                else:
                    try:
                        cov_text = json.dumps(coverage_result or {}, ensure_ascii=False, indent=2, default=str)
                    except Exception:
                        cov_text = repr(coverage_result)

                    last_build_diag = f"""
Build succeeded after compile_fix, but target coverage did not meet threshold.

Target:
 {func_id}

Required threshold:
 {cfg.coverage_threshold}

Observed coverage:
 {coverage_pct}

Coverage object:
{cov_text}

The next attempt should make the test execute lines {func.get("start_line")}-{func.get("end_line")}
in:
 {source_file_abs}
"""

            attempt += 1
            continue

        print(
            f"[pipeline] {func_id} attempt {attempt} coverage={coverage_pct}% make_ok={make_res['ok']}",
            file=sys.stderr,
        )

        try:
            cov_text = json.dumps(coverage_result or {}, ensure_ascii=False, indent=2, default=str)
        except Exception:
            cov_text = repr(coverage_result)

        last_build_diag = f"""
Build succeeded, but target coverage did not meet threshold.

Target:
 {func_id}

Required threshold:
 {cfg.coverage_threshold}

Observed coverage:
 {coverage_pct}

Coverage object:
{cov_text}

The next attempt should make the test execute lines {func.get("start_line")}-{func.get("end_line")}
in:
 {source_file_abs}
"""

        attempt += 1

    try:
        write_json(
            unit_dir / "unit_test_failed.json",
            {
                "passed": False,
                "func_id": func_id,
                "attempts": max_attempts,
                "coverage_pct": coverage_pct,
                "make_ok": last_make_ok,
                "last_judge": last_judge,
                "last_build_diag": last_build_diag,
                "unit_dir": str(unit_dir),
            },
        )
    except Exception:
        pass

    print(
        f"[pipeline] unit test FAILED after {max_attempts} attempts: "
        f"{func_id} coverage={coverage_pct}% make_ok={last_make_ok}",
        file=sys.stderr,
    )

    return func_id, {
        "passed": False,
        "coverage_pct": coverage_pct,
        "semantic_score": None if not last_judge else last_judge.get("score"),
        "verdict": last_judge,
        "unit_dir": str(unit_dir),
        "error": f"max attempts reached: {max_attempts}",
        "last_make_ok": last_make_ok,
    }

# endregion Per-function unit test generation


# region Parallel driver

def parallel_generate_unit_tests(
    cfg: PipelineConfig,
    paths: dict,
    analysis: dict,
    flags: dict,
) -> dict[str, dict]:
    """
    Stage 4: generate unit tests level-by-level, parallel within each level.
    Waits for level N to fully complete before starting level N-1 so semantic
    context flows upward correctly.
    """
    test_dir: Path = paths["test_dir"]
    funcs = functions_leaf_first(analysis)

    levels: dict[int, list[dict]] = {}
    for f in funcs:
        depth = int(f.get("depth", 0))
        levels.setdefault(depth, []).append(f)

    all_results: dict[str, dict] = {}
    workers = max(1, int(getattr(cfg, "max_unit_test_workers", 4)))

    for depth in sorted(levels.keys()):
        remaining = (cfg.max_functions - len(all_results)) if cfg.max_functions is not None else None
        if remaining is not None and remaining <= 0:
            break
        level_funcs = [
            f for f in levels[depth]
            if not (cfg.only_function and f["id"] != cfg.only_function)
            and not (cfg.only_level is not None and f.get("depth") != cfg.only_level)
        ]
        if remaining is not None:
            level_funcs = level_funcs[:remaining]
        if not level_funcs:
            continue

        semantic_context = _load_semantic_context(test_dir)
        print(f"[pipeline] Stage 4: depth={depth} {len(level_funcs)} funcs in parallel", file=sys.stderr)

        with ThreadPoolExecutor(max_workers=min(workers, len(level_funcs))) as pool:
            futs = {
                pool.submit(_generate_unit_test_for_func, cfg, paths, func, flags, semantic_context): func
                for func in level_funcs
            }
            for fut in as_completed(futs):
                func = futs[fut]
                try:
                    fid, result = fut.result()
                    all_results[fid] = result
                except Exception as e:
                    print(f"[pipeline] unit test error {func['id']}: {e}", file=sys.stderr)
                    all_results[func["id"]] = {"passed": False, "error": str(e)}

        for func in level_funcs:
            result = all_results.get(func["id"], {})
            if result.get("passed"):
                src_abs = _resolve_source_file(cfg, func["source_file"])
                _append_semantic_context(test_dir, {**func, "source_file": str(src_abs)}, result["verdict"])

    # Mark unit tests as completed in context if all targeted functions passed
    all_passed = True
    if not all_results:
        all_passed = False
    else:
        for fid, res in all_results.items():
            if not res.get("passed"):
                all_passed = False
                break

    if all_passed:
        context_file = test_dir / "_pipeline_context.json"
        if context_file.exists():
            ctx = load_json(context_file)
            ctx["unit_tests_completed"] = True
            ctx["unit_test_results"] = all_results
            write_json(context_file, ctx)
            print(f"[pipeline] All unit tests passed. Marked context as completed.", file=sys.stderr)

    return all_results

# endregion Parallel driver
