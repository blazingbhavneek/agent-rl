#!/usr/bin/env python3
"""
Simplified CUnit test-generation pipeline.

src/<process>  ->  tests/<process>/Makefile
                   tests/<process>/test_<process>.c
                   tests/<process>/agent_history/

Iterates functions leaf -> root.
Uses gcov line-range coverage as the source of truth.
Resumes from actual files (no state DB).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from difflib import SequenceMatcher

from stub.stub import ProjectAnalyzer          # type: ignore


# region Config & Infrastructure

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
    # Continuation threshold:
    # If a function already has coverage >= this value, skip it.
    coverage_threshold: float = 80.0
    # Stub integration batching:
    # Generated stubs are integrated in chunks instead of all at once.
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

def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

def now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat()

def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")

def parse_source_makefile_flags(source_makefile: Path) -> dict:
    """Extract build flags from Makefile assignments, including continuations."""
    text = read_text(source_makefile)
    flags: dict[str, str] = {}
    wanted = {"CFLAGS", "CFLAGS_LINUX", "CPPFLAGS", "INCLUDE", "LDFLAGS", "LDLIBS", "LIBS"}

    current: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:+?]?=\s*(.*)$", line)
        if m and m.group(1) in wanted:
            current = m.group(1)
            val = m.group(2).strip().rstrip("\\").strip()
            if val:
                flags[current] = (flags.get(current, "") + " " + val).strip()
            if not line.rstrip().endswith("\\"):
                current = None
            continue

        if current:
            val = line.strip().rstrip("\\").strip()
            if val and not val.startswith("#"):
                flags[current] = (flags.get(current, "") + " " + val).strip()
            if not line.rstrip().endswith("\\"):
                current = None

    return flags

# endregion Config & Infrastructure

# region Analysis
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
    for depth in sorted(levels.keys(), reverse=True):    # highest depth = leaves
        for fid in levels[depth]:
            if fid in index:
                ordered.append(index[fid])
    return ordered

def collect_stub_candidates(analysis: dict) -> list[str]:
    names: set[str] = set()
    for funcs in (analysis.get("stub_candidates") or {}).values():
        for n in funcs or []:
            names.add(n)
    # for f in analysis.get("functions", []) or []:
    #     for n in (f.get("calls_stub_candidates") or []):
    #         names.add(n)
    #     for n in (f.get("callback_refs_raw") or []):
    #         names.add(n)
    return sorted(names)

# endregion Analysis

# region Source File Helpers
TEST_FILE_MARKERS = [
    "/* === Includes === */",
    "/* === Compatibility Definitions === */",
    "/* === Test Globals === */",
    "/* === Test Helpers === */",
    "/* === Linker Wrapper Stubs === */",
    "/* === Test Cases === */",
    "/* === Test Registration === */",
]

def _project_source_files(cfg: PipelineConfig) -> list[Path]:
    """
    Return actual absolute .c files under cfg.source_dir.
    No guessing like:
        cfg.source_dir.parent / (process_name + ".c")
    If user passed:
        --source /.../src/dio100d
    and real file is:
        /.../src/dio100d/dio100d.c
    this returns that exact absolute path.
    """
    source_dir = cfg.source_dir.resolve()
    if source_dir.is_file() and source_dir.suffix == ".c":
        return [source_dir]
    if not source_dir.exists():
        return []
    return sorted(
        p.resolve()
        for p in source_dir.rglob("*.c")
        if p.is_file()
    )

def _resolve_source_file(cfg: PipelineConfig, source_file: str | Path) -> Path:
    source_dir = cfg.source_dir.resolve()
    src = Path(source_file)
    if src.is_absolute():
        return src.resolve()
    direct = (source_dir / src).resolve()
    if direct.exists():
        return direct
    matches = sorted(
        p.resolve()
        for p in source_dir.rglob(src.name)
        if p.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        wanted_suffix = src.as_posix()
        for m in matches:
            if m.as_posix().endswith(wanted_suffix):
                return m
        return matches[0]
    return direct

def _source_files_json_for_prompt(cfg: PipelineConfig) -> str:
    """
    Format actual source files for prompts.
    """
    files = _project_source_files(cfg)
    if not files:
        return " (no .c files found under cfg.source_dir)"
    return "\n".join(f" - {p}" for p in files)

def _source_includes_for_test_file(cfg: PipelineConfig, test_file: Path) -> list[str]:
    """
    Build production #include lines using actual discovered source files.
    Example:
    #include "../../src/dio100d/dio100d.c"
    This is computed from absolute paths, not guessed.
    """
    lines: list[str] = []
    for src in _project_source_files(cfg):
        rel = Path(os.path.relpath(src, start=test_file.parent)).as_posix()
        lines.append(f'#include "{rel}"')
    return lines

# endregion Source File Helpers

# region Agent & Build Infrastructure
def _snapshot_dir(d: Path) -> dict[Path, bytes]:
    snap: dict[Path, bytes] = {}
    try:
        for f in d.rglob("*"):
            if f.is_file():
                try:
                    snap[f] = f.read_bytes()
                except OSError:
                    pass
    except OSError:
        pass
    return snap

def _restore_from_snapshot(snap: dict[Path, bytes], protect_dir: Path) -> None:
    for f, orig in snap.items():
        try:
            if f.exists() and f.read_bytes() != orig:
                f.write_bytes(orig)
                print(f"[pipeline] GUARD: restored {f}", file=sys.stderr)
        except OSError:
            pass
    try:
        for f in protect_dir.rglob("*"):
            if f.is_file() and f not in snap:
                try:
                    f.unlink()
                    print(f"[pipeline] GUARD: deleted agent-created file {f}", file=sys.stderr)
                except OSError:
                    pass
    except OSError:
        pass

def run_agent(
    cfg: PipelineConfig,
    work_dir: Path,
    prompt: str,
    history_name: str,
    *,
    folder: Optional[Path] = None,
    history_dir: Optional[Path] = None,
    max_iterations: Optional[int] = None,
    timeout_sec: Optional[int] = None,
    protect_source: bool = True,
) -> dict:
    """Invoke agent.js with an external prompt."""
    agent_folder = folder or work_dir
    hist_dir = history_dir or (work_dir / "agent_history")
    history_path = hist_dir / history_name
    history_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_file = history_path.with_suffix(".prompt.txt")
    write_text(prompt_file, prompt)

    cmd = [
        "node", str(cfg.agent_js),
        "--folder", str(agent_folder),
        "--prompt-file", str(prompt_file),
        "--history", str(history_path),
        "--capture-raw-http-trace",
    ]

    env = os.environ.copy()
    env["MAX_ITERATIONS"] = str(
        max_iterations if max_iterations is not None else cfg.max_agent_iterations
    )
    env["PYTHON_BIN"] = cfg.python_bin

    actual_timeout = timeout_sec if timeout_sec is not None else cfg.agent_timeout_sec

    if cfg.dry_run:
        print(f"[pipeline][dry-run] would run: {' '.join(cmd)}", file=sys.stderr)
        return {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}

    _protect_dir = cfg.source_dir.parent.resolve() if protect_source else None
    _snap = _snapshot_dir(_protect_dir) if _protect_dir is not None else {}

    print(f"[pipeline] agent -> {history_name}", file=sys.stderr)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=actual_timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        proc = None

    elapsed = time.time() - t0
    res = {
        "exit_code": proc.returncode if proc else -1,
        "stdout": (proc.stdout if proc else "")[:4000],
        "stderr": (proc.stderr if proc else "")[:4000],
        "timed_out": timed_out,
        "elapsed": elapsed,
    }
    write_json(history_path.with_suffix(".result.json"), res)
    print(
        f"[pipeline] agent exit={res['exit_code']} "
        f"elapsed={elapsed:.1f}s timed_out={timed_out}",
        file=sys.stderr,
    )
    if _snap:
        _restore_from_snapshot(_snap, _protect_dir)
    return res

def run_make_test(test_dir: Path, timeout: int = 300) -> dict:
    cmd = ["make", "test"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(test_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "",
            "stderr": f"make test timed out after {timeout}s", "timed_out": True}

def check_function_coverage(
    test_dir: Path,
    source_file: str | Path,
    start_line: int,
    end_line: int,
    source_root: Optional[Path] = None,
) -> Optional[dict]:
    """
    Check coverage for original production source.
    Ignores test_*.c.gcov.
    Matches .gcov by Source: header, e.g.
    Source:../../src/dio100d/dio100d.c
    """
    test_dir = Path(test_dir).resolve()
    source_abs = Path(source_file).resolve()
    if source_root is not None:
        source_root = Path(source_root).resolve()

    # Refresh gcov outputs from generated .gcno files.
    for gcno in sorted(test_dir.glob("*.gcno")):
        subprocess.run(
            ["gcov", "-b", "-c", gcno.name],
            cwd=str(test_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    for _f in test_dir.glob("test_*.gcov"):
        try:
            _f.unlink()
        except Exception:
            pass

    gcov_files = sorted(test_dir.glob("*.gcov"))

    def _gcov_source(gcov_file: Path) -> Optional[Path]:
        try:
            lines = read_text(gcov_file).splitlines()
        except Exception:
            return None
        for line in lines[:40]:
            if "Source:" not in line:
                continue
            raw = line.split("Source:", 1)[1].strip()
            p = Path(raw)
            if p.is_absolute():
                return p.resolve()
            return (test_dir / p).resolve()
        return None

    matched_gcov = None
    print(f"[pipeline] coverage wanted source: {source_abs}", file=sys.stderr)
    for gcov_file in gcov_files:
        src = _gcov_source(gcov_file)
        print(
            f"[pipeline] coverage candidate: {gcov_file.name} -> {src}",
            file=sys.stderr,
        )
        if src is None:
            continue
        # Ignore test harness gcov.
        try:
            src.relative_to(test_dir)
            continue
        except ValueError:
            pass
        if src.name.startswith("test_"):
            continue
        # Only accept original project source files.
        if source_root is not None:
            try:
                src.relative_to(source_root)
            except ValueError:
                continue
        if src == source_abs:
            matched_gcov = gcov_file
            break

    if matched_gcov is None:
        print(
            "[pipeline] coverage: no matching production .gcov found\n"
            f" wanted source: {source_abs}\n"
            f" source_root: {source_root}\n"
            f" gcov files:\n"
            + "\n".join(f"    - {p}" for p in gcov_files),
            file=sys.stderr,
        )
        return None

    covered_lines = 0
    executable_lines = 0
    for line in read_text(matched_gcov).splitlines():
        m = re.match(r"^\s*([^:]+):\s*(\d+):", line)
        if not m:
            continue
        count_text = m.group(1).strip()
        line_no = int(m.group(2))
        if line_no < start_line or line_no > end_line:
            continue
        if count_text == "-":
            continue
        executable_lines += 1
        if count_text in ("#####", "====="):
            continue
        try:
            count = int(count_text.rstrip("*"))
        except ValueError:
            count = 0
        if count > 0:
            covered_lines += 1

    coverage_percent = (
        0.0
        if executable_lines == 0
        else (covered_lines / executable_lines) * 100.0
    )

    return {
        "source_file": str(source_abs),
        "gcov_file": str(matched_gcov),
        "range": {
            "start_line": start_line,
            "end_line": end_line,
        },
        "summary": {
            "covered_lines": covered_lines,
            "executable_lines": executable_lines,
            "coverage_percent": coverage_percent,
        },
    }

# endregion Agent & Build Infrastructure

# region Prompts
def prompt_for_compile_fix(
    makefile: str,
    test_file: str,
    build_output: str,
    source_dir: Optional[str] = None,
    source_makefile: Optional[str] = None,
    actual_source_files: Optional[list[str]] = None,
) -> str:
    source_dir_text = source_dir or "(not provided)"
    source_makefile_text = source_makefile or "(not provided)"
    if actual_source_files:
        actual_source_files_text = "\n".join(f" - {p}" for p in actual_source_files)
    else:
        actual_source_files_text = " (not provided)"

    return f"""

The CUnit test failed to compile, link, or run.

Fix the failure shown in the output below. It may be a compiler error, linker error, CUnit assertion failure, runtime crash, segfault, abort, or other execution failure.

After fix, make sure to compile them again and run them to make sure its ok now.

FILES YOU MAY EDIT
- Test Makefile:
  `{makefile}`
- Test C file:
  `{test_file}`

TEST HARNESS FIXING RULES
You are expected to fix the test harness, not production code.

You MAY change:
- the test C file,
- wrapper stubs,
- mock return values,
- fake/static test data,
- test setup/reset functions,
- local prototypes used only by the test,
- linker wrap flags in the test Makefile,
- include/compiler/linker settings in the test Makefile if needed.

You MUST NOT change:
- production source files,
- production headers,
- original source Makefile,
- generated external dependencies,
- system headers.

Important:
- Wrapper/stub signatures may be wrong. Inspect the real function prototypes and fix wrappers to match exactly.
- Mock return values may be wrong. If production dereferences a returned pointer, return valid static fake storage.
- Global fake data may be uninitialized. Initialize required fake globals in reset/setup or mock open/init functions.
- Linker wrap flags may be missing or stale. Preserve existing wrap flags, but add/remove test Makefile wrap flags if required by the test harness.
- Do not call `__real_*` from wrappers.
- Do not hide crashes by deleting assertions or replacing them with always-pass assertions.
- Do not edit production code to make the test pass.
- Do not create fake data, headers, or types just to make the test pass.

REFERENCE LOCATIONS
- Source folder passed by user:
  `{source_dir_text}`
- Original source Makefile:
  `{source_makefile_text}`

ACTUAL SOURCE FILES
{actual_source_files_text}

CRITICAL PATH RULES
- Do NOT invent source paths.
- Do NOT guess `src/<process>.c`.
- Do NOT guess a sibling file like:
  `source_dir.parent / "<process>.c"`
- Use only the actual source files listed above.
- If the test file includes production code, include from actual files under:
  `{source_dir_text}`

MAKEFILE RULE
The test Makefile should be based on:
```bash
cd <test_dir> && do_mkmf <source_folder>
```
Do NOT replace the whole Makefile.
If compile errors show missing headers, flags, defines, or libraries:
1. Read the original source Makefile:
   `{source_makefile_text}`
2. Copy or append missing relevant settings from the source Makefile into the test Makefile:
   - CFLAGS
   - CFLAGS_LINUX
   - CPPFLAGS
   - INCLUDE
   - LDFLAGS
   - LDLIBS
   - LIBS
3. Preserve existing do_mkmf-generated content.
4. Preserve existing WRAP_FUNCS and wrapper flags.

FAILING `make test` OUTPUT
{build_output}

COMMON FIXES
- Missing header:
  copy/include correct `INCLUDE += -I...` path from source Makefile.
- Missing macro/type:
  copy needed `CFLAGS += -D...` or include path from source Makefile.
- Missing library:
  copy needed `LDFLAGS`, `LDLIBS`, or `LIBS`.
- Stub signature mismatch:
  fix wrapper signature in `{test_file}`.
- Missing local prototype/type for wrapper:
  define it inside `{test_file}`, above `/* === Linker Wrapper Stubs === */`.

STRICT RULES
- You may edit only:
  - `{makefile}`
  - `{test_file}`
- Do NOT edit production source.
- Do NOT edit production headers.
- Do NOT create extra dependency headers.
- Preserve all existing tests and stubs.
- Do not call `__real_*` from wrappers.

After editing, run:
make test

When done, call submit_and_exit.
"""

def prompt_for_minimal_test(process_name: str, test_file: str, entry_sym: str) -> str:
    return f"""
You are editing an existing CUnit test file.

File to modify:
{test_file}

Process:
{process_name}

Entry function:
{entry_sym}

Goal:
Make a minimal startup test that compiles, links, runs, and exits quickly.

Do not over-engineer this stage.
Do not deeply rewrite the test framework.
Do not create many scenario tests here.
Do not run real blocking loops, hardware calls, IPC, timers, daemon loops, or real sleeps.

Required:
1. Add or fix wrappers/stubs for external dependencies needed by {entry_sym}.
2. Stub any mainloop/blocking function so it returns immediately.
3. Add configurable return values only where needed to make startup run.
4. Add simple call counters for important startup dependencies if easy:
   - process start/init
   - logging init
   - event registration
   - timer registration
   - mainloop
   - exit
5. Add/reset those counters in reset_mocks() if reset_mocks exists.
6. Add one minimal CUnit test that calls {entry_sym}.
7. Assert only basic startup behavior that is obvious from existing code.
8. Ensure CUnit main returns non-zero on test failure using CU_get_number_of_failures().

Important:
- This is only the bootstrap compile/run stage.
- Do not attempt full semantic coverage here.
- Keep changes small.
- Preserve existing tests.
- If a symbol/signature is unknown, inspect only the relevant header/source needed to fix it.

When done, call submit_and_exit.
"""

def prompt_for_hang_fix(process_name: str, test_file: str, makefile: str,
                        entry_sym: str, timeout_sec: int) -> str:
    return f"""The test binary timed out after {timeout_sec}s — a blocking function has no working stub.

TEST FILE: {test_file}
MAKEFILE: {makefile}
ENTRY SYMBOL: {entry_sym}

DIAGNOSIS:
`{entry_sym}(1, argv)` was called but the process never returned within {timeout_sec}s.
A function called (directly or indirectly) by `{entry_sym}` is blocking:
- Infinite event/main loop (pmf_mainloop, select loop, while(1))
- Blocking system call (read/accept/recv with no data ready)
- A function that calls exit()/_exit() which terminates the process before CUnit can finish

FIX:
1. Read the test file — find which `__wrap_*` stubs exist.
2. Identify the blocking function: look at what `{process_name}` calls early in startup.
3. Add or fix the stub so it returns immediately.
4. Stub pattern for blocking functions:
   ```c
   return_type __wrap_blocking_func(args) {{
       fprintf(stderr, "__wrap_blocking_func called\\n");
       return safe_return_value;  /* 0, NULL, or valid handle — do NOT loop or exit */
   }}
   ```
5. Run `make test` after fixing.

When done, call submit_and_exit.
"""

# endregion Prompts

# region Stage 1 — Test File Scaffolding
def ensure_test_file(cfg: PipelineConfig, paths: dict) -> None:
    """
    Ensure the CUnit test file exists and includes actual production .c files.

    Important:
    - Uses cfg.source_dir as the source of truth.
    - Does not guess src/<process>.c.
    - Includes every .c found under cfg.source_dir.
    - Defines main before production includes so production main is renamed.
    """
    test_file: Path = paths["test_file"]
    process_name: str = paths["process_name"]
    test_file.parent.mkdir(parents=True, exist_ok=True)

    production_include_lines = _source_includes_for_test_file(cfg, test_file)
    define_main = f"#define main {process_name}_entry_main"
    production_include_block = ""
    if production_include_lines:
        production_include_block = (
            f"\n/* Production sources — main renamed so CUnit owns int main(void). */\n"
            f"{define_main}\n"
            + "\n".join(production_include_lines)
            + "\n#undef main\n"
        )

    if not test_file.exists():
        skeleton = f"""/* CUnit tests for {process_name} */
{TEST_FILE_MARKERS[0]}
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
{production_include_block}
{TEST_FILE_MARKERS[1]}
/* Compatibility definitions go here if needed. */

{TEST_FILE_MARKERS[2]}
/* Static globals shared by stubs/tests go here. */

{TEST_FILE_MARKERS[3]}
/* Reusable helper functions go here. */

{TEST_FILE_MARKERS[4]}
/* Linker wrapper stubs go here. */

{TEST_FILE_MARKERS[5]}
/* Test cases go here. */

{TEST_FILE_MARKERS[6]}
int main(void)
{{
    CU_pSuite suite = NULL;
    if (CU_initialize_registry() != CUE_SUCCESS) {{
        return CU_get_error();
    }}
    suite = CU_add_suite("{process_name}_suite", NULL, NULL);
    if (suite == NULL) {{
        CU_cleanup_registry();
        return CU_get_error();
    }}
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    {{
        unsigned int failures = CU_get_number_of_failures();
        CU_cleanup_registry();
        return failures == 0 ? 0 : 1;
    }}
}}
"""
        write_text(test_file, skeleton)
        print(f"[pipeline] created test file: {test_file}", file=sys.stderr)
        print("[pipeline] actual source files included:", file=sys.stderr)
        for src in _project_source_files(cfg):
            print(f" - {src}", file=sys.stderr)
        return

    text = read_text(test_file)
    changed = False

    # Ensure includes marker exists.
    if TEST_FILE_MARKERS[0] not in text:
        text = TEST_FILE_MARKERS[0] + "\n" + text
        changed = True

    # Add any missing actual production includes.
    missing_includes = [
        line for line in production_include_lines
        if line not in text
    ]
    if missing_includes:
        has_define = define_main in text
        if has_define:
            # Insert AFTER the #define main line so new files stay inside the define/undef block.
            new_inc_block = "\n".join(missing_includes) + "\n"
            text = text.replace(
                define_main + "\n",
                define_main + "\n" + new_inc_block,
                1,
            )
        else:
            insertion = (
                f"\n/* Production sources — main renamed so CUnit owns int main(void). */\n"
                f"{define_main}\n"
                + "\n".join(missing_includes)
                + "\n#undef main\n"
            )
            text = text.replace(TEST_FILE_MARKERS[0], TEST_FILE_MARKERS[0] + insertion, 1)
        changed = True
    elif define_main not in text and any(ln in text for ln in production_include_lines):
        # Includes present but #define main wrapper missing.
        # Insert #define before first include, #undef after last include.
        present = [ln for ln in production_include_lines if ln in text]
        if present:
            text = text.replace(present[0], f"{define_main}\n{present[0]}", 1)
            # Insert #undef after last include (first occurrence in current text)
            last_inc = present[-1]
            idx = text.find(last_inc)
            if idx >= 0:
                end = idx + len(last_inc)
                text = text[:end] + "\n#undef main" + text[end:]
            changed = True

    # Ensure all markers exist.
    for marker in TEST_FILE_MARKERS:
        if marker not in text:
            text += f"\n\n{marker}\n"
            changed = True

    if changed:
        write_text(test_file, text)
        print(f"[pipeline] updated test file with actual source includes: {test_file}", file=sys.stderr)
        print("[pipeline] actual source files included:", file=sys.stderr)
        for src in _project_source_files(cfg):
            print(f" - {src}", file=sys.stderr)

# endregion Stage 1 — Test File Scaffolding

# region Stage 2 — Stub Generation & Integration
def _normalize_doc_name(s: str) -> str:
    """
    Normalize function/doc names for matching.

    Keeps this conservative:
    - lowercase
    - strip extension if accidentally present
    - remove common non-identifier separators
    """
    s = Path(s).stem
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        _normalize_doc_name(a),
        _normalize_doc_name(b),
    ).ratio()

def find_func_doc(
    docs_dir: Path,
    func_name: str,
    threshold: float = 0.90,
) -> tuple[Optional[Path], float, list[tuple[float, Path]]]:
    """Find the best markdown doc for a function.

    Matching policy:
    1. Prefer exact `<func_name>.md`.
    2. Otherwise fuzzy-match against markdown filename stems.
    3. Accept only if best score >= threshold.

    Returns:
        (matched_path, score, top_candidates)

    matched_path is None when no acceptable match exists.
    """
    docs_dir = docs_dir.resolve()
    if not docs_dir.exists():
        return None, 0.0, []

    # 1. Exact match first.
    exact = docs_dir / f"{func_name}.md"
    if exact.exists():
        return exact, 1.0, [(1.0, exact)]

    scored: list[tuple[float, Path]] = []
    for p in docs_dir.glob("*.md"):
        score = similarity_ratio(func_name, p.stem)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None, 0.0, []

    best_score, best_path = scored[0]
    if best_score >= threshold:
        return best_path, best_score, scored[:5]
    return None, best_score, scored[:5]

def generate_stub_code(
    cfg: PipelineConfig,
    test_dir: Path,
    func_name: str,
) -> Optional[str]:
    """
    Generate or reuse one wrapper stub.

    Continuation behavior:
    - If _stub_gen/<func>/stub.c already exists and contains a valid
      __wrap_<func> implementation, reuse it.
    - Do not call the agent again for already-generated valid stub files.
    - If the cached stub is invalid, regenerate it.

    The agent must not edit source or final test file.
    It only writes _stub_gen/<func>/stub.c.
    """
    safe_name = _safe_filename(func_name)
    stub_dir = test_dir / "_stub_gen" / safe_name
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_out = stub_dir / "stub.c"

    def _clean_stub_body(raw: str) -> str:
        body = raw.strip()
        body = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n", "", body)
        body = re.sub(r"\n```\s*$", "", body).strip()
        return body

    def _is_valid_stub_body(body: str) -> bool:
        if not body:
            return False
        if f"__wrap_{func_name}" not in body:
            return False
        if f"__real_{func_name}" in body:
            return False
        return True

    # Continuation: reuse existing generated stub ONLY if also validated.
    result_file = stub_dir / "result.json"
    if stub_out.exists():
        cached_body = _clean_stub_body(read_text(stub_out))
        if _is_valid_stub_body(cached_body):
            already_validated = False
            if result_file.exists():
                try:
                    already_validated = load_json(result_file).get("validated", False)
                except Exception:
                    pass
            if already_validated:
                print(
                    f"[pipeline] reuse cached validated stub: {func_name}",
                    file=sys.stderr,
                )
                return cached_body
            # Exists but not yet validated — run validation now, skip agent.
            print(
                f"[pipeline] stub exists unvalidated for {func_name}, validating",
                file=sys.stderr,
            )
            if not _validate_stub_locally(cfg, test_dir, func_name, stub_dir):
                print(
                    f"[pipeline] existing stub {func_name} failed validation, regenerating",
                    file=sys.stderr,
                )
                stub_out.unlink(missing_ok=True)
                result_file.unlink(missing_ok=True)
            else:
                return cached_body
        print(
            f"[pipeline] cached stub invalid for {func_name}, regenerating: {stub_out}",
            file=sys.stderr,
        )
        try:
            stub_out.unlink()
        except FileNotFoundError:
            pass

    repo_root = cfg.source_dir.parent.parent.resolve()
    docs_dir = cfg.func_docs_dir.resolve()
    func_doc, func_doc_score, func_doc_candidates = find_func_doc(
        docs_dir=docs_dir,
        func_name=func_name,
        threshold=0.90,
    )
    doc_candidates_text = "\n".join(
        f" score={score:.3f} path={path}"
        for score, path in func_doc_candidates
    ) or " (no markdown candidates found)"

    prompt = f"""You are generating ONE C linker-wrapper stub for unit testing.

TARGET FUNCTION
{func_name}

REQUIRED WRAPPER NAME
__wrap_{func_name}

REPOSITORY ROOT
{repo_root}

PRODUCTION SOURCE FOLDER
{cfg.source_dir}

FUNCTION DOCS DIRECTORY
{docs_dir}

MATCHED FUNCTION DOC FILE
{func_doc if func_doc else '(none matched with similarity >= 0.90)'}

MATCH SCORE
{func_doc_score:.3f}

TOP DOC CANDIDATES
{doc_candidates_text}

TEST FOLDER
{test_dir}

OUTPUT FILE
{stub_out}

TASK
Generate a deterministic fake/stub implementation for `{func_name}`.

Before writing the stub:
1. If a matched function doc file is listed above, read it first.
   Only trust it as the target docs if the match score is >= 0.90.
2. Search the production source folder for occurrences of `{func_name}`.
3. Read the call sites where production code uses `{func_name}`.
4. Read nearby headers/declarations to determine the correct signature.
5. Use docs + source occurrences to infer:
   - return type
   - output pointer behavior
   - callback registration/invocation behavior
   - required dummy values
   - whether NULL arguments are allowed

STRICT FILE RULES
- You may read anything under the repository.
- Do NOT modify production source files.
- Do NOT modify headers.
- Do NOT modify the final test file.
- Do NOT modify the Makefile.
- Write only this file:
  `{stub_out}`

STUB RULES
- Write raw C code only.
- No markdown fences.
- No explanations outside C comments.
- Generate exactly one wrapper function unless helper static state is required.
- The wrapper function must be named exactly:
  `__wrap_{func_name}`
- Include this log at the top of the wrapper:
  `fprintf(stderr, "__wrap_{func_name} called\n");`
- This is a unit-test fake, not a pass-through wrapper.
- Do NOT call `__real_{func_name}`.
- Do NOT forward to the real implementation.
- Keep behavior deterministic and safe.
- If docs/source show success is `0`, return `0`.
- If docs/source show failure/success conventions, follow them.
- If output pointers exist, guard NULL and fill minimal safe values.
- If callbacks are registered, store callback pointers in static globals or invoke them with safe dummy values when appropriate.

When done, call submit_and_exit.
"""

    run_agent(
        cfg,
        work_dir=repo_root,
        folder=repo_root,
        history_dir=test_dir / "agent_history",
        prompt=prompt,
        history_name=f"_gen_stub_{safe_name}.json",
        max_iterations=cfg.max_agent_iterations,
        timeout_sec=cfg.agent_timeout_sec,
    )

    if not stub_out.exists():
        return None

    body = _clean_stub_body(read_text(stub_out))
    if not body:
        return None
    if f"__wrap_{func_name}" not in body:
        print(
            f"[pipeline] rejecting stub for {func_name}: missing __wrap_{func_name}",
            file=sys.stderr,
        )
        return None
    if f"__real_{func_name}" in body:
        print(
            f"[pipeline] rejecting stub for {func_name}: calls __real_{func_name}",
            file=sys.stderr,
        )
        return None

    _validate_stub_locally(cfg, test_dir, func_name, stub_dir)
    return body


def _validate_stub_locally(
    cfg: PipelineConfig,
    test_dir: Path,
    func_name: str,
    stub_dir: Path,
) -> bool:
    """
    Compile-validate and runtime-validate a stub.
    Loops with fresh micro-fix agent until both pass.
    Writes result.json {validated:true} on success.
    """
    stub_c = stub_dir / "stub.c"
    validate_main = stub_dir / "stub_validate_main.c"
    validate_bin = stub_dir / "stub_validate"
    result_file = stub_dir / "result.json"

    if result_file.exists():
        try:
            if load_json(result_file).get("validated"):
                return True
        except Exception:
            pass

    # Harness exercises the --wrap linkage path.
    # Signature check is intentionally shallow (no-arg weak dummy):
    # real argument-type compatibility is caught in Stage 2 when the stub
    # is linked against production code that includes the real headers.
    harness = f"""#include <stdio.h>
/* Weak dummy — linker needs a real {func_name} to wrap; --wrap redirects calls to __wrap */
__attribute__((weak)) int {func_name}() {{ return 0; }}
int main(void) {{
    (void){func_name}();  /* exercised via --wrap_{func_name} */
    fprintf(stderr, "stub_validate OK\\n");
    return 0;
}}
"""
    write_text(validate_main, harness)

    repo_root = cfg.source_dir.parent.parent.resolve()
    context_file = test_dir / "_pipeline_context.json"
    flags: dict = {}
    if context_file.exists():
        try:
            flags = load_json(context_file).get("flags", {})
        except Exception:
            pass

    cflags = " ".join(filter(None, [
        flags.get("CFLAGS", ""),
        flags.get("CFLAGS_LINUX", ""),
        flags.get("CPPFLAGS", ""),
        flags.get("INCLUDE", ""),
    ]))

    attempt = 1
    while True:
        compile_res = subprocess.run(
            f"gcc -c {cflags} {stub_c} -o {stub_dir}/stub.o",
            shell=True, cwd=str(stub_dir),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60,
        )
        if compile_res.returncode != 0:
            err = (compile_res.stderr + compile_res.stdout)[:3000]
            print(f"[pipeline] stub compile error {func_name} attempt {attempt}: {err[:200]}", file=sys.stderr)
            _fix_stub_with_agent(cfg, stub_dir, func_name, err, repo_root, test_dir)
            attempt += 1
            continue

        link_res = subprocess.run(
            f"gcc {cflags} {stub_c} {validate_main} -Wl,--wrap={func_name} -o {validate_bin}",
            shell=True, cwd=str(stub_dir),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60,
        )
        if link_res.returncode != 0:
            err = (link_res.stderr + link_res.stdout)[:3000]
            print(f"[pipeline] stub link error {func_name} attempt {attempt}: {err[:200]}", file=sys.stderr)
            _fix_stub_with_agent(cfg, stub_dir, func_name, err, repo_root, test_dir)
            attempt += 1
            continue

        run_res = subprocess.run(
            [str(validate_bin)],
            cwd=str(stub_dir),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15,
        )
        if run_res.returncode != 0:
            err = f"runtime exit={run_res.returncode}\n{run_res.stderr}\n{run_res.stdout}"
            print(f"[pipeline] stub runtime error {func_name} attempt {attempt}: {err[:200]}", file=sys.stderr)
            _fix_stub_with_agent(cfg, stub_dir, func_name, err, repo_root, test_dir)
            attempt += 1
            continue

        write_json(result_file, {"validated": True, "func_name": func_name})
        print(f"[pipeline] stub validated: {func_name}", file=sys.stderr)
        return True


def _fix_stub_with_agent(
    cfg: PipelineConfig,
    stub_dir: Path,
    func_name: str,
    error: str,
    repo_root: Path,
    history_dir: Path,
) -> None:
    stub_c = stub_dir / "stub.c"
    prompt = f"""Fix the __wrap_{func_name} stub.

FILE: {stub_c}

ERROR:
{error[:3000]}

RULES:
- Edit ONLY {stub_c}
- Function must be named exactly __wrap_{func_name}
- Do NOT call __real_{func_name}
- Raw C only, no markdown
- Keep: fprintf(stderr, "__wrap_{func_name} called\\n");

When done, call submit_and_exit.
"""
    run_agent(
        cfg,
        work_dir=stub_dir,
        prompt=prompt,
        history_name=f"_stub_fix_{_safe_filename(func_name)}_{int(time.time())}.json",
        folder=repo_root,
        history_dir=history_dir / "agent_history",
        protect_source=True,
    )


def integrate_all_stubs_sequential(
    cfg: PipelineConfig,
    paths: dict,
    validated_bodies: dict[str, str],
    flags: dict,
) -> None:
    """
    Integrate validated stubs one at a time into master test file.
    After each insertion: make test on master. Loop compile-fix until pass.
    Master always passes make test after each stub.
    """
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    test_dir: Path = paths["test_dir"]
    source_dir = cfg.source_dir.resolve()
    repo_root = cfg.source_dir.parent.parent.resolve()

    for func_name, body in validated_bodies.items():
        if stub_exists(test_file, func_name):
            print(f"[pipeline] stub already integrated: {func_name}", file=sys.stderr)
            continue

        print(f"[pipeline] integrating stub: {func_name}", file=sys.stderr)
        insert_stub_into_test_file(test_file, func_name, body)
        ensure_wrap_flag(makefile, func_name)

        attempt = 1
        while True:
            sync_wrap_flags(test_file, makefile)
            res = run_make_test(test_dir)
            if res["ok"]:
                print(f"[pipeline] master OK after stub: {func_name}", file=sys.stderr)
                break
            print(f"[pipeline] make test failed after stub {func_name} attempt {attempt}, fixing", file=sys.stderr)
            diag = build_output_with_runtime_diagnostics(test_dir, test_file, res)
            run_agent(
                cfg,
                test_dir,
                prompt_for_compile_fix(
                    str(makefile), str(test_file), diag,
                    source_dir=str(source_dir),
                    source_makefile=str(source_dir / "Makefile"),
                    actual_source_files=[str(p) for p in _project_source_files(cfg)],
                ),
                f"_stub_integrate_fix_{_safe_filename(func_name)}_{int(time.time())}.json",
                folder=repo_root,
            )
            attempt += 1


def insert_stub_into_test_file(test_file: Path, func_name: str, body: str) -> bool:
    """
    Insert `body` after the `/* === Linker Wrapper Stubs === */` marker.
    Idempotent: if __wrap_<name> already exists, do nothing.
    """
    text = read_text(test_file)
    marker = "/* === Linker Wrapper Stubs === */"
    if re.search(rf"__wrap_{re.escape(func_name)}\b", text):
        return False
    if marker not in text:
        # Append at end if markers missing
        text += f"\n{marker}\n"
    insertion = f"\n/* --- stub: {func_name} --- */\n{body.strip()}\n"
    text = text.replace(marker, marker + insertion, 1)
    write_text(test_file, text)
    return True

def stub_exists(test_file: Path, name: str) -> bool:
    text = read_text(test_file)
    # Match a C identifier starting with __wrap_<name>
    return bool(re.search(rf"__wrap_{re.escape(name)}\b", text))

def handle_stubs(cfg: PipelineConfig, paths: dict, analysis: dict) -> None:
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    test_dir: Path = paths["test_dir"]
    batch_size = max(1, int(cfg.stub_batch_size))

    candidates = collect_stub_candidates(analysis)
    print(f"[pipeline] {len(candidates)} stub candidates total", file=sys.stderr)

    def _validated_stub_body(name: str) -> Optional[str]:
        stub_dir = test_dir / "_stub_gen" / _safe_filename(name)
        stub_out = stub_dir / "stub.c"
        result_file = stub_dir / "result.json"
        if not stub_out.exists() or not result_file.exists():
            return None
        try:
            if not load_json(result_file).get("validated"):
                return None
        except Exception:
            return None
        body = read_text(stub_out).strip()
        if not body or f"__wrap_{name}" not in body or f"__real_{name}" in body:
            return None
        return body

    bodies: dict[str, str] = {}
    for n in candidates:
        cached = _validated_stub_body(n)
        if cached:
            bodies[n] = cached

    # ------------------------------------------------------------------
    # Phase 1: generate stub bodies until every candidate has a validated body.
    # ------------------------------------------------------------------
    gen_round = 1
    while True:
        to_gen = [
            n for n in candidates
            if n not in bodies
        ]
        if not to_gen:
            break
        print(
            f"[pipeline] stub gen round {gen_round}: "
            f"{len(to_gen)} need bodies",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=min(6, len(to_gen))) as pool:
            futs = {pool.submit(generate_stub_code, cfg, test_dir, n): n for n in to_gen}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    body = fut.result()
                except Exception as e:
                    print(f"[pipeline] stub gen error {name}: {e}", file=sys.stderr)
                    body = None
                if body:
                    bodies[name] = body
                    print(f"[pipeline] body ready: {name} ({len(body)} chars)", file=sys.stderr)
                else:
                    print(f"[pipeline] NO body: {name} round {gen_round}", file=sys.stderr)
        gen_round += 1

    # Phase 2: sequential one-by-one integration with make test after each.
    context_file = test_dir / "_pipeline_context.json"
    flags: dict = {}
    if context_file.exists():
        try:
            flags = load_json(context_file).get("flags", {})
        except Exception:
            pass
    validated = {n: bodies[n] for n in candidates if n in bodies}
    integrate_all_stubs_sequential(cfg, paths, validated, flags)

# endregion Stage 2 — Stub Generation & Integration

# region Stage 3 — Makefile Setup
def build_annotated_makefile(cfg: PipelineConfig, paths: dict) -> dict:
    """
    Stage 0: create annotated master Makefile + write _pipeline_context.json.

    Returns extracted flags dict so every downstream stage can use them
    without re-reading the Makefile.

    Skips Makefile rebuild if _pipeline_context.json already exists.
    """
    test_dir: Path = paths["test_dir"]
    context_file = test_dir / "_pipeline_context.json"
    if context_file.exists():
        try:
            ctx = load_json(context_file)
            print("[pipeline] Stage 0: using cached _pipeline_context.json", file=sys.stderr)
            return ctx.get("flags", {})
        except Exception:
            pass
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    process_name: str = paths["process_name"]

    source_dir = cfg.source_dir.resolve()
    source_makefile = source_dir / "Makefile"

    test_dir.mkdir(parents=True, exist_ok=True)

    def _run_do_mkmf() -> None:
        print(
            f"[pipeline] generating Makefile with: cd {test_dir} && do_mkmf {source_dir}",
            file=sys.stderr,
        )
        res = subprocess.run(
            ["do_mkmf", str(source_dir)],
            cwd=str(test_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if res.returncode != 0:
            raise RuntimeError(
                "do_mkmf failed\n"
                f"command: cd {test_dir} && do_mkmf {source_dir}\n"
                f"exit={res.returncode}\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}\n"
            )
        if not makefile.exists():
            raise RuntimeError(
                "do_mkmf completed but Makefile was not created\n"
                f"test_dir={test_dir}\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}\n"
            )

    # -------------------------------------------------------------------------
    # 1. Ensure do_mkmf-generated Makefile exists.
    # -------------------------------------------------------------------------
    if not makefile.exists():
        _run_do_mkmf()
    else:
        text = read_text(makefile)
        # Detect old bad handmade Makefile generated by previous pipeline.
        trash_markers = [
            "# === Auto-generated CUnit pipeline rules ===",
            "# === End Auto-generated CUnit pipeline rules ===",
        ]
        if any(marker in text for marker in trash_markers):
            backup = makefile.with_name("Makefile.bad_pipeline_backup")
            write_text(backup, text)
            makefile.unlink()
            print(
                f"[pipeline] backed up bad generated Makefile to: {backup}",
                file=sys.stderr,
            )
            _run_do_mkmf()
        else:
            print(
                f"[pipeline] using existing Makefile template: {makefile}",
                file=sys.stderr,
            )

    # -------------------------------------------------------------------------
    # 2. Build actual production source list from cfg.source_dir.
    # -------------------------------------------------------------------------
    production_srcs: list[str] = []
    for src in _project_source_files(cfg):
        rel = Path(os.path.relpath(src.resolve(), start=test_dir)).as_posix()
        production_srcs.append(rel)
    production_srcs_text = " ".join(production_srcs)

    print(f"[pipeline] source folder: {source_dir}", file=sys.stderr)
    print(f"[pipeline] source Makefile: {source_makefile}", file=sys.stderr)
    print("[pipeline] production source files for gcov:", file=sys.stderr)
    if production_srcs:
        for src in production_srcs:
            print(f" - {src}", file=sys.stderr)
    else:
        print(" (none found)", file=sys.stderr)

    flags = parse_source_makefile_flags(source_makefile) if source_makefile.exists() else {}
    merged_flag_keys = ["CFLAGS", "CFLAGS_LINUX", "CPPFLAGS", "INCLUDE", "LDFLAGS", "LDLIBS", "LIBS"]
    merged_flag_lines: list[str] = []
    for key in merged_flag_keys:
        value = (flags or {}).get(key)
        if value:
            merged_flag_lines.append(f"{key} += {value}")
    merged_flag_block = "\n".join(merged_flag_lines)

    # -------------------------------------------------------------------------
    # 3. Append/update test target block.
    # -------------------------------------------------------------------------
    test_program = test_file.stem
    test_src = test_file.name

    block_start = f"# === TEST TARGET FOR {process_name} ==="
    block_end = f"# === END TEST TARGET FOR {process_name} ==="

    test_block = f"""
{block_start}
# TODO: review merged source Makefile flags below and keep the unit build aligned with production.
{merged_flag_block}
TEST_PROGRAM = {test_program}
TEST_SRCS = {test_src}
PRODUCTION_SRCS = {production_srcs_text}
TEST_LIBS += -lcunit
TEST_REPORT_FILE = {test_program}_report.txt
TEST_LOG_FILE = {test_program}_log.txt
COVERAGE_FLAGS += --coverage -ffunction-sections -fdata-sections
WRAP_FLAGS = $(WRAP_FUNCS)

.PHONY: test clean-test coverage-test

test: clean-test $(TEST_PROGRAM)
\t./$(TEST_PROGRAM) > $(TEST_REPORT_FILE) 2>$(TEST_LOG_FILE)
\t$(MAKE) coverage-test

$(TEST_PROGRAM): $(TEST_SRCS)
\t$(CC) $(CFLAGS_LINUX) $(CFLAGS) $(INCLUDE) $(COVERAGE_FLAGS) $(TEST_SRCS) -o $(TEST_PROGRAM) \\
\t$(TEST_LIBS) $(LDFLAGS) $(LDLIBS) $(LIBS) \\
\t-Wl,--gc-sections \\
\t$(WRAP_FLAGS)

coverage-test:
\t@gcov -b -c *.gcno >> $(TEST_REPORT_FILE) 2>&1 || true

clean-test:
\trm -f $(TEST_PROGRAM) $(TEST_REPORT_FILE) $(TEST_LOG_FILE) *.gcda *.gcno *.gcov *.o
{block_end}
""".strip() + "\n"

    text = read_text(makefile)
    pattern = re.compile(
        rf"{re.escape(block_start)}.*?{re.escape(block_end)}",
        re.DOTALL,
    )
    if pattern.search(text):
        new_text = pattern.sub(test_block.strip(), text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        new_text = text + "\n" + test_block

    if new_text != text:
        write_text(makefile, new_text)

    # -------------------------------------------------------------------------
    # 4. Parse source Makefile flags and write _pipeline_context.json.
    # -------------------------------------------------------------------------
    write_json(context_file, {
        "process_name": process_name,
        "source_dir": str(source_dir),
        "source_makefile": str(source_makefile),
        "actual_source_files": [str(p) for p in _project_source_files(cfg)],
        "flags": flags,
        "test_dir": str(paths["test_dir"]),
        "test_file": str(paths["test_file"]),
        "makefile": str(makefile),
    })
    sync_wrap_flags(paths["test_file"], makefile)
    print(f"[pipeline] Stage 0: Makefile + context ready: {makefile}", file=sys.stderr)
    return flags

def ensure_wrap_flag(makefile: Path, func_name: str) -> bool:
    """Append -Wl,--wrap=<name> to WRAP_FUNCS if not already present."""
    flag = f"-Wl,--wrap={func_name}"
    text = read_text(makefile)
    if flag in text:
        return False
    append_text(makefile, f"WRAP_FUNCS += {flag}\n")
    return True

def sync_wrap_flags(test_file: Path, makefile: Path) -> None:
    """Ensure every __wrap_* symbol in the test file has a WRAP_FUNCS entry."""
    text = read_text(test_file)
    for name in re.findall(r'__wrap_(\w+)\b', text):
        ensure_wrap_flag(makefile, name)

# endregion Stage 3 — Makefile Setup

# region Stage 5 & 6 Helpers

def _semantic_context_path(test_dir: Path) -> Path:
    return test_dir / "_leaf_to_root_semantic_context.json"


def _load_semantic_context(test_dir: Path) -> dict:
    path = _semantic_context_path(test_dir)
    if not path.exists():
        return {"functions": {}}

    data = _read_json_loose(path)
    if not isinstance(data, dict):
        return {"functions": {}}

    if "functions" not in data or not isinstance(data["functions"], dict):
        data["functions"] = {}

    return data


def _append_semantic_context(test_dir: Path, func: dict, verdict: dict) -> None:
    path = _semantic_context_path(test_dir)
    data = _load_semantic_context(test_dir)
    fid = func.get("id") or func.get("name") or "unknown"
    data["functions"][fid] = {
        "func": func,
        "verdict": verdict,
    }
    write_json(path, data)


import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional


def _read_json_loose(path: Path) -> dict:
    """
    Read JSON written by the agent.

    Tolerates:
    - surrounding text
    - markdown json fences
    - extra prose before/after object
    """
    if not path.exists():
        return {}

    raw = path.read_text(errors="ignore").strip()
    if not raw:
        return {}

    raw2 = raw.strip()

    # Strip markdown fences.
    raw2 = re.sub(r"^\s*```(?:json)?", "", raw2, flags=re.I).strip()
    raw2 = re.sub(r"```\s*$", "", raw2).strip()

    try:
        return json.loads(raw2)
    except Exception:
        pass

    start = raw2.find("{")
    end = raw2.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw2[start:end + 1])
        except Exception:
            return {}

    return {}


def _normalize_simple_judge_verdict(
    verdict: dict,
    *,
    min_score: int,
) -> Optional[dict]:
    """
    Required judge output:
      {"score": 85, "reason": "..."}

    Pipeline-compatible returned object:
      {
        "parse_ok": True,
        "score": 85,
        "reason": "...",
        "summary": "...",
        "passed": True/False
      }
    """
    if not isinstance(verdict, dict):
        return None

    if "score" not in verdict:
        return None

    try:
        score = int(verdict.get("score"))
    except Exception:
        return None

    score = max(0, min(100, score))
    reason = str(verdict.get("reason") or "").strip()

    return {
        "parse_ok": True,
        "score": score,
        "reason": reason,
        "summary": reason,
        "passed": score >= min_score,
    }


def prompt_for_semantic_test_judge(
    *,
    process_name: str,
    test_file: str,
    func: dict,
    coverage: dict,
    make_result: dict,
    semantic_context: dict,
    verdict_file: str,
    min_score: int,
) -> str:
    fid = func.get("id") or func.get("name") or "unknown_function"

    return f"""
You are a semantic unit-test judge for generated CUnit tests.

You are NOT fixing code in this step.

Judge whether the existing CUnit tests for the target function are meaningful.

Important:
- Do NOT automatically fail only because make_result says CUnit failed.
- If coverage is valid, decide whether the failing assertion is a meaningful
  production behavior check or just a bad/generated test.
- Fail smoke/coverage-only tests even if coverage is high.

Process:
{process_name}

Target function:
{fid}

Target function metadata:
{json.dumps(func, indent=2, default=str)}

Current coverage info:
{json.dumps(coverage, indent=2, default=str)}

Latest make/test result:
{json.dumps(make_result, indent=2, default=str)}

Existing leaf-to-root semantic context:
{json.dumps(semantic_context, indent=2, default=str)}

Test file:
{test_file}

Verdict output file:
{verdict_file}

Minimum acceptable semantic score:
{min_score}

Inspect:
1. target function implementation,
2. real headers/macros/globals,
3. callees/wrappers/mocks,
4. current CUnit tests,
5. parent/caller functions if useful.

Score high only if tests would fail when important production behavior is broken.

Evaluate:
- return values,
- state/global changes,
- struct field changes,
- timer/counter behavior,
- dependency call count,
- dependency arguments,
- payload structs passed to dependencies,
- error/cleanup behavior,
- boundary/null/error paths where relevant.

You MUST write ONLY valid JSON to:
{verdict_file}

Required JSON format, exactly two fields:
{{
 "score": 0,
 "reason": "short reason explaining semantic strength or weakness"
}}

Rules:
- No markdown.
- No code fence.
- No prose outside JSON.
- No extra fields.
- score must be integer 0 to 100.
- reason must be a string.
"""


def run_semantic_test_judge(
    cfg,
    *,
    test_dir: Path,
    repo_root: Path,
    process_name: str,
    test_file: Path,
    func: dict,
    coverage: dict,
    make_result: dict,
) -> dict:
    """
    Retry semantic judge until readable simple JSON is produced.

    Important:
    - JSON parse failure is judge infrastructure failure.
    - It does NOT become score=0.
    - It does NOT trigger unit-test regeneration.
    """
    min_score = int(getattr(cfg, "semantic_judge_min_score", 75))
    max_parse_retries = int(getattr(cfg, "semantic_judge_parse_retries", 10))

    fid = func.get("id") or func.get("name") or "unknown_function"
    safe_fid = _safe_filename(fid)

    verdict_file = test_dir / f"_semantic_judge_{safe_fid}.json"
    semantic_context = _load_semantic_context(test_dir)

    last_raw = ""

    for i in range(1, max_parse_retries + 1):
        try:
            verdict_file.unlink()
        except FileNotFoundError:
            pass

        retry_note = ""
        if i > 1:
            retry_note = f"""

PREVIOUS ATTEMPT FAILED TO PRODUCE PARSEABLE JSON.

Retry:
{i}/{max_parse_retries}

You must write exactly this shape:

{{"score": 75, "reason": "your reason"}}

No markdown.
No prose.
No code fence.
No extra fields.
"""

        prompt = prompt_for_semantic_test_judge(
            process_name=process_name,
            test_file=str(test_file),
            func=func,
            coverage=coverage or {},
            make_result=make_result or {},
            semantic_context=semantic_context,
            verdict_file=str(verdict_file),
            min_score=min_score,
        ) + retry_note

        run_agent(
            cfg,
            test_dir,
            prompt,
            f"{_safe_filename(fid)}_semantic_judge.json",
            folder=repo_root,
        )

        verdict_raw = {}
        if verdict_file.exists():
            last_raw = verdict_file.read_text(errors="ignore")[-4000:]
            verdict_raw = _read_json_loose(verdict_file)

        verdict = _normalize_simple_judge_verdict(
            verdict_raw,
            min_score=min_score,
        )

        if verdict is not None:
            verdict["judge_attempts"] = i
            return verdict

        print(
            f"[pipeline] judge JSON parse failed for {fid}, retry {i}/{max_parse_retries}",
            file=sys.stderr,
        )

    raise RuntimeError(
        "Semantic judge failed to produce parseable JSON after "
        f"{max_parse_retries} attempts for {fid}. "
        "Not regenerating tests because this is judge infrastructure failure.\n"
        f"Last judge output:\n{last_raw}"
    )


def _backup_root_for_func(unit_dir: Path, func: dict) -> Path:
    fid = func.get("id") or func.get("name") or "unknown_function"
    return unit_dir / "_cunit_backups" / _safe_filename(fid)


def _best_backup_meta_path(unit_dir: Path, func: dict) -> Path:
    return _backup_root_for_func(unit_dir, func) / "_best.json"


def _load_best_backup_meta(unit_dir: Path, func: dict) -> dict:
    p = _best_backup_meta_path(unit_dir, func)
    if not p.exists():
        return {}
    return _read_json_loose(p)


def backup_good_cunit_if_best(
    *,
    unit_dir: Path,
    unit_test_file: Path,
    unit_makefile: Path,
    func: dict,
    coverage_pct: Optional[float],
    coverage: dict,
    make_result: dict,
    judge_verdict: dict,
    cfg,
) -> None:
    """
    Save best known CUnit test and Makefile.

    Backup condition:
    - coverage exists
    - coverage >= cfg.coverage_threshold
    - judge score >= previous best score
    """
    if coverage_pct is None:
        return

    threshold = float(getattr(cfg, "coverage_threshold", 100.0))
    coverage_pct = float(coverage_pct)

    if coverage_pct < threshold:
        return

    score = int(judge_verdict.get("score") or 0)

    root = _backup_root_for_func(unit_dir, func)
    root.mkdir(parents=True, exist_ok=True)

    best_meta = _load_best_backup_meta(unit_dir, func)
    old_best = int(best_meta.get("score") or -1)
    if score < old_best:
        return

    fid = func.get("id") or func.get("name") or "unknown_function"
    ts = str(int(time.time()))
    backup_dir = root / f"score_{score}_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if unit_test_file.exists():
        shutil.copy2(unit_test_file, backup_dir / unit_test_file.name)

    if unit_makefile.exists():
        shutil.copy2(unit_makefile, backup_dir / unit_makefile.name)

    meta = {
        "function": fid,
        "timestamp": ts,
        "score": score,
        "reason": judge_verdict.get("reason", ""),
        "passed": bool(judge_verdict.get("passed")),
        "coverage_pct": coverage_pct,
        "coverage_threshold": threshold,
        "make_ok": bool(make_result.get("ok")),
        "backup_dir": str(backup_dir),
        "test_file": str(backup_dir / unit_test_file.name),
        "makefile": str(backup_dir / unit_makefile.name),
        "coverage": coverage or {},
    }

    write_json(backup_dir / "meta.json", meta)
    write_json(_best_backup_meta_path(unit_dir, func), meta)

    latest = root / "latest_best"
    latest.mkdir(parents=True, exist_ok=True)

    if unit_test_file.exists():
        shutil.copy2(unit_test_file, latest / unit_test_file.name)

    if unit_makefile.exists():
        shutil.copy2(unit_makefile, latest / unit_makefile.name)

    write_json(latest / "meta.json", meta)

    print(
        f"[pipeline] backed up best CUnit for {fid}: score={score} coverage={coverage_pct}%",
        file=sys.stderr,
    )


def _extract_coverage_pct(coverage: dict) -> Optional[float]:
    """
    Accept common coverage dict shapes.
    """
    if not isinstance(coverage, dict):
        return None

    for key in ("coverage_percent", "percent", "coverage", "pct"):
        if key in coverage and coverage.get(key) is not None:
            try:
                return float(coverage.get(key))
            except Exception:
                pass

    return None


def prompt_for_semantic_test_repair(
    *,
    process_name: str,
    test_file: str,
    func: dict,
    coverage: dict,
    judge_verdict: dict,
    semantic_context: dict,
) -> str:
    fid = func.get("id") or func.get("name") or "unknown_function"

    return f"""
You are editing an existing CUnit test file.

The tests already reached coverage threshold, but the semantic judge says
they are not meaningful enough.

Process:
{process_name}

Target function:
{fid}

Target function metadata:
{json.dumps(func, indent=2, default=str)}

Current coverage:
{json.dumps(coverage, indent=2, default=str)}

Semantic judge verdict:
{json.dumps(judge_verdict, indent=2, default=str)}

The judge returns only:
- score
- reason

Use judge_verdict["reason"] as the main repair instruction.

Existing leaf-to-root semantic context:
{json.dumps(semantic_context, indent=2, default=str)}

Test file to modify:
{test_file}

You must inspect the real source before editing:
- target function implementation,
- structs/macros/globals it uses,
- external dependencies it calls,
- callers/parents if useful,
- existing mocks/wrappers in the test file.

Do NOT add generic template tests.
Do NOT add coverage-only tests.
Do NOT guess field names or signatures.
Do NOT create fake project headers/types/macros/functions.

If compilation fails because of missing headers/types/macros, inspect real
production headers and Makefiles, then fix INCLUDE/CPPFLAGS/LDFLAGS/LIBS in
the unit Makefile.

Fix the tests so they semantically prove behavior.

Required repair behavior:
1. Address the judge reason.
2. Add or strengthen assertions so tests fail when production behavior is wrong.
3. Assert exact return values/state/mocks/arguments where possible.
4. Capture dependency arguments in wrappers when behavior depends on them.
5. If a struct pointer is passed to a dependency, copy it and assert important fields.
6. Add comments near the tests explaining:
   - what {fid} does,
   - why each scenario matters,
   - what parent/root functions can rely on from this function.
7. Extend reset_mocks/reset helpers for any new mock state.
8. Preserve existing meaningful tests and assertions.
9. Keep the test binary non-blocking and terminating.

Do not weaken tests just to compile.
If something does not compile, inspect headers/source and correct it.

When done, call submit_and_exit.
"""


def judge_and_backup_if_covered(
    cfg,
    *,
    test_dir: Path,
    unit_dir: Path,
    repo_root: Path,
    process_name: str,
    unit_test_file: Path,
    unit_makefile: Path,
    func: dict,
    coverage: dict,
    coverage_pct: Optional[float],
    make_result: dict,
) -> dict:
    """
    Run semantic judge when coverage is above threshold, regardless of make_ok.

    Also backs up the current CUnit test/Makefile if coverage is good and
    judge score is best so far.

    Raises RuntimeError if judge cannot produce parseable JSON after retries.
    """
    if coverage_pct is None:
        return {
            "covered": False,
            "accepted": False,
            "judge_verdict": None,
            "reason": "coverage_pct is None",
        }

    coverage_pct = float(coverage_pct)
    threshold = float(getattr(cfg, "coverage_threshold", 100.0))
    if coverage_pct < threshold:
        return {
            "covered": False,
            "accepted": False,
            "judge_verdict": None,
            "reason": "coverage below threshold",
        }

    judge_verdict = run_semantic_test_judge(
        cfg,
        test_dir=test_dir,
        repo_root=repo_root,
        process_name=process_name,
        test_file=unit_test_file,
        func=func,
        coverage=coverage or {},
        make_result=make_result or {},
    )

    backup_good_cunit_if_best(
        unit_dir=unit_dir,
        unit_test_file=unit_test_file,
        unit_makefile=unit_makefile,
        func=func,
        coverage_pct=coverage_pct,
        coverage=coverage or {},
        make_result=make_result or {},
        judge_verdict=judge_verdict,
        cfg=cfg,
    )

    fid = func.get("id") or func.get("name") or "unknown_function"

    if judge_verdict.get("passed"):
        print(
            f"[pipeline] judge PASSED: {fid} score={judge_verdict.get('score')}",
            file=sys.stderr,
        )
        _append_semantic_context(test_dir, func, judge_verdict)
        return {
            "covered": True,
            "accepted": True,
            "judge_verdict": judge_verdict,
            "reason": "semantic judge passed",
        }

    print(
        f"[pipeline] judge FAILED: {fid} score={judge_verdict.get('score')}",
        file=sys.stderr,
    )
    return {
        "covered": True,
        "accepted": False,
        "judge_verdict": judge_verdict,
        "reason": "semantic judge failed",
    }


def prompt_for_function_test_with_semantic_context(
    *,
    func: dict,
    coverage: dict,
    test_file: str,
    process_name: str,
    attempt: int,
    max_attempts: int,
    make_ok: bool,
    semantic_context: dict,
    last_judge_verdict: Optional[dict],
) -> str:
    fid = func.get("id") or func.get("name") or "unknown_function"

    return f"""
You are writing or repairing a real CUnit unit test file for production C code.

Process:
{process_name}

Target function:
{fid}

Target metadata:
{json.dumps(func, indent=2, default=str)}

Test file:
{test_file}

You must explore the source yourself before editing.
Inspect:
1. target implementation,
2. used headers,
3. structs/enums/macros/constants,
4. globals read/written,
5. callees,
6. callers/parent functions,
7. current test file,
8. current mocks/wrappers,
9. Makefile wrapping style.

Add meaningful tests for realistic use cases of {fid}.

The tests must include comments/explanations that describe:
- what {fid} is responsible for,
- which lower-level behavior it depends on,
- what each scenario proves,
- why parent/root functions can rely on this behavior later.

Do not create generic template tests.
Do not create smoke tests.
Do not merely execute lines for gcov.

A meaningful test should verify observable behavior:
- return value,
- global state change,
- struct field change,
- timer/counter transition,
- dependency call count,
- dependency argument values,
- payload passed to dependency,
- error log,
- cleanup,
- exit behavior,
- early return/no-op behavior.

Create separate tests for distinct source behaviors when present:
- normal path,
- boundary path,
- null/invalid path,
- external dependency failure,
- state transition,
- cleanup/error branch,
- early return/no-op.

Mocks/wrappers:
- capture call count,
- capture important arguments,
- allow configurable returns,
- capture payload structs with memcpy when needed,
- never block or call real hardware/IPC/mainloop.

reset_mocks/reset helper:
- reset all new mock state,
- reset production globals touched by tests,
- reset arrays/pointers/null-injection flags,
- do not rely on test order.

CUnit main:
- must return non-zero when CUnit has failures.

Do not delete meaningful tests.
Do not weaken assertions to pass build.
If something does not compile, inspect source/header definitions and fix it.

When done, call submit_and_exit.
"""


def collect_runtime_crash_diagnostics(test_dir: Path, test_binary_name: str) -> str:
    """
    Collect runtime diagnostics whenever make/test fails.

    If the binary does not exist because it was a compile/link failure, this
    function returns that fact. If the binary exists, it captures:
    - CUnit report/log files,
    - direct binary stdout/stderr,
    - gdb backtrace if gdb is available.
    """
    chunks: list[str] = []
    test_bin = test_dir / test_binary_name

    chunks.append("RUNTIME DIAGNOSTICS")
    chunks.append(f"test_dir: {test_dir}")
    chunks.append(f"test_binary: {test_bin}")

    for name in [f"{test_binary_name}_log.txt", f"{test_binary_name}_report.txt"]:
        p = test_dir / name
        if p.exists():
            try:
                chunks.append(
                    f"\n--- {name} ---\n"
                    f"{p.read_text(errors='ignore')[-12000:]}"
                )
            except Exception as e:
                chunks.append(f"\n--- {name} unreadable: {e} ---")

    if not test_bin.exists():
        chunks.append(
            "\n--- binary check ---\n"
            "Test binary does not exist. This is probably a compile/link failure, "
            "not a runtime crash."
        )
        return "\n".join(chunks)

    try:
        direct = subprocess.run(
            [str(test_bin)],
            cwd=str(test_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        chunks.append(
            "\n--- direct binary run ---\n"
            f"returncode={direct.returncode}\n"
            f"stdout:\n{direct.stdout[-8000:]}\n"
            f"stderr:\n{direct.stderr[-8000:]}"
        )
    except subprocess.TimeoutExpired:
        chunks.append(
            "\n--- direct binary run ---\n"
            "Timed out after 20 seconds. The test binary may be hanging."
        )
    except Exception as e:
        chunks.append(f"\n--- direct binary run failed ---\n{e}")

    try:
        gdb = subprocess.run(
            [
                "gdb", "-q", "-batch",
                "-ex", "set pagination off",
                "-ex", "run",
                "-ex", "bt",
                "-ex", "bt full",
                "--args", str(test_bin),
            ],
            cwd=str(test_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=40,
        )
        chunks.append(
            "\n--- gdb backtrace ---\n"
            f"returncode={gdb.returncode}\n"
            f"stdout:\n{gdb.stdout[-20000:]}\n"
            f"stderr:\n{gdb.stderr[-8000:]}"
        )
    except FileNotFoundError:
        chunks.append("\n--- gdb backtrace ---\ngdb not installed.")
    except subprocess.TimeoutExpired:
        chunks.append("\n--- gdb backtrace ---\ngdb timed out after 40 seconds.")
    except Exception as e:
        chunks.append(f"\n--- gdb backtrace failed ---\n{e}")

    return "\n".join(chunks)


def build_output_with_runtime_diagnostics(test_dir: Path, test_file: Path, res: dict) -> str:
    """
    Combine normal make output with runtime diagnostics.

    Use this instead of:
      (res.get("stderr") or "") + "\\n---\\n" + (res.get("stdout") or "")

    Gives the agent enough information to fix runtime crashes like Error 139.
    """
    base_output = collect_failure_diagnostics(test_dir, test_file, res)
    test_binary_name = Path(test_file).stem
    diagnostics = collect_runtime_crash_diagnostics(test_dir, test_binary_name)

    return (
        base_output
        + "\n\n==================== RUNTIME / EXECUTION DIAGNOSTICS ====================\n"
        + diagnostics
        + "\n======================\n"
        + "\nNOTE TO AGENT:\n"
        + "- If compilation/link failed, fix the compile/link error first.\n"
        + "- If the binary exists and direct run or gdb shows a crash, fix that runtime crash.\n"
        + "- For segfaults, inspect the backtrace and likely wrapper/mock/global pointer cause.\n"
        + "- Do not treat a runtime segfault as a normal compiler error.\n"
    )


def collect_failure_diagnostics(test_dir: Path, test_file: Path, res: dict) -> str:
    """
    Collect diagnostic information for any make/test failure.

    Safe for compile/link failures too:
    - If the binary does not exist, it records that.
    - If the binary exists, it runs it directly and tries gdb.
    """
    chunks: list[str] = []

    test_binary_name = Path(test_file).stem
    test_bin = test_dir / test_binary_name

    chunks.append("AUTOMATIC FAILURE DIAGNOSTICS")
    chunks.append(f"test_dir: {test_dir}")
    chunks.append(f"test_file: {test_file}")
    chunks.append(f"test_binary: {test_bin}")

    base_output = (res.get("stderr") or "") + "\n---\n" + (res.get("stdout") or "")
    chunks.append("\n--- original make/test output ---")
    chunks.append(base_output[-30000:])

    for name in [
        f"{test_binary_name}_log.txt",
        f"{test_binary_name}_report.txt",
    ]:
        p = test_dir / name
        if p.exists():
            try:
                chunks.append(f"\n--- {name} ---")
                chunks.append(p.read_text(errors="ignore")[-12000:])
            except Exception as e:
                chunks.append(f"\n--- {name} unreadable: {e} ---")

    if not test_bin.exists():
        chunks.append(
            "\n--- binary check ---\n"
            "Test binary does not exist. This is likely a compile or link failure."
        )
        return "\n".join(chunks)

    try:
        direct = subprocess.run(
            [str(test_bin)],
            cwd=str(test_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        chunks.append(
            "\n--- direct binary run ---\n"
            f"returncode={direct.returncode}\n"
            f"stdout:\n{direct.stdout[-8000:]}\n"
            f"stderr:\n{direct.stderr[-8000:]}"
        )
    except subprocess.TimeoutExpired:
        chunks.append(
            "\n--- direct binary run ---\n"
            "Timed out after 20 seconds. This is likely a runtime hang/blocking call."
        )
    except Exception as e:
        chunks.append(f"\n--- direct binary run failed ---\n{e}")

    try:
        gdb = subprocess.run(
            [
                "gdb", "-q", "-batch",
                "-ex", "set pagination off",
                "-ex", "run",
                "-ex", "bt",
                "-ex", "bt full",
                "--args", str(test_bin),
            ],
            cwd=str(test_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=40,
        )
        chunks.append(
            "\n--- gdb backtrace ---\n"
            f"returncode={gdb.returncode}\n"
            f"stdout:\n{gdb.stdout[-20000:]}\n"
            f"stderr:\n{gdb.stderr[-8000:]}"
        )
    except FileNotFoundError:
        chunks.append("\n--- gdb backtrace ---\ngdb not installed.")
    except subprocess.TimeoutExpired:
        chunks.append("\n--- gdb backtrace ---\ngdb timed out after 40 seconds.")
    except Exception as e:
        chunks.append(f"\n--- gdb backtrace failed ---\n{e}")

    return "\n".join(chunks)

# endregion Stage 5 & 6 Helpers

# region Stage 5 — Minimal Test Validation

def ensure_minimal_test_runs(cfg: PipelineConfig, paths: dict) -> bool:
    """
    Stage 5 infrastructure validation.

    First try the current test exactly as-is.
    If it already compiles, runs, terminates, and passes, do nothing else.
    """
    test_dir: Path = paths["test_dir"]
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    process_name: str = paths["process_name"]
    repo_root = cfg.source_dir.parent.parent.resolve()
    entry_sym = f"{process_name}_entry_main"

    print(
        f"[pipeline] minimal test phase: checking existing test binary first",
        file=sys.stderr,
    )

    # SIMPLE FIX:
    # If the current test already compiles/runs/passes, do not let the agent
    # rewrite it with a minimal smoke test.
    sync_wrap_flags(test_file, makefile)
    existing_res = run_make_test(test_dir, timeout=90)

    if existing_res["ok"]:
        print(
            f"[pipeline] existing test already compiles, runs, terminates, and passes; "
            f"skipping minimal-test generation",
            file=sys.stderr,
        )
        return True

    print(
        f"[pipeline] existing test did not pass; running minimal validation for {entry_sym}()",
        file=sys.stderr,
    )

    attempt = 1
    while True:
        print(
            f"[pipeline] minimal test attempt {attempt}",
            file=sys.stderr,
        )

        run_agent(
            cfg,
            test_dir,
            prompt_for_minimal_test(process_name, str(test_file), entry_sym),
            f"_minimal_test_{attempt:02d}.json",
            folder=repo_root,
        )

        sync_wrap_flags(test_file, makefile)
        res = run_make_test(test_dir, timeout=90)

        if res["ok"]:
            print(
                f"[pipeline] minimal test PASSED on attempt {attempt}; "
                f"binary compiles, runs, terminates, and infrastructure is usable",
                file=sys.stderr,
            )
            return True

        if res.get("timed_out"):
            print(
                f"[pipeline] minimal test HUNG binary did not exit within 90s; "
                f"fixing blocking stubs",
                file=sys.stderr,
            )
            run_agent(
                cfg,
                test_dir,
                prompt_for_hang_fix(
                    process_name,
                    str(test_file),
                    str(makefile),
                    entry_sym,
                    90,
                ),
                f"_hang_fix_{attempt:02d}.json",
                folder=repo_root,
            )
        else:
            print(
                f"[pipeline] minimal test compile/link/runtime error on attempt {attempt}; "
                f"collecting diagnostics and running compile-fix",
                file=sys.stderr,
            )
            _src = cfg.source_dir.resolve()
            diagnostic_build_output = build_output_with_runtime_diagnostics(
                test_dir,
                test_file,
                res,
            )
            run_agent(
                cfg,
                test_dir,
                prompt_for_compile_fix(
                    str(makefile),
                    str(test_file),
                    diagnostic_build_output,
                    source_dir=str(_src),
                    source_makefile=str(_src / "Makefile"),
                    actual_source_files=[str(p) for p in _project_source_files(cfg)],
                ),
                f"_minimal_compile_fix_{attempt:02d}.json",
                folder=repo_root,
            )
        attempt += 1

# endregion Stage 5 — Minimal Test Validation

# region Stage 4 — Parallel Unit Test Generation
import shutil as _shutil

def _stub_srcs_relative(test_dir: Path, unit_dir: Path) -> list[tuple[str, str]]:
    """
    Return [(relative_stub_path, func_name)] for all validated stubs.
    Relative path is from unit_dir (the cwd when building the unit test).
    Stubs are generated into _stub_gen/<func_name>/ by generate_stub_code.
    """
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
                rel = Path(os.path.relpath(stub_c.resolve(), start=unit_dir)).as_posix()
                result.append((rel, func_name))
        except Exception:
            pass
    return result


def _sync_stub_srcs(unit_test_file: Path, unit_makefile: Path, test_dir: Path, unit_dir: Path) -> None:
    """
    Update STUB_SRCS in unit Makefile to exclude stubs whose __wrap_ function
    is locally DEFINED (not just called) in the unit test file.
    Prevents duplicate-symbol link errors when the test overrides a stub.
    """
    text = read_text(unit_test_file)
    # Match definitions: return_type __wrap_name(...) {
    local_overrides = set(re.findall(r'__wrap_(\w+)\s*\([^)]*\)\s*\{', text))
    all_stubs = _stub_srcs_relative(test_dir, unit_dir)
    filtered = [path for path, fname in all_stubs if fname not in local_overrides]
    stub_srcs_line = "STUB_SRCS = " + " ".join(filtered)
    mk_text = read_text(unit_makefile)
    new_mk = re.sub(r'^STUB_SRCS\s*=.*$', stub_srcs_line, mk_text, flags=re.MULTILINE)
    if new_mk != mk_text:
        write_text(unit_makefile, new_mk)


def _scaffold_unit_test_dir(cfg: PipelineConfig, paths: dict, func: dict) -> Path:
    """Create _unit_tests/<func_id>/ with skeleton test file + generated Makefile."""
    test_dir: Path = paths["test_dir"]
    process_name: str = paths["process_name"]
    func_id = func["id"]
    safe_id = _safe_filename(func_id)

    unit_dir = test_dir / "_unit_tests" / safe_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "agent_history").mkdir(exist_ok=True)

    unit_test_file = unit_dir / f"test_{safe_id}.c"
    unit_makefile = unit_dir / "Makefile"

    if not unit_test_file.exists():
        skeleton = f"""/*
 * PLACEHOLDER ONLY.
 *
 * Target:
 * {func_id}
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

    if not unit_makefile.exists():
        _generate_unit_test_makefile(cfg, paths, func, safe_id, unit_dir, unit_test_file)

    return unit_dir


def _generate_unit_test_makefile(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    safe_id: str,
    unit_dir: Path,
    unit_test_file: Path,
) -> None:
    """Generate Makefile for a unit test dir, inheriting the master test Makefile."""
    context_file = paths["test_dir"] / "_pipeline_context.json"
    flags: dict = {}
    if context_file.exists():
        try:
            flags = load_json(context_file).get("flags", {})
        except Exception:
            pass

    func_id = func["id"]
    process_name: str = paths["process_name"]
    entry_sym = f"{process_name}_entry_main"

    # Include the master Makefile using a path relative to this unit test dir.
    master_makefile_rel = Path(
        os.path.relpath(paths["makefile"].resolve(), start=unit_dir)
    ).as_posix()

    # Production source is compiled separately so gcov can emit <source>.c.gcov.
    source_file_abs = _resolve_source_file(cfg, func["source_file"])
    prod_src_rel = Path(
        os.path.relpath(source_file_abs.resolve(), start=unit_dir)
    ).as_posix()

    test_program = f"test_{safe_id}"
    test_src = unit_test_file.name

    # Validated stub bodies to link.
    # This is refreshed by _sync_stub_srcs before each make test.
    stub_srcs_list = [path for path, _ in _stub_srcs_relative(paths["test_dir"], unit_dir)]
    stub_srcs_str = " ".join(stub_srcs_list)

    content = f"""# Unit test Makefile for {func_id}
# Auto-generated — do not edit manually

# Pull in the same include paths, libraries, architecture flags, and wrap flags
# as the master test Makefile. Unit-specific rules below override only what
# must differ for this per-function test.
MASTER_MAKEFILE = {master_makefile_rel}
include $(MASTER_MAKEFILE)

.DEFAULT_GOAL := test

#CC = gcc

# Fallbacks from pipeline_context.json, used only if the master Makefile did
# not define them.
CFLAGS ?= {flags.get('CFLAGS', '')}
CFLAGS_LINUX ?= {flags.get('CFLAGS_LINUX', '')}
CPPFLAGS ?= {flags.get('CPPFLAGS', '')}
INCLUDE ?= {flags.get('INCLUDE', '')}
LDFLAGS ?= {flags.get('LDFLAGS', '')}
LDLIBS ?= {flags.get('LDLIBS', '')}
LIBS ?= {flags.get('LIBS', '')}

# Unit-specific files
TEST_PROGRAM = {test_program}
TEST_SRCS = {test_src}
PROD_SRC = {prod_src_rel}
PROD_OBJ = prod_under_test.o

# Keep only the production gcov output.
# Example:
# PROD_SRC = ../../../../src/dio100d/dio100d.c
# TARGET_GCOV = dio100d.c.gcov
TARGET_GCOV = $(notdir $(PROD_SRC)).gcov

# Validated stub bodies.
# Local __wrap_* overrides in TEST_SRCS should be excluded by the pipeline.
STUB_SRCS = {stub_srcs_str}

TEST_LIBS += -lcunit
TEST_REPORT_FILE = {test_program}_report.txt
TEST_LOG_FILE = {test_program}_log.txt

# Coverage flags for this isolated unit build.
COVERAGE_FLAGS += --coverage -ffunction-sections -fdata-sections

# Use the same wrap functions as the master Makefile.
WRAP_FLAGS = $(WRAP_FUNCS)

.PHONY: test clean-test coverage-test

test: clean-test $(TEST_PROGRAM)
\t@set +e; \\
\t./$(TEST_PROGRAM) > $(TEST_REPORT_FILE) 2>$(TEST_LOG_FILE); \\
\tstatus=$$?; \\
\t$(MAKE) coverage-test; \\
\texit $$status

$(PROD_OBJ): $(PROD_SRC)
\t$(CC) $(CPPFLAGS) $(CFLAGS_LINUX) $(CFLAGS) $(INCLUDE) $(COVERAGE_FLAGS) -Dmain={entry_sym} -c $(PROD_SRC) -o $(PROD_OBJ)

$(TEST_PROGRAM): $(TEST_SRCS) $(PROD_OBJ) $(STUB_SRCS)
\t$(CC) $(CPPFLAGS) $(CFLAGS_LINUX) $(CFLAGS) $(INCLUDE) $(COVERAGE_FLAGS) $(TEST_SRCS) $(PROD_OBJ) $(STUB_SRCS) \\
\t-o $(TEST_PROGRAM) \\
\t$(TEST_LIBS) $(LDFLAGS) $(LDLIBS) $(LIBS) \\
\t-Wl,--gc-sections \\
\t$(WRAP_FLAGS)

coverage-test:
\t@echo "=== coverage-test ===" >> $(TEST_REPORT_FILE)
\t@echo "PWD=$$(pwd)" >> $(TEST_REPORT_FILE)
\t@echo "Target production gcov file: $(TARGET_GCOV)" >> $(TEST_REPORT_FILE)
\t@echo "Files before gcov:" >> $(TEST_REPORT_FILE)
\t@ls -la >> $(TEST_REPORT_FILE) 2>&1 || true
\t@found=0; \\
\tfor f in *.gcno; do \\
\t\t[ -e "$$f" ] || continue; \\
\t\tfound=1; \\
\t\techo "Running gcov -b -c $$f" >> $(TEST_REPORT_FILE); \\
\t\tgcov -b -c "$$f" >> $(TEST_REPORT_FILE) 2>&1 || true; \\
\tdone; \\
\tif [ "$$found" -eq 0 ]; then \\
\t\techo "WARNING: no .gcno files found for gcov" >> $(TEST_REPORT_FILE); \\
\tfi
\t@echo "Filtering .gcov files. Keeping only: $(TARGET_GCOV)" >> $(TEST_REPORT_FILE)
\t@find . -maxdepth 1 -name '*.gcov' ! -name '$(TARGET_GCOV)' -delete
\t@if [ ! -f "$(TARGET_GCOV)" ]; then \\
\t\techo "WARNING: target production gcov file was not generated: $(TARGET_GCOV)" >> $(TEST_REPORT_FILE); \\
\tfi
\t@echo "Files after gcov filtering:" >> $(TEST_REPORT_FILE)
\t@ls -la >> $(TEST_REPORT_FILE) 2>&1 || true

clean-test:
\trm -f $(TEST_PROGRAM) $(TEST_REPORT_FILE) $(TEST_LOG_FILE) *.gcda *.gcno *.gcov *.o
"""
    write_text(unit_dir / "Makefile", content)


def _generate_unit_test_for_func(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    flags: dict,
    semantic_context_snapshot: dict,
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

    unit_dir = _scaffold_unit_test_dir(cfg, paths, func)
    unit_test_file = unit_dir / f"test_{safe_id}.c"
    unit_makefile = unit_dir / "Makefile"
    judge_verdict_file = unit_dir / "judge_verdict.json"
    coverage_file = unit_dir / "coverage.json"

    master_test_file = test_dir / f"test_{process_name}.c"
    master_makefile = paths["makefile"]
    stub_gen_dir = test_dir / "_stub_gen"

    source_file_abs = _resolve_source_file(cfg, func["source_file"])

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

    def _safe_build_diag(make_res: dict) -> str:
        parts: list[str] = []
        try:
            diag = build_output_with_runtime_diagnostics(
                unit_dir,
                unit_test_file,
                make_res,
            )
            if diag:
                parts.append(str(diag))
        except Exception as e:
            parts.append(
                f"\nbuild_output_with_runtime_diagnostics failed: {type(e).__name__}: {e}\n"
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
            return "(no diagnostics available)"

        return "\n".join(parts)

    def _validated_stub_text() -> str:
        try:
            validated_stub_srcs = _stub_srcs_relative(test_dir, unit_dir)
            lines = []
            for rel_stub, stub_func_name in validated_stub_srcs:
                abs_stub = (unit_dir / rel_stub).resolve()
                lines.append(
                    f" - func={stub_func_name} rel_from_unit={rel_stub} abs={abs_stub}"
                )
            return "\n".join(lines) if lines else " - none"
        except Exception as e:
            return f" - could not list validated stubs: {type(e).__name__}: {e}"

    def _source_makefile_text() -> str:
        return (
            "\n".join(f" - {p}" for p in existing_source_makefiles)
            if existing_source_makefiles
            else " - none found"
        )

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
"""

        return f"""
You are running inside an agent with repository folder:

 {repo_root}

The current unit test file is only a placeholder or previous failed attempt.
Rewrite it from scratch if needed.

TARGET FUNCTION
 {func_id}

function object:
{json.dumps(func_for_prompt, ensure_ascii=False, indent=2)}

production source file:
 {source_file_abs}

source line range:
 {func.get("start_line")} - {func.get("end_line")}

Do not create fake project headers/types/macros/functions to spoof the build
harness just to make the build pass.

Unit test file to write/repair:
 {unit_test_file}

FILES YOU MAY EDIT
Unit Makefile to edit only if required for this one unit build:
 {unit_makefile}

FILES TO READ
Production source:
 {source_file_abs}

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

{previous_diag_block}
================================================================================
END UNIT TEST GENERATION FILESYSTEM CONTEXT
================================================================================
"""

    # Continuation: already passed,
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

    last_judge: Optional[dict] = None
    if judge_verdict_file.exists():
        try:
            last_judge = load_json(judge_verdict_file)
        except Exception:
            pass

    last_make_ok = True
    cov: Optional[dict] = None
    pct: Optional[float] = None
    last_build_diag: Optional[str] = None

    max_attempts = int(getattr(cfg, "max_unit_test_attempts", 4) or 4)

    # Fast path: if existing test already gives valid coverage, start from judge.
    if unit_test_file.exists() and unit_makefile.exists() and _has_real_cu_add_test(unit_test_file):
        try:
            _sync_stub_srcs(unit_test_file, unit_makefile, test_dir, unit_dir)
            sync_wrap_flags(unit_test_file, unit_makefile)

            existing_make_res = run_make_test(unit_dir)
            last_make_ok = bool(existing_make_res.get("ok"))

            cov, pct = _run_current_coverage()

            if pct is not None and pct >= float(cfg.coverage_threshold):
                print(
                    f"[pipeline] existing test has coverage={pct}%, starting from judge for {func_id}",
                    file=sys.stderr,
                )
                source_file_abs = _resolve_source_file(cfg, func["source_file"])
                func_for_prompt = {**func, "source_file": str(source_file_abs)}

                judge = run_semantic_test_judge(
                    cfg,
                    test_dir=unit_dir,
                    repo_root=repo_root,
                    process_name=process_name,
                    test_file=unit_test_file,
                    func=func_for_prompt,
                    coverage=cov or {},
                    make_result=existing_make_res or {},
                )

                backup_good_cunit_if_best(
                    unit_dir=unit_dir,
                    unit_test_file=unit_test_file,
                    unit_makefile=unit_makefile,
                    func=func_for_prompt,
                    coverage_pct=pct,
                    coverage=cov or {},
                    make_result=existing_make_res or {},
                    judge_verdict=judge,
                    cfg=cfg,
                )

                write_json(coverage_file, cov or {})
                write_json(judge_verdict_file, judge)
                last_judge = judge

                if judge.get("passed"):
                    print(
                        f"[pipeline] unit test PASSED from existing gcov: {func_id} score={judge.get('score')}",
                        file=sys.stderr,
                    )
                    return func_id, {
                        "passed": True,
                        "coverage_pct": pct,
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
                    f"[pipeline] existing test coverage={pct}% below threshold for {func_id}",
                    file=sys.stderr,
                )

        except RuntimeError:
            raise
        except Exception as e:
            print(
                f"[pipeline] existing-test judge fast path skipped for {func_id}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    attempt = 1

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
                func=func_for_prompt,
                coverage=cov or {},
                judge_verdict=last_judge,
                semantic_context=semantic_context,
            )
        else:
            prompt = prompt_for_function_test_with_semantic_context(
                func=func_for_prompt,
                coverage=cov or {},
                test_file=str(unit_test_file),
                process_name=process_name,
                attempt=attempt,
                max_attempts=max_attempts,
                make_ok=last_make_ok,
                semantic_context=semantic_context,
                last_judge_verdict=last_judge,
            )

        prompt = f"""{prompt}

{unit_agent_location_context}

FINAL TASK:
- Rewrite or repair this file with real CUnit tests:
    {unit_test_file}

FINAL REQUIREMENTS:
- Add at least one real CU_add_test().
- Execute target range {func.get("start_line")}-{func.get("end_line")} directly
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
            no_test_prompt = f"""
The previous edit is invalid because the unit test still has no real CU_add_test().

Rewrite the unit test file from scratch. Do not preserve placeholder content.

UNIT TEST FILE TO REWRITE:
 {unit_test_file}

TARGET SOURCE TO READ:
 {source_file_abs}

TARGET LINE RANGE:
 {func.get("start_line")} - {func.get("end_line")}

MASTER INTEGRATED TEST TO READ FOR WORKING WRAPPERS/STUBS/HELPERS:
 {master_test_file}

MASTER MAKEFILE:
 {master_makefile}

UNIT MAKEFILE:
 {unit_makefile}

VALIDATED GENERATED STUBS:
{_validated_stub_text()}

BUILD COMMAND:
 cd {unit_dir}
 make test

Requirements:
- Create a complete CUnit file.
- Include CUnit headers.
- Add at least one test function.
- Register it with CU_add_test().
- Execute the target line range directly or through a real caller.
- Use/copy only needed wrappers/helpers from the master integrated test.
- Do not blindly include the master integrated test.
- Do not leave an empty CUnit suite.
- Do not create fake project headers or spoof project types/macros/functions.
- If headers/types/macros are missing, inspect the real production headers and
  original Makefile, then fix INCLUDE/CPPFLAGS/LDFLAGS in the unit Makefile.
- Use real project definitions whenever available.
"""

            if last_build_diag:
                no_test_prompt = f"""{no_test_prompt}

PREVIOUS FAILURE CONTEXT:
{last_build_diag}
"""

            run_agent(
                cfg,
                unit_dir,
                no_test_prompt,
                f"{safe_id}_empty_test_fix_{int(time.time())}.json",
                folder=repo_root,
                history_dir=unit_dir / "agent_history",
            )

        if not _has_real_cu_add_test(unit_test_file):
            last_build_diag = (
                "The previous agent attempt still produced no real CU_add_test(). "
                "The unit test file is still empty or placeholder-only. "
                f"File: {unit_test_file}"
            )
            print(
                f"[pipeline] {func_id} attempt {attempt}: still no CU_add_test after repair",
                file=sys.stderr,
            )
            attempt += 1
            continue

        _sync_stub_srcs(unit_test_file, unit_makefile, test_dir, unit_dir)
        sync_wrap_flags(unit_test_file, unit_makefile)

        make_res = run_make_test(unit_dir)
        last_make_ok = bool(make_res.get("ok"))

        cov, pct = _run_current_coverage()

        # Judge immediately if coverage is good, even when make/test failed.
        if pct is not None and pct >= float(cfg.coverage_threshold):
            judge = run_semantic_test_judge(
                cfg,
                test_dir=unit_dir,
                repo_root=repo_root,
                process_name=process_name,
                test_file=unit_test_file,
                func=func_for_prompt,
                coverage=cov or {},
                make_result=make_res or {},
            )

            backup_good_cunit_if_best(
                unit_dir=unit_dir,
                unit_test_file=unit_test_file,
                unit_makefile=unit_makefile,
                func=func_for_prompt,
                coverage_pct=pct,
                coverage=cov or {},
                make_result=make_res or {},
                judge_verdict=judge,
                cfg=cfg,
            )

            write_json(coverage_file, cov or {})
            write_json(judge_verdict_file, judge)
            last_judge = judge

            if judge.get("passed"):
                print(
                    f"[pipeline] unit test PASSED: {func_id} score={judge.get('score')}",
                    file=sys.stderr,
                )
                return func_id, {
                    "passed": True,
                    "coverage_pct": pct,
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

        # Only compile-fix when coverage is not good enough and make/test failed.
        if not make_res["ok"]:
            diag = _safe_build_diag(make_res)

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
  real stub.
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
                last_build_diag = (
                    "The compile-fix attempt removed or failed to add real CU_add_test(). "
                    f"File: {unit_test_file}"
                )
                print(
                    f"[pipeline] {func_id} attempt {attempt}: no CU_add_test after compile_fix",
                    file=sys.stderr,
                )
                attempt += 1
                continue

            _sync_stub_srcs(unit_test_file, unit_makefile, test_dir, unit_dir)
            sync_wrap_flags(unit_test_file, unit_makefile)

            make_res = run_make_test(unit_dir)
            last_make_ok = bool(make_res.get("ok"))

            cov, pct = _run_current_coverage()

            if pct is not None and pct >= float(cfg.coverage_threshold):
                judge = run_semantic_test_judge(
                    cfg,
                    test_dir=unit_dir,
                    repo_root=repo_root,
                    process_name=process_name,
                    test_file=unit_test_file,
                    func=func_for_prompt,
                    coverage=cov or {},
                    make_result=make_res or {},
                )

                backup_good_cunit_if_best(
                    unit_dir=unit_dir,
                    unit_test_file=unit_test_file,
                    unit_makefile=unit_makefile,
                    func=func_for_prompt,
                    coverage_pct=pct,
                    coverage=cov or {},
                    make_result=make_res or {},
                    judge_verdict=judge,
                    cfg=cfg,
                )

                write_json(coverage_file, cov or {})
                write_json(judge_verdict_file, judge)
                last_judge = judge

                if judge.get("passed"):
                    print(
                        f"[pipeline] unit test PASSED after compile_fix: {func_id} score={judge.get('score')}",
                        file=sys.stderr,
                    )
                    return func_id, {
                        "passed": True,
                        "coverage_pct": pct,
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
                        cov_text = json.dumps(cov or {}, ensure_ascii=False, indent=2, default=str)
                    except Exception:
                        cov_text = repr(cov)

                    last_build_diag = f"""
Build succeeded after compile fix, but target coverage did not meet threshold.

Target:
 {func_id}

Required threshold:
 {cfg.coverage_threshold}

Coverage object:
{cov_text}

The next attempt should make the test execute lines {func.get("start_line")}-{func.get("end_line")}
in:
 {source_file_abs}
"""

            attempt += 1
            continue

        # Build/test was okay but coverage was low/missing.
        print(
            f"[pipeline] {func_id} attempt {attempt} coverage={pct}% make_ok={make_res['ok']}",
            file=sys.stderr,
        )

        try:
            cov_text = json.dumps(cov or {}, ensure_ascii=False, indent=2, default=str)
        except Exception:
            cov_text = repr(cov)

        last_build_diag = f"""
Build succeeded, but target coverage did not meet threshold.

Target:
 {func_id}

Required threshold:
 {cfg.coverage_threshold}

Observed coverage:
 {pct}

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
                "func_id": func_id,
                "attempts": max_attempts,
                "coverage_pct": pct,
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
        f"{func_id} coverage={pct}% make_ok={last_make_ok}",
        file=sys.stderr,
    )

    return func_id, {
        "passed": False,
        "coverage_pct": pct,
        "semantic_score": None if not last_judge else last_judge.get("score"),
        "verdict": last_judge,
        "unit_dir": str(unit_dir),
        "error": f"max attempts reached: {max_attempts}",
        "last_make_ok": last_make_ok,
    }


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

        # After level done: persist semantic context for accepted funcs
        for func in level_funcs:
            result = all_results.get(func["id"], {})
            if result.get("passed"):
                src_abs = _resolve_source_file(cfg, func["source_file"])
                _append_semantic_context(test_dir, {**func, "source_file": str(src_abs)}, result["verdict"])

    return all_results

# endregion Stage 4 — Parallel Unit Test Generation

# region Stage 5 — Unit Test Integration
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

        # Continuation check
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

    # Final full build+run of master suite after all integrations
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

# endregion Stage 5 — Unit Test Integration

# region Driver & CLI
def run(cfg: PipelineConfig) -> int:
    paths = derive_paths(cfg)
    for k, p in paths.items():
        if isinstance(p, Path):
            p.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] source : {cfg.source_dir}", file=sys.stderr)
    print(f"[pipeline] test dir: {paths['test_dir']}", file=sys.stderr)

    analysis = run_or_load_analysis(cfg, paths["analysis_path"])

    ensure_test_file(cfg, paths)                          # Stage 0a: bare skeleton
    flags = build_annotated_makefile(cfg, paths)          # Stage 0b: Makefile + context
    handle_stubs(cfg, paths, analysis)                    # Stage 1+2: stub gen+validate, integrate

    minimal_ok = ensure_minimal_test_runs(cfg, paths)     # Stage 3: smoke test master
    if not minimal_ok:
        print("[pipeline] minimal test never passed; aborting", file=sys.stderr)
        write_json(paths["test_dir"] / "DONE.json", {
            "finished_at": now_iso(), "source": str(cfg.source_dir),
            "error": "minimal_test_failed",
        })
        return 2

    unit_results = parallel_generate_unit_tests(cfg, paths, analysis, flags)  # Stage 4
    master_ok = integrate_all_unit_tests_sequential(       # Stage 5
        cfg, paths, analysis, unit_results, flags)

    passed = sum(1 for r in unit_results.values() if r.get("passed"))
    total = len(unit_results)
    exit_code = 0 if master_ok else 3
    write_json(paths["test_dir"] / "DONE.json", {
        "finished_at": now_iso(),
        "source": str(cfg.source_dir),
        "master_suite_ok": master_ok,
        "functions_total": total,
        "functions_done": passed,
        "functions_passed": passed,
        "functions_skipped": sum(
            1 for r in unit_results.values()
            if r.get("passed") and r.get("coverage_pct") is None
        ),
        "coverage": {
            k: round(v["coverage_pct"], 1) if v.get("coverage_pct") is not None else None
            for k, v in unit_results.items()
        },
        "semantic_score": {
            k: v.get("semantic_score") for k, v in unit_results.items()
        },
        "semantic_context_file": str(_semantic_context_path(paths["test_dir"])),
    })
    print(f"[pipeline] finished. {passed}/{total} functions passed. master_ok={master_ok}", file=sys.stderr)
    return exit_code

def parse_args(argv: Optional[list[str]] = None) -> PipelineConfig:
    ap = argparse.ArgumentParser(description="Simplified CUnit test-gen pipeline")
    ap.add_argument("--source", required=True, type=Path,
                    help=".../src/<process_name>")
    ap.add_argument("--agent-js", required=True, type=Path,
                    help="Path to agent.js")
    ap.add_argument("--system-json", type=Path, default=None)
    ap.add_argument("--agent-timeout-sec", type=int, default=1800)
    ap.add_argument("--max-agent-iterations", type=int, default=25)
    ap.add_argument(
        "--max-compile-fix-attempts",
        type=int,
        default=5,
        help="Legacy/v1 compatibility only; v2 repair loops run until success.",
    )
    ap.add_argument(
        "--max-test-attempts",
        type=int,
        default=4,
        help="Legacy/v1 compatibility only; v2 per-function loops run until success.",
    )
    ap.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Continuation skip threshold. Functions with coverage >= this percent are skipped.",
    )
    ap.add_argument(
        "--stub-batch-size",
        type=int,
        default=8,
        help="Number of generated stubs to give the integration agent at once.",
    )
    ap.add_argument("--only-function", type=str, default=None)
    ap.add_argument("--only-level", type=int, default=None)
    ap.add_argument("--max-functions", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python-bin", type=str, default=sys.executable,
                    help="Python binary used by agent.js for gcov analysis tool")
    ap.add_argument(
        "--max-stub-gen-retries",
        type=int,
        default=3,
        help="Legacy/v1 compatibility only; v2 stub generation runs until success.",
    )
    ap.add_argument(
        "--max-stub-integrate-retries",
        type=int,
        default=3,
        help="Legacy/v1 compatibility only; v2 stub integration runs until success.",
    )
    ap.add_argument(
        "--max-minimal-test-attempts",
        type=int,
        default=5,
        help="Legacy/v1 compatibility only; v2 minimal validation runs until success.",
    )
    ap.add_argument(
        "--func-docs-dir",
        type=Path,
        default=Path("/home/seigyo/rl/moove_docs/func"),
        help="Directory containing function docs as <func_name>.md",
    )
    ap.add_argument(
        "--semantic-judge-min-score",
        type=int,
        default=75,
        help="Minimum semantic score for a function to be considered done.",
    )
    ap.add_argument(
        "--max-unit-test-workers",
        type=int,
        default=4,
        help="ThreadPoolExecutor workers for parallel unit test generation.",
    )
    ap.add_argument(
        "--max-fix-attempts",
        type=int,
        default=20,
        help="Legacy/v1 compatibility only; v2 fix loops run until success.",
    )

    ns = ap.parse_args(argv)
    return PipelineConfig(
        source_dir=ns.source.resolve(),
        agent_js=ns.agent_js.resolve(),
        system_json=ns.system_json.resolve() if ns.system_json else None,
        agent_timeout_sec=ns.agent_timeout_sec,
        max_agent_iterations=ns.max_agent_iterations,
        max_compile_fix_attempts=ns.max_compile_fix_attempts,
        max_test_attempts=ns.max_test_attempts,
        coverage_threshold=ns.coverage_threshold,
        stub_batch_size=max(1, ns.stub_batch_size),
        only_function=ns.only_function,
        only_level=ns.only_level,
        max_functions=ns.max_functions,
        dry_run=ns.dry_run,
        python_bin=ns.python_bin,
        max_stub_gen_retries=ns.max_stub_gen_retries,
        max_stub_integrate_retries=ns.max_stub_integrate_retries,
        max_minimal_test_attempts=ns.max_minimal_test_attempts,
        func_docs_dir=ns.func_docs_dir.resolve(),
        semantic_judge_min_score=ns.semantic_judge_min_score,
        max_unit_test_workers=ns.max_unit_test_workers,
        max_fix_attempts=ns.max_fix_attempts,
    )

def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        return run(cfg)
    except KeyboardInterrupt:
        print("[pipeline] interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[pipeline] fatal: {e}", file=sys.stderr)
        return 1

# endregion Driver & CLI

if __name__ == "__main__":
    raise SystemExit(main())
