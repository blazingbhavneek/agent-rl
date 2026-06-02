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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from difflib import SequenceMatcher

from stub.stub import ProjectAnalyzer          # type: ignore
from tools.gcov import analyze_gcov_range      # type: ignore


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

    func_norm = _normalize_doc_name(func_name)
    scored: list[tuple[float, Path]] = []
    for p in docs_dir.glob("*.md"):
        stem_norm = _normalize_doc_name(p.stem)
        score = similarity_ratio(func_name, p.stem)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None, 0.0, []

    best_score, best_path = scored[0]
    if best_score >= threshold:
        return best_path, best_score, scored[:5]
    return None, best_score, scored[:5]


# ==========================================================================
# Config
# ==========================================================================
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


# ==========================================================================
# Directory derivation
# ==========================================================================
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


# ==========================================================================
# File IO helpers
# ==========================================================================
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


# ==========================================================================
# Analysis
# ==========================================================================
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


# ==========================================================================
# Test file / Makefile scaffolding
# ==========================================================================
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


def _project_source_files(cfg: PipelineConfig) -> list[Path]:
    """
    Return actual absolute .c files under cfg.source_dir.
    No guessing.
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


import os
import re
import subprocess
import sys
from pathlib import Path


def ensure_makefile(cfg: PipelineConfig, paths: dict) -> None:
    """
    Ensure Makefile exists by running do_mkmf first, then append only the pipeline test target.

    Coverage behavior:
    - Uses actual production .c files discovered under cfg.source_dir.
    - Does NOT guess src/<process>.c.
    - Generates production .gcov files such as:
        ^#^#src#dio100d#dio100d.c.gcov
    - test_*.c.gcov may also be generated, but function coverage checker should ignore it and match production gcov files by Source: header.
    """
    test_dir: Path = paths["test_dir"]
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

    # -------------------------------------------------------------------------
    # 3. Append/update test target block.
    # -------------------------------------------------------------------------
    test_program = test_file.stem
    test_src = test_file.name

    block_start = f"# === TEST TARGET FOR {process_name} ==="
    block_end = f"# === END TEST TARGET FOR {process_name} ==="

    test_block = f"""
{block_start}
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
\t$(CC) $(CFLAGS_LINUX) $(CFLAGS) $(INCLUDE) $(COVERAGE_FLAGS) $(TEST_SRCS) -o $(TEST_PROGRAM) $(TEST_LIBS) \\
\t-Wl,--gc-sections \\
\t$(WRAP_FLAGS)

coverage-test:
\t@for src in $(PRODUCTION_SRCS); do \\
\t\tgcov -b -c -p "$$src" >> $(TEST_REPORT_FILE) 2>&1 || true; \\
\tdone
\t@gcov -b -c -p *.gcno >> $(TEST_REPORT_FILE) 2>&1 || true

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

    print(f"[pipeline] Makefile ready: {makefile}", file=sys.stderr)


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
    production_include_block = ""
    if production_include_lines:
        production_include_block = (
            "\n/* Rename production main before including production sources. */\n"
            f"#define main {process_name}_entry_main\n"
            "\n/* Auto-included production sources for coverage. */\n"
            + "\n".join(production_include_lines)
            + "\n\n"
            "/* Restore test main name. */\n"
            "#undef main\n"
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
static void test_generated_stub_linkage_smoke(void)
{{
    CU_ASSERT_TRUE(1);
}}

{TEST_FILE_MARKERS[6]}
int main(void)
{{
    CU_pSuite suite = NULL;
    if (CU_initialize_registry() != CUE_SUCCESS) {{
        return CU_get_error();
    }}
    suite = CU_add_suite("{process_name}_generated_suite", NULL, NULL);
    if (suite == NULL) {{
        CU_cleanup_registry();
        return CU_get_error();
    }}
    if (CU_add_test(suite, "test_generated_stub_linkage_smoke",
        test_generated_stub_linkage_smoke) == NULL) {{
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

    # Remove bad late main define if it exists after the production include.
    bad_define = f"#define main {process_name}_entry_main"
    if bad_define in text:
        text = text.replace(bad_define + "\n", "")
        text = text.replace(bad_define, "")
        changed = True

    # Add any missing actual production includes.
    missing_includes = [
        line for line in production_include_lines
        if line not in text
    ]
    if missing_includes:
        insertion = (
            "\n/* Rename production main before including production sources. */\n"
            f"#define main {process_name}_entry_main\n"
            "\n/* Auto-included production sources for coverage. */\n"
            + "\n".join(missing_includes)
            + "\n\n"
            "/* Restore test main name. */\n"
            "#undef main\n"
        )
        text = text.replace(TEST_FILE_MARKERS[0], TEST_FILE_MARKERS[0] + insertion, 1)
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


# ==========================================================================
# Wrap flags in Makefile
# ==========================================================================
def ensure_wrap_flag(makefile: Path, func_name: str) -> bool:
    """Append -Wl,--wrap=<name> to WRAP_FUNCS if not already present."""
    flag = f"-Wl,--wrap={func_name}"
    text = read_text(makefile)
    if flag in text:
        return False
    append_text(makefile, f"WRAP_FUNCS += {flag}\n")
    return True


# ==========================================================================
# Stub / test-file scanning
# ==========================================================================
def stub_exists(test_file: Path, name: str) -> bool:
    text = read_text(test_file)
    # Match a C identifier starting with __wrap_<name>
    return bool(re.search(rf"__wrap_{re.escape(name)}\b", text))


# ==========================================================================
# Agent invocation
# ==========================================================================
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
    ]

    env = os.environ.copy()
    env["MAX_ITERATIONS"] = str(
        max_iterations if max_iterations is not None else cfg.max_agent_iterations
    )

    actual_timeout = timeout_sec if timeout_sec is not None else cfg.agent_timeout_sec

    if cfg.dry_run:
        print(f"[pipeline][dry-run] would run: {' '.join(cmd)}", file=sys.stderr)
        return {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}

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
    return res


# ==========================================================================
# Build
# ==========================================================================
def run_make_test(test_dir: Path) -> dict:
    cmd = ["make", "test"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(test_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "",
            "stderr": "make test timed out"}


# ==========================================================================
# Coverage
# ==========================================================================
def find_gcov_file(test_dir: Path, source_file: str) -> Optional[Path]:
    """
    Look for the .gcov that matches the function's original source.
    Try:
        - test_<process>.c.gcov    (if sources are #included into test file)
        - <basename>.gcov
        - any single .gcov in the directory
    """
    candidates: list[Path] = []
    base = Path(source_file).name
    for p in test_dir.glob("*.gcov"):
        if p.name in (f"{base}.gcov",):
            return p
        candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    # Fallback: prefer test_<*.c>.gcov
    for p in candidates:
        if p.name.endswith(".c.gcov"):
            return p
    return candidates[0] if candidates else None


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
            ["gcov", "-b", "-c", "-p", gcno.name],
            cwd=str(test_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )

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


# ==========================================================================
# Prompts (all external - nothing hardcoded in agent.js)
# ==========================================================================
def prompt_for_stub(func_name: str,
    test_file: str,
    process_name: str,
    test_dir: str) -> str:
    return f"""You are editing the CUnit test file `{test_file}` in `{test_dir}`.

TASK: add a linker-wrapper stub for ONE external function: `{func_name}`.

ABSOLUTE RULES:
- Modify ONLY `{test_file}` in the current directory. Do not touch production sources.
- Do NOT create stubs.c or any extra headers.
- Place the stub below the marker: `/* === Linker Wrapper Stubs === */`.
- Do NOT duplicate existing includes, helpers, globals, or other stubs.
- The wrapper name MUST be `__wrap_{func_name}`.
- Determine the correct signature by inspecting local headers and source files in `{test_dir}`'s parent project (look upward for `.h` files).
- Add a simple fprintf(stderr, ...) log at the top of the wrapper.
- Keep behaviour minimal and deterministic.
- If `{func_name}` looks like an event/callback registration stub, invoke the passed callback with reasonable arguments.
- Preserve ALL existing code, tests, and stubs in the file.
- Run `make test` if you want to validate; ignore failures unrelated to your stub.
- When done, call submit_and_exit.
"""


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

    return f"""The CUnit test does not compile.

FILES YOU MAY EDIT
- Test Makefile:
  `{makefile}`
- Test C file:
  `{test_file}`

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


def prompt_for_function_test(func: dict,
                        coverage: Optional[dict],
                        test_file: str,
                        process_name: str,
                        attempt: int,
                        max_attempts: int) -> str:
    fid = func["id"]
    name = func["name"]
    src = func["source_file"]
    s = func["start_line"]
    e = func["end_line"]
    cov_pct = None
    uncovered = []
    matched_gcov_file = None
    if coverage and isinstance(coverage, dict):
        summary = coverage.get("summary", {}) or {}
        cov_pct = summary.get("coverage_percent")
        if cov_pct is None:
            cov_pct = coverage.get("coverage_percent")
        uncovered = coverage.get("uncovered_executable_lines", []) or []
        matched_gcov_file = coverage.get("gcov_file")

    if uncovered:
        lines = []
        for u in uncovered[:80]:
            if isinstance(u, dict):
                line_no = u.get("line", "?")
                source_text = u.get("source", "")
                lines.append(f"{str(line_no):>5}: {source_text}")
            else:
                lines.append(str(u))
        uncovered_text = "\n".join(lines)
    else:
        if cov_pct == 0 or cov_pct == 0.0:
            uncovered_text = (
                "No uncovered-line list was extracted, but coverage is 0.0%.\n"
                "Assume the target function is not being executed yet.\n"
                "First add and register a CUnit test that directly calls the target production function."
            )
        elif cov_pct is None:
            uncovered_text = (
                "No gcov coverage data was available yet.\n"
                "First add and register a CUnit test that directly calls the target production function."
            )
        else:
            uncovered_text = (
                "No uncovered executable lines were reported for this range.\n"
                "If coverage is below threshold, inspect the matched production gcov file and add tests."
            )

    if matched_gcov_file:
        gcov_tool_text = f"""9. You may use the `analyze_function_coverage` tool if helpful.

IMPORTANT:
Do NOT analyze `test_{process_name}.c.gcov` or any `test_*.c.gcov` file.
Those files are CUnit test harness coverage, not production source coverage.
Use the matched PRODUCTION gcov file whose `Source:` header resolves to the target source file.

For this target, use:
    gcov_file: {matched_gcov_file}
    start_line: {s}
    end_line: {e}

Production gcov filenames may be path-encoded by `gcov -p`, for example:
    ^#^#src#dio100d#dio100d.c.gcov
That is normal. Do not assume the gcov filename matches the test filename."""
    else:
        gcov_tool_text = f"""9. You may use the `analyze_function_coverage` tool if helpful.

IMPORTANT:
Do NOT analyze `test_{process_name}.c.gcov` or any `test_*.c.gcov` file.
Those files are CUnit test harness coverage, not production source coverage.
No matched production gcov file is available yet.
First make the generated test directly call the target production function so gcov creates/updates the production `.gcov` file.

Target source:
```c
{src}
```

Line range:
  start_line: {s}
  end_line:   {e}"""

    if name == "main":
        production_call_rule = f"""CRITICAL HARNESS FACT
The production source may be included into the CUnit test file like this:
```c
#define main DISABLED_MAIN
#include "...production source..."
#undef main
```
Therefore the target production function:
```c
int main(int argc, char **argv)
```
may be compiled and callable inside the test file as:
```c
DISABLED_MAIN(argc, argv)
```
The test file has its own CUnit `main()`. Do NOT call `main()` directly.

For this target, add and register at least one CUnit test that directly calls the production main through the renamed symbol:
```c
char *argv[] = {{ "{process_name}", NULL }};
int ret = DISABLED_MAIN(1, argv);
```
Wrapper-only smoke tests are NOT enough.
The first goal is to make gcov change from:
```text
function DISABLED_MAIN called 0
```
to:
```text
function DISABLED_MAIN called 1
```
or higher."""
    else:
        production_call_rule = f"""MANDATORY PRODUCTION-CALL RULE
Add and register at least one CUnit test that directly calls the current target production function itself:
```text
{name}
```
Wrapper-only smoke tests are NOT enough.

If the function has static linkage but the production `.c` file is included directly into the test file, call it directly from the test because it is in the same translation unit.

Use safe/minimal arguments:
- For `int foo(void)`, call `foo()`.
- For scalar args, use safe values like `0`, `1`, boundary values, or known constants.
- For pointer args, create local zeroed objects or safe buffers when possible.
- Do not pass `NULL` unless the function is expected to handle `NULL`.

The first goal is to make gcov report the target function as called at least once."""

    report_name = f"{test_file.split('/')[-1]}_report.txt"

    return f"""You are improving CUnit coverage for ONE target function.

TARGET FUNCTION
id:           {fid}
name:         {name}
source:       {src}
line range:   {s} .. {e} (inclusive)

CURRENT STATUS
attempt:        {attempt} / {max_attempts}
coverage so far: {cov_pct if cov_pct is not None else 'unknown'}%

{production_call_rule}

UNCOVERED EXECUTABLE LINES (from PRODUCTION gcov, not test harness gcov)
{uncovered_text}

FILE YOU MAY EDIT
`{test_file}`
(do NOT modify any production sources)

RULES
0. MANDATORY:
Add and register at least one CUnit test that directly calls the current target production function itself. Do not only call `__wrap_*` stubs.

If the target function is `main` and the harness renames it with:
```c
#define main DISABLED_MAIN
```
then call:
    DISABLED_MAIN(argc, argv)
not:
    main(argc, argv).

1. Focus ONLY on the target function above. Do not modify tests for other functions.
2. Preserve ALL existing stubs, globals, helpers, and tests.
3. Place each new CUnit test under: `/* === Test Cases === */`.
4. Register every new test in the test registration area or in `all_tests()` using `CU_add_test()`. If the file does not literally contain `/* === Test Registration === */`, register it beside the existing `CU_add_test()` calls.
5. Each new test MUST start with:
```c
/**
 * @brief Tests {name} for <contract>.
 * @details Exercises: normal / edges / errors.
 * @rationale <why this matters>.
 */
static void test_<scenario>(void) {{
    fprintf(stderr, "===Test case test_<scenario>===\n");
    ...
}}
```
6. Cover: normal path, NULL/zero/empty edges, early-return / error paths, boundary values, loop entry/skip, switch cases.
7. Reuse existing `__wrap_*` stubs. If you need a NEW stub, add it under `/* === Linker Wrapper Stubs === */` with correct signature AND log.
8. After editing, run `make test` and inspect `{report_name}`.
{gcov_tool_text}
10. Do NOT consider your task complete until coverage for lines `{s}..{e}` is non-zero, or you have a concrete explanation why it cannot be improved.

When done, call submit_and_exit.
"""


# ==========================================================================
# Stub generation (parallel) + serial insertion
# ==========================================================================
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

    # Continuation: reuse existing generated stub if valid.
    if stub_out.exists():
        cached_body = _clean_stub_body(read_text(stub_out))
        if _is_valid_stub_body(cached_body):
            print(
                f"[pipeline] reuse cached generated stub for {func_name}: {stub_out}",
                file=sys.stderr,
            )
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
    return body


def integrate_stubs_and_compile_with_agent(
    cfg: PipelineConfig,
    paths: dict,
    generated_bodies: dict[str, str],
    missing_stub_names: list[str],
    *,
    batch_index: int = 1,
    batch_total: int = 1,
) -> None:
    """
    Integrate a batch of generated stubs.

    Critical:
    - Provide actual absolute source file list to the agent.
    - Tell it not to guess src/<process>.c.
    """
    test_dir: Path = paths["test_dir"]
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    process_name: str = paths["process_name"]

    source_dir = cfg.source_dir.resolve()
    source_makefile = source_dir / "Makefile"

    # Better repo_root fallback; not used for source guessing.
    repo_root = source_dir
    for parent in source_dir.parents:
        if (parent / ".git").exists():
            repo_root = parent
            break

    actual_source_files = _project_source_files(cfg)

    stub_gen_dir = test_dir / "_stub_gen"
    stub_gen_dir.mkdir(parents=True, exist_ok=True)
    stubs_json = stub_gen_dir / f"generated_stubs_batch_{batch_index:03d}_of_{batch_total:03d}.json"
    write_json(stubs_json, {
        "generated_at": now_iso(),
        "batch_index": batch_index,
        "batch_total": batch_total,
        "process_name": process_name,
        "source_dir": str(source_dir),
        "source_makefile": str(source_makefile),
        "actual_source_files": [str(p) for p in actual_source_files],
        "test_file": str(test_file),
        "makefile": str(makefile),
        "func_docs_dir": str(cfg.func_docs_dir),
        "missing_stub_names": missing_stub_names,
        "stubs": generated_bodies,
    })

    wrap_flags_text = "\n".join(
        f" -Wl,--wrap={name}" for name in missing_stub_names
    )

    actual_source_files_text = _source_files_json_for_prompt(cfg)

    prompt = f"""You are integrating a BATCH of generated linker-wrapper stub into a CUnit test.
This is batch {batch_index} / {batch_total}.

REPOSITORY ROOT
{repo_root}

SOURCE FOLDER PASSED BY USER
{source_dir}

ORIGINAL SOURCE MAKEFILE
{source_makefile}

ACTUAL SOURCE FILES UNDER SOURCE FOLDER
{actual_source_files_text}

TEST DIRECTORY
{test_dir}

TEST FILE TO EDIT
{test_file}

TEST MAKEFILE TO EDIT
{makefile}

GENERATED STUBS JSON FOR THIS BATCH
{stubs_json}

REQUIRED WRAP FLAGS FOR THIS BATCH
{wrap_flags_text or " (none)"}

CRITICAL PATH RULES
- Do NOT invent paths like:
  {source_dir.parent}/{process_name}.c
- Do NOT guess:
  `src/{process_name}.c`
- Use only the actual source files listed above or in `actual_source_files` from the JSON.
- The user already provided the source folder. Treat:
  {source_dir}
  as the source of truth.

IMPORTANT MAKEFILE CONTEXT
- The test Makefile should be generated by:
  `cd {test_dir} && do_mkmf {source_dir}`
- Do NOT replace the whole Makefile with a homemade Makefile.
- If missing headers, macros, or libraries occur, read:
  {source_makefile}
  and copy/append relevant settings into:
  {makefile}
- Preserve existing do_mkmf-generated content.
- Preserve existing wrapper flags.

TASK
1. Read `{stubs_json}`.
2. Read `{test_file}`.
3. Read `{makefile}`.
4. Read the original source Makefile:
  `{source_makefile}`
5. Integrate all usable `__wrap_*` stubs from this batch into `{test_file}`.
6. Add wrapper flags for this batch to `{makefile}`:
  {wrap_flags_text or " (none)"}
7. Ensure there is at least one simple CUnit smoke test.
8. Run:
   make test
   from:
   `{test_dir}`
9. If compile fails, fix only `{test_file}` and `{makefile}`.
   If failure is missing include/macro/library config, copy needed settings from `{source_makefile}`.

STRICT EDIT RULES
- You may modify only:
  - `{test_file}`
  - `{makefile}`
- Do NOT modify production source files.
- Do NOT modify production headers.
- Do NOT create random extra source/header files.

TEST FILE ORGANIZATION
Use these existing markers:
- `/* === Includes === */`
- `/* === Compatibility Definitions === */`
- `/* === Test Globals === */`
- `/* === Test Helpers === */`
- `/* === Linker Wrapper Stubs === */`
- `/* === Test Cases === */`
- `/* === Test Registration === */`

STUB RULES
- Put wrappers only under `/* === Linker Wrapper Stubs === */`.
- Do not duplicate wrappers.
- If a wrapper already exists, update/improve it.
- Remove any calls to `__real_*`.
- Every wrapper must be deterministic.
- Every wrapper should log:
  `fprintf(stderr, "__wrap_NAME called\n")`;
- Fix signatures if compile errors show mismatch.

MAKEFILE RULES
- Add each wrapper flag as:
  `WRAP_FUNCS += -Wl,--wrap=name`
- Do not remove existing WRAP_FUNCS.
- Do not replace the do_mkmf template.
- Copy missing build config from original source Makefile if needed.

When done, call submit_and_exit.
"""

    run_agent(
        cfg,
        work_dir=repo_root,
        folder=repo_root,
        history_dir=test_dir / "agent_history",
        prompt=prompt,
        history_name=f"__integrate_stubs_batch_{batch_index:03d}_of_{batch_total:03d}.json",
        max_iterations=max(cfg.max_agent_iterations, 60),
        timeout_sec=max(cfg.agent_timeout_sec, 3600),
    )


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


def handle_stubs(cfg: PipelineConfig, paths: dict, analysis: dict) -> None:
    """
    Generate/reuse missing stubs and integrate them in batches.

    Continuation behavior:
    - If a wrapper already exists in the final test file, do not regenerate it.
    - If a generated stub file already exists under _stub_gen/<func>/stub.c,
        generate_stub_code() will reuse it instead of calling the agent.
    - Wrapper flags are ensured for all candidates, even if the wrapper already
        exists in the test file.
    """
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    test_dir: Path = paths["test_dir"]

    candidates = collect_stub_candidates(analysis)
    print(f"[pipeline] {len(candidates)} stub candidates total", file=sys.stderr)

    # Always ensure wrapper flags for all candidates.
    # This fixes continuation cases where the wrapper exists but the Makefile flag
    # is missing.
    for name in candidates:
        ensure_wrap_flag(makefile, name)

    # Only generate/integrate stubs that are not already in the final test file.
    missing = [n for n in candidates if not stub_exists(test_file, n)]
    print(f"[pipeline] {len(missing)} missing stubs in final test file",
        file=sys.stderr)

    if not missing:
        print("[pipeline] all stubs already exist in test file; skipping stub generation/integration", file=sys.stderr)
        return

    bodies: dict[str, str] = {}

    # 1. Generate or reuse candidate stub files in parallel.
    with ThreadPoolExecutor(max_workers=min(6, len(missing))) as pool:
        futs = {
            pool.submit(generate_stub_code, cfg, test_dir, n): n
            for n in missing
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                body = fut.result()
            except Exception as e:
                print(
                    f"[pipeline] stub gen failed for {name}: {e}",
                    file=sys.stderr,
                )
                body = None
            if body:
                bodies[name] = body
                print(
                    f"[pipeline] generated/reused stub body for {name} "
                    f"({len(body)} chars)",
                    file=sys.stderr,
                )
            else:
                print(f"[pipeline] NO usable body for {name}", file=sys.stderr)

    # 2. Integrate stubs in smaller batches.
    batch_size = max(1, int(cfg.stub_batch_size))
    batches = [
        missing[i:i + batch_size]
        for i in range(0, len(missing), batch_size)
    ]

    print(
        f"[pipeline] integrating stubs in {len(batches)} batch(es), batch_size={batch_size}",
        file=sys.stderr,
    )

    for idx, batch_names in enumerate(batches, start=1):
        batch_bodies = {
            name: bodies[name]
            for name in batch_names
            if name in bodies
        }
        print(
            f"[pipeline] integrating stub batch {idx}/{len(batches)}: "
            f"names={batch_names}, usable_bodies={len(batch_bodies)}",
            file=sys.stderr,
        )
        integrate_stubs_and_compile_with_agent(
            cfg=cfg,
            paths=paths,
            generated_bodies=batch_bodies,
            missing_stub_names=batch_names,
            batch_index=idx,
            batch_total=len(batches),
        )


# ==========================================================================
# Compile-fix loop
# ==========================================================================
def ensure_skeleton_compiles(cfg: PipelineConfig, paths: dict) -> bool:
    """
    Compile-fix loop.
    Gives the agent:
    - exact source folder from args
    - original source Makefile
    - actual resolved source .c files
    """
    test_dir: Path = paths["test_dir"]
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]

    source_dir = cfg.source_dir.resolve()
    source_makefile = source_dir / "Makefile"
    actual_source_files = [str(p) for p in _project_source_files(cfg)]

    for attempt in range(cfg.max_compile_fix_attempts + 1):
        res = run_make_test(test_dir)
        if res["ok"]:
            print(
                f"[pipeline] skeleton compiles (attempt {attempt})",
                file=sys.stderr,
            )
            return True

        combined_output = (res.get("stderr") or "") + "\n---\n" + (res.get("stdout") or "")
        print(
            f"[pipeline] compile attempt {attempt + 1} failed, asking agent to fix",
            file=sys.stderr,
        )
        prompt = prompt_for_compile_fix(
            str(makefile),
            str(test_file),
            combined_output,
            source_dir=str(source_dir),
            source_makefile=str(source_makefile),
            actual_source_files=actual_source_files,
        )
        run_agent(
            cfg,
            test_dir,
            prompt,
            f"_compile_fix_{attempt:03d}.json",
        )

    return False


# ==========================================================================
# Per-function coverage loop
# ==========================================================================
def process_functions(cfg: PipelineConfig, paths: dict, analysis: dict) -> None:
    """
    Process target functions leaf -> root.

    Continuation behavior:
    - Before asking the agent to add tests for a function, run `make test`.
    - Analyze gcov line-range coverage for that function.
    - If coverage >= cfg.coverage_threshold, skip the function.
    - Default threshold is 80.0%.
    """
    test_dir: Path = paths["test_dir"]
    test_file: Path = paths["test_file"]
    process_name: str = paths["process_name"]

    funcs = functions_leaf_first(analysis)
    print(f"[pipeline] {len(funcs)} functions to process (leaf -> root)",
        file=sys.stderr)
    print(f"[pipeline] continuation skip threshold: {cfg.coverage_threshold:.1f}%",
        file=sys.stderr)

    done = 0
    for func in funcs:
        if cfg.only_function and func["id"] != cfg.only_function:
            continue
        if cfg.only_level is not None and func.get("depth") != cfg.only_level:
            continue
        if cfg.max_functions is not None and done >= cfg.max_functions:
            break

        # Continuation check:
        # Run current tests first, then inspect coverage for this exact function range.
        print(
            f"[pipeline] coverage pre-check for {func['id']} "
            f"lines {func['start_line']}..{func['end_line']}",
            file=sys.stderr,
        )
        make_res = run_make_test(test_dir)
        if not make_res["ok"]:
            print(
                f"[pipeline] make test failed during coverage pre-check for {func['id']}; "
                f"will still ask agent to work on this function",
                file=sys.stderr,
            )

        source_file_abs = _resolve_source_file(cfg, func["source_file"])
        print(
            f"[pipeline] resolved coverage source: {func['source_file']} -> {source_file_abs}",
            file=sys.stderr,
        )
        cov = check_function_coverage(
            test_dir,
            source_file_abs,
            func["start_line"],
            func["end_line"],
            source_root=cfg.source_dir.resolve(),
        )

        pct = None
        if cov is not None and isinstance(cov, dict):
            pct = (cov.get("summary", {}) or {}).get("coverage_percent")
            if pct is None:
                pct = cov.get("coverage_percent")

        if pct is not None and pct >= cfg.coverage_threshold:
            print(
                f"[pipeline] SKIP {func['id']} coverage={pct:.1f}% "
                f">= threshold={cfg.coverage_threshold:.1f}%",
                file=sys.stderr,
            )
            done += 1
            continue

        print(
            f"[pipeline] --> PROCESS {func['id']} current coverage={pct}%, "
            f"threshold={cfg.coverage_threshold:.1f}%",
            file=sys.stderr,
        )

        for attempt in range(1, cfg.max_test_attempts + 1):
            func_for_prompt = dict(func)
            func_for_prompt["source_file"] = str(source_file_abs)
            func_for_prompt["source_dir"] = str(cfg.source_dir.resolve())
            func_for_prompt["actual_source_files"] = [str(p) for p in _project_source_files(cfg)]
            func_for_prompt["initial_coverage_rule"] = (
                "For the first/initial CUnit test for this target function, "
                "add and register a test that directly calls the current target production function. "
                "Do not only call __wrap_* stubs. "
                "If the target function is main and the harness renames main using "
                "#define main DISABLED_MAIN, call DISABLED_MAIN(1, argv), not main. "
                "If the target is not main, call the target function itself with safe/minimal arguments. "
                "Use wrappers to prevent blocking, exiting, real I/O, database, networking, timers, "
                "shared memory, or logging side effects. "
                "The goal is that gcov reports the target function called at least once."
                """
When generating the first/initial CUnit test for a target source file, do not only create wrapper-smoke tests.
You must add at least one test that directly executes the current target production function being processed.

The current target function will be provided in the function metadata, for example:
- func["name"]
- func["source_file"]
- func["start_line"]
- func["end_line"]
- func["signature"] if available

Your goal for the first test is simple:
make gcov show that the target production function was called at least 1 time.

Rules:
1. If the target function is production main():
   The test harness may include the source with:
   #define main DISABLED_MAIN
   #include "..."
   #undef main
   In that case, call:
   char *argv[] = {"program_name", NULL};
   DISABLED_MAIN(1, argv);
   Do not call main() directly.

2. If the target function is not main():
   Call the actual target function by name using safe/minimal arguments.
   Examples:
   - For int foo(void): call foo();
   - For int foo(int x): call foo(0);
   - For int foo(char *s): call foo(NULL) first only if the function handles NULL safely; otherwise pass a local buffer.
   - For int foo(StructType *p): create a local StructType object, memset it to 0, and pass &object.
   - For void foo(void): call foo();

3. Do not settle for tests that only call __wrap_* stubs.
   Wrapper-only smoke tests are allowed, but they are not enough.
At least one registered CUnit test must call the target production function itself.

4. Register the production-call test in all_tests() using CU_add_test().

5. Keep the test safe:
- Stub or wrap functions that block forever.
- Stub or wrap functions that terminate the process.
- Stub or wrap external I/O, database, shared memory, networking, timers, and logging calls.
- Wrappers should return safe success values unless testing an error branch.
- If a function may loop forever, provide a wrapper/state variable that makes it return.

6. For production main():
Ensure wrappers make the normal path safe:
- pmf_mainloop returns immediately.
- pmf_exit does not exit.
- pmf_addevent returns 0.
- pmf_addtimer returns a valid nonnegative timer id, for example 1.
- initialization/open functions return success values.

7. The first test does not need perfect assertions.
It may use simple assertions like:
- CU_ASSERT_TRUE(1);
- CU_ASSERT_EQUAL(ret, expected);
- CU_ASSERT_TRUE(mock_called_count > 0);
The priority is production coverage:
the target function should appear in gcov as called 1 or more times.

8. If the function has static linkage but the source file is included directly into the test file:
call the static function directly from the test because it is in the same translation unit.

9. If the function name is replaced by a macro in the harness, call the rewritten name.
Specifically, if production main is renamed by:
#define main DISABLED_MAIN
then call DISABLED_MAIN(), not main().

10. Do not remove existing tests or wrappers. Add the production-call test alongside existing smoke/linkage tests.
"""
            )

            prompt = prompt_for_function_test(
                func=func_for_prompt,
                coverage=cov,
                test_file=str(test_file),
                process_name=process_name,
                attempt=attempt,
                max_attempts=cfg.max_test_attempts,
            )
            run_agent(
                cfg,
                test_dir,
                prompt,
                f"{_safe_filename(func['id'])}_attempt_{attempt:02d}.json",
            )

            make_res = run_make_test(test_dir)
            if not make_res["ok"]:
                print(
                    f"[pipeline] make test failed after attempt {attempt} for {func['id']}; "
                    f"coverage may be unavailable",
                    file=sys.stderr,
                )

            source_file_abs = _resolve_source_file(cfg, func["source_file"])
            print(
                f"[pipeline] resolved coverage source after attempt {attempt}: "
                f"{func['source_file']} -> {source_file_abs}",
                file=sys.stderr,
            )
            cov = check_function_coverage(
                test_dir,
                source_file_abs,
                func["start_line"],
                func["end_line"],
                source_root=cfg.source_dir.resolve(),
            )

            pct = None
            if cov is not None and isinstance(cov, dict):
                pct = (cov.get("summary", {}) or {}).get("coverage_percent")
                if pct is None:
                    pct = cov.get("coverage_percent")

            if pct is not None and pct >= cfg.coverage_threshold:
                print(
                    f"[pipeline] -> DONE {func['id']} coverage={pct:.1f}% "
                    f">= threshold={cfg.coverage_threshold:.1f}%",
                    file=sys.stderr,
                )
                break

            print(
                f"[pipeline] -> attempt {attempt} coverage={pct}%",
                file=sys.stderr,
            )

        done += 1

    print(f"[pipeline] processed {done}/{len(funcs)} functions", file=sys.stderr)


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


# ==========================================================================
# Driver
# ==========================================================================
def run(cfg: PipelineConfig) -> int:
    paths = derive_paths(cfg)
    for k, p in paths.items():
        if isinstance(p, Path):
            p.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] source : {cfg.source_dir}", file=sys.stderr)
    print(f"[pipeline] test dir: {paths['test_dir']}", file=sys.stderr)

    analysis = run_or_load_analysis(cfg, paths["analysis_path"])

    # ensure_makefile(cfg, paths)
    ensure_test_file(cfg, paths)
    handle_stubs(cfg, paths, analysis)

    if not ensure_skeleton_compiles(cfg, paths):
        print("[pipeline] skeleton failed to compile after all fix attempts",
            file=sys.stderr)
        return 2

    process_functions(cfg, paths, analysis)

    write_json(paths["test_dir"] / "DONE.json", {
        "finished_at": now_iso(),
        "source": str(cfg.source_dir),
    })
    print("[pipeline] finished.", file=sys.stderr)
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv: Optional[list[str]] = None) -> PipelineConfig:
    ap = argparse.ArgumentParser(description="Simplified CUnit test-gen pipeline")
    ap.add_argument("--source", required=True, type=Path,
                    help=".../src/<process_name>")
    ap.add_argument("--agent-js", required=True, type=Path,
                    help="Path to agent.js")
    ap.add_argument("--system-json", type=Path, default=None)
    ap.add_argument("--agent-timeout-sec", type=int, default=1800)
    ap.add_argument("--max-agent-iterations", type=int, default=25)
    ap.add_argument("--max-compile-fix-attempts", type=int, default=5)
    ap.add_argument("--max-test-attempts", type=int, default=4)
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
    ap.add_argument(
        "--func-docs-dir",
        type=Path,
        default=Path("/home/seigyo/rl/moove_docs/func"),
        help="Directory containing function docs as <func_name>.md",
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
        func_docs_dir=ns.func_docs_dir.resolve(),
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


if __name__ == "__main__":
    raise SystemExit(main())
