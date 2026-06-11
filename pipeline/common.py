from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .config import PipelineConfig
from .execution import assert_no_forbidden_host_paths, containerize_text, run_command, run_command_to_files


# region IO helpers

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

    # Strategy 1: clean JSON — the fast path.
    try:
        return json.loads(raw)
    except Exception:
        pass

    raw2 = raw.strip()
    # Strategy 2: strip markdown code fences the agent may have added, then retry.
    raw2 = re.sub(r"^\s*```(?:json)?", "", raw2, flags=re.I).strip()
    raw2 = re.sub(r"```\s*$", "", raw2).strip()

    try:
        return json.loads(raw2)
    except Exception:
        pass

    # Strategy 3: extract the first {...} block from surrounding prose.
    start = raw2.find("{")
    end = raw2.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw2[start:end + 1])
        except Exception:
            return {}

    return {}

# endregion IO helpers


# region Source file helpers

TEST_FILE_MARKERS = [
    "/* === Includes === */",
    "/* === Compatibility Definitions === */",
    "/* === Test Globals === */",
    "/* === Test Helpers === */",
    "/* === Linker Wrapper Stubs === */",
    "/* === Test Cases === */",
    "/* === Test Registration === */",
]

# Globally search all .c files in a folder
def _project_source_files(cfg: PipelineConfig) -> list[Path]:
    """
    Return actual absolute .c files under cfg.source_dir.
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
    # Already absolute — resolve symlinks and return.
    if src.is_absolute():
        return src.resolve()
    # Relative — try joining directly under source_dir first.
    direct = (source_dir / src).resolve()
    if direct.exists():
        return direct
    # Fall back to recursive filename search under source_dir.
    matches = sorted(
        p.resolve()
        for p in source_dir.rglob(src.name)
        if p.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    # Multiple hits: prefer the candidate whose path ends with the requested suffix
    # (handles subdirectory disambiguation like "subdir/foo.c").
    if len(matches) > 1:
        wanted_suffix = src.as_posix()
        for m in matches:
            if m.as_posix().endswith(wanted_suffix):
                return m
        return matches[0]
    return direct


def _source_files_json_for_prompt(cfg: PipelineConfig) -> str:
    files = _project_source_files(cfg)
    if not files:
        return " (no .c files found under cfg.source_dir)"
    return "\n".join(f" - {p}" for p in files)

# Give location of C files, gives and include containing the abosulute location of the file
def _source_includes_for_test_file(cfg: PipelineConfig, test_file: Path) -> list[str]:
    """Build production #include lines using absolute paths — no depth guessing."""
    lines: list[str] = []
    for src in _project_source_files(cfg):
        lines.append(f'#include "{src.resolve()}"')
    return lines

# endregion Source file helpers


# region Makefile wrap flag helpers

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

# endregion Makefile wrap flag helpers


# region Agent runner

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
    prompt_file = history_path.with_suffix(".prompt.txt")
    assert_no_forbidden_host_paths(
        cfg,
        "\n".join(str(p) for p in [work_dir, agent_folder, history_path, prompt_file]),
        f"agent paths {history_name}",
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Rewrite per-episode host paths to canonical so the agent only sees the
    # pristine container layout (no-op outside docker/path_map mode).
    prompt = containerize_text(cfg, prompt)
    assert_no_forbidden_host_paths(cfg, prompt, f"prompt {prompt_file}")
    write_text(prompt_file, prompt)

    # TODO: The raw HTTP trace didnt work recently, check this out
    cmd = [
        "node", str(cfg.agent_js),
        "--folder", str(agent_folder),
        "--prompt-file", str(prompt_file),
        "--history", str(history_path),
        "--capture-raw-http-trace",
    ]
    assert_no_forbidden_host_paths(
        cfg,
        "\n".join(str(part) for part in cmd),
        f"agent command {history_name}",
    )

    env = {
        "MAX_ITERATIONS": str(
            max_iterations if max_iterations is not None else cfg.max_agent_iterations
        ),
        "PYTHON_BIN": cfg.python_bin,
    }

    actual_timeout = timeout_sec if timeout_sec is not None else cfg.agent_timeout_sec

    if cfg.dry_run:
        print(f"[pipeline][dry-run] would run: {' '.join(cmd)}", file=sys.stderr)
        return {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}

    # Snapshot the source tree so we can restore any files the agent accidentally
    # modifies. Agents should only write inside test_dir; this is a safety net.
    use_snapshot = protect_source and getattr(cfg, "execution_mode", "local") != "docker"
    _protect_dir = cfg.source_dir.parent.resolve() if use_snapshot else None
    _snap = _snapshot_dir(_protect_dir) if _protect_dir is not None else {}

    print(f"[pipeline] agent -> {history_name}", file=sys.stderr)
    t0 = time.time()
    try:
        proc = run_command(
            cfg,
            cmd,
            cwd=work_dir,
            env=env,
            timeout=actual_timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        proc = None

    elapsed = time.time() - t0
    res = {
        "exit_code": proc.returncode if proc else -1,
        "stdout": (proc.stdout if proc else ""),
        "stderr": (proc.stderr if proc else ""),
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

# endregion Agent runner


# region Build & coverage

import re
import subprocess
from pathlib import Path

def safe_decode(data: bytes) -> str:
    if isinstance(data, str):
        return data

    for encoding in ("utf-8", "euc_jp", "cp932", "shift_jis", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")

# TODO: Take care of it when we need to pursue branching and move this pipelne code to the host
def run_make_test(cfg: PipelineConfig, test_dir: Path, timeout: int = 300) -> dict:
    try:
        # Use run_command instead of subprocess.run to support Docker mode
        proc = run_command_to_files(
            cfg,
            ["make", "test"],
            cwd=test_dir,
            timeout=timeout,
            stderr_path=test_dir / "stderr.txt",
            stdout_path=test_dir / "stdout.txt"
        )
    except subprocess.TimeoutExpired as e:
        stdout = safe_decode(e.stdout or b"")
        stderr = safe_decode(e.stderr or b"")

        return {
            "ok": False,
            "returncode": -1,
            "stdout": stdout,
            "stderr": stderr + f"\nmake test timed out after {timeout}s",
            "timed_out": True,
            "errors": [f"make test timed out after {timeout}s"],
        }

    # Normalize the result to match the expected dictionary format
    # run_command already returns decoded strings for stdout/stderr
    stdout = proc.stdout
    stderr = proc.stderr
    combined = stdout + "\n" + stderr

    blocking_error_patterns = [
        # GNU make hard failures
        r"make(?:\[\d+\])?: \*\*\* .* Error \d+",
        r"make(?:\[\d+\])?: .* Error \d+ \(ignored\)",

        # compiler/linker hard failures
        r"\berror:",
        r"undefined reference",
        r"collect2:\s+error",
        r"ld returned \d+ exit status",
        r"cannot open output file",
        r"cannot find -l",
        r"No such file or directory",
        r"No rule to make target",
        r"recipe for target .* failed",
        r"\bStop\.",
    ]

    errors = []
    for line in combined.splitlines():
        s = line.strip()
        if any(re.search(p, s, re.IGNORECASE) for p in blocking_error_patterns):
            errors.append(s)

    ok = proc.returncode == 0 and not errors

    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "errors": errors,
    }

# TODO: Take care of paths when dockerized
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
    Matches .gcov by Source: header.
    """
    test_dir = Path(test_dir).resolve()
    source_abs = Path(source_file).resolve()
    if source_root is not None:
        source_root = Path(source_root).resolve()

    # Step 1: find the .gcov file that corresponds to the production source.
    # gcov embeds the original source path in the header as "Source: <path>".
    # We read that header, skip test harness gcov files, and match by resolved path.
    gcov_files = sorted(test_dir.glob("*.gcov"))

    # Parse the source location of this gcov file
    def _gcov_source(gcov_file: Path) -> Optional[Path]:
        """Read the 'Source:' header line from a .gcov file and return the resolved path."""
        try:
            lines = read_text(gcov_file).splitlines()
        except Exception:
            return None
        for line in lines:
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
        # Skip gcov files whose source lives inside test_dir (the test harness itself).
        try:
            src.relative_to(test_dir)
            continue
        except ValueError:
            pass
        # Skip explicitly named test files.
        if src.name.startswith("test_"):
            continue
        # If a source root is given, skip files outside it (other libraries etc.).
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

    # Step 2: count covered vs executable lines inside the requested line range.
    # gcov format: "<count>: <lineno>: <source>".
    # count="-"     means the line is not executable (comments, blank lines).
    # count="#####" or "=====" means the line was never executed.
    # count is a positive integer or ends with "*" (partial coverage).
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
        if count_text == "-":          # non-executable line
            continue
        executable_lines += 1
        if count_text in ("#####", "====="):  # never executed
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

# endregion Build & coverage


# region Diagnostics

def collect_failure_diagnostics(
    cfg: PipelineConfig,
    test_dir: Path,
    test_file: Path,
    res: dict,
) -> str:
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
        direct = run_command(
            cfg,
            [str(test_bin)],
            cwd=test_dir,
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
        gdb = run_command(
            cfg,
            [
                "gdb", "-q", "-batch",
                "-ex", "set pagination off",
                "-ex", "run",
                "-ex", "bt",
                "-ex", "bt full",
                "--args", str(test_bin),
            ],
            cwd=test_dir,
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

def collect_runtime_crash_diagnostics(
    cfg: PipelineConfig,
    test_dir: Path,
    test_binary_name: str,
) -> str:
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
        direct = run_command(
            cfg,
            [str(test_bin)],
            cwd=test_dir,
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
        gdb = run_command(
            cfg,
            [
                "gdb", "-q", "-batch",
                "-ex", "set pagination off",
                "-ex", "run",
                "-ex", "bt",
                "-ex", "bt full",
                "--args", str(test_bin),
            ],
            cwd=test_dir,
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


def build_output_with_runtime_diagnostics(
    cfg: PipelineConfig,
    test_dir: Path,
    test_file: Path,
    res: dict,
) -> str:
    base_output = collect_failure_diagnostics(cfg, test_dir, test_file, res)
    test_binary_name = Path(test_file).stem
    diagnostics = collect_runtime_crash_diagnostics(cfg, test_dir, test_binary_name)

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

# endregion Diagnostics


# region Shared prompt

# TODO: This assumes test code compilation only? Are the checks too limited and bounded? any way to make this fixing easier
# Cline's patch tool often seems to be making mistakes with indentation and brackets, and they kind of accumulate v easily, we need to have it fix line ranges instead
# of trying to patch small stuff and creating more problems, optionally add a C checker or something, which gives exact syntax error line range which it can rewrite instead
# of micro patches that fail
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

CRITICAL — production source inclusion:
Production .c files are #included directly in the test file using this pattern:

  #define main <process>_entry_main
  #include "/absolute/path/to/production.c"
  #undef main

Do NOT add production .c files to the Makefile as separate compiled objects.
Do NOT remove or modify the #define/#include/#undef block in the test file.
Adding a production .c as a separate object while it is also #included causes
duplicate symbol linker errors. If symbols are unresolved, fix with linker
wrapper stubs (--wrap) or by fixing include paths — not by re-compiling the source.

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
- Some LIBS/INCLUDE that are included might not be present in the environment, based on error try to remove them and try so you know whats wrong.

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
- If the failure is a missing header, define, library, or include-path issue, fix the unit Makefile first instead of only changing the test C file.
- Do not call `__real_*` from wrappers.
- Do not hide crashes by deleting assertions or replacing them with always-pass assertions.
- Do not edit production code to make the test pass.


When modifying C code, avoid fragile inline edits whenever possible.

Do NOT try to surgically insert, delete, or replace a few characters or individual lines inside existing code. Small patches frequently introduce mismatched braces, broken indentation, malformed conditionals, and partial edits that leave the surrounding code inconsistent.

Instead:

* Identify the smallest logical unit that contains the problem.
* Rewrite that entire unit as a coherent block.
* Prefer replacing:

  * a complete statement block,
  * an entire if/else block,
  * a loop body,
  * a helper function,
  * a complete function,
  * a struct definition,
  * a header section,
    rather than making tiny in-place edits.

Guidelines:

* Treat code as blocks, not lines.
* Rewrite 5–20 lines cleanly if needed.
* Ensure braces, parentheses, and control flow are fully balanced within the rewritten block.
* Return the complete replacement block, not a diff of individual lines.
* Minimize the number of edit regions; prefer one clean block replacement over many scattered edits.

After every modification, run:

gcc -fsyntax-only <filename>

If a syntax error is reported:

1. Locate the logical block containing the error.
2. Rewrite the entire affected block.
3. Do not stack additional micro-patches on top of previous edits.
4. Repeat until `gcc -fsyntax-only` succeeds with no syntax errors.

A clean block rewrite is preferred over a minimal patch if it improves structural correctness and reliability.

When editing source code, output real source code, not an escaped representation of source code.

Do NOT introduce escape characters unless they are required by the target language syntax.

Examples:

Correct:
char *argv[] = {{ "dio110d", NULL }};

Incorrect:
char *argv[] = {{ \\"dio110d\\", NULL }};

But remember, CPP style comments are not allowd, so for comments use this sytax instead: \\* Comment body *\\

Only use escaped quotes (") when they are inside a string literal that itself contains quotation marks.

Treat the file as plain source code, not as JSON, Markdown, Python strings, shell strings, or serialized text.

Before finalizing edits, scan for suspicious escape sequences that commonly appear when code has been copied through another representation layer:

* "
* '
* \n outside string literals
* \t outside string literals

If such escapes appear in normal C code, remove them unless they are intentionally part of a string literal.


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

After you are done editing, make sure to verify by running:
make test

Keep fixing until its successfull

"""

# endregion Shared prompt
