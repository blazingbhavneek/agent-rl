from __future__ import annotations

import sys
from pathlib import Path

from .config import PipelineConfig
from .common import (
    _project_source_files,
    build_output_with_runtime_diagnostics,
    prompt_for_compile_fix,
    run_agent,
    run_make_test,
    sync_wrap_flags,
)


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
