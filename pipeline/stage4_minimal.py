from __future__ import annotations

import sys
from pathlib import Path

from .config import PipelineConfig
from .common import (
    _project_source_files,
    _source_files_json_for_prompt,
    build_output_with_runtime_diagnostics,
    prompt_for_compile_fix,
    run_agent,
    run_make_test,
    sync_wrap_flags,
)

def prompt_for_minimal_test(process_name: str, test_file: str, entry_sym: str,
                            actual_source_files: str) -> str:
    return f"""
You are editing an existing CUnit test file.

File to modify:
{test_file}

Process:
{process_name}

Entry function:
{entry_sym}

Actual production source files:
{actual_source_files}

Goal:
Make a minimal startup test that compiles, links, runs, exits quickly, and produces
non-zero coverage for the real production source file under src/.

Do not over-engineer this stage.
Do not deeply rewrite the test framework.
Do not create many scenario tests here.
Do not run real blocking loops, hardware calls, IPC, timers, daemon loops, or real sleeps.

CRITICAL:
A successful `make test` exit code is NOT enough.

You must personally verify the generated CUnit/gcov report before giving up.

After every edit, run:

  make test

Then read the generated report file in this directory.

The report file will usually have a name similar to one of these:

  {process_name}_report.txt
  test_{process_name}_report.txt
  unit_test_{process_name}_report.txt
  *_report.txt

Use commands such as:

  ls -lt *_report.txt
  cat *_report.txt

You must inspect the report.

The test is INVALID if the report shows zero CUnit tests, for example:

  tests      0      0
  asserts    0      0

The test is also INVALID if the production source file under src/ still has zero
coverage, for example:

  File '/path/to/src/{process_name}/{process_name}.c'
  Lines executed:0.00%

If either of those happens:
- do end the conversation, keep on going
- fix the test
- run `make test` again
- read the report again
- repeat until the report is valid

The report is valid only when:
1. CUnit shows at least one test registered and ran.
   The tests row must have Total > 0 and Ran > 0.

2. The production source file under src/ has non-zero line coverage.
   It must NOT say Lines executed:0.00% for the source file this test includes.

3. The test actually calls `{entry_sym}` or a real startup path that reaches the
   included production source.

CRITICAL — production source inclusion:
The test file already #includes the production source directly using this pattern:

  #define main {entry_sym}
  #include "/absolute/path/to/production.c"
  #undef main

Do NOT add production .c files to the Makefile as separate compiled objects.
Do NOT remove or modify this #define/#include/#undef block.
Adding production source as a separate object while it is also #included causes
duplicate symbol linker errors.

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
7. Register the test with CU_add_test().
8. Ensure CU_basic_run_tests() is called.
9. Ensure CUnit main returns non-zero on test failure using CU_get_number_of_failures().
10. Run `make test`.
11. Read the generated *_report.txt file.
12. If tests ran = 0 or production src coverage = 0.00%, fix and retry.

Expected CUnit main structure:

  int main(void) {{
      if (CU_initialize_registry() != CUE_SUCCESS) {{
          return 1;
      }}

      CU_pSuite suite = CU_add_suite("minimal_startup", NULL, NULL);
      if (suite == NULL) {{
          CU_cleanup_registry();
          return 1;
      }}

      if (CU_add_test(suite, "startup_runs", test_startup_runs) == NULL) {{
          CU_cleanup_registry();
          return 1;
      }}

      CU_basic_set_mode(CU_BRM_VERBOSE);
      CU_basic_run_tests();

      unsigned failures = CU_get_number_of_failures();
      CU_cleanup_registry();

      return failures == 0 ? 0 : 1;
  }}

Important:
- Do not submit only because compilation succeeded.
- Do not submit only because `make test` returned 0.
- Do not submit if the report says `tests 0 0`.
- Do not submit if the production source under src/ says `Lines executed:0.00%`.
- Keep editing/running/checking until the report proves real source execution.

When the report proves at least one CUnit test ran and the production source has non-zero coverage, only then end conversation, untill then keep trying. 
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

CRITICAL — production source inclusion:
The test file already #includes production source directly. Do NOT add production .c
files to the Makefile as separate compiled objects — that causes duplicate symbol errors.
Do NOT remove or modify the #define main / #include / #undef main block in the test file.

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
5. Run `make test` after fixing. Keep fixing until its successful

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

    # Strategy:
    # 1. Try the current test as-is first — avoid touching it if it already works.
    # 2. If it fails: ask the agent to add minimal startup stubs/tests.
    # 3. If the binary hangs (timeout): specifically ask for blocking-stub fixes.
    # 4. If compile/link/runtime error: run generic compile-fix agent.
    # Loop until the binary compiles, runs, terminates, and passes.
    print(
        f"[pipeline] minimal test phase: checking existing test binary first",
        file=sys.stderr,
    )

    sync_wrap_flags(test_file, makefile)
    existing_res = run_make_test(cfg, test_dir, timeout=90)

    if existing_res["ok"]:
        print(
            f"[pipeline] existing make test returned OK, but LLM must still inspect "
            f"*_report.txt and verify non-zero tests plus non-zero src coverage",
            file=sys.stderr,
        )

        # # TODO: Fix this later
        # return True

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
            prompt_for_minimal_test(
                process_name, str(test_file), entry_sym,
                actual_source_files=_source_files_json_for_prompt(cfg),
            ),
            f"_minimal_test_{attempt:02d}.json",
            folder=repo_root,
        )

        sync_wrap_flags(test_file, makefile)
        res = run_make_test(cfg, test_dir, timeout=90)

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
                cfg,
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
