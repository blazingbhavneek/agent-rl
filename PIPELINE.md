# CUnit Test Generation Pipeline

## What It Does

Automated pipeline that takes a C source process directory and produces a
CUnit test suite with line-range gcov coverage, iterating until each function
reaches a configurable coverage threshold.

```
src/<process>/  →  tests/<process>/Makefile
                   tests/<process>/test_<process>.c
                   tests/<process>/agent_history/
                   tests/<process>/DONE.json
```

---

## Original Intended Flow (workflow.md)

The original design described a sequential, agent-driven process:

1. Run `stub_identifier.py` on the project folder to find `external_undefined_funcs`
   (need stubs) vs `external_defined_funcs` (linked from other files, no stub needed).
2. Create test dir, run `do_mkmf <source_folder>` to generate base Makefile.
3. Port CFLAGS / INCLUDE / LDFLAGS from source Makefile into test Makefile.
4. Generate `test_<process>.c` containing:
   - `__wrap_*` linker stubs for all undefined external functions
   - CUnit test skeleton
5. Append test Makefile target block with WRAP_FLAGS, COVERAGE_FLAGS, gcov target.
6. Compile with `make test`, collect report + gcov coverage.

---

## What Was Wrong (Original Code)

### Ordering / structural bugs
- `ensure_makefile` ran before stubs existed — Makefile written with no WRAP_FUNCS.
- `ensure_test_file` created a premade smoke test (`test_generated_stub_linkage_smoke`)
  and hardcoded `#define main DISABLED_MAIN` unconditionally, even for processes
  with no `main` function.
- `handle_stubs` called `ensure_wrap_flag` for ALL candidates up front before
  integration — if a stub failed to integrate, its `--wrap` flag was still in the
  Makefile, causing a linker error `undefined reference to __wrap_X` that burned
  all `ensure_skeleton_compiles` attempts.
- `ensure_makefile` was commented out in `run()` entirely.

### Wrong symbol name in prompts
- All prompts told the agent to call `DISABLED_MAIN(1, argv)`.
- Actual rename in `ensure_test_file` was `#define main {process_name}_entry_main`.
- Agent wrote tests calling a symbol that didn't exist → compile error or silent no-call.

### Silent failures that cascaded
- Stub generation returned `None` on failure; pipeline continued without those stubs.
- Integration agent was called with `missing_stub_names` including stubs that had no
  generated body — agent confused, wrapped stubs that didn't exist.
- No retry on stub generation or integration — one pass, then move on regardless.
- After stubs failed, `ensure_skeleton_compiles` received a Makefile with orphaned
  `--wrap` flags → permanent linker errors → pipeline aborted.

### Coverage checking
- `gcov -p` flag produced path-encoded filenames like `^#^#src#dio100d#dio100d.c.gcov`.
- `test_dio100d.c.gcov` (test harness coverage) was never deleted, sat alongside
  production gcov files and got picked up by agents as the coverage source.
- Agent was pointed at the wrong gcov file → always saw 0% coverage → looped forever.
- Prompt told agent `matched_gcov_file` only when a file was already found; on first
  run (no prior gcov) the path was omitted entirely — agent had no idea where to look.

### Pre-check wasted every function's coverage data
- `process_functions` ran `make test` as a "pre-check" before processing each function.
- `make test` starts with `clean-test` which deletes all `.gcda/.gcno/.gcov`.
- So every function's pre-check wiped the coverage data that the previous function's
  tests had generated, then rebuilt, then checked coverage (always 0 for current function).
- Result: correct skip logic was impossible; every function appeared uncovered.

### Miscellaneous
- `_project_source_files` defined twice (duplicate).
- Duplicate `import os/re/subprocess/sys/pathlib` block mid-file.
- `DISABLED_MAIN` hardcoded in dead `initial_coverage_rule` block (built but never read).
- `prompt_for_stub` dead function (never called).
- `find_gcov_file` dead function (different approach, never called).
- `analyze_gcov_range` imported but never called from Python.
- `field` imported from dataclasses but unused.
- `python_bin` config field existed but was never wired into agent env.
- `ensure_test_file` on existing files removed `#define main` then didn't re-add it
  if all includes were already present → every second run broke compilation.
- `ensure_skeleton_compiles` returned `False` after last agent without doing a final
  `make test` — the last agent's fix was never verified.
- `DONE.json` was written with only `finished_at` and `source` — no coverage results,
  no indication of success or failure.
- `run()` did not abort if `ensure_skeleton_compiles` returned False (was returning 2
  but later was checked correctly — this varied across versions).

---

## What Was Fixed

| # | Fix | Reason |
|---|-----|--------|
| 1 | Renamed `DISABLED_MAIN` → `{process_name}_entry_main` in all prompts | Agent was calling a symbol that didn't exist |
| 2 | Uncommented `ensure_makefile` in `run()` | Makefile test target was never written |
| 3 | Removed duplicate `_project_source_files` + stale imports | Dead code confusion |
| 4 | Deleted `initial_coverage_rule` dead block (~75 lines) | Never read by prompt function, had wrong symbol |
| 5 | Deleted `prompt_for_stub` dead function | Never called |
| 6 | Deleted `find_gcov_file` dead function | Never called |
| 7 | Removed `analyze_gcov_range` import | Not used from Python; agent.js calls it via `PYTHON_BIN` subprocess |
| 8 | Removed `field` / `python_bin` from dataclass, restored `python_bin` properly | `python_bin` must be wired into `env["PYTHON_BIN"]` for agent.js gcov tool |
| 9 | Removed `bad_define` removal block from `ensure_test_file` | Was stripping `#define main` on every run, never re-adding it — broke compilation on 2nd+ run |
| 10 | Replaced `gcov -p` with `gcov` (no path-encoding) | Files now named `dio100d.c.gcov` not `^#^#src#...` |
| 11 | Added `rm -f test_*.gcov` after gcov in Makefile and in `check_function_coverage` | Test harness gcov files no longer pollute the directory |
| 12 | `expected_gcov_path` computed deterministically as `test_dir/{source_name}.gcov` | Agent always given the exact gcov path, even on first run before any gcov exists |
| 13 | `gcov_tool_text` made mandatory ("MUST call after make test") | Was optional "may use if helpful" — agent often skipped it |
| 14 | Added `_snapshot_dir` / `_restore_from_snapshot` in `run_agent` | Every agent call now restores source files after completion — hard guarantee |
| 15 | Added `folder=repo_root` to all agent calls | Agents confined to test_dir couldn't read production headers |
| 16 | Moved snapshot after dry_run check | Was snapshotting entire src/ on dry-run calls for no reason |
| 17 | `handle_stubs`: wrap flags moved out of stub phase entirely | Flags written by `ensure_makefile` via `sync_wrap_flags` AFTER stubs are integrated |
| 18 | Added `sync_wrap_flags(test_file, makefile)` | Derive WRAP_FUNCS from `__wrap_*` symbols actually in test file — called before every `make test` |
| 19 | Added stub generation retry loop (`max_stub_gen_retries=3`) | Single pass left missing stubs silently |
| 20 | Added stub integration retry loop (`max_stub_integrate_retries=3`) | Single pass left missing stubs silently |
| 21 | Integration agent only given stubs that have bodies | Was told to integrate stubs with no generated body — agent confused |
| 22 | Skip integration batch if `batch_bodies` is empty | Agent called with empty JSON, made spurious changes |
| 23 | Added `ensure_minimal_test_runs()` pipeline stage | Binary must compile + run + exit before coverage tests are written; blocking stubs detected here |
| 24 | Hang detection in `ensure_minimal_test_runs` (90s timeout) | `prompt_for_hang_fix` targets blocking stubs specifically vs generic compile fix |
| 25 | Reordered `run()`: `ensure_test_file` → `handle_stubs` → `ensure_makefile` | Makefile now written after stubs are integrated; WRAP_FUNCS reflect reality |
| 26 | `ensure_test_file` skeleton: bare includes only, no smoke test, no `#define main` | Agent writes the first test; `#define main` only added by compile-fix agent if needed |
| 27 | `ensure_skeleton_compiles` final `make test` after last agent | Last fix attempt was never verified |
| 28 | `process_functions` pre-check: reads existing gcov WITHOUT running `make test` | Running `make test` in pre-check wiped coverage data via `clean-test` on every function |
| 29 | Compile-fix agent added after failed `make test` in attempt loop | Failed builds burned all attempts with no fix |
| 30 | `make_ok` flag passed to `prompt_for_function_test` | Agent told explicitly when build is broken |
| 31 | `process_functions` returns coverage summary dict | `DONE.json` now records per-function coverage, skip counts, totals |
| 32 | `run_make_test` gains `timeout` parameter | Used for 90s hang detection in minimal test phase |
| 33 | `timed_out` key added to all `run_make_test` return dicts | Distinguishes hang from compile error |

---

## Current Pipeline Flow

```
run()
│
├─ run_or_load_analysis()
│   └─ ProjectAnalyzer → analysis.json (cached)
│
├─ ensure_test_file()          [Stage 1]
│   ├─ NEW: bare skeleton — standard headers + bare #include of production sources
│   │       NO #define main, NO premade tests, empty CUnit main
│   └─ EXISTING: add any missing production #includes only
│
├─ handle_stubs()              [Stage 2]
│   ├─ PHASE 1 — gen retry (max_stub_gen_retries rounds):
│   │   parallel (6 workers) per missing stub:
│   │   ├─ _stub_gen/<name>/stub.c valid? → reuse
│   │   └─ else: agent reads repo root, writes __wrap_<name> body
│   └─ PHASE 2 — integration retry (max_stub_integrate_retries rounds):
│       batched: agent reads stub JSON, writes __wrap_* into test file
│       (only stubs with generated bodies are integrated)
│
├─ ensure_makefile()           [Stage 3]
│   ├─ do_mkmf <source_dir>  (base Makefile, skipped if already exists)
│   ├─ append test target block (WRAP_FLAGS, COVERAGE_FLAGS, make targets)
│   └─ sync_wrap_flags(): scan test file for __wrap_* → WRAP_FUNCS += per symbol
│
├─ ensure_skeleton_compiles()  [Stage 4]
│   └─ loop (max_compile_fix_attempts):
│       ├─ sync_wrap_flags()
│       ├─ make test → ok → done
│       └─ fail → compile-fix agent (reads source Makefile for flags/includes)
│       └─ final make test after last agent
│
├─ ensure_minimal_test_runs()  [Stage 5]
│   └─ loop (max_minimal_test_attempts):
│       ├─ agent: write test calling <process>_entry_main(1, argv), fix blocking stubs
│       ├─ sync_wrap_flags()
│       ├─ make test (timeout=90s)
│       ├─ PASS → proceed
│       ├─ HANG (timed_out) → hang-fix agent: find blocking stub, make it return
│       └─ COMPILE/RUNTIME ERROR → build_output_with_runtime_diagnostics()
│           (gdb backtrace + direct binary run + report/log files) → compile-fix agent
│       (soft-fail — warns but doesn't abort, non-main functions may still work)
│
└─ process_functions()         [Stage 6]
    ├─ order: leaf → root (deepest callees first)
    ├─ semantic context: _leaf_to_root_semantic_context.json
    │   accumulates judge verdicts leaf → root so parent tests get child behavior context
    └─ per function:
        ├─ pre-check: read existing gcov WITHOUT rebuilding
        │   ├─ coverage < threshold → go to attempt loop
        │   └─ coverage ≥ threshold → run_semantic_test_judge()
        │       ├─ PASS (score ≥ semantic_judge_min_score) → skip, append context
        │       └─ FAIL → set last_judge, fall through to attempt loop
        ├─ attempt loop (max_test_attempts):
        │   ├─ if last_judge set → prompt_for_semantic_test_repair()
        │   │   (repair weaknesses identified by judge)
        │   └─ else → prompt_for_function_test_with_semantic_context()
        │       (semantic context from accepted lower-level functions passed in)
        │   ├─ run agent
        │   ├─ sync_wrap_flags()
        │   ├─ make test (timeout=300s)
        │   │   └─ FAIL → build_output_with_runtime_diagnostics() → compile-fix agent
        │   │           → sync_wrap_flags → make test again
        │   ├─ check_function_coverage()
        │   │   ├─ gcov -b -c *.gcno  (no -p flag, simple names)
        │   │   ├─ rm test_*.gcov
        │   │   └─ match <source>.gcov by Source: header → parse line range
        │   └─ coverage ≥ threshold AND make ok → run_semantic_test_judge()
        │       ├─ PASS → append context, break attempt loop
        │       └─ FAIL → set last_judge, continue (triggers repair next iteration)
        └─ record final pct and semantic score in results

DONE.json:
  finished_at, source,
  functions_total, functions_done, functions_skipped_at_threshold,
  coverage: { func_id → pct | null },
  semantic_score: { func_id → int | null },
  semantic_context_file: path to _leaf_to_root_semantic_context.json
```

---

## Semantic Judge Layer

Every function must pass both a coverage gate and an LLM semantic quality gate.

**`run_semantic_test_judge`** invokes the agent as a read-only judge:
- Reads target function implementation, headers, structs, globals, callees, callers
- Reads existing CUnit tests and mocks
- Uses accumulated leaf-to-root semantic context
- Writes a JSON verdict to `_semantic_judge_<func_id>.json`

Verdict fields: `passed`, `score` (0-100), `summary`, `function_explanation`,
`tested_behaviors`, `important_state`, `important_dependencies`,
`parent_context_value`, `missing_cases`, `weak_or_meaningless_checks`,
`required_fixes`, `remaining_risks`.

`passed=true` only if `score >= semantic_judge_min_score` (default 75).

**Repair loop**: when judge fails, the next attempt uses `prompt_for_semantic_test_repair`
which explicitly targets the judge's `missing_cases` and `required_fixes`. This continues
until the judge passes or `max_test_attempts` is exhausted.

**Leaf-to-root accumulation**: accepted function verdicts are stored in
`_leaf_to_root_semantic_context.json`. Parent function prompts receive this context
so tests verify orchestration of child behaviors, not just isolated smoke calls.

---

## Runtime Diagnostics

All compile-fix agent calls now receive enriched diagnostics via
`build_output_with_runtime_diagnostics()` / `collect_failure_diagnostics()`:
- Original `make test` stderr/stdout
- `<test>_log.txt` and `<test>_report.txt` content
- Direct binary run (20s timeout) — stdout, stderr, returncode
- `gdb -batch` backtrace (40s timeout) — full `bt full` output
- Annotated note distinguishing compile failure vs runtime crash vs hang

This gives the agent enough information to fix segfaults (Error 139) and other
runtime crashes, not just compile errors.

---

## Key Invariants

- **Source protection**: every agent call snapshots `src/` before and restores after — hard guarantee, not just prompt instructions.
- **Wrap flags**: derived entirely from `__wrap_*` symbols in the test file via `sync_wrap_flags`. No flag for a stub that isn't actually in the file. Called before every `make test`.
- **Stub completeness gate**: stubs retried until all generated AND integrated; pipeline does NOT proceed to compile with orphaned `--wrap` flags.
- **Hang detection**: binary timeout (90s) separate from compile error — different agent prompt for each.
- **No premade assumptions**: initial test file has bare includes. Whether production `main` exists, what rename to use, what stubs are needed — all handled by agents in response to compile errors.
- **Coverage path**: agent always given exact absolute path to production gcov file (`test_dir/<source_basename>.gcov`), even on first run.
- **Dual gate**: coverage threshold alone is not enough to skip a function — the LLM semantic judge must also pass (`score >= semantic_judge_min_score`). Smoke/coverage-only tests are rejected.
- **Leaf-to-root context**: each accepted function's behavioral explanation is persisted and fed to parent-function prompts, so higher-level tests verify orchestration, not just execution.
- **Runtime diagnostics**: every compile-fix call receives gdb backtrace + direct binary output in addition to make output, allowing segfault/crash root-cause analysis.

---

## Config Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | required | Path to `src/<process_name>` |
| `--agent-js` | required | Path to agent.js |
| `--python-bin` | `sys.executable` | Python binary for agent.js gcov tool (`PYTHON_BIN` env) |
| `--agent-timeout-sec` | 1800 | Per-agent subprocess timeout |
| `--max-agent-iterations` | 25 | MAX_ITERATIONS env for agent.js |
| `--max-compile-fix-attempts` | 5 | Retries in ensure_skeleton_compiles |
| `--max-stub-gen-retries` | 3 | Rounds of parallel stub generation |
| `--max-stub-integrate-retries` | 3 | Rounds of stub integration |
| `--max-minimal-test-attempts` | 5 | Attempts in ensure_minimal_test_runs |
| `--max-test-attempts` | 4 | Per-function coverage attempts |
| `--coverage-threshold` | 80.0 | Skip function if coverage ≥ this % |
| `--stub-batch-size` | 8 | Stubs per integration batch |
| `--func-docs-dir` | `/home/seigyo/rl/moove_docs/func` | Markdown docs for stub generation hints |
| `--only-function` | — | Process only this function id |
| `--only-level` | — | Process only functions at this call-depth level |
| `--max-functions` | — | Stop after N functions |
| `--dry-run` | false | Print agent commands without running |
| `--semantic-judge-min-score` | 75 | Minimum LLM semantic score for a function to be considered done |
