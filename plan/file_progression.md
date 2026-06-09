# File/Folder Progression for `main2.py` + `pipeline/`

This is a code-trace record of what the pipeline creates, rewrites, reuses, and expects on disk when started from `main2.py`. Line references point to the current repo state so we can map every artifact back to the code later.

## 1. Environment assumptions that are actually enforced

1. The positional `source_dir` must be the process folder directly under a folder literally named `src`, or `derive_test_dir()` raises immediately (`pipeline/config.py:36-42`).
   - Valid shape:
     - `/home/seigyo/test_env/haiden/dio/dio/src/<process_name>`
   - Derived test root:
     - `/home/seigyo/test_env/haiden/dio/dio/tests/<process_name>`
2. The deeper absolute location of `src` does not matter to path derivation. What matters is only:
   - `source_dir.parent.name == "src"` (`pipeline/config.py:36-42`)
   - `source_dir` exists and contains the production `.c` files that the pipeline will recursively include (`pipeline/common.py:114-127`, `pipeline/stage1_scaffold.py:30-42`)
3. The pipeline expects toolchain/runtime pieces inside the container:
   - `node` + `cfg.agent_js` (`pipeline/common.py:233-313`, `main2.py:46-70`)
   - `do_mkmf` (`pipeline/stage2_makefile.py:82-115`)
   - `make`, `gcc`, `gcov` (`pipeline/common.py:325-383`, `pipeline/stage2_makefile.py:203-218`, `pipeline/stage5_unit_tests.py:239-288`)
   - optionally `gdb` for diagnostics (`pipeline/common.py:575-603`, `pipeline/common.py:657-684`)
4. `system_json` defaults to `/home/seigyo/rl/system_functions.json` and function docs default to `/home/seigyo/rl/moove_docs/func` (`main2.py:49-50`, `pipeline/config.py:13-15`).
5. Important limitation: the master/unit Makefiles reuse flags copied from `source_dir/Makefile` mostly verbatim, and the code itself warns that relative paths in those flags are risky (`pipeline/stage2_makefile.py:157-164`, `pipeline/stage2_makefile.py:177-178`).
   - So "as long as `src` exists and include/libs are alongside it" is not a full guarantee.
   - It works best when the relevant `INCLUDE`, `CPPFLAGS`, `LDFLAGS`, `LDLIBS`, and `LIBS` entries are already valid from the generated test directories, or are absolute.

## 2. Entry arguments and path derivation

### CLI/config path

- `parse_args()` builds `PipelineConfig` from CLI flags (`main2.py:25-71`).
- The pipeline ignores `--output-dir` completely. It is parsed but never stored in `PipelineConfig` or used later (`main2.py:31-32`, `main2.py:58-71`).
- Effective defaults that matter:
  - `--coverage-threshold` default is `70.0` from CLI, even though `PipelineConfig` default is `80.0` (`main2.py:35`, `pipeline/config.py:19-20`)
  - `--max-test-attempts` default `4` (`main2.py:37`)
  - `--max-unit-test-workers` default `4` (`main2.py:40`)
  - `--agent-js` default `/home/seigyo/rl/agent.js` (`main2.py:47`)
  - `--agent-timeout-sec` default `1800` (`main2.py:53`)

### Derived path map

`derive_paths()` produces the base filesystem plan (`pipeline/config.py:45-63`):

- `test_dir = source_dir.parent.parent / "tests" / source_dir.name`
- `process_name = source_dir.name`
- `test_file = test_dir / f"test_{process_name}.c"`
- `makefile = test_dir / "Makefile"`
- `history_dir = test_dir / "agent_history"`
- `analysis_path = test_dir / "analysis.json"`
- `report_file = test_dir / f"test_{process_name}_report.txt"`
- `log_file = test_dir / f"test_{process_name}_logs.txt"`

Important interpretation notes:

1. `report_file` and `log_file` are derived but are not the names actually used by the generated master Makefile (`pipeline/config.py:51-53` vs `pipeline/stage2_makefile.py:184-198`).
2. The master Makefile uses `TEST_PROGRAM = unit_test_<process_name>` and therefore actually writes:
   - `unit_test_<process_name>_report.txt`
   - `unit_test_<process_name>_log.txt`
3. Several diagnostic helpers still look for `test_<process_name>`-based names, so master-suite diagnostics can be partially mismatched (`pipeline/common.py:521-550`, `pipeline/common.py:607-632`).

## 3. Actual runtime order from `main2.run()`

Use this order, not the inconsistent stage comments inside individual modules (`main2.py:74-159`):

1. derive paths
2. resume check via `_pipeline_context.json`
3. Stage 1 scaffold: `ensure_test_file()`
4. Stage 2 Makefile/context: `build_annotated_makefile()`
5. analysis cache/build: `run_or_load_analysis()`
6. Stage 3 stub generation/integration: `handle_stubs()`
7. Stage 4 minimal master test hardening: `ensure_minimal_test_runs()`
8. Stage 5 per-function unit tests: `parallel_generate_unit_tests()`
9. Stage 6 master integration: `integrate_all_unit_tests_sequential()`

Comment drift to remember:

- `build_annotated_makefile()` calls itself "Stage 0" (`pipeline/stage2_makefile.py:48-56`)
- `ensure_minimal_test_runs()` docstring calls itself "Stage 5 infrastructure validation" (`pipeline/stage4_minimal.py:202-208`)
- `parallel_generate_unit_tests()` docstring calls itself "Stage 4" (`pipeline/stage5_unit_tests.py:1395-1399`)

## 4. Resume gate before the normal path

Before any normal stage work, `main2.run()` checks `test_dir/_pipeline_context.json` (`main2.py:87-138`).

### Resume-only integration path

If `_pipeline_context.json` exists and contains `unit_tests_completed: true`:

1. Load cached `flags` and `unit_test_results` from the context file (`main2.py:90-99`).
2. Load or rebuild `analysis.json` (`main2.py:99`, `pipeline/analysis.py:12-31`).
3. Run `make test` in `test_dir` (`main2.py:101-103`, `pipeline/common.py:325-383`).
4. If the current master suite does not pass, run up to 5 pre-integration repair attempts:
   - read diagnostics
   - prompt agent
   - sync wrap flags
   - rerun `make test`
   - history names: `_pre_integration_fix_<timestamp>.json` and companion `.prompt.txt` / `.result.json` in `test_dir/agent_history` (`main2.py:104-136`, `pipeline/common.py:247-305`)
5. Then skip directly to Stage 6 integration (`main2.py:137-138`).

### Important consequence

Once `unit_tests_completed` is set, reruns do not redo scaffolding, Makefile generation, stub generation, minimal test generation, or per-function unit test generation unless the context file is changed or removed.

## 5. Stage 1: master scaffold creation/update

Code: `pipeline/stage1_scaffold.py:16-149`

### Files/folders affected

- creates `test_dir/` if missing (`pipeline/stage1_scaffold.py:26-28`)
- creates or updates `test_dir/test_<process_name>.c`

### First-time file contents

If `test_file` does not exist, the stage writes a full skeleton (`pipeline/stage1_scaffold.py:45-92`) containing:

- section markers from `TEST_FILE_MARKERS` (`pipeline/common.py:103-111`)
- CUnit includes
- stdlib/string/stdarg includes
- an absolute production-source include block:
  - `#define main <process_name>_entry_main`
  - `#include "<absolute path to each discovered .c>"`
  - `#undef main`
- placeholder sections for compatibility definitions, globals, helpers, wrappers, and test cases
- a CUnit `main()` that creates a suite and returns non-zero on failure

### Existing-file repair behavior

If the master test file already exists, the stage may rewrite it to restore:

- the includes marker if missing (`pipeline/stage1_scaffold.py:105-108`)
- missing production `#include` lines, inserted either inside the existing `#define main` block or as a fresh define/include/undef block (`pipeline/stage1_scaffold.py:110-134`)
- any missing section markers appended to the file (`pipeline/stage1_scaffold.py:136-145`)

### Production source discovery rule

This stage includes every `.c` recursively under `cfg.source_dir`, not one guessed file (`pipeline/common.py:114-127`, `pipeline/common.py:166-171`).

## 6. Stage 2: master Makefile generation + context checkpoint

Code: `pipeline/stage2_makefile.py:48-253`

### Files/folders affected

- ensures `test_dir/` exists (`pipeline/stage2_makefile.py:57-77`)
- creates/rewrites `test_dir/Makefile`
- conditionally creates `test_dir/Makefile.bad_pipeline_backup`
- writes `test_dir/_pipeline_context.json`
- may append `WRAP_FUNCS += -Wl,--wrap=<name>` lines to the Makefile via `sync_wrap_flags()` (`pipeline/stage2_makefile.py:250-252`, `pipeline/common.py:178-192`)

### Cache short-circuit

If `_pipeline_context.json` already exists and parses, this stage returns cached flags and does not rebuild the master Makefile (`pipeline/stage2_makefile.py:58-67`).

### Makefile creation path

1. If `Makefile` is missing, run `do_mkmf <test_program>` in `test_dir` (`pipeline/stage2_makefile.py:82-119`).
2. If `Makefile` exists but contains old pipeline markers, back it up to `Makefile.bad_pipeline_backup`, delete it, and rerun `do_mkmf` (`pipeline/stage2_makefile.py:120-140`).
3. Parse selected variables from `source_dir/Makefile`:
   - `CFLAGS`, `CFLAGS_LINUX`, `CPPFLAGS`, `INCLUDE`, `LDFLAGS`, `LDLIBS`, `LIBS` (`pipeline/stage2_makefile.py:20-45`, `pipeline/stage2_makefile.py:157-164`)
4. Append or replace a test block delimited by:
   - `# === TEST TARGET FOR <process_name> ===`
   - `# === END TEST TARGET FOR <process_name> ===`
   (`pipeline/stage2_makefile.py:166-235`)

### Master Makefile behavior after this stage

The generated block defines (`pipeline/stage2_makefile.py:179-220`):

- `TEST_PROGRAM = unit_test_<process_name>`
- `TEST_SRCS = test_<process_name>.c`
- `SOURCE_DIR = <source_dir>`
- `SRC_BUILD_DIR = <source_dir>/linux`
- `SRC_OBJS_ALL = $(wildcard $(SRC_BUILD_DIR)/*.o)`
- `SRC_OBJS = $(filter-out .../<process_name>.o .../main.o,$(SRC_OBJS_ALL))`
- coverage flags and wrap flags
- `test`, `coverage-test`, and `clean-test` targets

### Disk effects of each future `make test` on the master suite

Because the `test` target always depends on `clean-test`, every master run will remove and recreate:

- binary `unit_test_<process_name>`
- `unit_test_<process_name>_report.txt`
- `unit_test_<process_name>_log.txt`
- `*.gcda`, `*.gcno`, `*.gcov`, `*.o`
  (`pipeline/stage2_makefile.py:203-218`)

### `_pipeline_context.json` initial contents

Written fields (`pipeline/stage2_makefile.py:237-248`):

- `process_name`
- `source_dir`
- `source_makefile`
- `actual_source_files`
- `flags`
- `test_dir`
- `test_file`
- `makefile`

### Environment-sensitive caveats

1. Only `source_dir/Makefile` is parsed here, not parent Makefiles (`pipeline/stage2_makefile.py:74-75`, `main2.py:145-146`).
2. Relative include/library flags copied from that Makefile can become wrong from `test_dir` or deeper unit test dirs (`pipeline/stage2_makefile.py:177-178`).
3. The master link step expects usable production objects under `source_dir/linux/*.o` (`pipeline/stage2_makefile.py:174-193`).

## 7. Analysis cache/build

Code: `pipeline/analysis.py:12-31`

### File affected

- `test_dir/analysis.json`

### Behavior

1. If `analysis.json` exists and parses, reuse it (`pipeline/analysis.py:12-18`).
2. Otherwise run `ProjectAnalyzer` over `cfg.source_dir` and write the dumped result to `analysis.json` (`pipeline/analysis.py:20-31`).

This file feeds both stub candidate extraction and function iteration order.

## 8. Stage 3: stub generation, validation, and master-stub integration

Code: `pipeline/stage3_stubs.py:119-602`

### Directory tree introduced

- `test_dir/_stub_gen/<safe_func_name>/`
- `test_dir/agent_history/` entries for stub generation/fixing/integration

### Per-stub artifact progression

For each stub candidate name from `analysis["stub_candidates"]` (`pipeline/analysis.py:62-67`, `pipeline/stage3_stubs.py:531-533`):

1. Create `test_dir/_stub_gen/<safe_name>/` (`pipeline/stage3_stubs.py:136-139`).
2. Target stub file is `stub.c` in that directory (`pipeline/stage3_stubs.py:137-139`).
3. Reuse cached `stub.c` only if:
   - it contains `__wrap_<name>`
   - it does not contain `__real_<name>`
   - `result.json` exists with `validated: true`
   (`pipeline/stage3_stubs.py:157-185`)
4. Otherwise the agent is invoked and history is written under `test_dir/agent_history` with basename `_gen_stub_<safe_name>.json` (`pipeline/stage3_stubs.py:207-294`, `pipeline/common.py:247-305`).

### Stub validation artifacts

`_validate_stub_locally()` writes and/or regenerates inside each stub dir (`pipeline/stage3_stubs.py:319-424`):

- `stub_validate_main.c`
- `stub.o`
- `stub_validate` binary
- `result.json` with `{"validated": true, "func_name": <name>}`

If compile/link/runtime validation fails, `_fix_stub_with_agent()` runs and writes more history entries under `test_dir/agent_history` named `_stub_fix_<safe_name>_<timestamp>.json` (`pipeline/stage3_stubs.py:427-460`).

### Stage-level control flow

1. Load cached validated bodies if present (`pipeline/stage3_stubs.py:535-558`).
2. Repeatedly generate missing bodies in parallel until every candidate has one (`pipeline/stage3_stubs.py:560-590`).
   - concurrency is hard-coded as `min(6, len(to_gen))`
   - `cfg.stub_batch_size` is computed but not used (`pipeline/stage3_stubs.py:528-529`, `pipeline/stage3_stubs.py:576-577`)
3. Integrate validated stub bodies into the master test file one at a time (`pipeline/stage3_stubs.py:592-602`).

### Master-file mutation during stub integration

For each validated stub body (`pipeline/stage3_stubs.py:469-517`):

- insert `/* --- stub: <func_name> --- */` + body after `/* === Linker Wrapper Stubs === */` in `test_<process_name>.c` (`pipeline/stage3_stubs.py:93-107`, `pipeline/stage3_stubs.py:491-493`)
- append `WRAP_FUNCS += -Wl,--wrap=<func_name>` to the master Makefile if needed (`pipeline/stage3_stubs.py:492-493`, `pipeline/common.py:178-185`)
- rerun master `make test`
- if master build/test fails, run generic compile-fix agent against the master Makefile/test file and record history `_stub_integrate_fix_<safe_name>_<timestamp>.json` (`pipeline/stage3_stubs.py:495-516`)

### Important loop limits

- stub generation rounds are unbounded
- stub validation fix loop is unbounded
- stub integration fix loop is unbounded
- `max_stub_gen_retries` and `max_stub_integrate_retries` exist in config but are not used (`pipeline/config.py:28-29`)

## 9. Stage 4: minimal master test hardening

Code: `pipeline/stage4_minimal.py:202-319`

### Files affected

- master `test_<process_name>.c`
- master `Makefile`
- `test_dir/agent_history/` entries:
  - `_minimal_test_<NN>.json`
  - `_hang_fix_<NN>.json`
  - `_minimal_compile_fix_<NN>.json`

### Actual behavior

1. Sync master wrap flags from the current test file into the Makefile (`pipeline/stage4_minimal.py:227-228`).
2. Run `make test` once with a 90-second timeout (`pipeline/stage4_minimal.py:227-228`).
3. Even if that existing run succeeds, the function does **not** return early. The early return is commented out, so it always proceeds into the generation loop (`pipeline/stage4_minimal.py:230-243`).
4. On each loop iteration:
   - run agent with the minimal-startup prompt (`pipeline/stage4_minimal.py:245-261`)
   - sync wrap flags
   - rerun `make test` (`pipeline/stage4_minimal.py:263-264`)
   - if OK, return `True` (`pipeline/stage4_minimal.py:266-272`)
   - if timed out, run hang-fix prompt (`pipeline/stage4_minimal.py:274-292`)
   - else run generic compile-fix prompt (`pipeline/stage4_minimal.py:293-318`)

### Important loop limit note

This loop is unbounded. `cfg.max_minimal_test_attempts` exists but is not used (`pipeline/config.py:30`, `pipeline/stage4_minimal.py:245-319`).

## 10. Stage 5: per-function unit test workspaces

Code: `pipeline/stage5_unit_tests.py:87-1467`

### Directory tree introduced

For each function ID `func_id`, safe name `safe_id`:

- `test_dir/_unit_tests/<safe_id>/`
  - `test_<safe_id>.c`
  - `Makefile`
  - `coverage.json`
  - `judge_verdict.json`
  - optional `unit_test_failed.json`
  - `agent_history/`
    - `<safe_id>_test_<timestamp>.json` + `.prompt.txt` + `.result.json`
    - `<safe_id>_compile_fix_<timestamp>.json` + companions
    - `good_cunit_backups/<safe_id>/...`

### Unit workspace scaffold

`_scaffold_unit_test_dir()` creates the unit directory and `agent_history/`, then creates a placeholder `test_<safe_id>.c` if missing (`pipeline/stage5_unit_tests.py:87-125`).

`_generate_unit_test_makefile()` writes a unit-local Makefile (`pipeline/stage5_unit_tests.py:146-288`) that:

- includes the master Makefile
- sets unit-local `TEST_PROGRAM = test_<safe_id>`
- sets `TEST_SRCS = test_<safe_id>.c`
- stores `PROD_SRC = <absolute production source path>`
- uses `STUB_SRCS = ...`
- keeps only the target production `.gcov`
- provides legacy compatibility aliases for `test_<process_name>.c` and `unit_test_<process_name>`

### How stub files feed unit tests

Before each build, `_sync_stub_srcs()` rewrites the `STUB_SRCS = ...` line in the unit Makefile to exclude any stub whose `__wrap_*` is locally defined in the unit test file (`pipeline/stage5_unit_tests.py:42-80`).

### Fast-path reuse

If a unit test file and Makefile already exist and the test file already contains a real `CU_add_test()`, the stage may reuse it (`pipeline/stage5_unit_tests.py:397-480`):

1. sync stub sources and wrap flags
2. run `make test` in the unit dir
3. compute function coverage from unit-dir `.gcov`
4. if coverage meets threshold:
   - run deterministic semantic judge
   - backup best CUnit state
   - write `coverage.json`
   - write `judge_verdict.json`
   - return success if judge passes

### Important semantic reality

Despite the prompt helpers in `pipeline/semantic.py`, the currently active semantic judge is deterministic and always accepts any test that already reached the coverage threshold (`pipeline/semantic.py:408-448`).

That means `judge_verdict.json` currently contains fields like:

- `passed`
- `score`
- `reason`
- `judge_attempts`

not just the two-field LLM schema described by the unused judge prompt.

### Main per-function attempt loop

When not already complete, `_generate_unit_test_for_func()` runs up to `cfg.max_test_attempts` attempts (`pipeline/stage5_unit_tests.py:773-1350`):

1. generate or repair unit test via agent (`pipeline/stage5_unit_tests.py:788-1069`)
2. if the agent wrote no real `CU_add_test`, reset the unit file to a placeholder line and retry (`pipeline/stage5_unit_tests.py:1071-1082`)
3. sync stub source list and wrap flags (`pipeline/stage5_unit_tests.py:1084-1085`)
4. run `make test` in the unit dir (`pipeline/stage5_unit_tests.py:1087-1088`)
5. compute target-function coverage from unit-dir `.gcov` (`pipeline/stage5_unit_tests.py:1090`)
6. if coverage meets threshold:
   - run deterministic semantic judge (`pipeline/stage5_unit_tests.py:1092-1103`)
   - backup best snapshot (`pipeline/stage5_unit_tests.py:1104-1114`)
   - write `coverage.json` and `judge_verdict.json` (`pipeline/stage5_unit_tests.py:1116-1118`)
   - return success if accepted (`pipeline/stage5_unit_tests.py:1120-1131`)
7. if build failed:
   - gather large diagnostics bundle from logs / report / gdb / direct binary runs (`pipeline/stage5_unit_tests.py:482-554`, `pipeline/common.py:518-704`)
   - run compile-fix agent (`pipeline/stage5_unit_tests.py:1149-1214`)
   - if compile-fix removed `CU_add_test`, reset placeholder and retry (`pipeline/stage5_unit_tests.py:1216-1227`)
8. if build succeeded but coverage stayed below threshold, keep the unit file and retry with coverage feedback (`pipeline/stage5_unit_tests.py:1287-1349`)
9. after max attempts, write `unit_test_failed.json` (`pipeline/stage5_unit_tests.py:1351-1365`)

### Backup tree for passing/high-coverage unit tests

When a unit test reaches threshold and is at least as good as the previous best, `backup_good_cunit_if_best()` creates (`pipeline/semantic.py:471-558`):

- `unit_dir/agent_history/good_cunit_backups/<safe_id>/_best.json`
- `unit_dir/agent_history/good_cunit_backups/<safe_id>/score_<score>_<timestamp>/`
  - copied unit test file
  - copied Makefile
  - `meta.json`
- `unit_dir/agent_history/good_cunit_backups/<safe_id>/latest_best/`
  - copied unit test file
  - copied Makefile
  - `meta.json`

### Stage-level semantic context file

The shared semantic context file is:

- `test_dir/_leaf_to_root_semantic_context.json`

It is loaded before each depth batch and appended after passed results (`pipeline/semantic.py:23-50`, `pipeline/stage5_unit_tests.py:1425-1446`).

### Very important ordering detail

The analyzer helper `functions_leaf_first()` returns deepest functions first (`pipeline/analysis.py:34-59`), but `parallel_generate_unit_tests()` then rebuckets by depth and iterates `for depth in sorted(levels.keys())`, which is ascending (`pipeline/stage5_unit_tests.py:1403-1412`).

So actual per-level execution order in Stage 5 is **root-first, not leaf-first**, despite the comments.

### Final Stage 5 checkpoint update

If every targeted unit test passed, the stage reopens `_pipeline_context.json` and adds (`pipeline/stage5_unit_tests.py:1458-1465`):

- `unit_tests_completed: true`
- `unit_test_results: { ... }`

## 11. Stage 6: master integration of per-function tests

Code: `pipeline/stage6_integrate.py:280-590`

### Inputs this stage consumes

- master `test_<process_name>.c`
- master `Makefile`
- `analysis.json`
- `unit_test_results` from Stage 5 / context
- each passing unit workspace under `_unit_tests/<safe_id>/`

### Pre-integration repair pass

Before integrating any unit test, Stage 6 runs `make test` on the master suite (`pipeline/stage6_integrate.py:307-345`).

If that fails, it runs up to `cfg.max_fix_attempts` repair loops against the master test/Makefile using history entries `_pre_integration_fix_<timestamp>.json`.

### How unit tests are harvested

`extract_test_additions()` pulls only:

- test function bodies whose names appear in `CU_add_test(...)`
- the `CU_add_test(...)` registration lines themselves

from each standalone unit test (`pipeline/stage6_integrate.py:43-78`).

It does **not** automatically merge wrappers/helpers/includes from the unit file. Those are expected to be added later by the cleanup/fix agent if needed.

### Per-function master merge progression

For each passing function in `functions_leaf_first(analysis)` order (`pipeline/stage6_integrate.py:346-352`):

1. skip if result is not passed
2. skip if unit block marker already exists and current master coverage is close enough to the unit coverage (`pipeline/stage6_integrate.py:353-382`)
3. read unit test file and unit Makefile from `result["unit_dir"]` (`pipeline/stage6_integrate.py:357-365`)
4. splice into the master test file:
   - `/* --- unit: <func_id> --- */` + extracted test functions after `/* === Test Cases === */`
   - normalized `CU_add_test(...)` calls before `CU_basic_set_mode`
   (`pipeline/stage6_integrate.py:387-399`)
5. sync wrap flags into the master Makefile (`pipeline/stage6_integrate.py:398-399`)
6. run an initial cleanup agent over the merged master file:
   - history `_unit_integrate_cleanup_<safe_id>_<timestamp>.json`
   (`pipeline/stage6_integrate.py:414-442`)
7. enter a fix loop up to `cfg.max_fix_attempts`:
   - `gcc -fsyntax-only ... test_<process>.c`
   - if syntax fails, run `_unit_integrate_fix_<safe_id>_<timestamp>.json`
   - else run master `make test`
   - compute master function coverage
   - if master build passes and coverage is close enough to standalone coverage, accept
   - otherwise run another `_unit_integrate_fix_<safe_id>_<timestamp>.json`
   (`pipeline/stage6_integrate.py:444-535`)
8. store the last merge context in memory for possible final suite repair (`pipeline/stage6_integrate.py:537-546`)

### Final master-suite repair

After all per-function integrations, Stage 6 runs one final master `make test` (`pipeline/stage6_integrate.py:554-590`).

If it still fails, it runs up to `cfg.max_fix_attempts` final repairs using:

- `_final_master_fix_<timestamp>.json`

against the master test file and Makefile.

## 12. Agent-history file naming conventions

Every `run_agent()` call writes at least (`pipeline/common.py:247-305`):

- `<history_name>.prompt.txt`
- `<history_name>.result.json`

and passes `<history_name>` itself as the `--history` target to `agent.js`, so the agent can also create/update the main history JSON file.

Common history basenames by stage:

- Stage 3 stub gen: `_gen_stub_<safe>.json`
- Stage 3 stub fix: `_stub_fix_<safe>_<timestamp>.json`
- Stage 3 master stub integration fix: `_stub_integrate_fix_<safe>_<timestamp>.json`
- Stage 4 minimal generation: `_minimal_test_<NN>.json`
- Stage 4 hang fix: `_hang_fix_<NN>.json`
- Stage 4 compile fix: `_minimal_compile_fix_<NN>.json`
- Stage 5 unit generation: `<safe>_test_<timestamp>.json`
- Stage 5 unit compile fix: `<safe>_compile_fix_<timestamp>.json`
- Stage 6 pre-integration fix: `_pre_integration_fix_<timestamp>.json`
- Stage 6 cleanup merge pass: `_unit_integrate_cleanup_<safe>_<timestamp>.json`
- Stage 6 iterative merge fix: `_unit_integrate_fix_<safe>_<timestamp>.json`
- Stage 6 final suite fix: `_final_master_fix_<timestamp>.json`

### Source-tree write guard during agent runs

By default `run_agent()` snapshots `cfg.source_dir.parent` before invoking the agent and restores any changed files there afterward, also deleting new files created there (`pipeline/common.py:275-313`).

In this repo layout that means the guarded tree is usually the whole `src/` directory, not `tests/`.

Practical consequence:

- agent edits inside `tests/...` are meant to persist
- agent edits inside `src/...` are supposed to be rolled back after the run
- the pipeline is intentionally trying to keep production-source mutations out of the final filesystem state

## 13. Artifact tree after a typical successful full run

Not every transient file survives every `make test`, but the stable end-state usually looks like:

```text
tests/<process_name>/
  test_<process_name>.c
  Makefile
  _pipeline_context.json
  analysis.json
  _leaf_to_root_semantic_context.json
  agent_history/
    _gen_stub_*.json / .prompt.txt / .result.json
    _stub_integrate_fix_*.json / ...
    _minimal_test_*.json / ...
    _minimal_compile_fix_*.json / ...
    _pre_integration_fix_*.json / ...
    _unit_integrate_cleanup_*.json / ...
    _unit_integrate_fix_*.json / ...
    _final_master_fix_*.json / ...
  _stub_gen/
    <safe_stub_name>/
      stub.c
      stub_validate_main.c
      stub.o
      stub_validate
      result.json
  _unit_tests/
    <safe_func_id>/
      test_<safe_func_id>.c
      Makefile
      coverage.json
      judge_verdict.json
      agent_history/
        <safe>_test_*.json / ...
        <safe>_compile_fix_*.json / ...
        good_cunit_backups/
          <safe_func_id>/
            _best.json
            latest_best/
              test_<safe_func_id>.c
              Makefile
              meta.json
            score_<score>_<timestamp>/
              test_<safe_func_id>.c
              Makefile
              meta.json
```

Possible extra files:

- `Makefile.bad_pipeline_backup`
- `unit_test_failed.json` inside failed unit workspaces
- transient master/unit binaries, `.gcda`, `.gcno`, `.gcov`, `.o`, report/log files

## 14. Code quirks that matter when we use this record later

1. `--output-dir` is dead/ignored (`main2.py:31-32`, `main2.py:58-71`).
2. `main2.py` redundantly parses source Makefile flags once and then immediately overwrites them with `build_annotated_makefile()` output (`main2.py:143-146`).
3. `sync_wrap_flags` is imported from `pipeline.stage6_integrate` even though it originates in `pipeline.common`; this works only because `stage6_integrate.py` imports it into module scope (`main2.py:16-22`, `pipeline/stage6_integrate.py:12-23`).
4. Stage 4 always runs the minimal-test agent loop even if the current master suite already passed once (`pipeline/stage4_minimal.py:230-243`).
5. Stage 4 and Stage 3 fix/generation loops are unbounded despite config fields that suggest limits (`pipeline/config.py:28-30`, `pipeline/stage3_stubs.py:375-424`, `pipeline/stage4_minimal.py:245-319`).
6. Stage 5 unit-test depth ordering is root-first in actual execution, not leaf-first (`pipeline/analysis.py:34-59`, `pipeline/stage5_unit_tests.py:1401-1426`).
7. Master diagnostics use `test_<process>`-based names while the master Makefile builds `unit_test_<process>`, so direct-binary/report detection for master stages can be incomplete or wrong (`pipeline/config.py:48-53`, `pipeline/stage2_makefile.py:184-198`, `pipeline/common.py:521-550`, `pipeline/common.py:607-632`).
8. Master Makefile links `source_dir/linux/*.o` while Stage 1 also directly includes every `.c` under `source_dir`; depending on project layout this can create duplicate-symbol pressure that later stages then repair around (`pipeline/stage1_scaffold.py:30-42`, `pipeline/stage2_makefile.py:187-213`).
9. The resume-only pre-integration repair loop in `main2.run()` is hardcoded to 5 attempts and aborts the run if still failing (`main2.py:104-136`), while normal Stage 6 uses `cfg.max_fix_attempts` and can continue after its own preflight failure (`pipeline/stage6_integrate.py:302-345`).
