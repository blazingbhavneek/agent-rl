# Data Collection Stage Separation

This note explains two things:

1. What the current `main2.py` plus `pipeline/` flow does today, using the artifact progression captured in `file_progression.md`.
2. How to reshape it for stage-wise trace/data generation with the smallest possible code changes.

The main design constraint is safety. The current code barely has enough stable behavior to be useful, so the first implementation should avoid rewriting prompts, verifier logic, Makefiles, or integration behavior. The new behavior should be added as a thin data-collection mode around existing stage functions.

## Current Mental Model

Today the pipeline is end-to-end first.

Given:

```text
.../src/<process_name>
```

`derive_paths()` creates:

```text
.../tests/<process_name>/
```

Then `main2.run()` moves through the full pipeline:

```text
derive paths
resume/integration check
Stage 1: ensure master test scaffold
Stage 2: build master Makefile and _pipeline_context.json
analysis: build or load analysis.json
Stage 3: generate, validate, and integrate stubs
Stage 4: make the master test minimally compile/run
Stage 5: generate standalone per-function unit tests
Stage 6: integrate passing unit tests into the master test file
```

The key point: several stages already produce per-function artifacts, but they are treated as stepping stones toward one final master CUnit suite.

## Current Folder Progression

### Base Test Root

After path derivation and early setup:

```text
tests/<process_name>/
  test_<process_name>.c
  Makefile
  _pipeline_context.json
  analysis.json
  agent_history/
```

Important current behavior:

- `test_<process_name>.c` is the master integrated test file.
- `Makefile` is the master Makefile.
- `_pipeline_context.json` stores flags and later stores `unit_tests_completed` plus `unit_test_results`.
- `analysis.json` feeds both stub candidate extraction and function iteration.
- `agent_history/` stores master-level agent calls.

### Stage 1: Scaffold

Current stage function:

```text
pipeline/stage1_scaffold.py
  ensure_test_file(cfg, paths)
```

Current artifacts:

```text
tests/<process_name>/
  test_<process_name>.c
```

This stage creates or repairs the master test skeleton. It inserts section markers, CUnit includes, and absolute `#include` lines for production `.c` files.

For data generation, this is mostly preparation, not a rich RL/data stage.

### Stage 2: Makefile And Context

Current stage function:

```text
pipeline/stage2_makefile.py
  build_annotated_makefile(cfg, paths)
```

Current artifacts:

```text
tests/<process_name>/
  Makefile
  _pipeline_context.json
```

This stage creates the master Makefile and caches source Makefile flags into `_pipeline_context.json`.

For data generation, this is also preparation. Downstream episodes need this context because stub validation and unit Makefiles rely on the cached flags.

### Analysis

Current function:

```text
pipeline/analysis.py
  run_or_load_analysis(cfg, paths["analysis_path"])
```

Current artifact:

```text
tests/<process_name>/
  analysis.json
```

This is the item inventory:

- `analysis["stub_candidates"]` becomes the stub worklist.
- `analysis["functions"]` becomes the unit-test worklist.

For BFS data generation, this file defines the stage frontier.

### Stage 3: Stub Generation And Integration

Current stage function:

```text
pipeline/stage3_stubs.py
  handle_stubs(cfg, paths, analysis)
```

Current stable artifacts:

```text
tests/<process_name>/
  _stub_gen/
    <stub_func_safe_name>/
      stub.c
      stub_validate_main.c
      stub.o
      stub_validate
      result.json
  agent_history/
    _gen_stub_<safe>.json
    _gen_stub_<safe>.prompt.txt
    _gen_stub_<safe>.result.json
    _stub_fix_<safe>_<timestamp>.json
    _stub_integrate_fix_<safe>_<timestamp>.json
```

Current behavior:

1. `generate_stub_code()` creates or reuses one `_stub_gen/<safe>/stub.c`.
2. `_validate_stub_locally()` loops compile/link/runtime validation until the stub passes.
3. `handle_stubs()` gathers one validated body per candidate.
4. `integrate_all_stubs_sequential()` inserts those bodies into the master test file and fixes master build failures.

Why this is not enough for trace collection:

- `_stub_gen/<safe>/stub.c` is a single cached survivor.
- Repeated calls skip already validated stubs.
- The master file is mutated during integration.
- Agent histories exist, but the generated artifact is not stored as an independent episode.

### Stage 4: Minimal Master Test

Current stage function:

```text
pipeline/stage4_minimal.py
  ensure_minimal_test_runs(cfg, paths)
```

Current artifacts:

```text
tests/<process_name>/
  test_<process_name>.c
  Makefile
  agent_history/
    _minimal_test_<NN>.json
    _hang_fix_<NN>.json
    _minimal_compile_fix_<NN>.json
```

Current behavior:

- Runs `make test` on the master suite.
- Asks the agent to add minimal startup tests or repair build/runtime failures.
- Mutates the master test file and Makefile until the infrastructure works.

For data collection, this is a useful environment-bootstrap stage, but it should usually produce a selected master baseline before unit-test episodes begin.

### Stage 5: Per-Function Unit Tests

Current stage function:

```text
pipeline/stage5_unit_tests.py
  parallel_generate_unit_tests(cfg, paths, analysis, flags)
```

Current per-function artifacts:

```text
tests/<process_name>/
  _unit_tests/
    <func_safe_id>/
      test_<func_safe_id>.c
      Makefile
      coverage.json
      judge_verdict.json
      unit_test_failed.json
      agent_history/
        <safe>_test_<timestamp>.json
        <safe>_test_<timestamp>.prompt.txt
        <safe>_test_<timestamp>.result.json
        <safe>_compile_fix_<timestamp>.json
        <safe>_compile_fix_<timestamp>.prompt.txt
        <safe>_compile_fix_<timestamp>.result.json
        good_cunit_backups/
          <safe>/
            _best.json
            latest_best/
              test_<safe>.c
              Makefile
              meta.json
            score_<score>_<timestamp>/
              test_<safe>.c
              Makefile
              meta.json
```

Current behavior:

1. One active workspace is created at `_unit_tests/<safe>/`.
2. The agent writes `test_<safe>.c`.
3. The stage runs `make test`.
4. Coverage is checked for the target function.
5. The deterministic judge accepts tests that reached coverage threshold.
6. Compile-fix and semantic repair can run inside the same workspace.
7. Passing/high-coverage tests are backed up under `good_cunit_backups/`.

Why this is close but still not a dataset stage:

- Each function has one active mutable workspace.
- A failed attempt can be overwritten by the next attempt.
- Internal attempts are a repair trajectory, not separate clean episode directories.
- Existing passing workspaces are skipped by cache checks.
- The stage ultimately feeds Stage 6 integration, not a stage-wise corpus.

### Stage 6: Master Integration

Current stage function:

```text
pipeline/stage6_integrate.py
  integrate_all_unit_tests_sequential(cfg, paths, analysis, unit_results, flags)
```

Current artifacts:

```text
tests/<process_name>/
  test_<process_name>.c
  Makefile
  agent_history/
    _pre_integration_fix_<timestamp>.json
    _unit_integrate_cleanup_<safe>_<timestamp>.json
    _unit_integrate_fix_<safe>_<timestamp>.json
    _final_master_fix_<timestamp>.json
```

Current behavior:

- Reads passing unit workspaces.
- Extracts test functions and `CU_add_test(...)` registrations.
- Merges them into the master file.
- Runs cleanup/fix agents until the master build and coverage are acceptable.

For the new data-generation goal, this becomes optional. It is useful later for final validation, but it should not be required just to collect stub or unit-test traces.

## New Data-Collection Vision

The new pipeline should be stage-wise and repeatable.

Old priority:

```text
one source -> full end-to-end pipeline -> one final master test suite
```

New priority:

```text
many sources/functions -> run one chosen stage many times -> save all traces -> select a frontier -> move to next stage
```

This means the product is no longer only the final CUnit file. The product is a corpus of agent episodes:

```text
prompt/context + agent history/trace + produced files + verifier output + metadata
```

The selected artifacts are just the moving frontier that lets later stages run.

## New File Structure

Keep the current folders because existing code expects them.

Add a dataset archive beside them:

```text
tests/<process_name>/
  test_<process_name>.c
  Makefile
  _pipeline_context.json
  analysis.json

  _stub_gen/
    <safe_stub>/
      stub.c
      result.json

  _unit_tests/
    <safe_func>/
      test_<safe_func>.c
      Makefile
      coverage.json
      judge_verdict.json

  _trace_dataset/
    manifest.json
    frontiers/
      stubs/
        <safe_stub>/
          selected.json
          stub.c
          result.json
      minimal_master/
        selected.json
        test_<process_name>.c
        Makefile
      unit_tests/
        <safe_func>/
          selected.json
          test_<safe_func>.c
          Makefile
          coverage.json
          judge_verdict.json

    episodes/
      stubs/
        <safe_stub>/
          <episode_id>/
            metadata.json
            prompt.txt
            history.json
            result.json
            files/
              stub.c
              stub_validate_main.c
            verifier/
              validation.json

      minimal_master/
        <episode_id>/
          metadata.json
          prompt.txt
          history.json
          result.json
          files/
            test_<process_name>.c
            Makefile
          verifier/
            make_result.json

      unit_tests/
        <safe_func>/
          <episode_id>/
            metadata.json
            agent_history/
              <safe>_test_*.json
              <safe>_compile_fix_*.json
              *.prompt.txt
              *.result.json
            files/
              test_<safe_func>.c
              Makefile
            verifier/
              make_result.json
              coverage.json
              judge_verdict.json
              unit_test_failed.json
```

Important compatibility rule:

```text
_trace_dataset/...
  stores all attempts

_stub_gen/... and _unit_tests/...
  store only selected/current artifacts for existing downstream code
```

That keeps the current pipeline mostly intact.

## New Stages

### `prepare`

Purpose:

```text
Create the stable test root and item inventory.
```

Runs:

```text
ensure_test_file()
build_annotated_makefile()
run_or_load_analysis()
```

Produces:

```text
test_<process_name>.c
Makefile
_pipeline_context.json
analysis.json
```

This stage should be safe and idempotent.

### `collect-stubs`

Purpose:

```text
Run N independent stub episodes for each selected stub candidate.
```

Needs:

```text
analysis.json
_pipeline_context.json
```

Produces:

```text
_trace_dataset/episodes/stubs/<safe_stub>/<episode_id>/
```

Does not need to mutate:

```text
test_<process_name>.c
_stub_gen/<safe_stub>/
```

Selection is separate.

### `select-stubs`

Purpose:

```text
Pick the stub artifact that downstream code should use.
```

For now, since scoring is not ready, selection can be simple:

- Prefer `validated == true`.
- If multiple validated stubs exist, pick the first passing one or latest passing one.
- Store the decision in `selected.json`.

Produces:

```text
_trace_dataset/frontiers/stubs/<safe_stub>/
  selected.json
  stub.c
  result.json
```

### `materialize-stubs`

Purpose:

```text
Copy selected stubs into the legacy active location.
```

Produces:

```text
_stub_gen/<safe_stub>/
  stub.c
  result.json
```

This is what keeps Stage 5 working without teaching it a new stub lookup system.

### `integrate-stubs`

Purpose:

```text
Use selected stubs to make the master baseline work.
```

Runs only the current integration half:

```text
integrate_all_stubs_sequential()
```

This stage mutates:

```text
test_<process_name>.c
Makefile
```

It should remain separate from `collect-stubs` because data collection should not require master mutation.

### `minimal-master`

Purpose:

```text
Produce a selected master baseline that unit-test episodes can reference.
```

Runs:

```text
ensure_minimal_test_runs()
```

Produces current artifacts:

```text
test_<process_name>.c
Makefile
agent_history/_minimal_*.json
```

Also snapshots selected baseline into:

```text
_trace_dataset/frontiers/minimal_master/
  selected.json
  test_<process_name>.c
  Makefile
```

### `collect-unit-tests`

Purpose:

```text
Run N independent unit-test episodes for each selected function.
```

Needs:

```text
analysis.json
selected/materialized stubs
minimal master baseline
```

Produces:

```text
_trace_dataset/episodes/unit_tests/<safe_func>/<episode_id>/
```

Should not overwrite:

```text
_unit_tests/<safe_func>/
```

Selection is separate.

### `select-unit-tests`

Purpose:

```text
Pick the unit test artifact that downstream code should use.
```

For now:

- Prefer `passed == true`.
- Then prefer higher `coverage_pct`.
- Then prefer higher `semantic_score`.
- If score is not meaningful yet, keep the metadata and pick the first passing artifact.

Produces:

```text
_trace_dataset/frontiers/unit_tests/<safe_func>/
  selected.json
  test_<safe_func>.c
  Makefile
  coverage.json
  judge_verdict.json
```

### `materialize-unit-tests`

Purpose:

```text
Copy selected unit tests into the legacy active location.
```

Produces:

```text
_unit_tests/<safe_func>/
  test_<safe_func>.c
  Makefile
  coverage.json
  judge_verdict.json
```

This keeps Stage 6 working without changing integration code.

### `integrate`

Purpose:

```text
Optional final master-suite integration.
```

Runs:

```text
integrate_all_unit_tests_sequential()
```

This is not the primary data-generation product. It remains useful as a later validation stage.

## BFS Scheduler Shape

The important orchestration difference:

```text
for stage in selected_stages:
  for source in selected_sources:
    prepare source if needed
    for item in stage_items:
      run N episodes
  select/materialize frontier for this stage
```

This allows commands like:

```text
main2.py <source_dir> --stage prepare
main2.py <source_dir> --stage collect-stubs --episodes-per-item 16
main2.py <source_dir> --stage select-stubs
main2.py <source_dir> --stage materialize-stubs
main2.py <source_dir> --stage integrate-stubs
main2.py <source_dir> --stage minimal-master
main2.py <source_dir> --stage collect-unit-tests --episodes-per-item 8
main2.py <source_dir> --stage select-unit-tests
main2.py <source_dir> --stage materialize-unit-tests
main2.py <source_dir> --stage integrate
```

The current end-to-end behavior should stay available:

```text
main2.py <source_dir>
```

No flag should mean current behavior.

## Minimal Surgical Code Plan

### Rule 1: Keep The Default Path Untouched

`main2.run()` currently executes the end-to-end flow. The safest change is:

```python
if cfg.stage != "end-to-end":
    run_selected_stage(cfg)
    return

# existing current code continues here
```

That means normal usage keeps the current behavior.

### Rule 2: Add A Thin Stage Module

Add one new module:

```text
pipeline/data_collection.py
```

This module should contain orchestration only:

- dataset path helpers
- episode id helper
- copy/snapshot helpers
- stage entrypoints
- simple selector/materializer helpers

It should call existing functions rather than duplicate their internals.

### Rule 3: Add Only Tiny Optional Override Parameters

The most fragile current behavior is hard-coded active workspaces.

Current stub generation writes here:

```text
_stub_gen/<safe_stub>/stub.c
```

Current unit generation writes here:

```text
_unit_tests/<safe_func>/test_<safe_func>.c
```

For repeated independent episodes, add optional override paths.

For stubs:

```python
generate_stub_code(
    cfg,
    test_dir,
    func_name,
    *,
    stub_dir_override: Path | None = None,
    history_dir_override: Path | None = None,
) -> Optional[str]
```

Default behavior remains identical:

```python
stub_dir = stub_dir_override or test_dir / "_stub_gen" / safe_name
history_dir = history_dir_override or test_dir / "agent_history"
```

Also pass the override history directory into stub-fix validation so fix calls for the episode stay with the episode.

For unit tests:

```python
_generate_unit_test_for_func(
    cfg,
    paths,
    func,
    flags,
    semantic_context_snapshot,
    *,
    unit_dir_override: Path | None = None,
) -> tuple[str, dict]
```

Default behavior remains identical:

```python
unit_dir = unit_dir_override or test_dir / "_unit_tests" / safe_id
```

The collector calls the same function with a fresh episode workspace. Normal Stage 5 calls it without the override.

### Rule 4: Do Not Split Every Internal Repair Yet

Eventually we may want each agent call to be its own RL sample. That is real, but the first safe implementation should collect a full trajectory episode:

```text
one stub generation run, including any stub-fix calls
one unit-test generation run, including any compile-fix calls
```

Why:

- `run_agent()` already writes prompt, history, and result files.
- The verifier outputs already exist.
- Splitting every repair into separate samples can be done later by post-processing `agent_history`.

This avoids changing the core loops while still collecting useful traces.

### Rule 5: Selection Should Copy Into Existing Legacy Paths

Do not teach every downstream stage a new data layout yet.

Instead:

```text
selected stub episode -> _stub_gen/<safe>/stub.c
selected unit episode -> _unit_tests/<safe>/test_<safe>.c
```

That lets:

- `_sync_stub_srcs()`
- `parallel_generate_unit_tests()`
- `integrate_all_unit_tests_sequential()`

keep using the paths they already understand.

### Rule 6: Stage Dispatch Should Be Small

Add config fields:

```python
stage: str = "end-to-end"
episodes_per_item: int = 1
trace_dataset_dirname: str = "_trace_dataset"
```

Add CLI flags:

```text
--stage
--episodes-per-item
--trace-dataset-dirname
```

Start with these stage names:

```text
end-to-end
prepare
collect-stubs
select-stubs
materialize-stubs
integrate-stubs
minimal-master
collect-unit-tests
select-unit-tests
materialize-unit-tests
integrate
```

No multi-source scheduler is required inside `main2.py` at first. A shell loop or separate launcher can run the same stage over many source dirs. That is safer than immediately changing the CLI to manage source manifests.

## Tests To Add Before Trusting Any Runtime Change

Use `test_main.py` style tests with mocks. Do not run real agents, real `make`, or real `gcc` in unit tests.

### CLI And Dispatch Tests

Test:

- default `--stage` is `end-to-end`
- `--stage collect-stubs` is parsed into config
- default path still calls the normal current functions
- selected stage path returns before Stage 6 unless stage is `integrate`

### Dataset Path Tests

Test:

- episode ids create isolated directories
- dataset path helpers build:

```text
_trace_dataset/episodes/stubs/<safe>/<episode_id>
_trace_dataset/episodes/unit_tests/<safe>/<episode_id>
_trace_dataset/frontiers/stubs/<safe>
_trace_dataset/frontiers/unit_tests/<safe>
```

### Stub Episode Tests

With `run_agent`, `_validate_stub_locally`, and filesystem writes mocked or simplified:

- `collect-stubs` writes into an episode directory
- active `_stub_gen/<safe>/` is not touched during collection
- `select-stubs` picks a validated episode
- `materialize-stubs` copies selected `stub.c` and `result.json` into `_stub_gen/<safe>/`

### Unit Episode Tests

With `run_make_test`, `check_function_coverage`, and `run_semantic_test_judge` mocked:

- `collect-unit-tests` writes into an episode unit workspace
- active `_unit_tests/<safe>/` is not touched during collection
- selected unit test copies into `_unit_tests/<safe>/`
- `coverage.json` and `judge_verdict.json` are preserved

### Regression Test For Default End-To-End

Mock the stage functions and assert the existing order remains:

```text
ensure_test_file
build_annotated_makefile
run_or_load_analysis
handle_stubs
ensure_minimal_test_runs
parallel_generate_unit_tests
integrate_all_unit_tests_sequential
```

This is the most important safety test. If the default path changes accidentally, the patch is too risky.

## Why This Is The Lowest-Risk Path

This plan avoids:

- rewriting prompts
- changing Makefile generation
- changing coverage parsing
- changing semantic judging
- changing Stage 6 integration
- changing default end-to-end behavior
- making the pipeline understand multiple source dirs at once

It adds:

- one stage dispatcher
- one data-collection module
- optional workspace overrides for stub and unit generators
- simple copy-based frontier materialization
- unit tests around dispatch and artifact isolation

That is the smallest useful cut: stage-wise trace generation becomes possible, while the existing fragile runtime path remains the reference path.
