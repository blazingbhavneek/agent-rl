# Final Combined Plan

This combines:

- `plan/data_coll_stage_seperation.md`
- `plan/host_docker_seperaton.md`
- the current `main2.py` and `pipeline/` runtime shape

The target is to get both features into code with the smallest practical
surface-area change:

1. stage-wise dataset/trace collection
2. host Python harness with container-side agent/build/test execution

The guiding rule is: keep the existing end-to-end pipeline as the reference
path, and add narrow boundaries around it.

## Success Criteria

After implementation:

- `python main2.py <source_dir>` still follows the current end-to-end path by
  default.
- Local execution remains the default and behaves like the current code.
- Docker/container execution is opt-in.
- Generated prompts, C files, Makefiles, `_pipeline_context.json`, coverage
  files, and history paths use canonical `/home/seigyo/...` paths only.
- Dataset episodes are saved under `_trace_dataset/` without changing the
  legacy active locations until a select/materialize stage is run.
- Selected dataset artifacts are copied back into the legacy locations that
  existing downstream code already understands:
  - `_stub_gen/<safe_stub>/`
  - `_unit_tests/<safe_func>/`
- No container lifecycle manager is introduced in this pass. The code assumes
  an already-created, correctly mounted container.

## Non-Goals For This Pass

Do not change:

- prompt wording, except for tiny path-safety checks if needed
- Makefile structure
- analyzer behavior
- coverage parsing
- semantic judge behavior
- retry semantics
- Stage 6 merge logic
- multi-source scheduling
- container creation/deletion
- the ignored `--output-dir` flag

Do not introduce host/container path mapping as a first step. Because the code
uses `Path.resolve()` heavily, path mapping would touch prompts, includes,
Makefiles, coverage, context files, and history metadata. That is too risky for
the first combined implementation.

## Current Runtime Facts To Preserve

`main2.run()` is the active spine:

1. `derive_paths(cfg)`
2. optional resume gate through `test_dir/_pipeline_context.json`
3. `ensure_test_file(cfg, paths)`
4. `build_annotated_makefile(cfg, paths)`
5. `run_or_load_analysis(cfg, paths["analysis_path"])`
6. `handle_stubs(cfg, paths, analysis)`
7. `ensure_minimal_test_runs(cfg, paths)`
8. `parallel_generate_unit_tests(cfg, paths, analysis, flags)`
9. `integrate_all_unit_tests_sequential(cfg, paths, analysis, unit_results, flags)`

Important couplings:

- `PipelineConfig.source_dir` is both the source Python reads and the path shown
  to the agent/generated files.
- `derive_test_dir()` requires `source_dir.parent.name == "src"`.
- `test_dir` is derived as `source_dir.parent.parent / "tests" / source_dir.name`.
- Stage 3 currently combines stub generation, validation, and master
  integration.
- Stage 5 already has the best unit-test episode boundary:
  `_generate_unit_test_for_func()`.
- `_pipeline_context.json` caches absolute paths and controls the resume gate.
- `unit_tests_completed: true` skips straight to integration on later normal
  runs.

## Implementation Order

Use this order so each step has a clean fallback:

1. Add execution config/CLI fields, defaulting to local behavior.
2. Add `pipeline/execution.py` and route command execution through it.
3. Update direct command call sites to pass `cfg` explicitly.
4. Add tests proving local end-to-end behavior still dispatches in the old
   order.
5. Add stage/dataset config fields, defaulting to `end-to-end`.
6. Add `pipeline/data_collection.py` with stage dispatch and dataset helpers.
7. Add tiny workspace override parameters to stub/unit generation functions.
8. Add select/materialize stages that copy chosen artifacts into legacy paths.
9. Add Docker smoke tests using an already-created container.
10. Only then run a real stage-wise collection attempt.

This order makes the command boundary stable before dataset episodes start
using it.

## Part 1: Execution Boundary

### Config Fields

Add to `PipelineConfig` in `pipeline/config.py`:

```python
execution_mode: str = "local"
container_name: Optional[str] = None
container_profile: Path = Path("/home/seigyo/.bash_profile")
forbidden_host_prefixes: tuple[str, ...] = ()
```

Add matching CLI flags in `main2.py::parse_args()`:

```text
--execution-mode local|docker
--container-name <name>
--container-profile /home/seigyo/.bash_profile
--forbid-host-prefix <prefix>   # appendable, docker-mode guard
```

Rules:

- `local` is the default.
- `docker` requires `--container-name`.
- The profile is sourced for every docker-side command.
- `--forbid-host-prefix` is optional so local development is not blocked by
  over-eager checks. On the work PC, pass the real host checkout/output prefix
  if host path leakage is a concern.

### New Runner Module

Add:

```text
pipeline/execution.py
```

Primary API:

```python
def run_command(
    cfg: PipelineConfig,
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    ...
```

Local mode:

- Calls `subprocess.run()` with the same capture/text/timeout behavior the code
  uses today.
- Raises `subprocess.TimeoutExpired` on timeout, like `subprocess.run()` does.

Docker mode:

- Builds a `docker exec` command.
- Uses canonical `cwd` inside the container.
- Sources `cfg.container_profile`.
- Then runs the requested command.
- For list commands, use `shlex.join()` to build the inner shell command.
- For string commands with `shell=True`, pass the string through as the inner
  shell command.
- Pass `MAX_ITERATIONS` and `PYTHON_BIN` through as environment variables when
  `run_agent()` provides them.

Docker inner command shape:

```text
source /home/seigyo/.bash_profile
cd <canonical cwd>
exec <command>
```

Keep this helper small. Do not add container creation or mount logic here.

### Command Call Sites To Replace

Replace only command execution, not surrounding logic.

In `pipeline/common.py`:

- `run_agent()`
  - keep prompt writing exactly where it is
  - keep history/result file names
  - call `run_command(cfg, ["node", str(cfg.agent_js), ...], cwd=work_dir, ...)`
  - local mode keeps snapshot/restore
  - docker mode skips snapshot/restore by default and relies on read-only source
    mounts

- `run_make_test()`
  - change signature to:

    ```python
    def run_make_test(cfg: PipelineConfig, test_dir: Path, timeout: int = 300) -> dict:
        ...
    ```

  - call `run_command(cfg, ["make", "test"], cwd=test_dir, timeout=timeout)`
  - return the same dict shape as today

- diagnostics:

  ```python
  collect_failure_diagnostics(cfg, test_dir, test_file, res)
  collect_runtime_crash_diagnostics(cfg, test_dir, test_binary_name)
  build_output_with_runtime_diagnostics(cfg, test_dir, test_file, res)
  ```

  - direct binary runs use `run_command(cfg, [str(test_bin)], ...)`
  - `gdb` runs use `run_command(cfg, ["gdb", ...], ...)`
  - keep the current fallback text when `gdb` is absent or times out

In `pipeline/stage2_makefile.py`:

- `_run_do_mkmf()`
  - replace direct `subprocess.run(["do_mkmf", test_program], ...)`
  - use `run_command(cfg, ["do_mkmf", test_program], cwd=test_dir, timeout=300)`
  - preserve the current error message content

In `pipeline/stage3_stubs.py`:

- `_validate_stub_locally()`
  - compile command through `run_command(..., shell=True)`
  - link command through `run_command(..., shell=True)`
  - validation binary through `run_command(..., shell=False)`
- `integrate_all_stubs_sequential()`
  - `run_make_test(cfg, test_dir)`
  - `build_output_with_runtime_diagnostics(cfg, ...)`

In `pipeline/stage4_minimal.py`:

- all `run_make_test(test_dir, ...)` calls become `run_make_test(cfg, test_dir, ...)`
- diagnostics call gets `cfg`

In `pipeline/stage5_unit_tests.py`:

- all `run_make_test(unit_dir, ...)` calls become `run_make_test(cfg, unit_dir, ...)`
- diagnostics call gets `cfg`
- `check_function_coverage()` stays unchanged because it only reads generated
  `.gcov` files from the mounted canonical path

In `pipeline/stage6_integrate.py`:

- all `run_make_test(test_dir, ...)` calls become `run_make_test(cfg, test_dir, ...)`
- diagnostics call gets `cfg`
- the `gcc -fsyntax-only ...` syntax check uses `run_command(cfg, ..., shell=True)`

In `main2.py`:

- resume-gate `run_make_test()` calls get `cfg`
- resume-gate diagnostics get `cfg`

### Path Hygiene Guard

Add a tiny helper, probably in `pipeline/execution.py` or `pipeline/common.py`:

```python
def assert_no_forbidden_host_paths(cfg: PipelineConfig, text: str, label: str) -> None:
    ...
```

Use it in docker mode before writing/running prompts:

- prompt text in `run_agent()`
- optionally generated `_pipeline_context.json` after Stage 2

This guard should only check `cfg.forbidden_host_prefixes`. Do not hard-code the
current laptop path, because the work PC host prefix may differ.

## Part 2: Dataset Stage Separation

### Config Fields

Add to `PipelineConfig`:

```python
stage: str = "end-to-end"
episodes_per_item: int = 1
trace_dataset_dirname: str = "_trace_dataset"
```

Add CLI flags:

```text
--stage <stage-name>
--episodes-per-item <N>
--trace-dataset-dirname _trace_dataset
```

Valid stage names for the first implementation:

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

No flag means `end-to-end`.

### Main Dispatch

In `main2.run()` keep the current code as the default path.

Add the stage gate immediately after `paths = derive_paths(cfg)`:

```python
paths = derive_paths(cfg)

if cfg.stage != "end-to-end":
    from pipeline.data_collection import run_selected_stage
    run_selected_stage(cfg, paths)
    return

# existing resume gate and existing end-to-end code continue below
```

This is the most important safety property. The normal path should not be
rewritten into the new stage framework.

### New Dataset Module

Add:

```text
pipeline/data_collection.py
```

Keep this module orchestration-only:

- dataset path helpers
- episode id helper
- metadata writer
- small copy helpers
- stage dispatch
- simple selectors
- simple materializers

It should call existing stage functions rather than duplicate their logic.

### Dataset Layout

Use a slightly simpler layout than the first dataset plan so implementation is
small and reliable:

```text
tests/<process_name>/
  _trace_dataset/
    manifest.json

    episodes/
      stubs/
        <safe_stub>/
          <episode_id>/
            metadata.json
            workspace/
              stub.c
              stub_validate_main.c
              stub.o
              stub_validate
              result.json
            agent_history/
              *.json
              *.prompt.txt
              *.result.json

      unit_tests/
        <safe_func>/
          <episode_id>/
            metadata.json
            workspace/
              test_<safe_func>.c
              Makefile
              coverage.json
              judge_verdict.json
              unit_test_failed.json
              good_cunit_backups/
              agent_history/

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
```

Compatibility rule:

```text
_trace_dataset/ stores all attempts.
_stub_gen/ and _unit_tests/ store only the selected active artifacts.
```

Downstream stages should keep reading the legacy active paths.

### Workspace Overrides

Add only optional override parameters. Defaults must preserve current behavior.

In `pipeline/stage3_stubs.py`:

```python
def generate_stub_code(
    cfg: PipelineConfig,
    test_dir: Path,
    func_name: str,
    *,
    stub_dir_override: Path | None = None,
    history_dir_override: Path | None = None,
) -> Optional[str]:
    ...
```

Default:

```python
stub_dir = stub_dir_override or test_dir / "_stub_gen" / safe_name
history_dir = history_dir_override or test_dir / "agent_history"
```

Also thread `history_dir_override` into `_validate_stub_locally()` and
`_fix_stub_with_agent()` so stub repair histories stay inside the episode.

In `pipeline/stage5_unit_tests.py`:

```python
def _scaffold_unit_test_dir(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    *,
    unit_dir_override: Path | None = None,
) -> Path:
    ...

def _generate_unit_test_for_func(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    flags: dict,
    semantic_context_snapshot: dict,
    *,
    unit_dir_override: Path | None = None,
) -> tuple[str, dict]:
    ...
```

Default:

```python
unit_dir = unit_dir_override or test_dir / "_unit_tests" / safe_id
```

Keep `test_dir` as the active master test root even when `unit_dir_override` is
used. That lets unit episodes still read selected stubs from active `_stub_gen/`
and the active master Makefile.

### Stage Details

#### `prepare`

Runs:

```python
ensure_test_file(cfg, paths)
flags = build_annotated_makefile(cfg, paths)
analysis = run_or_load_analysis(cfg, paths["analysis_path"])
```

Produces:

- `test_<process_name>.c`
- `Makefile`
- `_pipeline_context.json`
- `analysis.json`

This stage is idempotent and safe to call before collection.

#### `collect-stubs`

Prerequisite:

- `prepare`

For each stub candidate from `collect_stub_candidates(analysis)` and for each
episode:

1. Create:

   ```text
   _trace_dataset/episodes/stubs/<safe_stub>/<episode_id>/
   ```

2. Call:

   ```python
   generate_stub_code(
       cfg,
       test_dir,
       func_name,
       stub_dir_override=episode_root / "workspace",
       history_dir_override=episode_root / "agent_history",
   )
   ```

3. Write `metadata.json` with:

   - stage name
   - process name
   - stub function name
   - episode id
   - canonical source/test paths
   - validation status from `workspace/result.json` if present
   - command mode/container name

Do not touch `_stub_gen/<safe_stub>/` during collection.

#### `select-stubs`

For each stub:

1. Read all stub episodes.
2. Prefer `workspace/result.json` with `validated: true`.
3. If multiple are valid, pick the first valid episode by stable sort, or the
   latest valid episode if that is easier to reason about.
4. Write:

   ```text
   _trace_dataset/frontiers/stubs/<safe_stub>/selected.json
   _trace_dataset/frontiers/stubs/<safe_stub>/stub.c
   _trace_dataset/frontiers/stubs/<safe_stub>/result.json
   ```

Selection is intentionally simple for the first pass.

#### `materialize-stubs`

Copies selected frontier files into:

```text
_stub_gen/<safe_stub>/stub.c
_stub_gen/<safe_stub>/result.json
```

Write/copy only from selected validated episodes.

Do not integrate into the master file here.

#### `integrate-stubs`

Load materialized validated stubs from `_stub_gen/`, then call:

```python
integrate_all_stubs_sequential(cfg, paths, validated_bodies, flags)
```

Do not call `handle_stubs()` from this stage, because `handle_stubs()` would
generate missing stubs and defeat collection/materialization separation.

#### `minimal-master`

Runs:

```python
ensure_minimal_test_runs(cfg, paths)
```

Then snapshots the active master baseline into:

```text
_trace_dataset/frontiers/minimal_master/
  selected.json
  test_<process_name>.c
  Makefile
```

This gives unit-test collection a named baseline.

#### `collect-unit-tests`

Prerequisites:

- `prepare`
- selected/materialized stubs, if the source needs stubs
- a current master baseline from `minimal-master`

For each targeted function from `functions_leaf_first(analysis)` and for each
episode:

1. Create:

   ```text
   _trace_dataset/episodes/unit_tests/<safe_func>/<episode_id>/
   ```

2. Call:

   ```python
   _generate_unit_test_for_func(
       cfg,
       paths,
       func,
       flags,
       semantic_context_snapshot,
       unit_dir_override=episode_root / "workspace",
   )
   ```

3. Write `metadata.json` with:

   - stage name
   - function metadata
   - episode id
   - result object
   - coverage percent
   - semantic score
   - pass/fail
   - canonical source/test paths
   - command mode/container name

Do not touch active `_unit_tests/<safe_func>/` during collection.

First pass can run episodes sequentially. Parallel multi-container scheduling
belongs outside `main2.py`.

#### `select-unit-tests`

For each function:

1. Read unit-test episode metadata/results.
2. Prefer:
   - `passed: true`
   - higher `coverage_pct`
   - higher `semantic_score`
3. Copy selected files into:

   ```text
   _trace_dataset/frontiers/unit_tests/<safe_func>/
   ```

Keep the selector deterministic.

#### `materialize-unit-tests`

Copy selected unit-test frontier files into:

```text
_unit_tests/<safe_func>/
  test_<safe_func>.c
  Makefile
  coverage.json
  judge_verdict.json
```

Update `_pipeline_context.json` carefully:

- Always write/update `unit_test_results` for materialized selected units.
- Set `unit_tests_completed: true` only if the selected passing frontier covers
  every function targeted by the current filters:
  - `only_function`
  - `only_level`
  - `max_functions`
- If coverage is partial, leave `unit_tests_completed` false or absent.

This prevents the existing resume gate from skipping too much on the next normal
run.

#### `integrate`

Load unit results from selected frontiers or `_pipeline_context.json`, then call:

```python
integrate_all_unit_tests_sequential(cfg, paths, analysis, unit_results, flags)
```

This keeps final master-suite integration optional. Dataset collection does not
depend on it.

## Combined Work-PC / Container Contract

The host and container must both see the same canonical paths.

Canonical shape:

```text
/home/seigyo/rl/
  main2.py
  pipeline/
  agent.js
  system_functions.json
  moove_docs/func/

/home/seigyo/.../<project_root>/
  src/<process_name>/
  tests/<process_name>/
```

Mount rules:

- Project/source root visible at the canonical path inside the container.
- Production source and build neighbors read-only in the container.
- `tests/<process_name>/` writable and unique per attempt/container.
- If production objects are expected under `source_dir/linux/*.o`, they must be
  visible in the container.
- Do not use symlinks that resolve to host-only prefixes.
- If host and container cannot share the canonical path, create a staging/bind
  mount at the canonical path before running the harness. Do not add path
  translation to this pipeline in the first implementation.

For many independent episodes:

- safest first version: one external attempt orchestrator creates one container
  and one writable test mount per attempt, then invokes one `main2.py --stage`
  command
- acceptable local smoke version: one already-created container runs
  `--episodes-per-item N` sequentially in one writable test mount

## Example Commands

Prepare:

```text
python /home/seigyo/rl/main2.py \
  /home/seigyo/.../<project_root>/src/<process_name> \
  --execution-mode docker \
  --container-name <attempt_container> \
  --stage prepare
```

Collect one unit-test episode for a specific function after stubs/minimal master
are ready:

```text
python /home/seigyo/rl/main2.py \
  /home/seigyo/.../<project_root>/src/<process_name> \
  --execution-mode docker \
  --container-name <attempt_container> \
  --stage collect-unit-tests \
  --only-function <function_id> \
  --episodes-per-item 1
```

Full staged flow:

```text
python main2.py <source_dir> --stage prepare
python main2.py <source_dir> --stage collect-stubs --episodes-per-item 4
python main2.py <source_dir> --stage select-stubs
python main2.py <source_dir> --stage materialize-stubs
python main2.py <source_dir> --stage integrate-stubs
python main2.py <source_dir> --stage minimal-master
python main2.py <source_dir> --stage collect-unit-tests --episodes-per-item 4
python main2.py <source_dir> --stage select-unit-tests
python main2.py <source_dir> --stage materialize-unit-tests
python main2.py <source_dir> --stage integrate
```

The original behavior remains:

```text
python main2.py <source_dir>
```

## Tests To Add

Important: the current `test_main.py` imports `main.py`, not `main2.py`. Do not
trust it alone for this change. Add focused tests for `main2.py` and the new
modules.

Suggested new file:

```text
test_main2_combined.py
```

### Execution Tests

Mock subprocess/Docker. Do not require a real container.

Test:

- `PipelineConfig` defaults to local/end-to-end.
- `parse_args()` parses execution fields.
- local `run_command()` calls `subprocess.run()` with the provided cwd.
- docker `run_command()` builds `docker exec` with:
  - container name
  - profile source
  - canonical cwd
  - inner command
- docker mode without `container_name` raises a clear error.
- `run_make_test(cfg, test_dir)` returns the same dict shape as before.
- `run_agent()` in dry-run/local mode still avoids subprocess.
- docker `run_agent()` skips source snapshot/restore.

### Default Runtime Regression Test

Mock the stage functions in `main2.run()` and assert default order remains:

```text
ensure_test_file
build_annotated_makefile
run_or_load_analysis
handle_stubs
ensure_minimal_test_runs
parallel_generate_unit_tests
integrate_all_unit_tests_sequential
```

Also assert `cfg.stage != "end-to-end"` returns through
`data_collection.run_selected_stage()` before the resume gate and normal stages.

### Dataset Helper Tests

Test:

- dataset root path is `test_dir / cfg.trace_dataset_dirname`
- episode ids produce isolated directories
- frontier paths are stable
- metadata files contain canonical paths
- forbidden host-prefix guard catches configured prefixes

### Stub Collection Tests

With `generate_stub_code()` and validation mocked:

- `collect-stubs` writes under `_trace_dataset/episodes/stubs/...`
- active `_stub_gen/` is untouched during collection
- `select-stubs` chooses a validated episode
- `materialize-stubs` copies `stub.c` and `result.json` into `_stub_gen/`
- `integrate-stubs` calls `integrate_all_stubs_sequential()` and not
  `handle_stubs()`

### Unit Collection Tests

With `_generate_unit_test_for_func()`, `run_make_test()`, coverage, and judge
mocked:

- `collect-unit-tests` writes under `_trace_dataset/episodes/unit_tests/...`
- active `_unit_tests/` is untouched during collection
- `select-unit-tests` prefers passed, then higher coverage, then higher semantic
  score
- `materialize-unit-tests` copies selected files into `_unit_tests/`
- `materialize-unit-tests` does not set `unit_tests_completed` when only a
  partial frontier exists
- `materialize-unit-tests` sets `unit_tests_completed` only when all targeted
  functions have selected passing results

## Verification Commands

Local syntax:

```text
/home/blaze/miniconda3/envs/rl/bin/python -m py_compile main2.py pipeline/*.py
```

Local tests:

```text
/home/blaze/miniconda3/envs/rl/bin/python -m pytest test_main.py test_main2_combined.py
```

Docker environment smoke before a pipeline run:

```text
source /home/seigyo/.bash_profile
command -v node
command -v do_mkmf
command -v gcc
command -v make
command -v gcov
test -f /home/seigyo/rl/agent.js
```

Pipeline Docker smoke:

```text
python /home/seigyo/rl/main2.py \
  /home/seigyo/.../<project_root>/src/<process_name> \
  --execution-mode docker \
  --container-name <attempt_container> \
  --stage prepare \
  --forbid-host-prefix <host_checkout_or_output_prefix>
```

Then inspect:

```text
tests/<process_name>/test_<process_name>.c
tests/<process_name>/Makefile
tests/<process_name>/_pipeline_context.json
tests/<process_name>/agent_history/*.prompt.txt
```

They should contain canonical `/home/seigyo/...` paths only.

## Main Risks And Guards

### Risk: `Path.resolve()` Leaks Host Paths

Guard:

- require canonical bind mounts or real canonical directories
- no symlinked canonical paths
- use `--forbid-host-prefix` in docker smoke tests

### Risk: Old `_pipeline_context.json` Reuses Wrong Absolute Paths

Guard:

- use a fresh `tests/<process_name>/` per container attempt
- delete context when changing mount layout
- do not share one context across different canonical roots

### Risk: `unit_tests_completed` Skips Stages Too Early

Guard:

- `materialize-unit-tests` writes the flag only for a complete passing targeted
  frontier
- partial dataset collection leaves the flag false/absent

### Risk: Dataset Collection Mutates Legacy Active Artifacts

Guard:

- collection stages only write under `_trace_dataset/episodes/...`
- only materialize stages copy into `_stub_gen/` or `_unit_tests/`
- tests assert active directories are untouched during collection

### Risk: Docker Runner Changes Local Behavior

Guard:

- local mode is default
- `run_command(local)` mirrors `subprocess.run()`
- default-order regression test around `main2.run()`
- no command is moved into Docker unless `cfg.execution_mode == "docker"`

### Risk: Source Is Modified By The Agent

Guard:

- local mode keeps snapshot/restore
- docker mode relies on read-only source mounts
- only `tests/<process_name>/` is writable

### Risk: Per-Episode Container Isolation Is Overbuilt Too Soon

Guard:

- first implementation assumes an existing container
- external orchestration handles one-container-per-attempt later
- `main2.py` only needs `--container-name`

## Minimal File Change Checklist

Expected code files:

```text
main2.py
pipeline/config.py
pipeline/common.py
pipeline/stage2_makefile.py
pipeline/stage3_stubs.py
pipeline/stage4_minimal.py
pipeline/stage5_unit_tests.py
pipeline/stage6_integrate.py
pipeline/execution.py
pipeline/data_collection.py
test_main2_combined.py
```

Expected untouched behavior:

- default end-to-end stage order
- existing active artifact layout
- existing Makefile contents, except paths still canonical as before
- existing prompts, except optional path guard failure before agent execution

## Bottom Line

The safest combined implementation is:

1. Add one execution runner so host Python can ask the container to run
   `node`, `do_mkmf`, `make`, `gcc`, test binaries, and `gdb`.
2. Keep canonical `/home/seigyo/...` paths everywhere.
3. Add a stage dispatcher that is bypassed by default.
4. Collect episodes only under `_trace_dataset/`.
5. Select/materialize by copying chosen artifacts into the legacy paths that the
   existing stages already use.

That gets both requested features without asking the fragile CUnit pipeline to
learn a second filesystem model or a new artifact model all at once.
