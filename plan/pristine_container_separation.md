# Pristine-Container / Host-Trace Separation

This note records, with code references, why the current `collect-*` stages
break the host/container separation we wanted, and the smallest-risk plan to
fix it. It is written so that if the implementation goes wrong later, the exact
current behavior and the intended behavior can be recovered from here.

Related notes:

- `plan/data_coll_stage_seperation.md` (host-side dataset/trace layout we want)
- `plan/file_progression.md` (pristine on-disk layout the agent/container must see)
- `plan/host_docker_seperaton.md` (execution boundary already implemented)
- `plan/final_plan.md` (the combined pass that produced the current code)

## 1. The Problem (with evidence)

Goal restated:

- The **container / agent** must only ever see the canonical pristine layout
  from `file_progression.md`:
  `tests/<proc>/_stub_gen/<safe>/stub.c`, `tests/<proc>/_unit_tests/<safe>/...`,
  `tests/<proc>/test_<proc>.c`. No `_trace_dataset`, no `<episode_id>`.
- The **host** owns the `_trace_dataset/` archive from
  `data_coll_stage_seperation.md` and stores every attempt there.
- `_trace_dataset` must never be visible to / referenced by any container.

What the current code actually does: it points the agent's workspace directly
at the dataset directory via override parameters, so the dataset path leaks
into everything the agent sees.

### 1.1 Leak in collect-stubs

`pipeline/data_collection.py::_stage_collect_stubs` builds
`episode_root = root / "episodes" / "stubs" / safe / eid`, then
`workspace = episode_root / "workspace"`, and calls:

```python
generate_stub_code(
    cfg, test_dir, func_name,
    stub_dir_override=workspace,           # <- dataset path
    history_dir_override=history_dir,      # <- dataset path
)
```

Inside `pipeline/stage3_stubs.py::generate_stub_code` the override becomes
`stub_dir`, and `stub_out = stub_dir / "stub.c"`. That `stub_out` is then
embedded verbatim in the agent prompt:

```
TEST FOLDER
{test_dir}

OUTPUT FILE
{stub_out}        # e.g. .../tests/proc/_trace_dataset/episodes/stubs/foo/<eid>/workspace/stub.c
...
Write only this file:
  `{stub_out}`
```

So the agent is explicitly told to write into a `_trace_dataset/...` path. The
container is no longer pristine. `_validate_stub_locally` then runs `gcc`/link
with `cwd=stub_dir` (same dataset path) and `history_dir_override` flows into
`run_agent`'s `--history` and `.prompt.txt`/`.result.json` files, leaking again.

Note: `--forbid-host-prefix` (the docker hygiene guard in
`pipeline/execution.py::assert_no_forbidden_host_paths`) does NOT catch this,
because `_trace_dataset` lives under the canonical `tests/<proc>` path, not
under a forbidden host prefix.

### 1.2 Leak in collect-unit-tests

`pipeline/data_collection.py::_stage_collect_unit_tests` calls
`_generate_unit_test_for_func(..., unit_dir_override=workspace)` where again
`workspace = .../_trace_dataset/episodes/unit_tests/<safe>/<eid>/workspace`.

In `pipeline/stage5_unit_tests.py`, that override becomes `unit_dir`, and:

- the unit prompt embeds `{unit_test_file}` and `{unit_makefile}` (both under
  the dataset path),
- `_generate_unit_test_makefile` writes `unit_dir/Makefile` containing
  `TEST_PROGRAM`, `PROD_SRC`, etc. at that dataset path,
- `backup_good_cunit_if_best` writes `unit_dir/agent_history/good_cunit_backups/...`
  under the dataset path.

Same leak, same loss of pristineness.

### 1.3 Missing: per-trace container

The original intent (`host_docker_seperaton.md` §"Mount Contract", "Per parallel
agent later") was: each attempt gets its own container with its own writable
bind mount over `tests/<proc>`, so many agents run in parallel while each sees
the same pristine canonical path. The current code does none of this:
`pipeline/execution.py::run_command` only ever `docker exec`s into one
pre-existing `cfg.container_name`. `final_plan.md` explicitly deferred container
lifecycle ("No container lifecycle manager is introduced in this pass").

## 2. Target Model

Mirror images:

- Container/agent side = `file_progression.md` pristine layout, identical every
  run. The override hack is removed.
- Host side = `data_coll_stage_seperation.md` archive. Host harvests each
  pristine run into `_trace_dataset/episodes/...`.

Replacement for the override hack: run each stage into the **canonical pristine
location** (`_stub_gen/<safe>/`, `_unit_tests/<safe>/`), then the host **copies
the artifacts out** into the episode dir and **resets** the canonical location
for the next episode.

Isolation between same-item episodes comes from EITHER:

- Phase 1: running them serially with reset between (no new infra), or
- Phase 2: giving each episode its own container + its own writable mount of
  the canonical path (true parallel, real "one container per trace").

## 3. Phase 1 — Kill the leak (surgical)

Scope: `pipeline/data_collection.py` only. No changes to `execution.py`,
stage3/stage5 internals, select/materialize, or the end-to-end spine.

### 3.1 New helpers (in data_collection.py)

```python
def _reset_dir(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)

def _harvest(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    """Copy each name (file or dir) from src_dir to dst_dir if it exists."""
    for name in names:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

def _harvest_glob(src_dir: Path, dst_dir: Path, pattern: str) -> None:
    """Copy files matching glob (shared agent_history entries)."""
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.glob(pattern):
        shutil.copy2(src, dst_dir / src.name)
```

### 3.2 Rewrite `_stage_collect_stubs`

Current (leaky) body passes overrides. New body runs canonical + harvest +
reset. Sketch:

```python
def _stage_collect_stubs(cfg, paths):
    _flags, analysis = _prepare(cfg, paths)
    test_dir = Path(paths["test_dir"])
    history_dir = test_dir / "agent_history"      # shared, canonical
    candidates = collect_stub_candidates(analysis)
    root = dataset_root(cfg, paths)
    for func_name in candidates:
        safe = _safe_filename(func_name)
        active = test_dir / "_stub_gen" / safe
        for _ in range(max(1, int(cfg.episodes_per_item))):
            eid = _episode_id()
            episode_root = root / "episodes" / "stubs" / safe / eid
            workspace = episode_root / "workspace"
            ep_history = episode_root / "agent_history"

            _reset_dir(active)                     # pristine start, kill cache
            body = generate_stub_code(cfg, test_dir, func_name)  # NO overrides

            # harvest produced artifacts into the episode (keep current layout)
            _harvest(active, workspace,
                     ["stub.c", "result.json", "stub_validate_main.c"])
            _harvest_glob(history_dir, ep_history, f"_gen_stub_{safe}.*")
            _harvest_glob(history_dir, ep_history, f"_stub_fix_{safe}_*")

            result = {}
            rp = workspace / "result.json"
            if rp.exists():
                try: result = load_json(rp)
                except Exception: result = {}
            metadata = { **_metadata_base(cfg, paths, "collect-stubs", eid),
                         "func_name": func_name, "safe_name": safe,
                         "validated": bool(result.get("validated")),
                         "body_chars": len(body or ""),
                         "workspace": str(workspace),
                         "agent_history": str(ep_history),
                         "result": result }
            write_json(episode_root / "metadata.json", metadata)
            _reset_dir(active)                     # leave pristine for next
```

Why reset before AND after: `generate_stub_code` reuses a cached validated
`stub.c` (see `stage3_stubs.py` cache block: if `result.json.validated` it
returns the cached body without calling the agent). Independent episodes must
not reuse — reset before forces a fresh agent run. Reset after keeps the
canonical tree clean so it does not pollute later stages / other items.

The episode `workspace/` keeps the SAME shape as today
(`workspace/stub.c`, `workspace/result.json`), so `select-stubs` /
`materialize-stubs` and frontier code are untouched.

### 3.3 Rewrite `_stage_collect_unit_tests`

Same pattern, target `_unit_tests/<safe>/`:

```python
def _stage_collect_unit_tests(cfg, paths):
    flags, analysis = _prepare(cfg, paths)
    test_dir = Path(paths["test_dir"])
    root = dataset_root(cfg, paths)
    semantic_context = _load_semantic_context(test_dir)
    for func in _target_functions(cfg, analysis):
        safe = _safe_filename(func["id"])
        active = test_dir / "_unit_tests" / safe
        for _ in range(max(1, int(cfg.episodes_per_item))):
            eid = _episode_id()
            workspace = root / "episodes" / "unit_tests" / safe / eid / "workspace"

            _reset_dir(active)                         # pristine start
            fid, result = _generate_unit_test_for_func( # NO override
                cfg, paths, func, flags, semantic_context)

            _harvest(active, workspace, [
                f"test_{safe}.c", "Makefile", "coverage.json",
                "judge_verdict.json", "unit_test_failed.json",
                "agent_history",        # includes good_cunit_backups/
            ])
            metadata = { ... same fields as today ... }
            write_json(workspace.parent / "metadata.json", metadata)
            _reset_dir(active)                         # leave pristine
```

Inputs that must survive between episodes (do NOT reset): selected
`_stub_gen/*` (read by `_sync_stub_srcs`), the master baseline
`test_<proc>.c` + `Makefile` from `minimal-master`, `_pipeline_context.json`,
`analysis.json`. We only ever reset the single `_unit_tests/<safe>/` dir for the
item under collection.

### 3.4 Remove override usage; keep override params

Drop the `stub_dir_override` / `history_dir_override` / `unit_dir_override`
arguments from the two call sites only. Leave the parameters DEFINED on
`generate_stub_code`, `_scaffold_unit_test_dir`, `_generate_unit_test_for_func`
(default `None` = canonical). This avoids editing stage3/stage5 signatures and
any other caller, minimizing blast radius. They simply become unused.

### 3.5 What Phase 1 delivers / does not

Delivers: agent/container sees only canonical pristine paths; host accumulates
all attempts under `_trace_dataset/`. Works in `local` mode and in `docker`
mode against a single pre-created container.

Does NOT deliver: parallel same-item episodes, and "one container per trace."
Episodes are serial because they share one canonical `_unit_tests/<safe>/`.

### 3.6 Phase 1 risks

1. `_reset_dir(active)` deletes the canonical `_stub_gen/<safe>` /
   `_unit_tests/<safe>`. If a user mixes `collect-*` with a normal end-to-end
   run in the same `tests/<proc>`, collection will wipe active artifacts. Mitigate
   by documenting that `collect-*` owns/clears those item dirs; selection still
   restores via `materialize-*`.
2. Stub `agent_history` is SHARED (`test_dir/agent_history`), not per-stub. The
   glob harvest (`_gen_stub_<safe>.*`, `_stub_fix_<safe>_*`) couples to the
   naming in `stage3_stubs.py` / `file_progression.md §12`. If those names change,
   harvest silently misses history. Acceptable, low impact (history is
   supplementary).
3. Serial episodes are slower. Acceptable until Phase 2.
4. Time-based `_episode_id()` plus serial loop is fine; no collision risk.

## 4. Phase 2 — True per-trace container (IMPLEMENTED, opt-in)

Implemented as "option 1": per-episode bind mount at the canonical path plus a
single host->canonical prefix rewrite at the two execution choke points. Source
is immutable and shared; only the test dir varies per episode; episodes run in
parallel.

### 4.0 How it actually works

Key idea: keep `paths` canonical so generated content (unit Makefile
`MASTER_MAKEFILE`, `PROD_SRC`, master-test refs) stays canonical and untouched.
Write per-episode output via the EXISTING override params pointed at a host
episode dir `H`. Bind-mount `H` at the canonical test path; rewrite the
`H -> canonical` prefix for anything the agent/container sees.

Per episode (`pipeline/data_collection.py::_collect_one_stub` /
`_collect_one_unit`, dispatched by `_run_jobs`):

1. `H = _trace_dataset/episodes/<kind>/<safe>/<eid>/testdir`. Copy the prepared
   canonical test dir into `H` (`_prepare_episode_testdir`), excluding
   `_trace_dataset`, `_unit_tests` (always) and `_stub_gen` (stub episodes
   only; unit episodes keep selected stubs). So `H` is a pristine full test dir.
2. `container.create(cfg, name, host_test_dir=H, canonical_test_dir=<canonical>,
   repo_root=<repo>)` runs `docker run -d`:
   - `-v <repo>:<repo>:ro` (immutable source/project, shared, never copied)
   - `-v <H>:<canonical_test_dir>` (per-episode writable, overlaid)
   - `+ cfg.container_run_args` (e.g. mount `/home/seigyo/rl`)
   - `<image> sleep infinity`
3. `run_cfg = replace(cfg, execution_mode="docker", container_name=name,
   per_episode_container=False, path_map=((str(H), str(canonical)),))`.
4. Run the stage with canonical `paths`/`test_dir` but overrides pointed at `H`:
   `generate_stub_code(run_cfg, canonical, fn, stub_dir_override=H/_stub_gen/safe,
   history_dir_override=H/agent_history)` /
   `_generate_unit_test_for_func(run_cfg, paths, ..., unit_dir_override=H/_unit_tests/safe)`.
5. Translation makes the agent see canonical only:
   - `pipeline/execution.py::containerize_text` rewrites `H -> canonical` in the
     `docker exec` cwd and every command arg/string (handles gcc/make/node,
     `--folder/--prompt-file/--history`, binary paths).
   - `pipeline/common.py::run_agent` rewrites the prompt text before writing the
     prompt file, so the file the agent reads is canonical.
   - Both are no-ops unless docker mode + a non-empty `path_map`.
6. Inside the container, canonical == `H` (the mount), so writes land in `H` on
   the host; the host reads results straight from `H`.
7. `container.teardown(name)` in a `finally`. Harvest `H/_stub_gen|_unit_tests/<safe>/`
   into the episode `workspace/` (keeps select/materialize layout). `H` itself
   persists as the per-episode full test dir.

Parallelism: `_run_jobs` uses a `ThreadPoolExecutor(max_workers=episode_concurrency)`
in per-episode-container mode (serial otherwise). Each job has its own `H`,
container, and `run_cfg`, so there is no shared mutable disk state.

Why the agent never sees `_trace_dataset`: `H` lives under `_trace_dataset` on
the host, but the mount remaps it to the canonical path and `containerize_text`
rewrites any `H` string to canonical. The `_trace_dataset` prefix is exactly the
part that gets rewritten away.

### 4.1 Config / CLI

- `per_episode_container: bool` (`--per-episode-container`)
- `container_image: Optional[str]` (`--container-image`, required for the above)
- `container_run_args: tuple[str,...]` (`--container-run-arg`, repeatable; mounts)
- `episode_concurrency: int` (`--episode-concurrency`)
- `path_map: tuple[tuple[str,str],...]` (runtime only, set per episode)

### 4.2 Operating requirements

1. Run the prep/select stages first so the canonical base test dir is complete
   before collection: `prepare`, then for unit episodes
   `collect-stubs/select-stubs/materialize-stubs` and `minimal-master`. In
   per-episode-container mode `_prepare` must hit cache (it reads the existing
   `_pipeline_context.json` / `analysis.json` and rewrites scaffold host-side
   only); it should not need to run `do_mkmf`, which would require a container.
2. `cfg.container_run_args` must mount `/home/seigyo/rl` (agent.js, system json,
   docs) unless the image already contains it.
3. The image/profile must provide `node, do_mkmf, gcc, make, gcov, gdb`.
4. The repo ro mount must expose `source_dir/linux/*.o` for the master link.

### 4.3 Risks / caveats

1. Per-episode `copytree` of the base test dir is modest (test artifacts, not
   src), but scales with base size; acceptable, optimize later if needed.
2. `containerize_text` is a string prefix replace. `H` is a long unique path
   that contains the canonical prefix, so the longest-first ordering replaces
   the full `H` and leaves plain-canonical paths untouched. Don't add a second
   overlapping map entry without checking ordering.
3. Parallel `docker run`/`rm` plus many `H` dirs — teardown is in `finally`;
   set `episode_concurrency` to match host resources.
4. `coverage.json` / `result.json` keep host `H` paths in their data (host-read
   only, never shown to the agent) — acceptable, same trade the plans named.

## 5. Tests (`test_main2_combined.py`, 13 passing)

- Phase 1: `test_collect_stubs_writes_episode_not_active_stub_dir`,
  `test_collect_unit_tests_writes_episode_and_resets_active` — episode harvested,
  active dir reset; no override kwargs.
- Translation: `test_containerize_text_maps_host_to_canonical` (docker maps
  `H->canonical`, local is a no-op).
- Phase 2: `test_per_episode_container_lifecycle` (create/teardown, mount
  `canonical_test_dir`, `path_map`, `testdir/`+`workspace/` on host, no
  `_trace_dataset` in container name) and
  `test_per_episode_container_parallel_runs_all_episodes` (N episodes ->
  N containers under `--episode-concurrency`).
- Unchanged: parse/dispatch, run_command local/docker, default stage order,
  select/materialize.

## 6. Files touched

- `pipeline/config.py` — `per_episode_container`, `container_image`,
  `container_run_args`, `episode_concurrency`, `path_map`.
- `pipeline/execution.py` — `containerize_text` + cwd/cmd rewrite in docker mode.
- `pipeline/common.py` — `run_agent` rewrites the prompt before writing it.
- `pipeline/container.py` (new) — per-episode `create`/`teardown` with ro source
  + rw `H` mounts.
- `pipeline/data_collection.py` — `_reset_dir`/`_harvest`/`_harvest_glob`,
  `_prepare_episode_testdir`, `_episode_container`, `_run_jobs`, and
  `_collect_one_stub`/`_collect_one_unit` workers (serial Phase 1 / parallel
  Phase 2).
- `main2.py` — matching CLI flags.

Stage3/stage5 internals, select/materialize, and the end-to-end spine are
unchanged; Phase 2 reuses the existing override params.

## 7. Bottom line (as built)

Both phases shipped:

- Phase 1 (default): pristine canonical collection, serial, reset between
  episodes. No container required (local or single shared docker container).
- Phase 2 (`--per-episode-container`): one fresh container per trace, immutable
  shared source ro-mounted, per-episode test dir `H` mounted at the canonical
  path, `H->canonical` rewrite via `containerize_text` + `run_agent`, parallel
  via `--episode-concurrency`. Agent only ever sees canonical pristine paths;
  host keeps episode-wise full test dirs under `_trace_dataset`.

Files touched: `pipeline/config.py`, `pipeline/execution.py`,
`pipeline/common.py` (run_agent prompt rewrite), `pipeline/container.py` (new),
`pipeline/data_collection.py`, `main2.py`, `test_main2_combined.py`.
Stage internals (stage3/stage5), select/materialize, and the end-to-end spine
are unchanged; Phase 2 reuses the existing override params rather than editing
stage signatures.

Tests (`test_main2_combined.py`, 13 passing) cover: flag parsing, run_command
local/docker, default stage order, stage dispatch, stub/unit episode harvest +
active reset (Phase 1), `containerize_text` mapping, per-episode container
lifecycle + mount/path_map, and parallel multi-episode collection.