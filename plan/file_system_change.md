# File System Change — What Went Wrong and What Was Actually Wanted

## What I Was Building

RL training data collection pipeline. The goal:

- Use an existing fragile CUnit test generation pipeline as an **environment**
- Wrap thin collection/selection/materialization around it
- Harvest RL trajectories at scale: `(state, action, reward)` tuples
- Product is a **corpus of agent episodes**, not a final CUnit file

## The Plans (plan/ folder)

### `file_progression.md`
Code-trace of what `main2.py` + `pipeline/` creates on disk. Accurate record of artifact layout, stage order, cache behavior, known quirks. Good reference document.

### `host_docker_seperaton.md`
Plan to move Python harness to host while keeping agent.js/make/gcc/gcov/gdb inside Docker. Key constraint: canonical paths everywhere — agent must never see host prefixes. Added `pipeline/execution.py` as command boundary. Good plan.

### `data_coll_stage_seperation.md`
Plan to reshape pipeline from end-to-end into stage-wise repeatable collection:
```
for stage in stages:
  for item in items:
    run N episodes
  select best → materialize → advance
```
Stages: prepare → collect-stubs → select-stubs → materialize-stubs → integrate-stubs → minimal-master → collect-unit-tests → select-unit-tests → materialize-unit-tests → integrate

### `final_plan.md`
Combined plan: execution boundary + stage separation. Defines the dataset layout, workspace overrides for stub/unit generators, select/materialize contract. Good on paper.

## Why The Plans Are Dogshit (In Practice)

Plans were correct. **Implementation broke the core isolation invariant.**

### What Actually Happened (tree.txt from work PC)

```
dio100d/_trace_dataset/episodes/stubs/DioGetPtr/<outer_episode>/
  ├── agent_history/
  ├── testdir/                          ← WRONG: container bind mount artifact
  │   ├── _inner_trace_dataset/         ← WRONG: recursive collection
  │   │   └── episodes/stubs/
  │   │       ├── DioGetPtr/            ← ALL 15 stubs re-collected inside
  │   │       ├── DioReqAns/
  │   │       ├── NcmConnect/
  │   │       └── ... (15 stubs total)
  │   ├── _stub_gen/
  │   └── agent_history/
  └── workspace/
```

**The bug:** `_collect_one_stub` in `data_collection.py` creates a per-episode container, binds the episode's `testdir/` at the canonical `tests/<proc>/` path inside the container, then calls `generate_stub_code` with `run_cfg` (which has `per_episode_container=False`). But `run_cfg` still has `trace_dataset_dirname="_trace_dataset"`. Inside the container, when `generate_stub_code` runs, it triggers the full `collect-stubs` stage again for ALL stub candidates — writing into `testdir/_inner_trace_dataset/` (the canonical path inside container = the episode's testdir on host). Recursive collection.

**Secondary:** The outer loop only got to the first stub candidate per process before timing out, because each "single stub" episode was actually running ALL stubs.

### Root Cause in One Line

The per-episode container ran `collect-stubs` for ALL items instead of `generate_stub_code` for ONE item.

## What I Actually Wanted

### Intended structure per process

```
tests/<process_name>/
  test_<process_name>.c
  Makefile
  _pipeline_context.json
  analysis.json

  _stub_gen/                        # legacy active (materialized selected)
    <safe_stub>/
      stub.c
      result.json

  _unit_tests/                      # legacy active (materialized selected)
    <safe_func>/
      test_<safe_func>.c
      Makefile
      coverage.json
      judge_verdict.json

  _trace_dataset/
    manifest.json
    episodes/
      stubs/
        <safe_stub>/
          <episode_id>/             # ONE stub, ONE attempt, NOTHING else
            metadata.json
            workspace/
              stub.c
              result.json
              stub_validate_main.c
            agent_history/
              _gen_stub_<safe>.prompt.txt
              _gen_stub_<safe>.result.json
      unit_tests/
        <safe_func>/
          <episode_id>/             # ONE function, ONE attempt
            metadata.json
            workspace/
              test_<safe_func>.c
              Makefile
              coverage.json
              judge_verdict.json
            agent_history/
              <safe>_test_*.prompt.txt
              <safe>_test_*.result.json
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

### Key invariants

1. **One episode = one item, one attempt.** Episode dir has metadata + workspace + agent_history. Nothing else.
2. **No nesting.** `_trace_dataset` never appears inside an episode dir.
3. **No `testdir/`.** Container bind mount path must not leak into episode artifacts.
4. **Collection doesn't touch legacy paths.** Only `materialize-*` stages copy into `_stub_gen/` and `_unit_tests/`.
5. **Per-episode container = isolation, not recursion.** Container runs `generate_stub_code` for ONE function, writes into the episode workspace, tears down. No stage dispatching inside the container.

### The fix needed

`_collect_one_stub` with `per_episode_container=True` should:
1. Create container with episode testdir mounted at canonical path
2. Call `generate_stub_code(run_cfg, canonical_test_dir, func_name, stub_dir_override=host_episode_stub_dir)` — for ONE func only
3. Harvest artifacts from host episode dir
4. Tear down container

It must NOT call `run_selected_stage(collect-stubs)` inside the container. The container is just an execution environment for one function's stub generation.

## What The Test Does

`test_pipeline.py` — real integration test. Mocks ONLY at subprocess boundary (`run_command`, `container.create/teardown`). All Python pipeline code runs as-is.

- `TestLocalCollectStubs` — local mode, shows correct flat structure
- `TestContainerCollectStubs` — container mode, **reproduces the bug**: `testdir/_inner_trace_dataset/` nesting visible in tree output
- `TestFullLocalFlow` — all stages end to end in local mode
- `TestStructuralInvariants` — cross-cutting negative assertions

Run:
```
python -m pytest test_pipeline.py -v -s
```

The container tree output directly matches `diff/tree.txt` from the work PC — confirming the bug is real and reproduced locally.
