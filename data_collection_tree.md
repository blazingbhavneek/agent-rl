# Data Collection Tree

## Environment

| Item | Value |
|---|---|
| Orchestrator | Python, runs on **host** machine |
| Agent runtime | Node.js (`agent.js`), runs **inside Docker container** |
| Docker image | `my_image` — contains gcc, gcov, make, do_mkmf, CUnit, Node.js, Python venv, production source build artifacts |
| Source truth | `/home/seigyo/src/{process}/` inside Docker — **never mutated** |
| LLM endpoint | `http://10.160.144.101:51027/v1` (OpenAI-compatible, env `OPENAI_BASE_URL`) |
| RAG service | `http://10.160.152.38:51029/` (env `RAG_SERVICE_URL`) |
| Python venv | `/home/seigyo/rl/.venv/bin/python` inside Docker |
| Func docs | `/home/seigyo/rl/moove_docs/func/` inside Docker |
| Host data root | `/data/` on host, bind-mounted per run |

---

## Pipeline Stages (existing `pipeline/` code)

### Stage 1 — Scaffold (`stage1_scaffold.py`)
Creates CUnit test file skeleton with section markers. Includes all production `.c` files via absolute `#include` paths. Defines `main` rename so CUnit owns `int main(void)`. Idempotent — updates missing markers on re-run.

### Stage 2 — Makefile (`stage2_makefile.py`)
Runs `do_mkmf` to generate base Makefile. Appends test target block with gcov flags, WRAP_FUNCS, coverage-test rule. Parses production Makefile for CFLAGS/LDFLAGS and merges them. Writes `_pipeline_context.json` caching flags + source paths.

### Stage 3 — Stubs (`stage3_stubs.py`)
For each function: agent generates `__wrap_*` linker wrapper stubs for external dependencies. Validates stub compiles. Integrates into test file under `/* === Linker Wrapper Stubs === */` marker. Retries up to `max_stub_gen_retries=3`. Uses RAG doc lookup via `ask_specialists` tool to find correct signatures.

### Stage 4 — Minimal Test (`stage4_minimal.py`)
Agent writes the smallest possible test that achieves non-zero gcov coverage for the target function. Success = `make test` passes + `coverage_pct > 0`. Up to `max_minimal_test_attempts=5`.

### Stage 5 — Unit Tests (`stage5_unit_tests.py`)
Agent writes comprehensive unit tests for isolated function. Target: `coverage_pct >= coverage_threshold` (default 80%). Parallel workers (`max_unit_test_workers=4`). Semantic judge validates test quality (`semantic_judge_min_score=75`). Compile fix loop up to `max_fix_attempts=20`. Backs up best version seen.

### Stage 6 — Integration (`stage6_integrate.py`)
Extracts test functions + `CU_add_test` calls from per-function unit test files. Merges into master `test_{process}.c` under correct markers. Runs `make test` on master. Compile fix loop if needed.

---

## Cline SDK Features Used

### `Agent.snapshot()` → `AgentRuntimeStateSnapshot`
Captures full runtime state (messages + metadata) at any point during a run. Used at branch points to fork.

### `Agent.restore(messages)`
Replaces conversation history, resets runtime state. Preserves tools, hooks, model, subscribers. Used to initialize a branch agent from a snapshot.

### `Agent.continue(input?)`
Resumes execution with optional steering message. Called after `restore()` on branch agent — e.g. `branchAgent.continue("use approach B")`.

### `agent.subscribe(listener)`
Streaming event listener. Used to detect `<explore>` tags in `assistant-text-delta` events, `apply_patch` failures in `tool-finished` events, `make test` failures in `run_commands` tool results.

### `AgentRunResult.messages`
Full message history returned directly from `agent.run()`. No need to re-read history JSON — this IS the trajectory.

### Plugin `beforeTool(context)` hook
Intercepts every tool call before execution. Records tool name + input. A thrown error here counts as tool failure (useful for forced branching experiments). Used for DPO data: capture every `apply_patch` attempt and its outcome.

### `ClineCore.send({ sessionId, prompt })`
Injects follow-up message into running session. Alternative to branching — steer existing session mid-run.

### `AgentRunResult`
```typescript
{
  status: "completed" | "aborted" | "failed"
  iterations: number
  outputText: string
  messages: readonly AgentMessage[]   // full trajectory
  usage: AgentUsage                   // input/output tokens
  error?: Error
}
```

---

## Isolation Model

```
HOST (orchestrator.py runs here)
  │
  │  docker run --rm
  │    -v /root_fs/home/seigyo/src:/home/seigyo/src:ro       ← source, read-only
  │    -v /data/traces/s{N}/{func}/{retry_K}:/home/seigyo/rl/tests/{process}:rw
  │    -e MAX_ITERATIONS=25
  │    -e OPENAI_BASE_URL=...
  │    -e MODEL_NAME=...
  │    my_image
  │    node /home/seigyo/rl/agent.js
  │      --folder /home/seigyo/rl/tests/{process}
  │      --prompt-file /home/seigyo/rl/tests/{process}/agent_history/run.prompt.txt
  │      --history  /home/seigyo/rl/tests/{process}/agent_history/run.json
  │
  ▼
DOCKER CONTAINER
  /home/seigyo/src/{process}/        ← real source (ro mount)
  /home/seigyo/rl/tests/{process}/   ← branch dir (rw mount)
    branch_root/
    branch_B1/                       ← created at branch point (cp -r)
    branch_B1_C/                     ← nested branch
    tree.json
  /home/seigyo/rl/.venv/             ← inside image
  /home/seigyo/rl/moove_docs/        ← inside image
```

Agent always sees `/home/seigyo/...` — unaware of `/data/traces/...` on host.

---

## Full Collection Flow

```
ROOT SOURCE (never touched)
/home/seigyo/src/moove/
════════════════════════════════════════════════════════════════════════

SOURCE: moove    STAGE: 3 (stubs)

Analysis → function list leaf-first: [func_1, func_2, func_3, ...]

  func_1 → N=4 retries (each independent, sequential by default)
  │
  ├── retry_0 ── docker run ──► agent runs
  │                               step 1: reads source
  │                               step 2: writes stub attempt
  │                               step 3: make → FAIL
  │                               step 4: <explore>A: weak symbol | B: wrap flag
  │                                         BRANCH POINT
  │                                         snapshot S1
  │                                         cp branch_root/ → branch_B1/
  │                                         agent_A continues in branch_root/
  │                                         agent_B starts  in branch_B1/
  │                                         (both run inside same container)
  │                               agent_A: make PASS cov=72%
  │                               agent_B: make PASS cov=88%
  │                             container exits
  │                             → 2 leaf traces
  │
  ├── retry_1 ── docker run ──► no branch → 1 leaf trace  (make FAIL, cov=0%)
  │
  ├── retry_2 ── docker run ──► 2 branch points → 3 leaf traces
  │
  └── retry_3 ── docker run ──► no branch → 1 leaf trace  (make PASS, cov=61%)

  total for func_1: 7 traces

  /data/traces/stage3/func_1/
    retry_0/
      branch_root/
        test_moove.c
        Makefile
        _pipeline_context.json
        agent_history/
          run.json          ← AgentRunResult.messages (full trajectory)
          run.prompt.txt    ← prompt used
          run.result.json   ← exit_code, elapsed, timed_out
        score.json          ← make_ok, coverage_pct, reward
      branch_B1/
        test_moove.c        ← copy of branch_root at branch point + new writes
        Makefile
        agent_history/
          run.json          ← messages DELTA from branch point onward
          run.prompt.txt
          run.result.json
        score.json
      tree.json             ← branch structure (see Tree Format below)
    retry_1/
      branch_root/
        ...
    retry_2/
      branch_root/
      branch_B1/
      branch_B1_C/
      tree.json
    retry_3/
      branch_root/
        ...
    registry.jsonl          ← all 7 leaf traces for func_1 stage3

════════════════════════════════════════════════════════════════════════

  func_2 → N=4 retries → stored at /data/traces/stage3/func_2/
  func_3 → N=4 retries → stored at /data/traces/stage3/func_3/
  ...

════════════════════════════════════════════════════════════════════════

STAGE 3 COMPLETE FOR SOURCE moove
  → for each func: pick best trace (max reward) → best_artifacts/func_X/stage3_stubs.c
  → funcs with reward=0 across all traces: retry more OR skip to stage 4 anyway
  → start stage 4 queue

════════════════════════════════════════════════════════════════════════

STAGE 4 (minimal test) — same retry pattern
  each container additionally mounts:
    best_artifacts/{func_id}/:ro → /home/seigyo/rl/artifacts/{func_id}/
  agent sees stubs already present, builds minimal test on top

STAGE 5 (unit tests) — same retry pattern
  mounts: stage3 stubs + stage4 scaffold

STAGE 6 (integration) — same retry pattern
  mounts: stage5 unit tests
  reward = master test suite passes

════════════════════════════════════════════════════════════════════════

RE-RUNNING SAME SOURCE
  run collect.py again on same source → new timestamps → more traces
  registry grows, best selection always picks from full history
  no limit on how many times a source is collected
```

---

## Branch Points (what triggers a branch)

| Event | How detected | Branch cost |
|---|---|---|
| `<explore>A\|B\|C</explore>` in assistant text | `assistant-text-delta` event, regex match | 1 branch per alternative |
| `apply_patch` tool returns error | `tool-finished` event, error in result | 1 branch (alt patch strategy) |
| `run_commands make test` returns non-zero | `tool-finished` event, parse make output | 1 branch (alt fix strategy) |
| `make test` passes but coverage < threshold | scored after run, triggers next retry | no mid-run branch, drives retry count |

Branch budget per container run: configurable (default 3 total branches). When exhausted, agent continues linearly.

---

## Data Formats

### score.json (per leaf trace)
```json
{
  "make_ok": true,
  "coverage_pct": 88.0,
  "reward": 0.88,
  "compile_errors": [],
  "n_attempts": 1,
  "elapsed_sec": 142.3,
  "scored_at": "2026-06-09T14:32:00Z"
}
```

### tree.json (per retry dir)
```json
{
  "retry_id": "retry_0",
  "func_id": "foo_parse",
  "stage": 3,
  "nodes": [
    {
      "node_id": "branch_root",
      "parent_id": null,
      "branch_point_step": null,
      "branch_point_type": null,
      "reward": 0.72
    },
    {
      "node_id": "branch_B1",
      "parent_id": "branch_root",
      "branch_point_step": 4,
      "branch_point_type": "explore",
      "alternative": "use wrap flag instead of weak symbol",
      "reward": 0.88
    }
  ]
}
```

### registry.jsonl (per func per stage, append-only)
```jsonl
{"func_id":"foo_parse","stage":3,"retry":0,"node_id":"branch_root","parent_node":null,"trace_dir":"traces/stage3/foo_parse/retry_0/branch_root","reward":0.72,"make_ok":true,"coverage_pct":72.0,"collected_at":"2026-06-09T14:32:00Z"}
{"func_id":"foo_parse","stage":3,"retry":0,"node_id":"branch_B1","parent_node":"branch_root","trace_dir":"traces/stage3/foo_parse/retry_0/branch_B1","reward":0.88,"make_ok":true,"coverage_pct":88.0,"collected_at":"2026-06-09T14:32:01Z"}
{"func_id":"foo_parse","stage":3,"retry":1,"node_id":"branch_root","parent_node":null,"trace_dir":"traces/stage3/foo_parse/retry_1/branch_root","reward":0.0,"make_ok":false,"coverage_pct":0.0,"collected_at":"2026-06-09T14:40:00Z"}
```

### agent_history/run.json
Direct `AgentRunResult.messages` array from Cline SDK. Full sequence of:
- `{role: "user", content: [{type: "text", text: "..."}]}`
- `{role: "assistant", content: [{type: "text", text: "..."}, {type: "tool_use", name: "apply_patch", input: {...}}]}`
- `{role: "tool", content: [{type: "tool_result", content: "..."}]}`

For branch nodes: messages start from branch point onward (delta only). Full trajectory = parent `run.json` messages `[0..branch_point_step]` + this file.

---

## Directory Layout (complete)

```
/data/
  traces/
    stage3/
      {func_id}/
        retry_0/
          branch_root/
            test_{process}.c
            Makefile
            _pipeline_context.json
            *.gcov  *.gcno  *.gcda
            agent_history/
              run.json
              run.prompt.txt
              run.result.json
            score.json
          branch_B1/            (if branched)
            ...
          tree.json
        retry_1/
          branch_root/
        ...
        registry.jsonl
    stage4/
      {func_id}/
        ...
    stage5/
      {func_id}/
        ...
    stage6/
      {func_id}/
        ...

  best_artifacts/
    {func_id}/
      stage3_stubs.c
      stage4_scaffold.c
      stage5_unit.c

  registry.jsonl              ← global cross-stage registry
```

---

## Files to Build

```
collect/
  orchestrator.py   entry point: iterate sources → functions → stages → retries
  runner.py         docker run wrapper + in-process branch manager (Agent.snapshot/restore)
  trace.py          mkdir, seed branch dir, cp-at-branch-point, score after container
  registry.py       append/read registry.jsonl (per-func and global)
  prompts.py        per-stage prompt builders (takes func dict + stage + artifacts path)
  tree.py           tree.json read/write, branch node tracking
```

---

## Future (future_to_do.md)

- Parallel workers: `ThreadPoolExecutor` across functions (K containers at once)
- DPO extraction: scan `run.json` for `apply_patch FAIL → retry → success` pairs
- GRPO training: group traces by `(func_id, stage, retry)` sharing same prompt, normalize rewards
- Dynamic budget: smarter branch point scoring (not all branch points equal)
- Cross-stage chaining: start stage 4 per-func as soon as its stage 3 completes (pipeline overlap)
- Scorer container: separate Docker run after agent container exits to re-run make test cleanly
