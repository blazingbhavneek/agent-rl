# Explore-Branch Data Collection — Per-File Implementation Plan (v2, SDK-verified)

> Phase goal: **collect the maximum number of distinct trajectory variations per function.**
> Scoring is **secondary/optional** — never block collection on it.
> Everything below is grounded in the **actually installed SDK**
> (`local-sdk-test/node_modules/@cline/sdk/dist/index.js`, build 2026-05-27). Line numbers cited so you
> can re-read. Written so a small model can implement file-by-file with no guessing.

---

# PART A — Ground-truth SDK facts (do NOT re-derive; trust these)

The agent runtime class is `AgentRuntime` (exported as `Agent`). Verified methods/behavior:

```
new Agent(config)                       // config fields below
agent.run(input)      -> Promise<AgentRunResult>   // input: string | Message | Message[]
agent.continue(input) -> Promise<AgentRunResult>   // IDENTICAL to run() (both call execute())
agent.snapshot()      -> { agentId, status, iteration, messages, pendingToolCalls, usage, lastError, ... }
agent.restore(msgs)   -> void   // replaces state.messages + config.initialMessages, resets run state,
                                //   KEEPS tools/hooks/plugins/subscribers. (line 266003)
agent.subscribe(fn)   -> unsubscribeFn   // fn(event); events listed below
agent.abort(reason)   -> void   // aborts the in-flight run loop
```

**Constructor config fields that matter** (mirror what `ClineCore.create` was given in `agent.js` today):
`providerId, modelId, baseUrl, apiKey, cwd, enableTools, enableSpawnAgent, systemPrompt, maxIterations,
extraTools (array of createTool(...)), tools, toolPolicies, plugins, hooks, initialMessages, modelOptions`.
- `initialMessages` seeds the conversation at construction (line 265973: `state.messages =
  cloneMessages(config.initialMessages ?? [])`). **This is how we fork a branch into its own agent.**
- `enableTools: true` turns on the built-in coding tools. Built-in tool names (line ~263059,
  `DefaultToolNames`): `read_files, search_codebase, run_commands, fetch_web_content, apply_patch,
  editor, skills, ask_question, submit_and_exit`. `extraTools` adds custom ones
  (today: `ask_specialists`, `analyze_function_coverage`).

**Run loop behavior** (`execute()`, line ~266100 — this dictates the whole branch mechanism):
- Each iteration: generate ONE assistant message, push it, emit `assistant-message`.
- If that assistant message has **zero tool calls** → emit `turn-finished` then `finishRun("completed")`
  and **return** (UNLESS `completionPolicy.requireCompletionTool === true`, which instead injects a
  reminder user message and loops). **We will NOT enable requireCompletionTool**, so a tool-less turn
  ends the run. This is exactly our explore stop point.
- If there are tool calls → run them; if one is a terminal/completing tool (`submit_and_exit`) →
  `finishRun("completed")` and return; else loop.
- Hits `maxIterations` → throws "exceeded maxIterations".

**Events** from `subscribe` (verified names): `run-started, message-added, turn-started,
assistant-message, turn-finished, run-finished, run-failed`. Plus the streaming/UX names the current
`agent.js` already handles (`assistant-text-delta, tool-started, tool-finished, status-notice`). The
lower-level runtime uses `assistant-message` (whole message) and `turn-finished`.

**Hooks** (registered from `config.hooks`, line 266050): `beforeRun, afterRun, beforeModel, afterModel,
beforeTool, afterTool, onEvent`. A throw in `beforeTool` aborts that tool call.

**AgentRunResult shape** (verified, line ~266240 + the success path):
```
{ agentId, agentRole, runId, status: "completed"|"aborted"|"failed",
  iterations, outputText, messages: Message[], usage, error? }
```
`Message` = `{ role: "user"|"assistant"|"tool", content: Part[] }`,
`Part` ∈ `{type:"text",text}` | `{type:"tool-call",toolCallId,toolName,input}` |
`{type:"tool-result",toolCallId,output,isError}`. (helpers `textFromMessage`, `normalizeInput` at 265900.)

> **Consequence:** we do NOT need ClineCore sessions for collection. We build ONE `Agent` per node
> (root + each branch), each with its own `cwd`, drive it with `run`/`continue`, read `result.messages`,
> and fork by constructing a new `Agent` whose `initialMessages` is the permuted snapshot. Clean.

---

# PART B — The mechanism (explore + slot-1 permutation), in one screen

1. **Model rule (system prompt):** when multiple viable approaches exist, emit each in its own
   `<explore>…</explore>` tag (most-preferred first), then **stop the turn — call no tool**. You will get
   "Proceed with the first option." **Always implement the FIRST `<explore>` block.**
2. **Runner at an explore turn** (assistant message has ≥2 `<explore>` blocks `[B0,B1,…]`):
   - Root path keeps order → model takes `B0`.
   - Branch `i` (i=1..N-1, within budget): build a NEW agent in a COPIED workspace, seed it with the same
     history but the explore block **`Bi` moved to slot 0**, then send the identical
     `"Proceed with the first option."` → model takes `Bi`, believing it always took option 1.
3. **Why permute slot-1 instead of "take option 2":** every trajectory in the dataset is literally
   "emitted options, took the first." Canonical and branch paths are indistinguishable, so RL never sees a
   signal that "fewer options = better" → the model keeps proposing diverse options (exploration stays
   alive) while the collector harvests every option as a real explored path. Permutation = exploration
   augmentation done in the collection layer, not via model sampling.
4. **Self-contained options:** each `<explore>` block must NOT reference siblings (no "option 1/2",
   "above", "previous", "former/latter") so any permutation still reads as the model's own message.
5. Branches recurse (a branch can hit another explore), bounded by a global branch budget.

Constant used everywhere: `STEER = "Proceed with the first option."` (identical for root + all branches).

---

# PART C — EXISTING files: what they have, what to change

## C1. `agent.js`  (BIG change — swap the run engine, keep everything else)
**Has today:** `CONFIG`/`RUN_STATE`; helpers (`log`, `safeAsync`, `shortText`, `summarizeToolInput`, …);
`parseArgs` (flags `--folder --prompt --prompt-file --history --capture-raw-http-trace
--disable-streaming`); `RAGClient`; `createAnalyzeFunctionCoverageTool`; `createSearchTool`;
`buildModelOptions`; `answerWithSpecialist` (already uses `new Agent({...})` — proof the class works);
`createAskSpecialistsTool`; `buildSystemPrompt` (currently trivial); `buildToolPolicies`;
`createEventHandler` (turn logging); ClineCore plumbing (`subscribeIfPossible`, `pollSessionUntilDone`,
`waitForSessionEnd`, `getSessionId`, …); `writeHistory`; `collectFinalResult`; `main()` which does
`ClineCore.create(coreOptions)` + `cline.start(startOptions)` + wait + `writeHistory`.

**KEEP unchanged:** all tool builders, `RAGClient`, `buildModelOptions`, `createEventHandler` (reuse for
logging via `agent.subscribe`), `parseArgs` (add 3 flags, see below), `buildToolPolicies`,
`writeHistory` helper shape.

**CHANGE 1 — `buildSystemPrompt()`**: append the DECISION PROTOCOL block (verbatim):
```
DECISION PROTOCOL
When you hit a point with multiple distinct, viable approaches (how to stub a symbol, how to fix a
compile/link error, which test inputs to use, how to structure a test), do NOT silently pick one.
Instead:
  1. Emit each approach inside its own tag, on its own lines:
        <explore>
        <one self-contained approach. Never write "option 1/2", "above", or "previous".>
        </explore>
  2. List 2 to 4 approaches, the one you most prefer FIRST.
  3. After the final </explore>, STOP. Call NO tool in this message.
You will then receive exactly: "Proceed with the first option."
Always implement the FIRST <explore> block. The other blocks are alternatives you are NOT taking now.
Never collapse to a single option to save effort — list real alternatives whenever they exist.
When the whole task is done, call submit_and_exit.
```

**CHANGE 2 — `parseArgs()` / `CONFIG`**: add
- `--trace-dir <path>` → `CONFIG.traceDir` (where to write per-leaf `agent_history/` + `tree.json`).
- env `MAX_BRANCHES` → `CONFIG.maxBranches` (default 3).
- env `MAX_BRANCH_DEPTH` → `CONFIG.maxBranchDepth` (default 4).

**CHANGE 3 — add small pure helpers** (new functions in this file, see PART D for bodies):
`extractExploreBlocks(text)`, `hasUnconsumedExplore(message)`, `permuteToFront(messages, branchPointIndex,
i)`, `lastAssistantMessage(messages)`, `makeBranchDir(parentDir, i)`, `cpWorkspace(src,dst)`,
`buildAgentConfig(cwd, initialMessages)`, `writeLeaf(traceDir, result, promptText)`,
`appendTreeNode(traceDir, node)`.

**CHANGE 4 — add the branch-aware `beforeTool` guard plugin** (backup enforcement):
```
const exploreGuardHook = {
  beforeTool: async (ctx) => {
    // ctx exposes the current messages/state via snapshot; if the latest assistant message
    // still contains an un-acted <explore> block, throw to cancel this tool call.
    if (latestAssistantHasExplore(ctx)) {
      throw new Error("explore-pending: emit options then stop, do not call tools");
    }
  }
};
```
Pass via `hooks: exploreGuardHook` (or `plugins:[{hooks:exploreGuardHook}]`) in the agent config.
VERIFY the exact `ctx` shape (it carries a snapshot per the emit calls); if `ctx` lacks messages, keep a
module-level `EXPLORE_PENDING` boolean set by the driver instead.

**CHANGE 5 — replace `main()` body** (everything after building tools/handler) with the single-turn
branch-driver (PART D). Remove the ClineCore-only plumbing you no longer call (`pollSessionUntilDone`,
`waitForSessionEnd`, `collectFinalResult`, session-id helpers) — or keep them dead; not required.
The root run must STILL write the same `--history` file (`writeHistory`) so existing tooling keeps working;
branch leaves are written additionally under `--trace-dir`.

**`buildAgentConfig(cwd, initialMessages)` must mirror today’s `commonRuntimeFields`:**
```
{ providerId: CONFIG.providerId, modelId: CONFIG.modelId, baseUrl: CONFIG.baseUrl, apiKey: CONFIG.apiKey,
  cwd, enableTools: true, enableSpawnAgent: true, systemPrompt: buildSystemPrompt(),
  maxIterations: CONFIG.maxIter, extraTools: [askSpecialistsTool, analyzeFunctionCoverageTool],
  toolPolicies: buildToolPolicies(), hooks: exploreGuardHook,
  ...(initialMessages ? { initialMessages } : {}), ...buildModelOptions() }
```
> **VERIFY FIRST (probe in PART E #1):** that `new Agent(buildAgentConfig(cwd))` actually exposes
> `apply_patch`/`run_commands`/`read_files` and writes inside `cwd`. The specialist subagent used
> `tools:` with no built-ins; built-ins come from `enableTools:true`. If field names differ
> (`enable_tools` etc.), copy the exact keys from the current `commonRuntimeFields` object.

## C2. `pipeline/config.py`  (small additions)
**Has:** `PipelineConfig` dataclass (`source_dir, agent_js, system_json, func_docs_dir,
agent_timeout_sec, max_agent_iterations, max_compile_fix_attempts, max_test_attempts,
coverage_threshold, …, max_fix_attempts`); `derive_test_dir`, `derive_paths`.
**Add fields:** `data_root: Path = Path("/data")`, `max_branches: int = 3`, `max_branch_depth: int = 4`,
`retries_per_func: int = 4` (alias of `max_test_attempts` for collection), `docker_image: str =
"my_image"`, `enable_scoring: bool = False`, `stages: list[int] = field(default_factory=lambda:[3])`.
Keep everything else.

## C3. `pipeline/common.py`  (reuse helpers; `run_agent` gets a docker variant)
**Has:** IO helpers; `_read_json_loose`; source-file helpers; `ensure_wrap_flag`/`sync_wrap_flags`;
`_snapshot_dir`/`_restore_from_snapshot` (source guard); **`run_agent()`** (spawns `node agent.js`
locally, snapshots+restores source tree as a guard); `run_make_test()`; `check_function_coverage()`.
**Keep all of it.** `run_make_test` + `check_function_coverage` are the optional scorer later.
**Change:** factor the `node agent.js` command builder so `collect/runner.py` can wrap it in
`docker run`. Easiest: leave `run_agent` as-is for local use, and have `runner.py` build its own docker
command (it needs the container mounts anyway). Add `--trace-dir` + `MAX_BRANCHES`/`MAX_BRANCH_DEPTH`
to whichever path launches the agent.

## C4. `pipeline/analysis.py`  (reuse as-is)
**Has:** `run_or_load_analysis(cfg, out_path)` (runs `ProjectAnalyzer`, caches `analysis.json`);
`functions_leaf_first(analysis)` (orders deepest-callees first); `collect_stub_candidates(analysis)`.
**Use unchanged** from `collect/orchestrator.py` to get the leaf-first function list. Each function dict
has at least `id`, name, `source_file`, `start_line`, `end_line` (used by the coverage tool/scorer).

## C5. `pipeline/stage3_stubs.py` … `stage6_integrate.py`  (reuse the PROMPT text)
**Have:** per-stage agent orchestration + prompt strings (stage3 stubs via `ask_specialists`; stage4
minimal test; stage5 unit tests + semantic judge; stage6 integration). For collection we do NOT run these
file orchestrators — we only **reuse their prompt-building text** inside `collect/prompts.py`. Action:
lift the prompt strings (the big f-strings each stage sends to the agent) into `collect/prompts.py`
functions. Do not delete the stage files; they remain the reference + the non-branching pipeline.

## C6. `pipeline/semantic.py`  (optional, scoring-time only)
Semantic judge (`semantic_judge_min_score`). Not used during raw collection. Wire later inside the
optional scorer if you want quality labels. Ignore for now.

---

# PART D — NEW code: the branch driver inside `agent.js` (the hard part)

All of this lives in `agent.js` (or a sibling `explore.js` imported by it). Pseudocode is explicit;
translate 1:1.

### D1. Pure helpers
```
const STEER = "Proceed with the first option.";
const EXPLORE_RE = /<explore>([\s\S]*?)<\/explore>/g;

function extractExploreBlocks(text) {
  // returns array of { full: "<explore>...</explore>", inner: "...", start, end } in document order
  const out = []; let m;
  EXPLORE_RE.lastIndex = 0;
  while ((m = EXPLORE_RE.exec(text)) !== null)
    out.push({ full: m[0], inner: m[1].trim(), start: m.index, end: m.index + m[0].length });
  return out;
}

function lastAssistantMessage(messages) {
  for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === "assistant") return messages[i];
  return null;
}

function assistantText(msg) {
  return (msg?.content || []).filter(p => p.type === "text").map(p => p.text).join("");
}

// Build a permuted copy of the FULL messages array where, inside the branch-point assistant message,
// explore block index i is moved to the front. Everything else byte-identical.
function permuteToFront(messages, i) {
  const clone = structuredClone(messages);
  const a = lastAssistantMessage(clone);
  // operate on the single text part that holds the explore blocks
  const part = a.content.find(p => p.type === "text" && p.text.includes("<explore>"));
  const blocks = extractExploreBlocks(part.text);
  if (i <= 0 || i >= blocks.length) return clone;
  // reorder: [Bi, B0, B1, ... (without Bi)]
  const order = [i, ...blocks.map((_, k) => k).filter(k => k !== i)];
  // rebuild text: keep any prose before first block and after last block, swap the block region
  const firstStart = blocks[0].start, lastEnd = blocks[blocks.length - 1].end;
  const head = part.text.slice(0, firstStart);
  const tail = part.text.slice(lastEnd);
  const joiner = "\n";
  const newRegion = order.map(k => blocks[k].full).join(joiner);
  part.text = head + newRegion + tail;
  return clone;
}
```

### D2. The recursive driver
```
// budget is a shared mutable object: { remaining: CONFIG.maxBranches }
async function driveNode(agent, traceDir, promptOrNull, budget, depth, nodeMeta) {
  // depth 0 root: promptOrNull = CONFIG.prompt. Branches: agent already seeded via initialMessages,
  // promptOrNull = STEER (sent as first input to take the slot-0 option).
  attachLogging(agent);                       // agent.subscribe(createEventHandler())
  let res = await agent.run(promptOrNull);     // runs until: tool-less turn (explore or finish) OR submit_and_exit

  while (true) {
    const lastA = lastAssistantMessage(res.messages);
    const opts  = lastA ? extractExploreBlocks(assistantText(lastA)) : [];

    const isExploreTurn = opts.length >= 2;
    if (isExploreTurn && budget.remaining > 0 && depth < CONFIG.maxBranchDepth) {
      // ---- BRANCH POINT ----
      const snap = agent.snapshot();           // snap.messages == res.messages
      appendTreeNode(traceDir, { node_id: nodeMeta.id, branch_point_step: res.messages.length,
                                 branch_point_type: "explore", n_options: opts.length });
      for (let i = 1; i < opts.length; i++) {
        if (budget.remaining <= 0) break;
        budget.remaining -= 1;
        const bDir   = makeBranchDir(traceDir, i);          // mkdir + cpWorkspace(agent.cwd -> bDir)
        const bAgent = new Agent(buildAgentConfig(bDir, permuteToFront(snap.messages, i)));
        await driveNode(bAgent, bDir, STEER, budget, depth + 1,
                        { id: `${nodeMeta.id}_B${i}`, parent: nodeMeta.id,
                          alternative: opts[i].inner, slot1_from_index: i });
      }
      // parent continues on option 0 with the SAME steer
      res = await agent.continue(STEER);
      continue;
    }

    if (isTerminal(res)) break;                 // status completed/failed/aborted AND not an explore turn
    res = await agent.continue();               // safety: nudge once more (rare: tool-less non-explore)
    if (isTerminal(res)) break;
  }
  writeLeaf(traceDir, res, promptOrNull);       // run.json (full res.messages), run.prompt.txt, run.result.json
  return res;
}

function isTerminal(res) {
  return res && (res.status === "completed" || res.status === "failed" || res.status === "aborted");
}
```
Notes:
- A normal coding turn calls a tool, so `run()`/`continue()` keeps looping internally and only returns at
  (a) a tool-less turn (our explore stop, or a genuine stop) or (b) `submit_and_exit`. So the outer
  `while` mostly iterates once per *branch point*, not once per turn — cheap.
- If the model emits explore but ALSO sneaks a tool call, the `beforeTool` guard throws → the tool errors;
  the run continues and on the next tool-less turn we catch the explore. Belt + suspenders.
- **Sequential branches** (one container, recursion) — matches `data_collection_tree.md`. Each branch has
  its own `cwd` copy so file writes never collide.

### D3. `main()` wiring (replaces the ClineCore block)
```
await parseArgs();
process.chdir(CONFIG.workspace);
const rootDir = CONFIG.traceDir || CONFIG.workspace;       // root leaf dir
const rootAgent = new Agent(buildAgentConfig(CONFIG.workspace, /*initialMessages*/ undefined));
const budget = { remaining: CONFIG.maxBranches };
const res = await driveNode(rootAgent, rootDir, CONFIG.prompt, budget, 0, { id: "branch_root", parent: null });
await writeHistory(res);                                   // keep legacy --history output for root
process.exit(res.status === "completed" ? 0 : res.status === "failed" ? 1 : 124);
```

### D4. Trace writers (write into `--trace-dir`)
```
writeLeaf(dir, res, promptText):
  mkdir dir/agent_history
  write dir/agent_history/run.json        = JSON(res.messages)          // FULL messages (self-contained)
  write dir/agent_history/run.prompt.txt  = promptText ?? ""
  write dir/agent_history/run.result.json = { status: res.status, iterations: res.iterations,
                                              outputText: res.outputText, usage: res.usage }
appendTreeNode(dir, node):  // node = {node_id,parent_id,branch_point_step,branch_point_type,alternative,...}
  read-or-init dir/tree.json {retry_id, func_id, stage, nodes:[]}; push node; write back.
```
> **Simplification (intended):** store **FULL** `res.messages` per leaf, NOT deltas. Deltas save space but
> need parent+child stitching. Full messages are trivial to consume for SFT/DPO/GRPO later.

---

# PART E — NEW files: the Python `collect/` package

Directory `collect/` (sibling of `pipeline/`). Six files.

## E1. `collect/prompts.py`
**What:** per-stage prompt builders. **Does:** `build_prompt(stage:int, func:dict, artifacts_dir:Path,
cfg)->str`. Body = lift the prompt f-strings from `pipeline/stage{3..6}_*.py`. Must instruct the agent to:
work only inside its folder, use `ask_specialists` for signatures, run `make test`, and follow the
DECISION PROTOCOL when stuck. Returns one big string written to `run.prompt.txt`.
Signature per stage: `build_stage3(func, cfg)`, `build_stage4(func, artifacts, cfg)`, etc.; `build_prompt`
dispatches on `stage`.

## E2. `collect/runner.py`
**What:** one container run = one retry. **Does:**
```
run_retry(cfg, stage, func, retry_idx, retry_dir, prompt_text) -> dict:
  # retry_dir is on host: /data/traces/stage{N}/{func_id}/retry_{k}/
  write prompt to retry_dir/run.prompt.txt
  cmd = ["docker","run","--rm",
     "-v", f"{HOST_SRC}:/home/seigyo/src:ro",
     "-v", f"{retry_dir}:/home/seigyo/rl/tests/{cfg.process}:rw",
     "-e", f"MAX_ITERATIONS={cfg.max_agent_iterations}",
     "-e", f"MAX_BRANCHES={cfg.max_branches}",
     "-e", f"MAX_BRANCH_DEPTH={cfg.max_branch_depth}",
     "-e", f"MODEL_NAME={...}", "-e", f"OPENAI_BASE_URL={...}", "-e", f"RAG_SERVICE_URL={...}",
     cfg.docker_image, "node", "/home/seigyo/rl/agent.js",
       "--folder",   f"/home/seigyo/rl/tests/{cfg.process}/branch_root",
       "--trace-dir",f"/home/seigyo/rl/tests/{cfg.process}",
       "--prompt-file", f"/home/seigyo/rl/tests/{cfg.process}/run.prompt.txt",
       "--history",  f"/home/seigyo/rl/tests/{cfg.process}/branch_root/agent_history/run.json"]
  proc = subprocess.run(cmd, timeout=cfg.agent_timeout_sec, capture_output=True, text=True)
  return {"exit_code":proc.returncode, "timed_out":..., "stdout":..[:4000], "stderr":..[:4000]}
```
**Seed before run:** `trace.seed_retry_dir(retry_dir)` creates `branch_root/` with the stage scaffold
(stage1/stage2 outputs: `test_<process>.c`, `Makefile`, `_pipeline_context.json`) so the agent starts
from a compiling skeleton. For stages 4–6 also mount/copy the chosen best artifacts from earlier stages
(`-v best_artifacts/{func}:/home/seigyo/rl/artifacts/{func}:ro`).
Note: `agent.js` writes branch dirs (`branch_root`, `branch_B1`, …) + `tree.json` INTO the mounted
`/home/seigyo/rl/tests/{process}` which IS `retry_dir`. After the container exits they’re on the host.

## E3. `collect/trace.py`
**What:** filesystem + post-run harvest. **Does:**
- `retry_dir(cfg, stage, func_id, k) -> Path` = `cfg.data_root/traces/stage{N}/{func_id}/retry_{k}`.
- `seed_retry_dir(retry_dir, cfg, stage, func)`: mkdir; create `branch_root/`; run stage1 scaffold +
  stage2 makefile generators (reuse `pipeline.stage1_scaffold` / `pipeline.stage2_makefile` helpers, or
  copy a prebuilt skeleton) so `make test` exists before the agent starts.
- `harvest(retry_dir) -> list[dict]`: walk `retry_dir` for every dir containing `agent_history/run.json`
  (each = one leaf), read `tree.json`, return leaf records `{node_id, parent_node, trace_dir, status,
  messages_path}`. **No scoring here** unless `cfg.enable_scoring`.
- optional `score_leaf(leaf, func, cfg)`: run `pipeline.common.run_make_test(leaf.dir)` +
  `check_function_coverage(...)`, write `leaf.dir/score.json` `{make_ok, coverage_pct, reward}`.

## E4. `collect/registry.py`
**What:** append-only ledgers. **Does:** `append(path, row:dict)` (one JSON per line);
`read(path)->list[dict]`. Writes per-func `registry.jsonl` (in `stage{N}/{func_id}/`) and the global
`cfg.data_root/registry.jsonl`. Row schema = `data_collection_tree.md` §registry.jsonl
(`func_id,stage,retry,node_id,parent_node,trace_dir,status[,reward,make_ok,coverage_pct],collected_at`).
With scoring off, omit reward/coverage or set null.

## E5. `collect/tree.py`
**What:** read helper for the `tree.json` that `agent.js` writes. **Does:** `read(retry_dir)->dict`;
`leaves(tree)->list[node]` (nodes with no children); `best_leaf(tree)` (max reward if scored, else
`branch_root`). Python mostly READS; `agent.js` is the writer. Keep `write/merge` for the seed.

## E6. `collect/orchestrator.py`  (entry point)
**What:** the fan-out loop. **Does:**
```
main(source_dir, stages=[3], retries=4):
  cfg = PipelineConfig(source_dir=..., agent_js=..., data_root=Path("/data"), max_branches=3, ...)
  analysis = pipeline.analysis.run_or_load_analysis(cfg, cfg... analysis_path)
  funcs = pipeline.analysis.functions_leaf_first(analysis)
  for stage in stages:
    for func in funcs:
      for k in range(retries):
        rdir = trace.retry_dir(cfg, stage, func["id"], k)
        trace.seed_retry_dir(rdir, cfg, stage, func)
        prompt = prompts.build_prompt(stage, func, artifacts_dir(cfg,func), cfg)
        res = runner.run_retry(cfg, stage, func, k, rdir, prompt)
        leaves = trace.harvest(rdir)
        for leaf in leaves:
          if cfg.enable_scoring: trace.score_leaf(leaf, func, cfg)
          registry.append(per_func_registry(rdir), row_from(leaf, func, stage, k))
          registry.append(cfg.data_root/"registry.jsonl", row_from(leaf, func, stage, k))
      # pick best artifact for next stage (if scoring) else branch_root
      best = select_best(cfg, stage, func); copy_to(best, best_artifacts_dir(cfg, func, stage))
  print summary: total leaves collected per func.
```
No `ThreadPoolExecutor` yet — sequential. Parallelism is a later flip (K containers at once).

---

# PART F — Data file formats (reuse `data_collection_tree.md` exactly)

- **`tree.json`** (per retry dir) — written by `agent.js`. Add per branch node:
  `"alternative": "<inner text of the explore block this branch took>"`, `"slot1_from_index": i`.
- **`registry.jsonl`** (per func + global) — written by `collect/registry.py`.
- **`score.json`** (per leaf) — only if `enable_scoring`.
- **`agent_history/run.json`** — FULL `res.messages` array per leaf (decided: not deltas).

Directory layout: identical to `data_collection_tree.md` §"Directory Layout"
(`/data/traces/stage{N}/{func_id}/retry_{k}/{branch_root,branch_B1,…}/`, `tree.json`, `registry.jsonl`;
`/data/best_artifacts/{func_id}/`; global `/data/registry.jsonl`).

---

# PART G — Build + verification order (do in sequence; each step has a check)

1. **PROBE the Agent (most important).** Tiny scratch script: `new Agent(buildAgentConfig(tmpdir))`,
   `await agent.run("create file hi.txt containing 'x' using apply_patch, then submit_and_exit")`.
   PASS = `hi.txt` exists in `tmpdir` and result.status=="completed". This proves `enableTools:true`
   gives coding tools + honors `cwd`. If it fails, copy the exact field keys from today’s
   `commonRuntimeFields` (e.g. `enable_tools`, `workspace`, `localRuntime`) until tools appear.
2. **PROBE tool-less stop.** `await agent.run("Reply with the single word DONE and call no tool.")`.
   PASS = returns `status=="completed"` immediately (confirms a tool-less turn ends the run, so explore
   turns will end it). If it loops, you have `requireCompletionTool` on — turn it OFF, or switch the
   driver to abort-on-`assistant-message`-event.
3. **System prompt + helpers** (`buildSystemPrompt` block, `extractExploreBlocks`, `permuteToFront`).
   Unit-test `permuteToFront` on a hand-written 3-option message: assert block i lands first and head/tail
   prose unchanged.
4. **Driver, no branching yet** (`driveNode` with `budget.remaining=0`). Run a real stage-3 prompt in a
   scratch dir; confirm it behaves like today and writes `run.json`.
5. **Enable branching** (`budget=3`). Use a prompt that forces a decision; confirm N leaf dirs
   (`branch_root`, `branch_B1`, …) + a `tree.json`, and each branch’s `run.json` shows a DIFFERENT option
   in slot 0 followed by `STEER`.
6. **`beforeTool` guard** — verify a model that calls a tool right after `<explore>` gets the tool
   cancelled and still branches.
7. **`collect/` end-to-end** — one function, stage 3, `retries=2`, inside docker; confirm traces under
   `/data/traces/stage3/{func}/` and rows in both registries.
8. **Widen** — more funcs, stages 4–6 (mount best_artifacts), optional scoring, then parallel workers.

---

# PART H — Gotchas (each verified or flagged)

- **`continue` === `run`** (both `execute`): the FIRST `run`/`continue` you give a branch agent must be
  `STEER`, because the seeded `initialMessages` already ends at the explore turn; `STEER` is the user
  message that makes it act on slot 0. Do NOT re-send the original prompt to a branch.
- **`restore()` vs new `Agent`**: branches need a DIFFERENT `cwd`, so build a **new `Agent`** with
  `initialMessages=permuted` (don’t `restore()` the parent — that keeps the parent’s `cwd`). `restore()`
  is only handy if you ever reuse one agent on one workspace.
- **Byte-faithful permutation**: only the `<explore>` block substrings move; head/tail prose identical, or
  training data gets noisy. `permuteToFront` keeps `head`/`tail` slices verbatim.
- **Self-contained options**: enforce in prompt; optionally lint extracted `inner` for banned words
  (`option`, `above`, `previous`, `former`, `latter`) and, if found, `continue("Re-emit each <explore>
  option self-contained; do not reference the others.")` before branching.
- **Workspace copy** (`cpWorkspace`): copy the agent `cwd` tree EXCLUDING `agent_history/` (each branch
  gets its own) and regenerable build junk (`*.gcda *.gcno *.o`). Use `cp -r` then delete those.
- **Budget exhaustion** degrades to linear (no fork), never errors.
- **maxIterations**: keep it generous (25) — it bounds a SINGLE node’s internal turns, and branches each
  get a fresh counter (new Agent). It is NOT the branch budget.
- **Docker mount = retry_dir**: `agent.js --trace-dir` points at the in-container tests dir which is the
  bind-mounted host `retry_dir`; all branch dirs + `tree.json` land on the host automatically.
- **Source guard**: the old `run_agent` snapshotted the source tree to restore agent edits. In docker the
  source is mounted `:ro`, so that guard is unnecessary — read-only mount is the guard.

---

# PART I — One-paragraph summary

The installed SDK’s `Agent` exposes `run/continue/snapshot/restore/subscribe`, seeds a conversation from
`config.initialMessages`, gives coding tools when `enableTools:true`, honors `cwd`, and **ends a run on any
tool-less assistant turn**. So: teach the model (system prompt) to dump 2–4 self-contained `<explore>`
options then stop; that tool-less turn ends `run()`. In `agent.js`, a recursive `driveNode` inspects the
last assistant message — if it holds ≥2 `<explore>` blocks, it snapshots, and for each other option spins
up a NEW `Agent` in a COPIED workspace seeded with the history **permuted so that option sits in slot 0**,
then sends the identical "Proceed with the first option." Every branch is therefore a legitimate
"took option 1" trajectory with a different option in the lead — so RL never learns to emit fewer options,
and exploration stays alive. A thin Python `collect/` package fans this across functions (leaf-first from
`analysis.py`) × stages × retries via `docker run`, harvesting every leaf’s full `messages` + `tree.json`.
Scoring is an optional bolt-on; right now maximize the count of distinct explored trajectories.
