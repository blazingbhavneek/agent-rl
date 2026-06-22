# Prompt: Produce a Technical-Strategy Evaluation Report

You are writing a concise internal report for an engineering manager. The report
must explain the results of an evaluation of three C code-generation approaches:

- Cloud model with RAG
- Local model with RAG
- RL-trained direct model

The reader is technical enough to understand model quality, retrieval, latency,
and infrastructure tradeoffs. They do not need implementation-level details such
as JSONL schemas, prompt wording, retry file names, or judge internals.

## Input material

Use only the evidence supplied below. Do not invent metrics, dollar amounts,
performance claims, or conclusions that are not supported by the data.

<evaluation_data>
Paste the evaluation summary JSON, per-setup score JSONLs, selected plots, cost
inputs, and any relevant notes here.
</evaluation_data>

<experiment_context>
Paste model identifiers, task-set description, evaluation dates, task count,
known environment constraints, and any cloud-budget interruption notes here.
</experiment_context>

## Evaluation context

The evaluation uses the same task set for each setup. RAG-based setups retrieve
relevant library/API context before code generation. The RL baseline generates
directly. Outputs are filtered for coherent C source before scoring so provider
error messages, empty output, and tool/stdout transcripts are retried rather
than treated as code.

Valid outputs are evaluated independently against the task and its reference
answer. The scores are:

- `task_correctness`: requested behavior and required API use.
- `reference_similarity`: semantic alignment with intended/reference behavior.
- `code_quality`: clarity, safety, and implementation quality.
- `overall`: `0.45 * task_correctness + 0.40 * reference_similarity + 0.15 * code_quality`.

Score distributions, means, and standard deviations describe both typical quality
and consistency. Compilation/verification outcomes are supporting reliability
evidence; do not overstate them as full functional proof if the supplied data
does not establish that.

## Required report structure

Write the report using these sections. Target roughly 1,000–1,500 words unless
the supplied evidence is too limited to support that length.

### 1. Objective

State the engineering decision: identify the most practical way to improve
task-specific C code generation while balancing quality, reliability, cost,
delivery time, and engineering effort.

### 2. Experiment setup

Briefly explain:

- the three approaches evaluated;
- that they used the same task set and reference behavior;
- the role of RAG for library/API context;
- the types of evidence collected: quality, consistency, completion/retry,
  latency/tokens if present, and operational interruptions.

Do not describe file formats, implementation details, or prompt templates.

### 3. Results and observed trends

Use the supplied charts and statistics. Include a compact comparison table:

| Setup | Judged tasks | Mean overall | Std. deviation | Main observed strength | Main observed limitation |
|---|---:|---:|---:|---|---|

Then interpret the results:

- Which setup has the strongest average answer quality?
- Which setup is most consistent or variable?
- How do correctness, reference similarity, and code quality differ?
- Are there meaningful completion, retry, latency, or token-use trends?
- If coverage differs because a cloud budget stopped a run, say so prominently
  and avoid claiming a definitive winner from unequal samples.

Use specific values from the data. If a value is absent, say it was not measured.

### 4. Technical interpretation

Explain the findings at an engineering level:

- What RAG appears to help with: task-specific APIs, library context, and
  grounding generation in relevant information.
- Distinguish retrieval quality from generation/reasoning quality.
- Explain that coherence retries protect the benchmark from counting provider
  failures or transcript fallbacks as model answers, but do not improve a
  coherent answer's correctness.
- Explain what the RL result indicates about specialization potential without
  overstating the current evidence.

### 5. Cost, effort, and operational tradeoffs

Include this table and fill it only with evidence-backed statements:

| Approach | Technical value | Primary cost/effort | Operational constraint |
|---|---|---|---|
| Cloud + RAG |  | Recurring API/token spend |  |
| Local + RAG |  | Hardware, serving, maintenance, electricity |  |
| RL direct |  | GPUs, training, environment/data/reward engineering |  |
| Hybrid cloud + local RAG |  | Integration plus reduced recurring API use |  |

Cover these practical points:

- Cloud models provide a fast path to strong reasoning but have recurring
  token cost, budget limits, and provider dependency.
- Local inference has infrastructure and maintenance costs but low marginal
  generation cost after deployment and greater control over data/context.
- Agentic RL is materially more complex than RAG: it requires task
  environments, automated rewards/verification, curated failures, GPU capacity,
  repeated experiments, and experienced engineering/research staffing.
- RAG is comparatively faster to deploy and can improve task grounding without
  retraining a model.

Do not make up dollar estimates. If actual pricing, GPU, or staffing numbers are
provided, use them and state the assumptions.

### 6. Hybrid architecture option

Assess this concrete option:

- Run retrieval, API/document lookup, filtering, and context compression locally.
- Send only relevant context to a stronger cloud model for difficult tasks.
- Use local generation for lower-risk or high-volume tasks when its measured
  quality is adequate.
- Route tasks to cloud when task complexity or low confidence warrants stronger
  reasoning.

Explain whether the supplied data supports this direction. Describe it as a
hypothesis to validate when there is not enough routing or token data.

### 7. Recommendation and next steps

Make a clear, staged recommendation:

1. Name the best current baseline for quality, with an evidence-based caveat.
2. State the near-term RAG/local-context work that is justified.
3. State the next measurement needed for a hybrid design: quality retention,
   cloud-token reduction, latency, and routing threshold.
4. State whether RL should be treated as a near-term production path or a longer
   research investment, based on the measured advantage over simpler approaches.

Include 3–5 concrete next experiments or decisions. Examples include a balanced
cloud-budget rerun, larger task coverage, category-level breakdown, human review
of borderline cases, or hybrid routing/token-ablation measurement.

## Writing rules

- Lead with the conclusion, but preserve uncertainty where data is incomplete.
- Use plain engineering language. Avoid investor language, hype, and generic AI claims.
- Separate observed facts from interpretation and recommendations.
- Do not claim statistical significance unless the supplied data includes an
  appropriate analysis.
- Do not call a setup "cheaper" without stating whether this means recurring
  cost, marginal inference cost, total ownership cost, or engineering effort.
- Prefer a few evidence-rich charts/tables over implementation detail.
- End with a short decision statement: what to use now, what to measure next,
  and what investment to defer or pursue.
