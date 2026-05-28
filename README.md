# Agent RL Workspace

This repo is set up as a root-level consumer project for a locally patched Cline SDK.

The shape is intentionally simple:

- `cline/` contains the SDK source you patch and build.
- `agent.js` is the real root-level agent entrypoint.
- `test.js` is the root-level regression test for exact raw request/response capture.

## Goal

Use a Cline SDK agent with custom tools while also persisting the exact provider request and response into assistant-message history metadata so the saved transcript can be used for training and analysis.

The trace is stored on the assistant message at:

- `message.metadata.rawHttpTrace.interactions[n].request.body`
- `message.metadata.rawHttpTrace.interactions[n].response.body`

Headers like `authorization` are redacted before persistence.

## Repo Layout

```text
.
├── agent.js
├── test.js
├── package.json
├── README.md
└── cline/
    └── sdk/
        ├── package.json
        └── packages/
```

## How It Works

The SDK patch lives in `cline/sdk/packages/llms/src/providers/ai-sdk.ts`.

It captures the real HTTP request and response at the provider `fetch` boundary after the SDK has already formatted:

- system prompt
- conversation messages
- tool schemas
- tool results
- file content

The runtime then attaches that raw trace to the final assistant message before history is written.

If `disableStreaming` is enabled, the SDK uses a non-streaming provider call so the stored response body is the final JSON response instead of SSE chunks.

## Build

Install the root dependency:

```bash
npm install
```

Install SDK workspace dependencies:

```bash
npm run sdk:install
```

Build the SDK:

```bash
npm run build:sdk
```

Or do both in one step:

```bash
npm run setup:sdk
```

`agent.js` and `test.js` import the SDK from:

```text
./cline/sdk/packages/sdk/dist/index.js
```

So after SDK changes, rebuild before running the root scripts.

## Run The Trace Test

```bash
node ./test.js
```

What it verifies:

- a root-level agent can use the locally built SDK
- raw trace metadata is persisted on the assistant message
- the stored request body exactly matches what hit the fake provider
- the stored response body exactly matches the provider JSON response
- streaming is disabled when requested, so the request does not send `"stream": true`

The test writes:

- `./history.json`

### Live OpenAI-Compatible Test

`test.js` can also run against a real OpenAI-compatible endpoint such as vLLM or NVIDIA's compatible API.

Set these environment variables:

- `TEST_BASE_URL`
- `TEST_API_KEY`
- `TEST_MODEL_ID`

Optional:

- `TEST_PROMPT`
- `TEST_SYSTEM_PROMPT`

Example:

```bash
TEST_BASE_URL="https://integrate.api.nvidia.com/v1" \
TEST_API_KEY="$NVIDIA_API_KEY" \
TEST_MODEL_ID="minimaxai/minimax-m2.7" \
node ./test.js
```

In live mode, the test still uses `providerId: "openai-compatible"` and still enables:

- `captureRawHttpTrace: true`
- `disableStreaming: true`

That matches the intended production shape for OpenAI-compatible backends like vLLM while ensuring the persisted trace stores a final JSON response body instead of SSE chunks.

## Run The Agent

Example:

```bash
node ./agent.js \
  --folder /absolute/path/to/workspace \
  --prompt "Explain the SDK extension points" \
  --history traces/run-01.json \
  --capture-raw-http-trace
```

Useful environment variables:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `MODEL_NAME`
- `PROVIDER_ID`
- `RAG_SERVICE_URL`
- `MAX_ITERATIONS`
- `CAPTURE_RAW_HTTP_TRACE`
- `DISABLE_STREAMING`

History is written inside the workspace passed to `--folder`.

Default:

- `<workspace>/agent_history.json`

Custom:

- `<workspace>/<value passed to --history>`
