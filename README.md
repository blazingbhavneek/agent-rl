# Agent RL Workspace

This repo has three moving parts:

- `cline/sdk/`: the real Cline SDK workspace. This is where the SDK source lives and where the raw request/response capture change was made.
- `local-sdk-test/`: a separate consumer project used to prove that a locally built `@cline/sdk` package works when installed from disk like a real downstream app.
- `agent.js`: a small CLI agent that uses `@cline/sdk`, queries a RAG service, and writes conversation history to disk.

The raw trace capture lives in the SDK provider/runtime path, not in `agent.js`. `agent.js` just opts into it when you ask for it.

## What Changed

The SDK now persists the exact provider request and exact provider response text in assistant-message history metadata:

- request body: `message.metadata.rawHttpTrace.interactions[n].request.body`
- response body: `message.metadata.rawHttpTrace.interactions[n].response.body`

This is captured at the HTTP `fetch` boundary in the SDK, so it includes the real provider wire format after Cline has already formatted system prompts, user messages, tool results, file reads, and other content for the model.

Two important notes:

- authorization-style headers are redacted on purpose
- the trace is opt-in and only appears when you enable `captureRawHttpTrace`
- if you want the raw trace to store one final JSON response body instead of SSE chunks, also enable `disableStreaming`

## Prerequisites

- Node.js `>= 22`
- Bun `1.3.13`

The SDK workspace itself is Bun-based. The local consumer test project is a normal npm project on purpose.

## Repo Layout

```text
.
├── agent.js
├── README.md
├── cline/
│   └── sdk/
│       ├── package.json
│       └── packages/
└── local-sdk-test/
    ├── build-local-sdk-bundle.mjs
    ├── package.json
    ├── test.js
    └── history.json
```

## Build The SDK Workspace

If you want to build the actual Cline SDK packages:

```bash
cd cline/sdk
bun install
bun run build:sdk
```

Useful variants:

- build packages plus the CLI app:

```bash
cd cline/sdk
bun run build
```

- run the full package test suite:

```bash
cd cline/sdk
bun run test
```

- run the unit-oriented suite:

```bash
cd cline/sdk
bun run test:unit
```

- run a focused package test:

```bash
cd cline/sdk
bun -F @cline/agents test
```

The raw trace implementation was added in these SDK source files:

- `cline/sdk/packages/llms/src/providers/ai-sdk.ts`
- `cline/sdk/packages/agents/src/agent-runtime.ts`
- `cline/sdk/packages/shared/src/agent.ts`

## Why `local-sdk-test/` Exists

`cline/sdk/` is a Bun workspace for SDK development. That is not the same thing as a real downstream app installing `@cline/sdk`.

`local-sdk-test/` exists to test the downstream case:

- bundle a local `@cline/sdk` package from the source in `cline/sdk/`
- install that package into a separate npm project
- run a real `Agent`
- verify that history includes `rawHttpTrace`

This catches packaging and runtime issues that do not show up when everything runs from source inside the workspace.

## Set Up The Local Consumer Test Project

From a clean checkout:

```bash
cd local-sdk-test
npm run setup-local-sdk
```

That does four things:

1. prepares `.local-sdk-builder/`
2. installs the builder dependencies inside `.local-sdk-builder/`
3. bundles a local `@cline/sdk` package from `cline/sdk/packages/`
4. installs that local package into `local-sdk-test/node_modules/`

If you already bootstrapped once and only changed SDK source files, the faster rebuild path is:

```bash
cd local-sdk-test
npm run bundle-local-sdk
npm run install-local-sdk
```

## Run The Raw Trace Integration Test

Once `local-sdk-test/` is set up:

```bash
cd local-sdk-test
npm test
```

What the test does:

- starts a fake OpenAI-compatible HTTP server
- installs and uses the locally built `@cline/sdk`
- runs an `Agent`
- writes `history.json`
- asserts that the persisted assistant message contains:
  - the exact request JSON body sent to the server
  - the exact SSE response body returned by the server

The test file is:

- `local-sdk-test/test.js`

The history file written by the test is:

- `local-sdk-test/history.json`

If the test passes, you should see:

```text
PASS raw trace persisted to /.../local-sdk-test/history.json
```

## Where To Find History

There are two history locations in this repo, depending on what you ran.

### 1. Local SDK Integration Test

The test writes:

- `local-sdk-test/history.json`

The raw trace is on the assistant message:

```json
{
  "role": "assistant",
  "metadata": {
    "rawHttpTrace": {
      "version": 1,
      "providerId": "openai-compatible",
      "modelId": "demo-model",
      "interactions": [
        {
          "request": {
            "body": "{... exact JSON sent to the provider ...}"
          },
          "response": {
            "body": "data: {... exact SSE returned by the provider ...}"
          }
        }
      ]
    }
  }
}
```

### 2. `agent.js`

`agent.js` changes directory into the target workspace before it writes history. That means the history file is written inside the folder you pass to `--folder`, not next to `agent.js`.

Default:

- `<your-workspace>/agent_history.json`

Custom:

- `<your-workspace>/<whatever-you-passed-to---history>`

Example:

```bash
node ./agent.js \
  --folder /tmp/my-workspace \
  --prompt "Explain the extension points in this SDK" \
  --history traces/run-01.json \
  --capture-raw-http-trace
```

That writes history here:

- `/tmp/my-workspace/traces/run-01.json`

## Running `agent.js`

`agent.js` is just a script file in this repo root. The repo root is not a preconfigured npm app, so you have two ways to use it.

### Option A: Run It From Your Own Consumer Project

Create a normal Node project anywhere you want and install:

```bash
npm install @cline/sdk zod
```

Then copy `agent.js` there or import the same logic into your app.

### Option B: Run The Included `agent.js` From This Repo Root

If you want to run the root `agent.js` directly from this repo, initialize the repo root as a small Node consumer project first:

```bash
npm init -y
npm install ./local-sdk-test/.local-sdk-builder/package zod
```

Then run the agent:

```bash
node ./agent.js \
  --folder /absolute/path/to/workspace \
  --prompt "Answer a question about the library"
```

If you want raw wire capture in the saved history:

```bash
node ./agent.js \
  --folder /absolute/path/to/workspace \
  --prompt "Answer a question about the library" \
  --capture-raw-http-trace \
  --disable-streaming
```

You can also enable it with an environment variable:

```bash
CAPTURE_RAW_HTTP_TRACE=1 node ./agent.js \
  --folder /absolute/path/to/workspace \
  --prompt "Answer a question about the library"
```

And for a final non-SSE provider response body in history:

```bash
CAPTURE_RAW_HTTP_TRACE=1 DISABLE_STREAMING=1 node ./agent.js \
  --folder /absolute/path/to/workspace \
  --prompt "Answer a question about the library"
```

### `agent.js` Environment Variables

`agent.js` supports these environment variables:

- `PROVIDER_ID`
- `MODEL_NAME`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `RAG_SERVICE_URL`
- `MAX_ITERATIONS`
- `CAPTURE_RAW_HTTP_TRACE`
- `DISABLE_STREAMING`

Defaults in `agent.js` currently point at:

- an OpenAI-compatible model endpoint
- a separate RAG service for doc search / cache lookup

If those services are not running, the script will fail or the specialist lookups will return errors.

## Integrating Raw Trace Capture Into Your Own Agent

The only thing your agent needs to do is opt in through `modelOptions.metadata`:

```js
import { Agent } from "@cline/sdk";

const agent = new Agent({
  providerId: "openai-compatible",
  modelId: "demo-model",
  baseUrl: "http://127.0.0.1:8000/v1",
  apiKey: "EMPTY",
  systemPrompt: "You are a helpful assistant.",
  modelOptions: {
    metadata: {
      captureRawHttpTrace: true,
      disableStreaming: true,
    },
  },
});
```

That is enough for the SDK to:

- capture the exact HTTP request body sent to the provider
- capture the exact HTTP response body returned by the provider
- use a non-streaming provider call so the stored response body is a final JSON payload instead of chunked SSE
- attach that trace to the assistant message metadata in `result.messages`

The provided `agent.js` now supports this through:

- `--capture-raw-http-trace`
- `--disable-streaming`
- `CAPTURE_RAW_HTTP_TRACE=1`
- `DISABLE_STREAMING=1`

## How To Read The Trace Programmatically

When a run finishes:

```js
const result = await agent.run("Reply with OK.");
const assistant = [...result.messages].reverse().find((m) => m.role === "assistant");
const trace = assistant?.metadata?.rawHttpTrace;

console.log(trace?.interactions?.[0]?.request?.body);
console.log(trace?.interactions?.[0]?.response?.body);
```

## Quick Start Summary

If you only want to validate the raw trace feature end to end:

```bash
cd local-sdk-test
npm run setup-local-sdk
npm test
```

If you want to build and test the real SDK workspace:

```bash
cd cline/sdk
bun install
bun run build:sdk
bun run test
```

If you want to use the included research agent:

```bash
npm init -y
npm install ./local-sdk-test/.local-sdk-builder/package zod
node ./agent.js \
  --folder /absolute/path/to/workspace \
  --prompt "What are the main extension points in this codebase?" \
  --capture-raw-http-trace
```
