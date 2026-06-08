#!/usr/bin/env node

import {
  Agent,
  ClineCore,
  createTool,
} from "./cline/sdk/packages/sdk/dist/index.js";
import { z } from "zod";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import process from "node:process";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, writeFile, readFile } from "node:fs/promises";

const execFileAsync = promisify(execFile);

const COVERAGE_PYTHON = "/home/seigyo/rl/.venv/bin/python";

const CONFIG = {
  providerId: process.env.PROVIDER_ID || "openai-compatible",
  modelId: process.env.MODEL_NAME || "openai/gpt-oss-120b",
  baseUrl: process.env.OPENAI_BASE_URL || "http://10.160.144.101:51027/v1",
  apiKey: process.env.OPENAI_API_KEY || "EMPTY",
  ragUrl: process.env.RAG_SERVICE_URL || "http://10.160.152.38:51029/",
  maxIter: Number.parseInt(process.env.MAX_ITERATIONS || "15", 10),
  // Do not default to 0/infinite.
  maxWaitMs: Number.parseInt(process.env.MAX_WAIT_MS || "900000", 10),
  idleWaitMs: Number.parseInt(process.env.IDLE_WAIT_MS || "180000", 10),
  heartbeatMs: Number.parseInt(process.env.HEARTBEAT_MS || "30000", 10),
  captureRawHttpTrace: /^(1|true|yes)$/i.test(
    process.env.CAPTURE_RAW_HTTP_TRACE || "",
  ),
  disableStreaming: /^(1|true|yes)$/i.test(
    process.env.DISABLE_STREAMING || "",
  ),
  workspace: null,
  prompt: null,
  historyFile: "agent_history.json",
};

const RUN_STATE = {
  turn: 0,
  currentTool: null,
  currentSessionId: null,
  startedAt: Date.now(),
  lastActivityAt: Date.now(),
  submitAndExitFinished: false,
};

function nowIso() {
  return new Date().toISOString();
}

function elapsedSec() {
  return Math.round((Date.now() - RUN_STATE.startedAt) / 1000);
}

function markActivity() {
  RUN_STATE.lastActivityAt = Date.now();
}

function idleSec() {
  return Math.round((Date.now() - RUN_STATE.lastActivityAt) / 1000);
}

function log(message) {
  process.stderr.write(`[${nowIso()}] ${message}\n`);
}

function formatError(error) {
  if (error instanceof Error) {
    return error.stack || error.message;
  }
  return String(error);
}

function shortText(value, max = 300) {
  const text =
    typeof value === "string" ? value : JSON.stringify(value ?? null);
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max)}...`;
}

function summarizeToolInput(input) {
  if (!input || typeof input !== "object") {
    return shortText(input);
  }

  if (typeof input.query === "string") {
    return `query=${JSON.stringify(shortText(input.query, 500))}`;
  }

  if (typeof input.question === "string") {
    return `question=${JSON.stringify(shortText(input.question, 500))}`;
  }

  if (Array.isArray(input.questions)) {
    return `questions=${JSON.stringify(input.questions.map((q) => shortText(q, 200)))}`;
  }

  if (typeof input.command === "string") {
    return `command=${JSON.stringify(shortText(input.command, 500))}`;
  }

  if (typeof input.path === "string") {
    return `path=${JSON.stringify(input.path)}`;
  }

  if (Array.isArray(input.paths)) {
    return `paths=${JSON.stringify(input.paths)}`;
  }

  if (typeof input.file === "string") {
    return `file=${JSON.stringify(input.file)}`;
  }

  if (typeof input.url === "string") {
    return `url=${JSON.stringify(input.url)}`;
  }

  return shortText(input, 500);
}

function getOutputLength(value) {
  if (value == null) {
    return 0;
  }
  if (typeof value === "string") {
    return value.length;
  }
  try {
    return JSON.stringify(value).length;
  } catch {
    return String(value).length;
  }
}

async function safeAsync(fn, context = "unknown") {
  try {
    return await fn();
  } catch (error) {
    log(`[ERROR:${context}] ${formatError(error)}`);
    return null;
  }
}

process.on("uncaughtException", (error) => {
  log(`[UNCAUGHT EXCEPTION] ${formatError(error)}`);
});

process.on("unhandledRejection", (reason) => {
  log(`[UNHANDLED REJECTION] ${formatError(reason)}`);
});

async function parseArgs() {
  let promptFile = null;

  for (let i = 2; i < process.argv.length; i += 1) {
    const arg = process.argv[i];

    if (arg === "--folder") {
      CONFIG.workspace = resolve(process.argv[++i] || "");
      continue;
    }

    if (arg === "--prompt") {
      CONFIG.prompt = process.argv[++i] || "";
      continue;
    }

    if (arg === "--prompt-file") {
      promptFile = resolve(process.argv[++i] || "");
      continue;
    }

    if (arg === "--history") {
      CONFIG.historyFile = process.argv[++i] || CONFIG.historyFile;
      continue;
    }

    if (arg === "--capture-raw-http-trace") {
      CONFIG.captureRawHttpTrace = true;
      continue;
    }

    if (arg === "--disable-streaming") {
      CONFIG.disableStreaming = true;
      continue;
    }
  }

  if (promptFile) {
    if (!existsSync(promptFile)) {
      throw new Error(`Prompt file not found: ${promptFile}`);
    }
    CONFIG.prompt = await readFile(promptFile, "utf8");
  }

  if (!CONFIG.workspace || !CONFIG.prompt) {
    throw new Error(
      "Missing required arguments. Usage: agent.js --folder <path> (--prompt <text> | --prompt-file <file>) [--history <file>]",
    );
  }

  if (!existsSync(CONFIG.workspace)) {
    throw new Error(`Folder not found: ${CONFIG.workspace}`);
  }
}

class RAGClient {
  async postJson(path, payload, context) {
    return safeAsync(async () => {
      if (path === "/query") {
        log(`[RAG] cache lookup: ${shortText(payload.query, 180)}`);
      } else if (path === "/search") {
        log(`[RAG] search: ${shortText(payload.query, 180)}`);
      }

      const response = await fetch(new URL(path, CONFIG.ragUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(
          `RAG ${path} failed: ${response.status} ${response.statusText}`,
        );
      }

      const data = await response.json();
      if (path === "/query") {
        log(
          `[RAG] cache result: action=${data?.action || "unknown"} hasAnswer=${Boolean(
            data?.answer,
          )}`,
        );
      } else if (path === "/search") {
        log(`[RAG] search result: chunks=${data?.chunks?.length || 0}`);
      }
      return data;
    }, context);
  }

  async query(question) {
    return this.postJson("/query", { query: question }, "RAG.query");
  }

  async search(query) {
    return this.postJson("/search", { query }, "RAG.search");
  }

  async cacheAnswer(question, answer) {
    return safeAsync(async () => {
      const response = await fetch(new URL("/cache-answer", CONFIG.ragUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer }),
      });

      log(
        `[RAG] cached specialist answer: status=${response.status} answerChars=${getOutputLength(
          answer,
        )}`,
      );
    }, "RAG.cacheAnswer");
  }
}

function formatChunk(chunk, index) {
  if (typeof chunk === "string") {
    return `${index + 1}. ${chunk}`;
  }

  if (chunk && typeof chunk === "object") {
    const content =
      chunk.content || chunk.text || chunk.chunk || JSON.stringify(chunk);
    return `${index + 1}. ${content}`;
  }

  return `${index + 1}. ${String(chunk)}`;
}

function createAnalyzeFunctionCoverageTool() {
  return createTool({
    name: "analyze_function_coverage",
    description:
      "Analyze gcov line coverage for one C function using the project Python venv coverage parser.",
    inputSchema: z.object({
      gcov_file: z.string().min(1).describe("Path or basename of .gcov file."),
      function_id: z.string().min(1).describe("Function identifier."),
      function_name: z.string().min(1).describe("Function name."),
      source_file: z.string().min(1).describe("Original source file."),
      start_line: z.number().int().positive().describe("Function start line."),
      end_line: z.number().int().positive().describe("Function end line."),
    }),
    async execute(input) {
      log(
        `[TOOL] analyze_function_coverage requested function=${input.function_name} range=${input.start_line}-${input.end_line} gcov=${input.gcov_file}`,
      );

      const payload = {
        ...input,
        workspace: CONFIG.workspace,
      };

      const payloadB64 = Buffer.from(JSON.stringify(payload), "utf8").toString(
        "base64",
      );

      const script = String.raw`
import base64
import json
import os
import sys
from pathlib import Path

def find_file_recursive(root, wanted_name, max_depth=8):
    root = Path(root)

    def walk(path, depth):
        if depth > max_depth:
            return None
        try:
            entries = list(path.iterdir())
        except Exception:
            return None

        for entry in entries:
            if entry.is_file() and entry.name == wanted_name:
                return entry

        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name in {".git", "node_modules", ".cache"}:
                continue
            found = walk(entry, depth + 1)
            if found:
                return found
        return None

    return walk(root, 0)

def parse_gcov_line(line):
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None

    raw_count = parts[0].strip()
    try:
        line_number = int(parts[1].strip())
    except Exception:
        return None

    source_text = parts[2]
    executable = False
    covered = False
    count = None

    if raw_count == "-":
        executable = False
    elif raw_count in {"#####", "====="}:
        executable = True
        covered = False
        count = 0
    else:
        cleaned = raw_count.replace("*", "").strip()
        try:
            numeric = int(cleaned)
            executable = True
            count = numeric
            covered = numeric > 0
        except Exception:
            executable = False

    return {
        "raw_count": raw_count,
        "line_number": line_number,
        "source_text": source_text,
        "executable": executable,
        "covered": covered,
        "count": count,
    }

try:
    payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    workspace = Path(payload["workspace"]).resolve()
    gcov_file = payload["gcov_file"]
    function_id = payload["function_id"]
    function_name = payload["function_name"]
    source_file = payload["source_file"]
    start_line = int(payload["start_line"])
    end_line = int(payload["end_line"])

    gcov_path = Path(gcov_file)
    if not gcov_path.is_absolute():
        gcov_path = workspace / gcov_file

    if not gcov_path.exists():
        found = find_file_recursive(workspace, Path(gcov_file).name)
        if found:
            gcov_path = found

    if not gcov_path.exists():
        print(json.dumps({
            "ok": False,
            "error": f"gcov file not found: {gcov_file}",
            "searched_from": str(workspace),
            "summary": {
                "function_id": function_id,
                "function_name": function_name,
                "source_file": source_file,
                "start_line": start_line,
                "end_line": end_line,
                "executable_lines": 0,
                "covered_lines": 0,
                "uncovered_lines": 0,
                "coverage_percent": 0,
            },
            "uncovered": [],
        }))
        sys.exit(0)

    text = gcov_path.read_text(encoding="utf-8", errors="replace")
    executable = []
    covered = []
    uncovered = []

    for line in text.splitlines():
        parsed = parse_gcov_line(line)
        if not parsed:
            continue

        line_number = parsed["line_number"]
        if line_number < start_line or line_number > end_line:
            continue
        if not parsed["executable"]:
            continue

        item = {
            "line": line_number,
            "count": parsed["count"],
            "raw_count": parsed["raw_count"],
            "source": parsed["source_text"].rstrip(),
        }
        executable.append(item)
        if parsed["covered"]:
            covered.append(item)
        else:
            uncovered.append(item)

    executable_lines = len(executable)
    covered_lines = len(covered)
    uncovered_lines = len(uncovered)
    coverage_percent = 100.0 if executable_lines == 0 else round((covered_lines / executable_lines) * 100.0, 2)

    print(json.dumps({
        "ok": True,
        "gcov_path": str(gcov_path),
        "summary": {
            "function_id": function_id,
            "function_name": function_name,
            "source_file": source_file,
            "start_line": start_line,
            "end_line": end_line,
            "executable_lines": executable_lines,
            "covered_lines": covered_lines,
            "uncovered_lines": uncovered_lines,
            "coverage_percent": coverage_percent,
        },
        "uncovered": uncovered,
    }))
except Exception as e:
    print(json.dumps({
        "ok": False,
        "error": repr(e),
    }))
    sys.exit(0)
`;

      try {
        const { stdout, stderr } = await execFileAsync(
          COVERAGE_PYTHON,
          ["-c", script, payloadB64],
          {
            cwd: CONFIG.workspace,
            maxBuffer: 20 * 1024 * 1024,
          },
        );

        if (stderr && stderr.trim()) {
          log(`[TOOL] analyze_function_coverage stderr: ${shortText(stderr, 1000)}`);
        }

        const result = JSON.parse(stdout.trim());
        log(
          `[TOOL] analyze_function_coverage done function=${input.function_name} coverage=${result?.summary?.coverage_percent ?? "unknown"}%`,
        );
        return result;
      } catch (error) {
        return {
          ok: false,
          error: `failed to run ${COVERAGE_PYTHON}: ${formatError(error)}`,
          summary: {
            function_id: input.function_id,
            function_name: input.function_name,
            source_file: input.source_file,
            start_line: input.start_line,
            end_line: input.end_line,
            executable_lines: 0,
            covered_lines: 0,
            uncovered_lines: 0,
            coverage_percent: 0,
          },
          uncovered: [],
        };
      }
    },
  });
}

function createSearchTool(rag) {
  return createTool({
    name: "search_knowledge_base",
    description: "Search indexed documentation chunks for library-specific context.",
    inputSchema: z.object({
      query: z.string().min(1).describe("Documentation search query."),
    }),
    async execute({ query }) {
      log(`[SPECIALIST TOOL] search_knowledge_base query="${shortText(query, 250)}"`);

      const data = await rag.search(query);
      if (!data?.chunks?.length) {
        log("[SPECIALIST TOOL] search_knowledge_base done: chunks=0");
        return "No relevant context found.";
      }

      const output = data.chunks
        .map((chunk, index) => formatChunk(chunk, index))
        .join("\n\n---\n\n");

      log(
        `[SPECIALIST TOOL] search_knowledge_base done: chunks=${data.chunks.length} outputChars=${output.length}`,
      );
      return output;
    },
  });
}

function buildModelOptions() {
  if (!CONFIG.captureRawHttpTrace && !CONFIG.disableStreaming) {
    return {};
  }

  return {
    modelOptions: {
      metadata: {
        ...(CONFIG.captureRawHttpTrace
          ? { captureRawHttpTrace: true }
          : {}),
        ...(CONFIG.disableStreaming || CONFIG.captureRawHttpTrace
          ? { disableStreaming: true }
          : {}),
      },
    },
  };
}

async function answerWithSpecialist(rag, question) {
  log(`[SPECIALIST] question: ${shortText(question, 250)}`);

  const cached = await rag.query(question);
  if (cached?.action === "answer" && cached.answer) {
    log(`[SPECIALIST] cache hit: answerChars=${getOutputLength(cached.answer)}`);
    return `Q: ${question}\nA: ${cached.answer}\n[source:cache]`;
  }

  log("[SPECIALIST] cache miss, starting subagent");

  const subagent = new Agent({
    providerId: CONFIG.providerId,
    modelId: CONFIG.modelId,
    baseUrl: CONFIG.baseUrl,
    apiKey: CONFIG.apiKey,
    systemPrompt: [
      "You are a strict documentation specialist.",
      "Use the search_knowledge_base tool to gather context before answering.",
      "Do not invent APIs or function signatures.",
      // "<|think|>",
    ].join(" "),
    tools: [createSearchTool(rag)],
    toolPolicies: {
      search_knowledge_base: { autoApprove: true },
    },
    maxIterations: 5,
    ...buildModelOptions(),
  });

  subagent.subscribe?.((event) => {
    try {
      if (event.type === "tool-started") {
        log(
          `[SPECIALIST TOOL] ${event.toolCall?.toolName || "unknown"} ${summarizeToolInput(
            event.toolCall?.input,
          )}`,
        );
      }

      if (event.type === "tool-finished") {
        log(`[SPECIALIST TOOL] ${event.toolCall?.toolName || "unknown"} finished`);
      }

      if (event.type === "run-failed") {
        log(`[SPECIALIST FAILED] ${formatError(event.error)}`);
      }
    } catch (error) {
      log(`[SPECIALIST EVENT ERROR] ${formatError(error)}`);
    }
  });

  const result = await subagent.run(question);
  const answer = result?.outputText?.trim() || "[No answer]";

  log(
    `[SPECIALIST] done: status=${result?.status || "unknown"} answerChars=${answer.length}`,
  );

  if (result?.error) {
    log(`[SPECIALIST ERROR] ${formatError(result.error)}`);
  }

  await rag.cacheAnswer(question, answer);
  return `Q: ${question}\nA: ${answer}\n[source:subagent]`;
}

function createAskSpecialistsTool(rag) {
  return createTool({
    name: "ask_specialists",
    description:
      "Ask one to three library questions using cache first, then documentation-only subagents.",
    inputSchema: z.object({
      questions: z
        .array(
          z
            .string()
            .min(10)
            .describe(
              "A concrete question about the library, API, or expected behavior.",
            ),
        )
        .min(1)
        .max(3)
        .describe("One to three questions to send to specialists."),
    }),
    async execute({ questions }) {
      log(`[TOOL] ask_specialists started: count=${questions.length}`);
      questions.forEach((question, index) => {
        log(`[TOOL] ask_specialists q${index + 1}: ${shortText(question, 250)}`);
      });

      const results = await Promise.allSettled(
        questions.map((question) =>
          safeAsync(
            () => answerWithSpecialist(rag, question),
            `specialist:${question}`,
          ),
        ),
      );

      const output = results
        .map((result) => {
          if (result.status === "fulfilled" && result.value) {
            return result.value;
          }
          if (result.status === "rejected") {
            return `[ERR] ${formatError(result.reason)}`;
          }
          return "[ERR] Specialist returned no result.";
        })
        .join(`\n\n${"-".repeat(50)}\n\n`);

      log(`[TOOL] ask_specialists finished: outputChars=${output.length}`);
      return output;
    },
  });
}

function buildSystemPrompt() {
  return [
    "You are a helpful Assistant, do whatever the user tells you to do. <|think|>",
  ].join(" ");
}

function buildToolPolicies() {
  return {
    ask_specialists: { autoApprove: true },
    read_files: { autoApprove: true },
    search_codebase: { autoApprove: true },
    run_commands: { autoApprove: true },
    fetch_web_content: { autoApprove: true },
    apply_patch: { autoApprove: true },
    editor: { autoApprove: true },
    skills: { autoApprove: true },
    ask_question: { autoApprove: true },
    submit_and_exit: { autoApprove: true },
    analyze_function_coverage: { autoApprove: true },
  };
}

function getMessageText(message) {
  return (message?.content || [])
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function getToolResultOutput(event) {
  const toolResult = event.message?.content?.find(
    (part) => part.type === "tool-result",
  );
  if (toolResult?.type === "tool-result") {
    return toolResult.output;
  }
  return event.result;
}

function createEventHandler() {
  let sawTextDeltaThisTurn = false;
  let currentTurnStarted = false;

  return function handleEvent(event) {
    try {
      const type = event?.type || "unknown";
      markActivity();

      switch (type) {
        case "assistant-text-delta": {
          if (!currentTurnStarted) {
            RUN_STATE.turn += 1;
            currentTurnStarted = true;
            sawTextDeltaThisTurn = false;
            log(`[TURN ${RUN_STATE.turn}] assistant responding`);
          }

          const text = event.text || event.delta || "";
          if (text) {
            sawTextDeltaThisTurn = true;
            process.stdout.write(text);
          }
          break;
        }

        case "assistant-message": {
          if (!currentTurnStarted) {
            RUN_STATE.turn += 1;
            currentTurnStarted = true;
            sawTextDeltaThisTurn = false;
            log(`[TURN ${RUN_STATE.turn}] assistant message`);
          }

          if (!sawTextDeltaThisTurn) {
            const finalText = getMessageText(event.message);
            if (finalText) {
              process.stdout.write(`${finalText}\n`);
            }
          }

          log(`[TURN ${RUN_STATE.turn}] done`);
          currentTurnStarted = false;
          sawTextDeltaThisTurn = false;
          break;
        }

        case "tool-started": {
          const toolName = event.toolCall?.toolName || "unknown";
          RUN_STATE.currentTool = toolName;
          if (toolName === "submit_and_exit") {
            log("[EXIT] submit_and_exit called");
          } else {
            log(
              `[TOOL] ${toolName} started: ${summarizeToolInput(
                event.toolCall?.input,
              )}`,
            );
          }
          break;
        }

        case "tool-finished": {
          const toolName =
            event.toolCall?.toolName || RUN_STATE.currentTool || "unknown";
          const output = getToolResultOutput(event);
          if (toolName === "submit_and_exit") {
            RUN_STATE.submitAndExitFinished = true;
            log("[EXIT] submit_and_exit finished");
          } else {
            log(
              `[TOOL] ${toolName} finished: outputChars=${getOutputLength(output)}`,
            );
          }
          RUN_STATE.currentTool = null;
          break;
        }

        case "status-notice": {
          log(`[NOTICE] ${event.message}`);
          break;
        }

        case "run-failed": {
          log(`[RUN FAILED] ${formatError(event.error)}`);
          break;
        }

        case "ended":
        case "session-ended":
        case "run-completed":
        case "completed": {
          log(`[SESSION] ${type}`);
          break;
        }

        default:
          break;
      }
    } catch (error) {
      log(`[EVENT HANDLER ERROR] ${formatError(error)}`);
    }
  };
}

function subscribeIfPossible(target, handler, label = "target") {
  if (!target) {
    return null;
  }

  if (typeof target.subscribe === "function") {
    try {
      const unsubscribe = target.subscribe(handler);
      log(`[SUBSCRIBE] ${label}: ok`);
      return typeof unsubscribe === "function" ? unsubscribe : null;
    } catch {
      try {
        const unsubscribe = target.subscribe("*", handler);
        log(`[SUBSCRIBE] ${label}: ok wildcard`);
        return typeof unsubscribe === "function" ? unsubscribe : null;
      } catch {
        log(`[SUBSCRIBE] ${label}: failed`);
      }
    }
  }

  if (typeof target.on === "function") {
    const eventNames = [
      "event",
      "assistant-text-delta",
      "assistant-message",
      "tool-started",
      "tool-finished",
      "status-notice",
      "run-failed",
      "ended",
      "session-ended",
      "run-completed",
      "completed",
    ];

    for (const eventName of eventNames) {
      target.on(eventName, (event) =>
        handler({ type: eventName, ...(event || {}) }),
      );
    }

    log(`[SUBSCRIBE] ${label}: ok EventEmitter`);
    return null;
  }

  return null;
}

async function consumeAsyncEventsIfPossible(target, handler) {
  if (!target || typeof target[Symbol.asyncIterator] !== "function") {
    return false;
  }

  for await (const event of target) {
    handler(event);
  }

  return true;
}

function getSessionId(value) {
  return (
    value?.sessionId ||
    value?.id ||
    value?.payload?.sessionId ||
    value?.session?.sessionId ||
    value?.session?.id ||
    null
  );
}

function isTerminalEvent(event, wantedSessionId) {
  const type = event?.type;
  const eventSessionId = getSessionId(event);

  if (wantedSessionId && eventSessionId && eventSessionId !== wantedSessionId) {
    return false;
  }

  if (
    type === "ended" ||
    type === "session-ended" ||
    type === "run-completed" ||
    type === "completed" ||
    type === "run-failed"
  ) {
    return true;
  }

  const status = event?.status || event?.payload?.status;
  return (
    status === "completed" ||
    status === "failed" ||
    status === "cancelled" ||
    status === "canceled" ||
    status === "aborted"
  );
}

function getStatus(value) {
  return (
    value?.status ||
    value?.payload?.status ||
    value?.session?.status ||
    value?.result?.status ||
    null
  );
}

function normalizeTerminalEvent(event) {
  if (!event || typeof event !== "object") {
    return { status: "completed" };
  }

  const existingStatus = getStatus(event);
  if (existingStatus) {
    return {
      ...event,
      status: existingStatus,
    };
  }

  switch (event.type) {
    case "run-failed":
      return {
        ...event,
        status: "failed",
      };
    case "completed":
    case "run-completed":
    case "session-ended":
    case "ended":
      return {
        ...event,
        status: "completed",
      };
    default:
      return {
        ...event,
        status: "completed",
      };
  }
}

function statusFromFinishReason(reason) {
  if (reason === "timeout") {
    return "timeout";
  }
  if (reason === "idle-timeout") {
    return "idle-timeout";
  }
  if (reason === "submit_and_exit") {
    return "completed";
  }
  return "completed";
}

function isTerminalStatus(status) {
  return (
    status === "completed" ||
    status === "failed" ||
    status === "cancelled" ||
    status === "canceled" ||
    status === "aborted"
  );
}

async function pollSessionUntilDone(cline, sessionId) {
  if (!cline || !sessionId || typeof cline.get !== "function") {
    return null;
  }

  log(`[POLL] watching session=${sessionId}`);
  const startedAt = Date.now();
  let lastStatus = null;

  while (true) {
    if (RUN_STATE.submitAndExitFinished) {
      log("[POLL] submit_and_exit already finished");
      return { status: "completed" };
    }

    const session = await safeAsync(() => cline.get(sessionId), "cline.get.poll");
    const status = getStatus(session);

    if (status !== lastStatus) {
      log(`[POLL] status=${status || "unknown"} elapsed=${elapsedSec()}s`);
      lastStatus = status;
    }

    if (isTerminalStatus(status)) {
      return session;
    }

    const elapsed = Date.now() - startedAt;
    const idle = Date.now() - RUN_STATE.lastActivityAt;

    if (
      Number.isFinite(CONFIG.maxWaitMs) &&
      CONFIG.maxWaitMs > 0 &&
      elapsed > CONFIG.maxWaitMs
    ) {
      log(`[POLL] timeout after ${CONFIG.maxWaitMs}ms`);
      return {
        ...(session || {}),
        status: status || "timeout",
      };
    }

    if (
      !RUN_STATE.currentTool &&
      Number.isFinite(CONFIG.idleWaitMs) &&
      CONFIG.idleWaitMs > 0 &&
      idle > CONFIG.idleWaitMs
    ) {
      log(`[POLL] idle timeout after ${CONFIG.idleWaitMs}ms`);
      return {
        ...(session || {}),
        status: status || "idle-timeout",
      };
    }

    await new Promise((resolve) => setTimeout(resolve, CONFIG.heartbeatMs));
  }
}

async function waitForSessionEnd(cline, sessionId) {
  if (!cline || !sessionId) {
    return null;
  }

  log(`[WAIT] session=${sessionId}`);
  return new Promise((resolve) => {
    let done = false;
    const startedAt = Date.now();

    const finish = (reason, event = null) => {
      if (done) {
        return;
      }

      done = true;
      clearInterval(heartbeat);
      clearTimeout(timeout);
      try {
        unsubscribe?.();
      } catch {
        // ignore unsubscribe errors
      }

      log(`[WAIT] finished: ${reason} elapsed=${elapsedSec()}s`);
      resolve(
        event
          ? normalizeTerminalEvent(event)
          : { status: statusFromFinishReason(reason) },
      );
    };

    const monitor = (event) => {
      markActivity();
      if (isTerminalEvent(event, sessionId)) {
        finish(event?.type || "terminal-event", event);
      }
    };

    const unsubscribe = subscribeIfPossible(cline, monitor, "waiter");
    const heartbeat = setInterval(() => {
      const idle = Date.now() - RUN_STATE.lastActivityAt;
      log(
        `[RUNNING] elapsed=${elapsedSec()}s idle=${idleSec()}s turn=${RUN_STATE.turn} currentTool=${
          RUN_STATE.currentTool || "none"
        }`,
      );

      if (RUN_STATE.submitAndExitFinished) {
        finish("submit_and_exit");
        return;
      }

      if (
        !RUN_STATE.currentTool &&
        Number.isFinite(CONFIG.idleWaitMs) &&
        CONFIG.idleWaitMs > 0 &&
        idle > CONFIG.idleWaitMs
      ) {
        finish("idle-timeout");
      }
    }, CONFIG.heartbeatMs);

    const timeout =
      Number.isFinite(CONFIG.maxWaitMs) && CONFIG.maxWaitMs > 0
        ? setTimeout(() => finish("timeout"), CONFIG.maxWaitMs)
        : null;
  });
}

async function writeHistory(result) {
  const historyPath = resolve(process.cwd(), CONFIG.historyFile);
  await mkdir(dirname(historyPath), { recursive: true });
  await writeFile(
    historyPath,
    JSON.stringify(result?.messages || [], null, 2),
    "utf8",
  );
  log(`[SAVED] ${historyPath}`);
}

async function collectFinalResult(cline, resultOrSession) {
  const sessionId = getSessionId(resultOrSession);
  let session = null;
  let messages = resultOrSession?.messages || null;

  if (cline && sessionId && typeof cline.get === "function") {
    session = await safeAsync(() => cline.get(sessionId), "cline.get");
  }

  if (cline && sessionId && typeof cline.readMessages === "function") {
    const readMessages = await safeAsync(
      () => cline.readMessages(sessionId),
      "cline.readMessages",
    );
    if (readMessages) {
      messages = readMessages;
    }
  }

  return {
    ...(session || {}),
    ...(resultOrSession || {}),
    ...(messages ? { messages } : {}),
  };
}

async function main() {
  try {
    await parseArgs();
    process.chdir(CONFIG.workspace);
  } catch (error) {
    log(`[INIT ERROR] ${formatError(error)}`);
    process.exit(1);
  }

  if (!ClineCore?.create) {
    log(
      "[INIT ERROR] ClineCore.create was not found in ./cline/sdk/packages/sdk/dist/index.js",
    );
    process.exit(1);
  }

  log("[START] ClineCore agent");
  log(`[CONFIG] provider=${CONFIG.providerId} model=${CONFIG.modelId}`);
  log(`[CONFIG] baseUrl=${CONFIG.baseUrl}`);
  log(`[CONFIG] workspace=${CONFIG.workspace}`);
  log(`[CONFIG] maxIterations=${CONFIG.maxIter}`);
  log(`[CONFIG] history=${CONFIG.historyFile}`);

  const rag = new RAGClient();
  const askSpecialistsTool = createAskSpecialistsTool(rag);
  const analyzeFunctionCoverageTool = createAnalyzeFunctionCoverageTool();
  const handler = createEventHandler();

  const commonRuntimeFields = {
    cwd: CONFIG.workspace,
    workspace_root: CONFIG.workspace,
    workspace: CONFIG.workspace,
    workspaceRoot: CONFIG.workspace,
    enable_tools: true,
    enable_spawn: true,
    enable_teams: false,
    enableTools: true,
    enableSpawnAgent: true,
    enableAgentTeams: false,
    systemPrompt: buildSystemPrompt(),
    maxIterations: Number.isFinite(CONFIG.maxIter) ? CONFIG.maxIter : 15,
    extraTools: [askSpecialistsTool, analyzeFunctionCoverageTool],
    toolPolicies: buildToolPolicies(),
    ...buildModelOptions(),
  };

  const runtimeConfig = {
    providerId: CONFIG.providerId,
    modelId: CONFIG.modelId,
    baseUrl: CONFIG.baseUrl,
    apiKey: CONFIG.apiKey,
    ...commonRuntimeFields,
  };

  const localRuntime = {
    extraTools: [askSpecialistsTool, analyzeFunctionCoverageTool],
  };

  const coreOptions = {
    ...runtimeConfig,
    localRuntime,
  };

  const startOptions = {
    prompt: CONFIG.prompt,
    ...commonRuntimeFields,
    localRuntime,
    config: {
      ...runtimeConfig,
    },
  };

  try {
    log("[CLINE] creating core");
    const cline = await ClineCore.create(coreOptions);
    log("[CLINE] core ready");

    if (cline?.host?.startSession) {
      const originalStartSession = cline.host.startSession.bind(cline.host);
      cline.host.startSession = async (input) => {
        log(
          `[SESSION] startSession tools=${input?.config?.enableTools ?? input?.enableTools} spawn=${
            input?.config?.enableSpawnAgent ?? input?.enableSpawnAgent
          } cwd=${input?.config?.cwd || input?.cwd || "unknown"}`,
        );
        const result = await originalStartSession(input);
        RUN_STATE.currentSessionId = getSessionId(result);
        log(`[SESSION] started id=${RUN_STATE.currentSessionId || "unknown"}`);
        return result;
      };
    }

    subscribeIfPossible(cline, handler, "cline");

    log("[CLINE] starting run");
    const resultOrSession = await cline.start(startOptions);
    const sessionId = getSessionId(resultOrSession);
    RUN_STATE.currentSessionId = sessionId;
    log(`[CLINE] start returned session=${sessionId || "unknown"}`);
    subscribeIfPossible(resultOrSession, handler, "session");

    let terminalSession = null;
    if (sessionId) {
      const waiters = [
        waitForSessionEnd(cline, sessionId),
        pollSessionUntilDone(cline, sessionId),
      ];
      if (resultOrSession?.[Symbol.asyncIterator]) {
        waiters.push(consumeAsyncEventsIfPossible(resultOrSession, handler));
      }
      terminalSession = await Promise.race(waiters);
    } else if (resultOrSession?.[Symbol.asyncIterator]) {
      terminalSession = await Promise.race([
        consumeAsyncEventsIfPossible(resultOrSession, handler),
        new Promise((resolve) =>
          setTimeout(
            () => resolve({ status: "timeout" }),
            CONFIG.maxWaitMs,
          ),
        ),
      ]);
    }

    const result = {
      ...(await collectFinalResult(cline, resultOrSession)),
      ...(terminalSession || {}),
    };

    if (result?.error) {
      log(`[CLINE ERROR] ${formatError(result.error)}`);
    }

    if (result?.outputText && typeof result.outputText === "string") {
      process.stdout.write(`${result.outputText.trim()}\n`);
    }

    if (result?.status && result.status !== "completed" && !result?.error) {
      log(`[CLINE INCOMPLETE] status=${result.status}`);
    }

    await safeAsync(() => writeHistory(result), "write_history");

    const status = result?.status || resultOrSession?.status || "completed";
    log(`[DONE] status=${status} elapsed=${elapsedSec()}s turns=${RUN_STATE.turn}`);

    const okStatuses = new Set(["completed", "success"]);
    const timeoutStatuses = new Set(["timeout", "idle-timeout"]);

    if (okStatuses.has(status)) {
      process.exit(0);
    }

    if (timeoutStatuses.has(status)) {
      process.exit(124);
    }

    process.exit(1);
  } catch (error) {
    log(`[FATAL] ${formatError(error)}`);
    process.exit(2);
  }
}

await main();
