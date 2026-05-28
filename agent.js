#!/usr/bin/env node

import { Agent, createTool } from "./cline/sdk/packages/sdk/dist/index.js";
import { z } from "zod";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import process from "node:process";

const CONFIG = {
  providerId: process.env.PROVIDER_ID || "openai-compatible",
  modelId: process.env.MODEL_NAME || "openai/gpt-oss-120b",
  baseUrl: process.env.OPENAI_BASE_URL || "http://10.160.144.101:51027/v1",
  apiKey: process.env.OPENAI_API_KEY || "EMPTY",
  ragUrl: process.env.RAG_SERVICE_URL || "http://10.160.152.38:51029/",
  maxIter: Number.parseInt(process.env.MAX_ITERATIONS || "15", 10),
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

process.on("uncaughtException", (error) => {
  process.stderr.write(`\n[UNCAUGHT EXCEPTION] ${formatError(error)}\n`);
});

process.on("unhandledRejection", (reason) => {
  process.stderr.write(`\n[UNHANDLED REJECTION] ${formatError(reason)}\n`);
});

function formatError(error) {
  if (error instanceof Error) {
    return error.stack || error.message;
  }
  return String(error);
}

async function safeAsync(fn, context = "unknown") {
  try {
    return await fn();
  } catch (error) {
    process.stderr.write(`\n[ERROR:${context}] ${formatError(error)}\n`);
    return null;
  }
}

function parseArgs() {
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

  if (!CONFIG.workspace || !CONFIG.prompt) {
    throw new Error(
      "Missing required arguments. Usage: agent.js --folder <path> --prompt <text> [--history <file>]",
    );
  }

  if (!existsSync(CONFIG.workspace)) {
    throw new Error(`Folder not found: ${CONFIG.workspace}`);
  }
}

class RAGClient {
  async postJson(path, payload, context) {
    return safeAsync(async () => {
      const response = await fetch(new URL(path, CONFIG.ragUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`RAG ${path} failed: ${response.status} ${response.statusText}`);
      }

      return response.json();
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
      await fetch(new URL("/cache-answer", CONFIG.ragUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer }),
      });
    }, "RAG.cacheAnswer");
  }
}

function formatChunk(chunk, index) {
  if (typeof chunk === "string") {
    return `${index + 1}. ${chunk}`;
  }

  if (chunk && typeof chunk === "object") {
    const content = chunk.content || chunk.text || chunk.chunk || JSON.stringify(chunk);
    return `${index + 1}. ${content}`;
  }

  return `${index + 1}. ${String(chunk)}`;
}

function createSearchTool(rag) {
  return createTool({
    name: "search_knowledge_base",
    description: "Search indexed documentation chunks for library-specific context.",
    inputSchema: z.object({
      query: z.string().min(1).describe("Documentation search query."),
    }),
    async execute({ query }) {
      const data = await rag.search(query);

      if (!data?.chunks?.length) {
        return "No relevant context found.";
      }

      return data.chunks.map((chunk, index) => formatChunk(chunk, index)).join("\n\n---\n\n");
    },
  });
}

async function answerWithSpecialist(rag, question) {
  const cached = await rag.query(question);
  if (cached?.action === "answer" && cached.answer) {
    return `Q: ${question}\nA: ${cached.answer}\n[source:cache]`;
  }

  const subagent = new Agent({
    providerId: CONFIG.providerId,
    modelId: CONFIG.modelId,
    baseUrl: CONFIG.baseUrl,
    apiKey: CONFIG.apiKey,
    systemPrompt: [
      "You are a strict documentation specialist.",
      "Use the search_knowledge_base tool to gather context before answering.",
      "Do not invent APIs or function signatures.",
      "If the docs are insufficient, say exactly what is missing.",
    ].join(" "),
    tools: [createSearchTool(rag)],
    toolPolicies: {
      search_knowledge_base: { autoApprove: true },
    },
    maxIterations: 5,
  });

  const result = await subagent.run(question);
  const answer = result?.outputText?.trim() || "[No answer]";

  await rag.cacheAnswer(question, answer);

  return `Q: ${question}\nA: ${answer}\n[source:subagent]`;
}

function createAskSpecialistsTool(rag) {
  return createTool({
    name: "ask_specialists",
    description: "Ask one to three library questions using cache first, then documentation-only subagents.",
    inputSchema: z.object({
      questions: z
        .array(
          z
            .string()
            .min(10)
            .describe("A concrete question about the library, API, or expected behavior."),
        )
        .min(1)
        .max(3)
        .describe("One to three questions to send to specialists."),
    }),
    async execute({ questions }) {
      const results = await Promise.allSettled(
        questions.map((question) =>
          safeAsync(() => answerWithSpecialist(rag, question), `specialist:${question}`),
        ),
      );

      return results
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
    },
  });
}

function logToolResult(event) {
  const toolResult = event.message?.content?.find((part) => part.type === "tool-result");
  if (!toolResult || toolResult.type !== "tool-result") {
    return;
  }

  const output =
    typeof toolResult.output === "string"
      ? toolResult.output
      : JSON.stringify(toolResult.output, null, 2);

  process.stderr.write(`\n[TOOL RESULT] ${event.toolCall.toolName}\n${output}\n`);
}

function getMessageText(message) {
  return (message?.content || [])
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}

async function writeHistory(result) {
  const historyPath = resolve(process.cwd(), CONFIG.historyFile);
  await mkdir(dirname(historyPath), { recursive: true });
  await writeFile(historyPath, JSON.stringify(result?.messages || [], null, 2), "utf8");
  process.stderr.write(`[SAVED] ${historyPath}\n`);
}

async function main() {
  try {
    parseArgs();
    process.chdir(CONFIG.workspace);
  } catch (error) {
    process.stderr.write(`[INIT ERROR] ${formatError(error)}\n`);
    process.exit(1);
  }

  const rag = new RAGClient();
  const agent = new Agent({
    providerId: CONFIG.providerId,
    modelId: CONFIG.modelId,
    baseUrl: CONFIG.baseUrl,
    apiKey: CONFIG.apiKey,
    systemPrompt: [
      "You are an engineer who is new to this library.",
      "Even if the prompt seems specific, ask specialists first so you understand the library before acting.",
      "Do not assume function signatures or behavior.",
      `Workspace: ${CONFIG.workspace}`,
    ].join(" "),
    tools: [createAskSpecialistsTool(rag)],
    toolPolicies: {
      ask_specialists: { autoApprove: true },
    },
    maxIterations: Number.isFinite(CONFIG.maxIter) ? CONFIG.maxIter : 15,
    ...((CONFIG.captureRawHttpTrace || CONFIG.disableStreaming)
      ? {
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
        }
      : {}),
  });

  let sawTextDelta = false;
  agent.subscribe((event) => {
    try {
      switch (event.type) {
        case "assistant-text-delta":
          if (event.text) {
            sawTextDelta = true;
            process.stdout.write(event.text);
          }
          break;
        case "assistant-message":
          if (!sawTextDelta) {
            const finalText = getMessageText(event.message);
            if (finalText) {
              process.stdout.write(`${finalText}\n`);
            }
          }
          process.stderr.write("\n[FINAL MESSAGE]\n");
          process.stderr.write(`${JSON.stringify(event.message, null, 2)}\n`);
          break;
        case "tool-started":
          process.stderr.write(`\n[TOOL CALL] ${event.toolCall.toolName}\n`);
          process.stderr.write(`${JSON.stringify(event.toolCall.input, null, 2)}\n`);
          break;
        case "tool-finished":
          logToolResult(event);
          break;
        case "status-notice":
          process.stderr.write(`\n[NOTICE] ${event.message}\n`);
          break;
        case "run-failed":
          process.stderr.write(`\n[RUN FAILED] ${formatError(event.error)}\n`);
          break;
        default:
          break;
      }
    } catch (error) {
      process.stderr.write(`\n[EVENT HANDLER ERROR] ${formatError(error)}\n`);
    }
  });

  try {
    const result = await agent.run(CONFIG.prompt);

    if (result?.error) {
      process.stderr.write(`\n[AGENT ERROR] ${formatError(result.error)}\n`);
    }

    if (result?.status !== "completed" && !result?.error) {
      process.stderr.write(`\n[AGENT INCOMPLETE] ${JSON.stringify(result, null, 2)}\n`);
    }

    await safeAsync(() => writeHistory(result), "write_history");

    process.stderr.write(`\n[DONE] status=${result?.status}\n`);
    process.exit(result?.status === "completed" ? 0 : 1);
  } catch (error) {
    process.stderr.write(`\n[FATAL] ${formatError(error)}\n`);
    process.exit(2);
  }
}

await main();
