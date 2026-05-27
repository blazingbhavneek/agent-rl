import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";
import * as ClineSdk from "@cline/sdk";

const { Agent } = ClineSdk;

const RESPONSE_BODY = {
  id: "cmpl-test-1",
  object: "chat.completion",
  created: 0,
  model: "demo-model",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: "OK",
      },
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: 17,
    completion_tokens: 1,
    total_tokens: 18,
    prompt_tokens_details: {
      cached_tokens: 0,
      cache_write_tokens: 0,
    },
    completion_tokens_details: {
      reasoning_tokens: 0,
    },
  },
};

const RESPONSE_TEXT = JSON.stringify(RESPONSE_BODY);

const capturedRequests = [];

function createServer() {
  return http.createServer(async (req, res) => {
    if (req.method !== "POST" || req.url !== "/v1/chat/completions") {
      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      res.end("not found");
      return;
    }

    const chunks = [];
    for await (const chunk of req) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }

    const requestBody = Buffer.concat(chunks).toString("utf8");
    capturedRequests.push({
      method: req.method,
      url: req.url,
      body: requestBody,
      headers: req.headers,
    });

    res.writeHead(200, {
      "content-type": "application/json; charset=utf-8",
    });
    res.end(RESPONSE_TEXT);
  });
}

async function main() {
  const server = createServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");

  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Failed to bind test server");
  }

  const historyPath = resolve(process.cwd(), "history.json");

  try {
    const agent = new Agent({
      providerId: "openai-compatible",
      modelId: "demo-model",
      baseUrl: `http://127.0.0.1:${address.port}/v1`,
      apiKey: "test-key",
      systemPrompt: "You are a concise assistant.",
      maxIterations: 1,
      modelOptions: {
        metadata: {
          captureRawHttpTrace: true,
          disableStreaming: true,
        },
      },
    });

    const result = await agent.run("Reply with the single word OK.");

    await writeFile(historyPath, JSON.stringify(result.messages, null, 2), "utf8");

    const persistedMessages = JSON.parse(await readFile(historyPath, "utf8"));
    const assistantMessage = [...persistedMessages]
      .reverse()
      .find((message) => message.role === "assistant");

    assert.ok(assistantMessage, "Expected a persisted assistant message");

    const trace = assistantMessage.metadata?.rawHttpTrace;
    assert.ok(trace, "Expected rawHttpTrace metadata on the persisted assistant message");
    assert.equal(trace.version, 1);
    assert.equal(trace.providerId, "openai-compatible");
    assert.equal(trace.modelId, "demo-model");
    assert.equal(trace.interactions.length, 1, "Expected one captured HTTP interaction");

    const interaction = trace.interactions[0];
    assert.equal(interaction.request.url, `http://127.0.0.1:${address.port}/v1/chat/completions`);
    assert.equal(interaction.request.method, "POST");
    assert.equal(interaction.request.body, capturedRequests[0]?.body);
    assert.equal(interaction.response?.status, 200);
    assert.equal(interaction.response?.body, RESPONSE_TEXT);

    assert.match(interaction.request.body, /"Reply with the single word OK\."/);
    assert.doesNotMatch(interaction.request.body, /"stream":true/);
    assert.match(interaction.response.body, /"object":"chat\.completion"/);
    assert.equal(result.outputText.trim(), "OK");

    process.stdout.write(`PASS raw trace persisted to ${historyPath}\n`);
  } finally {
    server.close();
  }
}

await main();
