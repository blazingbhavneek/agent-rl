import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";
import { Agent } from "./cline/sdk/packages/sdk/dist/index.js";

loadDotEnv();

const LIVE_BASE_URL = process.env.TEST_BASE_URL || process.env.OPENAI_BASE_URL;
const LIVE_API_KEY = process.env.TEST_API_KEY || process.env.OPENAI_API_KEY;
const LIVE_MODEL_ID = process.env.TEST_MODEL_ID || process.env.MODEL_NAME;
const LIVE_SYSTEM_PROMPT =
	process.env.TEST_SYSTEM_PROMPT || "You are a concise assistant.";
const LIVE_PROMPT =
	process.env.TEST_PROMPT ||
	"Please complete this sentence and nothing else: The quick copper fox cataloged ...";

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

function loadDotEnv() {
	const envPath = resolve(process.cwd(), ".env");
	if (!existsSync(envPath)) {
		return;
	}

	const lines = requireEnvFile(envPath).split(/\r?\n/);
	for (const rawLine of lines) {
		const line = rawLine.trim();
		if (!line || line.startsWith("#")) {
			continue;
		}

		const separatorIndex = line.indexOf("=");
		if (separatorIndex <= 0) {
			continue;
		}

		const key = line.slice(0, separatorIndex).trim();
		let value = line.slice(separatorIndex + 1).trim();
		if (
			(value.startsWith('"') && value.endsWith('"')) ||
			(value.startsWith("'") && value.endsWith("'"))
		) {
			value = value.slice(1, -1);
		}

		value = value.replace(/\$\{([^}]+)\}/g, (_match, name) => process.env[name] || "");

		if (!(key in process.env)) {
			process.env[key] = value;
		}
	}
}

function requireEnvFile(envPath) {
	try {
		return process.getBuiltinModule("node:fs").readFileSync(envPath, "utf8");
	} catch (error) {
		throw new Error(
			`Failed to read .env file at ${envPath}: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
}

function isLiveMode() {
	return Boolean(LIVE_BASE_URL && LIVE_API_KEY && LIVE_MODEL_ID);
}

async function writeHistoryAndGetAssistantMessage(result, historyPath) {
	await writeFile(historyPath, JSON.stringify(result.messages, null, 2), "utf8");

	const persistedMessages = JSON.parse(await readFile(historyPath, "utf8"));
	const assistantMessage = [...persistedMessages]
		.reverse()
		.find((message) => message.role === "assistant");

	assert.ok(assistantMessage, "Expected a persisted assistant message");
	return assistantMessage;
}

function assertTraceBasics(trace, expectedModelId) {
	assert.ok(trace, "Expected rawHttpTrace metadata on the persisted assistant message");
	assert.equal(trace.version, 1);
	assert.equal(trace.providerId, "openai-compatible");
	assert.equal(trace.modelId, expectedModelId);
	assert.ok(trace.interactions.length >= 1, "Expected at least one captured HTTP interaction");
}

async function runLiveMode(historyPath) {
	const agent = new Agent({
		providerId: "openai-compatible",
		modelId: LIVE_MODEL_ID,
		baseUrl: LIVE_BASE_URL,
		apiKey: LIVE_API_KEY,
		systemPrompt: LIVE_SYSTEM_PROMPT,
		maxIterations: 1,
		modelOptions: {
			metadata: {
				captureRawHttpTrace: true,
				disableStreaming: true,
			},
		},
	});

	const result = await agent.run(LIVE_PROMPT);
	const assistantMessage = await writeHistoryAndGetAssistantMessage(
		result,
		historyPath,
	);
	const trace = assistantMessage.metadata?.rawHttpTrace;

	assertTraceBasics(trace, LIVE_MODEL_ID);

	const interaction = trace.interactions.at(-1);
	assert.ok(interaction, "Expected a captured HTTP interaction");
	assert.match(interaction.request.url, /\/chat\/completions|\/responses/);
	assert.equal(interaction.request.method, "POST");
	assert.equal(interaction.request.headers.authorization, "[REDACTED]");
	assert.ok(
		typeof interaction.request.body === "string" &&
			interaction.request.body.includes(LIVE_PROMPT),
		"Expected persisted request body to contain the live test prompt",
	);
	assert.ok(
		typeof interaction.response?.body === "string" &&
			interaction.response.body.length > 0,
		"Expected persisted response body text",
	);
	assert.doesNotMatch(
		interaction.request.body,
		/"stream":true/,
	);
	assert.ok(result.outputText.trim().length > 0, "Expected non-empty model output");

	process.stdout.write(`PASS live raw trace persisted to ${historyPath}\n`);
}

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
	const historyPath = resolve(process.cwd(), "history.json");

	if (isLiveMode()) {
		await runLiveMode(historyPath);
		return;
	}

	const server = createServer();
	server.listen(0, "127.0.0.1");
	await once(server, "listening");

	const address = server.address();
	if (!address || typeof address === "string") {
		throw new Error("Failed to bind test server");
	}

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
		const assistantMessage = await writeHistoryAndGetAssistantMessage(
			result,
			historyPath,
		);

		const trace = assistantMessage.metadata?.rawHttpTrace;
		assertTraceBasics(trace, "demo-model");
		assert.equal(trace.interactions.length, 1, "Expected one captured HTTP interaction");

		const interaction = trace.interactions[0];
		assert.equal(
			interaction.request.url,
			`http://127.0.0.1:${address.port}/v1/chat/completions`,
		);
		assert.equal(interaction.request.method, "POST");
		assert.equal(interaction.request.body, capturedRequests[0]?.body);
		assert.equal(interaction.response?.status, 200);
		assert.equal(interaction.response?.body, RESPONSE_TEXT);
		assert.equal(interaction.request.headers.authorization, "[REDACTED]");

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
