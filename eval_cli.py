import os
import re
import json
import time
import asyncio
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import requests
from openai import OpenAI
from collections import defaultdict
from tree_sitter import Language, Parser
from tree_sitter_c import language as c_language
import subprocess

# =========================
# Config
# =========================

LOCAL_BASE_URL = os.getenv(
    "LOCAL_LLM_BASE_URL",
    "http://10.160.144.101:51021/v1",
)

VERIFY_SERVER_URL = os.getenv(
    "VERIFY_SERVER_URL",
    "http://10.160.152.38:10000",
).rstrip("/")

MCP_URL = os.getenv(
    "MOOVE_MCP_URL",
    "http://10.160.152.38:9001/mcp",
)

# Cloud config used only when --dev is NOT passed.
CLOUD_BASE_URL = os.getenv(
    "CLOUD_LLM_BASE_URL",
    "https://api.openai.com/v1",
)

CLOUD_API_KEY = os.getenv(
    "CLOUD_LLM_API_KEY",
    os.getenv("OPENAI_API_KEY", ""),
)

ORIGINAL_MODEL = os.getenv("ORIGINAL_MODEL", "openai/gpt-oss-120b")
RL_MODEL = os.getenv("RL_MODEL", "openai/gpt-oss-120b")

CLOUD_CODE_MODEL = os.getenv("CLOUD_CODE_MODEL", "gpt-5.4-mini")
CLOUD_JUDGE_MODEL = os.getenv("CLOUD_JUDGE_MODEL", "gpt-5.4")
LOCAL_JUDGE_MODEL = os.getenv("LOCAL_JUDGE_MODEL", ORIGINAL_MODEL)

# Keep output here.
OUT_DIR = Path("eval_runs")
OUT_DIR.mkdir(exist_ok=True)


# =========================
# Data structures
# =========================

@dataclass
class Problem:
    id: str
    question: str
    metadata: dict[str, Any]


@dataclass
class VerifyResult:
    compiled: bool
    passed: int
    total: int
    error: Optional[str]
    details: dict[str, Any]


@dataclass
class RunResult:
    task_id: str
    setup: str
    model: str
    invocation: str
    rag_enabled: bool

    prompt: str
    raw_response: str
    extracted_code: str

    required: list[str]
    called: list[str]
    missing: list[str]
    required_pass: bool

    compile_pass: bool
    compile_logs: str

    hard_pass: bool

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    token_source: str

    latency_ms: int

    rag_query_count: int
    rag_queries: list[str]


# =========================
# Function call analyzer
# Use your local analyzer, not broken remote analyzer.
# =========================

class FunctionCallAnalyzer:
    def __init__(self):
        self.parser = Parser(Language(c_language()))

    def extract_called_functions(self, code: str) -> set[str]:
        tree = self.parser.parse(code.encode("utf8"))
        called = set()

        def visit(node):
            if node.type == "call_expression":
                fn = node.child_by_field_name("function")
                if fn:
                    name = self._extract_name(fn)
                    if name:
                        called.add(name)

            for c in node.children:
                visit(c)

        visit(tree.root_node)
        return called

    def _extract_name(self, node):
        if node.type == "identifier":
            return node.text.decode()

        if node.type == "field_expression":
            f = node.child_by_field_name("field")
            if f:
                return f.text.decode()

        return None


analyzer = FunctionCallAnalyzer()


# =========================
# Helpers
# =========================

def extract_c_code(text: str) -> str:
    m = re.search(r"```c\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # fallback: accept whole answer as code
    return text.strip()


def approx_tokens(text: str) -> int:
    # Development fallback only.
    # Later replace with provider/proxy usage logs for cloud.
    return max(1, len(text) // 4)


def make_prompt(
    question: str,
    required: list[str],
    main_c_path: Optional[Path] = None,
) -> str:
    if main_c_path is not None:
        target_file_text = (
            f"The target file already exists at this absolute path:\n"
            f"{main_c_path.resolve()}\n"
        )
        write_rule = (
            f"- Write the complete C solution into this exact file:\n"
            f"  {main_c_path.resolve()}\n"
            f"- Do not write to ./main.c unless it is the same file as the absolute path above.\n"
        )
    else:
        target_file_text = (
            "You are in a workspace that contains a file named main.c.\n"
        )
        write_rule = (
            "- Write the complete C solution into main.c.\n"
        )

    return f"""Solve this C code-generation task.

{target_file_text}
Research properly using the given tools first, dont make assumptions, the functions are part of the library.

Rules:
{write_rule}- Do not create or edit any other files.
- Do not ask the user questions.
- Do not include explanations.
- The final answer in chat can be brief, but the actual solution must be in the target C file.

Task:
{question}
"""

def load_tasks(path: str) -> list[Problem]:
    problems = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            problems.append(
                Problem(
                    id=row.get("id") or f"task_{len(problems):05d}",
                    question=row["question"],
                    metadata={"required": row.get("required", [])},
                )
            )

    return problems


# =========================
# Verifier
# Server compile logic + local required-function analyzer.
# =========================

def verify_code(
    problem: Problem,
    code: str,
    apl_variant: int = 2,
    timeout: float = 60.0,
) -> VerifyResult:
    required = problem.metadata.get("required", [])

    if not required:
        return VerifyResult(
            compiled=False,
            passed=0,
            total=0,
            error="No required functions specified",
            details={"required": [], "missing": []},
        )

    try:
        payload = {
            "code": code,
            "apl_variant": apl_variant,
            "required": required,
        }

        r = requests.post(
            f"{VERIFY_SERVER_URL}/compile",
            json=payload,
            timeout=timeout,
        )

    except requests.RequestException as e:
        return VerifyResult(
            compiled=False,
            passed=0,
            total=len(required),
            error=str(e),
            details={
                "required": required,
                "missing": required,
                "compile_logs": str(e),
            },
        )

    if r.status_code != 200:
        return VerifyResult(
            compiled=False,
            passed=0,
            total=len(required),
            error=r.text,
            details={
                "required": required,
                "missing": required,
                "compile_logs": r.text,
            },
        )

    data = r.json()
    compiled = bool(data.get("compiled"))

    compile_logs = (
        data.get("compile_logs")
        or data.get("logs")
        or data.get("error")
        or ""
    )

    called = analyzer.extract_called_functions(code)
    missing = [fn for fn in required if fn not in called]
    present = [fn for fn in required if fn in called]

    total_external = data.get("total_external_functions_executed")
    total_correct = data.get("total_correct_functions_executed")

    details = {
        "required": required,
        "called": sorted(called),
        "present": present,
        "missing": missing,
        "compile_logs": compile_logs,
        "total_external_functions_executed": total_external,
        "total_correct_functions_executed": total_correct,
    }

    # Your rule:
    # if any required function missing => fail
    required_pass = len(missing) == 0

    # Also keep your external-function rule if server provides it.
    external_ok = True
    if total_external is not None and total_correct is not None:
        if total_external == 0 or total_correct != total_external:
            external_ok = False

    final_compiled = compiled and required_pass and external_ok

    return VerifyResult(
        compiled=final_compiled,
        passed=len(present) if final_compiled else 0,
        total=len(required),
        error=None if final_compiled else f"Missing={missing}; compile_logs={compile_logs}",
        details=details,
    )


# =========================
# Copilot CLI setup
# =========================
import subprocess
from pathlib import Path

def prepare_copilot_home(name: str, rag_enabled: bool) -> Path:
    home = (OUT_DIR / f"copilot_home_{name}").resolve()
    home.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["COPILOT_HOME"] = str(home)

    if rag_enabled:
        print(
            f"[prepare_copilot_home] registering MCP server in COPILOT_HOME={home}",
            flush=True,
        )

        proc = subprocess.run(
            [
                "copilot",
                "mcp",
                "add",
                "--transport",
                "http",
                "moove",
                MCP_URL,
            ],
            env=env,
            cwd=str(Path.cwd()),  # cwd no longer matters because COPILOT_HOME is absolute
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print("[prepare_copilot_home] copilot mcp add stdout:", flush=True)
        print(proc.stdout, flush=True)

        if proc.stderr.strip():
            print("[prepare_copilot_home] copilot mcp add stderr:", flush=True)
            print(proc.stderr, flush=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"copilot mcp add failed with returncode={proc.returncode}"
            )

    with open(home / "copilot-instructions.md", "w", encoding="utf-8") as f:
        f.write(
            "You are being evaluated on C code generation.\n"
            "Follow the user prompt exactly.\n"
            "If the prompt asks you to edit a specific file, edit that exact file.\n"
            "Do not create unrelated files.\n"
        )

    return home

async def print_mcp_tools_direct():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({
        "moove": {
            "transport": "http",
            "url": MCP_URL,
        },
    })

    tools = await client.get_tools()

    print("\n========== MCP TOOLS DIRECT ==========")
    for t in tools:
        print(f"- {t.name}")
        desc = getattr(t, "description", "")
        if desc:
            print(f"  {desc[:300]}")
    print("======================================\n")


def extract_mcp_queries(raw: str) -> list[str]:
    """
    Extract visible Copilot MCP calls from stdout transcript.
    Example matched line:
      ● function_info (MCP: moove) · name: "pmf_preexit"
    """

    queries = []

    pattern = re.compile(
        r"●\s+([^\n]+?)\s+\(MCP:\s*([^)]+)\).*?(?:name:\s*\"([^\"]+)\")?",
        re.DOTALL,
    )

    for m in pattern.finditer(raw):
        tool = (m.group(1) or "").strip()
        server = (m.group(2) or "").strip()
        name = (m.group(3) or "").strip()

        if name:
            queries.append(f"{server}.{tool}(name={name})")
        else:
            queries.append(f"{server}.{tool}")

    return queries


def parse_copilot_token_line(raw: str) -> tuple[int | None, int | None]:
    """
    Parse Copilot CLI token line if present:
      Tokens     ↑ 14.4k • ↓ 2.5k

    Returns:
      input_tokens, output_tokens
    """

    def parse_num(s: str) -> int:
        s = s.strip().lower()
        if s.endswith("k"):
            return int(float(s[:-1]) * 1000)
        return int(float(s))

    m = re.search(
        r"Tokens\s+↑\s*([0-9.]+k?)\s*•\s*↓\s*([0-9.]+k?)",
        raw,
        re.IGNORECASE,
    )

    if not m:
        return None, None

    return parse_num(m.group(1)), parse_num(m.group(2))


async def _stream_live(stream, prefix: str, buf: list[str]):
    while True:
        chunk = await stream.readline()
        if not chunk:
            break

        text = chunk.decode("utf-8", errors="replace")
        buf.append(text)

        print(f"[{prefix}] {text}", end="", flush=True)


async def call_copilot_cli(
    setup: str,
    model: str,
    prompt: str,
    rag_enabled: bool,
    provider_base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
    timeout: float = 600.0,
    max_retries: int = 2,
    retry_delay_sec: float = 5.0,
    workspace_dir: Optional[Path] = None,
) -> tuple[str, int, int, int, str, list[str]]:
    """
    Copilot CLI call with retries.

    Important behavior:
    - Uses the workspace_dir passed by run_one if provided.
    - Pre-creates main.c before launching Copilot.
    - Runs Copilot with cwd=workspace_dir.
    - Reads final code from workspace_dir/main.c.
    - Prepends final main.c contents to raw output as a C code block so existing
      extract_c_code(raw) behavior still works.
    """

    if provider_base_url is None:
        provider_base_url = LOCAL_BASE_URL

    if provider_api_key is None:
        provider_api_key = "dummy"

    attempts = max_retries + 1
    last_exc: Optional[Exception] = None
    last_raw = ""
    overall_t0 = time.time()

    # Use caller-provided workspace if available.
    # Do NOT overwrite this with OUT_DIR / f"copilot_workspace_{setup}".
    if workspace_dir is None:
        workspace = (OUT_DIR / f"copilot_workspace_{setup}").resolve()
    else:
        workspace = Path(workspace_dir).resolve()

    workspace.mkdir(parents=True, exist_ok=True)

    main_c_path = (workspace / "main.c").resolve()

    for attempt in range(1, attempts + 1):
        print(
            f"\n========== COPILOT ATTEMPT {attempt}/{attempts} ==========",
            flush=True,
        )
        print(f"setup={setup}", flush=True)
        print(f"model={model}", flush=True)
        print(f"rag_enabled={rag_enabled}", flush=True)
        print(f"provider_base_url={provider_base_url}", flush=True)
        print(f"workspace={workspace}", flush=True)
        print(f"main_c_path={main_c_path}", flush=True)
        print("=========================================================\n", flush=True)

        try:
            # Critical:
            # Pre-create main.c. Do NOT delete it.
            # Copilot is much more reliable when the target file exists.
            initial_main_c = (
                "/*\n"
                "Target file for Copilot evaluation.\n"
                "Replace this entire file with the complete C solution.\n"
                "*/\n"
            )

            main_c_path.write_text(initial_main_c, encoding="utf-8")

            exact_main_c_path = main_c_path.resolve()

            prompt_for_copilot = f"""{prompt}

You must use the MCP server tools for library/function research before coding.

IMPORTANT WORKSPACE INSTRUCTION:
- Your current working directory is:
  {workspace}
- The target file already exists.
- The exact file you must edit is:
  {exact_main_c_path}
- Replace the entire contents of that file with the complete C solution.
- Do not create any other files.
- Do not edit any other files.
- Do not only describe the solution in chat.
- Do not stop after research.
- Before finishing, make sure {exact_main_c_path} contains the final C code.
- The final chat response can be brief, but the actual solution must be written into that file.
"""

            copilot_home = prepare_copilot_home(
                f"{setup}_attempt_{attempt}",
                rag_enabled,
            )

            env = os.environ.copy()
            env.update(
                {
                    "COPILOT_HOME": str(copilot_home.resolve()),
                    "COPILOT_PROVIDER_TYPE": "openai",
                    "COPILOT_PROVIDER_BASE_URL": provider_base_url,
                    "COPILOT_PROVIDER_API_KEY": provider_api_key,
                    "COPILOT_MODEL": model,
                }
            )

            print("\n========== COPILOT LAUNCH DEBUG ==========", flush=True)
            print(f"setup={setup}", flush=True)
            print(f"model={model}", flush=True)
            print(f"rag_enabled={rag_enabled}", flush=True)
            print(f"provider_base_url={provider_base_url}", flush=True)
            print(f"workspace={workspace}", flush=True)
            print(f"cwd_for_copilot={workspace}", flush=True)
            print(f"main_c_path={main_c_path}", flush=True)
            print(f"main_c_exists={main_c_path.exists()}", flush=True)
            print(
                f"main_c_size={main_c_path.stat().st_size if main_c_path.exists() else 'missing'}",
                flush=True,
            )
            print(f"COPILOT_HOME={env.get('COPILOT_HOME')}", flush=True)
            print("prompt_first_1500_chars:", flush=True)
            print(prompt_for_copilot[:1500], flush=True)
            print("==========================================\n", flush=True)

            # -------------------------
            # MCP list check
            # -------------------------

            print("\n========== COPILOT MCP LIST ==========", flush=True)
            print(f"[DEBUG] COPILOT_HOME for mcp list = {env['COPILOT_HOME']}", flush=True)
            print(f"[DEBUG] cwd for mcp list = {workspace}", flush=True)

            mcp_proc = await asyncio.create_subprocess_exec(
                "copilot",
                "mcp",
                "list",
                env=env,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                mcp_stdout, mcp_stderr = await asyncio.wait_for(
                    mcp_proc.communicate(),
                    timeout=30.0,
                )

            except asyncio.TimeoutError:
                try:
                    mcp_proc.kill()
                except ProcessLookupError:
                    pass

                await mcp_proc.wait()

                raise RuntimeError("copilot mcp list timed out after 30 seconds")

            print(mcp_stdout.decode("utf-8", errors="replace"), flush=True)

            if mcp_stderr:
                print("[copilot mcp stderr]", flush=True)
                print(mcp_stderr.decode("utf-8", errors="replace"), flush=True)

            print("======================================\n", flush=True)

            # -------------------------
            # Main Copilot run
            # -------------------------

            if rag_enabled:
                cmd = [
                    "copilot",
                    "-p",
                    prompt_for_copilot,
                    "--allow-all-tools",
                    "--no-ask-user",
                ]
            else:
                cmd = [
                    "copilot",
                    "-p",
                    prompt_for_copilot,
                    "--no-ask-user",
                ]

            print("========== COPILOT RUN START ==========", flush=True)
            print(f"setup={setup}", flush=True)
            print(f"model={model}", flush=True)
            print(f"attempt={attempt}/{attempts}", flush=True)
            print(f"timeout_sec={int(timeout)}", flush=True)
            print(f"workspace={workspace}", flush=True)
            print(f"main_c_path={exact_main_c_path}", flush=True)
            print("command=", " ".join(cmd[:1] + ["-p", "<PROMPT>"] + cmd[3:]), flush=True)
            print("=======================================\n", flush=True)

            t0 = time.time()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_buf: list[str] = []
            stderr_buf: list[str] = []

            stdout_task = asyncio.create_task(
                _stream_live(proc.stdout, "copilot stdout", stdout_buf)
            )
            stderr_task = asyncio.create_task(
                _stream_live(proc.stderr, "copilot stderr", stderr_buf)
            )

            timed_out = False

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)

            except asyncio.TimeoutError:
                timed_out = True

                print(
                    f"\n[COPILOT TIMEOUT] Killing process after "
                    f"{int(timeout)} seconds",
                    flush=True,
                )

                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

                await proc.wait()

            await asyncio.gather(
                stdout_task,
                stderr_task,
                return_exceptions=True,
            )

            latency_ms = int((time.time() - t0) * 1000)

            stdout_text = "".join(stdout_buf)
            stderr_text = "".join(stderr_buf)

            transcript_raw = stdout_text

            if timed_out:
                transcript_raw += "\n\n[TIMEOUT]"

            if stderr_text.strip():
                transcript_raw += "\n\n[stderr]\n" + stderr_text

            generated_code = ""
            if main_c_path.exists():
                generated_code = main_c_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).strip()

            print("\n========== COPILOT EXIT DEBUG ==========", flush=True)
            print(f"returncode={proc.returncode}", flush=True)
            print(f"timed_out={timed_out}", flush=True)
            print(f"main_c_exists={main_c_path.exists()}", flush=True)
            print(f"main_c_path={main_c_path}", flush=True)
            print(f"main_c_chars={len(generated_code)}", flush=True)
            print("main_c_first_1000_chars_after:", flush=True)
            print(generated_code[:1000], flush=True)
            print("========================================\n", flush=True)

            # -------------------------
            # Retryable failure checks
            # -------------------------

            if timed_out:
                raise RuntimeError(
                    f"Copilot CLI timed out after {int(timeout)} seconds"
                )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Copilot CLI exited with non-zero returncode={proc.returncode}"
                )

            # If Copilot left the placeholder unchanged, it did not solve the task.
            placeholder_stripped = initial_main_c.strip()

            if not generated_code or generated_code == placeholder_stripped:
                raise RuntimeError(
                    f"Copilot did not write generated code to main.c at {exact_main_c_path}"
                )

            # Also reject if it only left a comment-like placeholder.
            if (
                "Target file for Copilot evaluation" in generated_code
                and "Replace this entire file" in generated_code
                and len(generated_code) < 500
            ):
                raise RuntimeError(
                    f"Copilot left placeholder content in main.c at {exact_main_c_path}"
                )

            # Put main.c first so existing extract_c_code(raw) picks this code.
            raw = (
                "```c\n"
                f"{generated_code}\n"
                "```\n\n"
                "[COPILOT_TRANSCRIPT]\n"
                f"{transcript_raw}"
            )

            last_raw = raw

            parsed_input_tokens, parsed_output_tokens = parse_copilot_token_line(
                transcript_raw
            )

            if parsed_input_tokens is not None and parsed_output_tokens is not None:
                input_tokens = parsed_input_tokens
                output_tokens = parsed_output_tokens
                token_source = "copilot_cli_stdout"
            else:
                input_tokens = approx_tokens(prompt_for_copilot)
                output_tokens = approx_tokens(transcript_raw)
                token_source = "estimated"

            rag_queries = extract_mcp_queries(transcript_raw)

            print("\n========== COPILOT RUN END ==========", flush=True)
            print(f"setup={setup}", flush=True)
            print(f"attempt={attempt}/{attempts}", flush=True)
            print(f"returncode={proc.returncode}", flush=True)
            print(f"timed_out={timed_out}", flush=True)
            print(f"latency_ms={latency_ms}", flush=True)
            print(f"input_tokens={input_tokens}", flush=True)
            print(f"output_tokens={output_tokens}", flush=True)
            print(f"token_source={token_source}", flush=True)
            print(f"rag_query_count={len(rag_queries)}", flush=True)
            print(f"main_c_path={exact_main_c_path}", flush=True)
            print(f"main_c_bytes={len(generated_code.encode('utf-8'))}", flush=True)

            for q in rag_queries:
                print(f"rag_query={q}", flush=True)

            print("=====================================\n", flush=True)

            return (
                raw,
                latency_ms,
                input_tokens,
                output_tokens,
                token_source,
                rag_queries,
            )

        except Exception as e:
            last_exc = e

            print(
                f"[WARN] copilot attempt={attempt}/{attempts} failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

            if attempt < attempts:
                sleep_sec = retry_delay_sec * attempt

                print(
                    f"[copilot] retrying after {sleep_sec:.1f}s",
                    flush=True,
                )

                await asyncio.sleep(sleep_sec)

    total_latency_ms = int((time.time() - overall_t0) * 1000)

    raw_excerpt = last_raw[-2000:] if last_raw else ""

    raise RuntimeError(
        f"call_copilot_cli failed after {attempts} attempts "
        f"in {total_latency_ms} ms. "
        f"Last error: "
        f"{type(last_exc).__name__ if last_exc else 'UnknownError'}: {last_exc}\n"
        f"Last raw excerpt:\n{raw_excerpt}"
    ) from last_exc

# =========================
# Direct OpenAI-compatible local call
# =========================

async def call_direct_local(
    model: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant. <|think|>",
    timeout: float = 180.0,
    max_retries: int = 2,
    retry_delay_sec: float = 5.0,
) -> tuple[str, int, Optional[int], Optional[int], str]:
    """
    Direct OpenAI-compatible local call with retries.

    max_retries=2 means:
      - attempt 1
      - retry 1
      - retry 2

    Total attempts = 3.
    """

    attempts = max_retries + 1
    last_exc: Optional[Exception] = None
    overall_t0 = time.time()

    for attempt in range(1, attempts + 1):
        try:
            print(
                f"[direct_local] attempt={attempt}/{attempts} model={model}",
                flush=True,
            )

            client = OpenAI(
                base_url=LOCAL_BASE_URL,
                api_key="dummy",
            )

            t0 = time.time()

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                timeout=timeout,
            )

            latency_ms = int((time.time() - t0) * 1000)

            raw = resp.choices[0].message.content or ""

            usage = getattr(resp, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                token_source = "provider_usage"
            else:
                input_tokens = approx_tokens(system_prompt + "\n" + prompt)
                output_tokens = approx_tokens(raw)
                token_source = "estimated"

            return raw, latency_ms, input_tokens, output_tokens, token_source

        except Exception as e:
            last_exc = e

            print(
                f"[WARN] direct_local attempt={attempt}/{attempts} failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

            if attempt < attempts:
                sleep_sec = retry_delay_sec * attempt
                print(
                    f"[direct_local] retrying after {sleep_sec:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(sleep_sec)

    latency_ms = int((time.time() - overall_t0) * 1000)

    raise RuntimeError(
        f"call_direct_local failed after {attempts} attempts "
        f"in {latency_ms} ms: "
        f"{type(last_exc).__name__ if last_exc else 'UnknownError'}: {last_exc}"
    ) from last_exc

# =========================
# Run one setup
# =========================

async def run_one(problem: Problem, setup_cfg: dict[str, Any]) -> RunResult:
    setup = setup_cfg["id"]
    model = setup_cfg["model"]
    invocation = setup_cfg["invocation"]
    rag_enabled = setup_cfg.get("rag_enabled", False)

    required = problem.metadata.get("required", [])

    workspace_dir: Optional[Path] = None
    main_c_path: Optional[Path] = None

    if invocation == "copilot":
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", problem.id)
        safe_setup = re.sub(r"[^A-Za-z0-9_.-]+", "_", setup)

        workspace_dir = (
            OUT_DIR
            / f"copilot_workspace_{safe_setup}"
            / safe_task_id
        ).resolve()

        workspace_dir.mkdir(parents=True, exist_ok=True)

        main_c_path = (workspace_dir / "main.c").resolve()

        # Critical: Copilot is much more reliable if the file already exists.
        if not main_c_path.exists():
            main_c_path.write_text(
                "/* Write the complete solution for this task here. */\n",
                encoding="utf-8",
            )

        prompt = make_prompt(problem.question, required, main_c_path=main_c_path)
    else:
        prompt = make_prompt(problem.question, required)

    run_t0 = time.time()

    try:
        if invocation == "copilot":
            raw, latency_ms, input_tokens, output_tokens, token_source, rag_queries = await call_copilot_cli(
                setup=setup,
                model=model,
                prompt=prompt,
                rag_enabled=rag_enabled,
                provider_base_url=setup_cfg.get("provider_base_url", LOCAL_BASE_URL),
                provider_api_key=setup_cfg.get("provider_api_key", "dummy"),
                workspace_dir=workspace_dir,
            )

        elif invocation == "direct":
            raw, latency_ms, input_tokens, output_tokens, token_source = await call_direct_local(
                model=model,
                prompt=prompt,
            )
            rag_queries = []

        else:
            raise ValueError(f"Unknown invocation: {invocation}")

    except Exception as e:
        latency_ms = int((time.time() - run_t0) * 1000)

        error_text = (
            f"[GENERATION_FAILED]\n"
            f"setup={setup}\n"
            f"model={model}\n"
            f"invocation={invocation}\n"
            f"rag_enabled={rag_enabled}\n"
            f"error_type={type(e).__name__}\n"
            f"error={e}\n"
        )

        print(
            f"[ERROR] setup={setup} task={problem.id} generation failed "
            f"after retries, marking as fail and moving on: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        input_tokens = approx_tokens(prompt)
        output_tokens = approx_tokens(error_text)
        total_tokens = input_tokens + output_tokens

        return RunResult(
            task_id=problem.id,
            setup=setup,
            model=model,
            invocation=invocation,
            rag_enabled=rag_enabled,

            prompt=prompt,
            raw_response=error_text,
            extracted_code="",

            required=required,
            called=[],
            missing=required,
            required_pass=False,

            compile_pass=False,
            compile_logs=error_text,

            hard_pass=False,

            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_source="estimated_after_failure",

            latency_ms=latency_ms,

            rag_query_count=0,
            rag_queries=[],
        )

    if invocation == "copilot" and main_c_path is not None and main_c_path.exists():
        code = main_c_path.read_text(encoding="utf-8", errors="replace").strip()

        if not code:
            code = extract_c_code(raw)
    else:
        code = extract_c_code(raw)

    verify = verify_code(problem, code)

    called = verify.details.get("called", [])
    missing = verify.details.get("missing", required)
    compile_logs = verify.details.get("compile_logs", "") or ""

    required_pass = len(missing) == 0
    compile_pass = bool(verify.compiled)
    hard_pass = required_pass and compile_pass

    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return RunResult(
        task_id=problem.id,
        setup=setup,
        model=model,
        invocation=invocation,
        rag_enabled=rag_enabled,

        prompt=prompt,
        raw_response=raw,
        extracted_code=code,

        required=required,
        called=called,
        missing=missing,
        required_pass=required_pass,

        compile_pass=compile_pass,
        compile_logs=compile_logs,

        hard_pass=hard_pass,

        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        token_source=token_source,

        latency_ms=latency_ms,

        rag_query_count=len(rag_queries),
        rag_queries=rag_queries,
    )

# =========================
# Judge
# Local judge in --dev mode.
# Cloud judge when --dev is not passed.
# =========================

def make_judge_prompt(problem: Problem, results: list[RunResult]) -> str:
    compact = []

    for r in results:
        compact.append(
            {
                "setup": r.setup,
                "model": r.model,
                "invocation": r.invocation,
                "rag_enabled": r.rag_enabled,

                "code": r.extracted_code,
                "required": r.required,
                "called": r.called,
                "missing": r.missing,
                "required_pass": r.required_pass,

                "compile_pass": r.compile_pass,
                "compile_logs": r.compile_logs,

                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "latency_ms": r.latency_ms,

                "rag_query_count": r.rag_query_count,
                "rag_queries": r.rag_queries,
            }
        )

    return f"""You are judging multiple model outputs for the same C code-generation task.

Judge relatively, not absolutely.

Task:
{problem.question}

Required functions:
{problem.metadata.get("required", [])}

You are given each setup's code, compile result, required-function result, token count, latency, and RAG queries.
Use only the RAG queries, not hidden retrieved responses.

Categories:
1. compile_result
2. required_function_usage
3. efficiency
4. rag_efficiency

Rules:
- A model that compiles and calls all required functions should usually beat one that does not.
- If two models both pass, prefer the one with simpler code, fewer tokens, lower latency, and fewer unnecessary RAG queries.
- If RAG was used but the result still failed, penalize it.
- If the RL/no-RAG model matches RAG quality with much less effort, reward that.
- Scores must be relative 0-10 among these candidates.

Return only valid JSON:
{{
  "ranking": ["best_setup", "second_setup", "third_setup"],
  "scores": {{
    "setup_id": {{
      "compile_result": 0,
      "required_function_usage": 0,
      "efficiency": 0,
      "rag_efficiency": 0,
      "overall": 0
    }}
  }},
  "reasoning": {{
    "setup_id": "brief reason"
  }},
  "winner": "setup_id"
}}

Candidates:
{json.dumps(compact, indent=2)}
"""


async def judge_task(
    problem: Problem,
    results: list[RunResult],
    dev: bool,
) -> dict[str, Any]:
    if dev:
        judge_base_url = LOCAL_BASE_URL
        judge_api_key = "dummy"
        judge_model = LOCAL_JUDGE_MODEL
        judge_mode = "local_dev_judge"
    else:
        judge_base_url = CLOUD_BASE_URL
        judge_api_key = CLOUD_API_KEY
        judge_model = CLOUD_JUDGE_MODEL
        judge_mode = "cloud_judge"

        if not judge_api_key:
            raise RuntimeError(
                "Cloud judge requested, but no API key was found. "
                "Set OPENAI_API_KEY or CLOUD_LLM_API_KEY, or run with --dev."
            )

    client = OpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key,
    )

    prompt = make_judge_prompt(problem, results)

    print("\n========== JUDGE START ==========", flush=True)
    print(f"judge_mode={judge_mode}", flush=True)
    print(f"judge_model={judge_model}", flush=True)
    print(f"judge_base_url={judge_base_url}", flush=True)
    print("=================================\n", flush=True)

    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful evaluation judge. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
    )

    raw = resp.choices[0].message.content or ""

    try:
        parsed = json.loads(raw)
        parsed["_judge_mode"] = judge_mode
        parsed["_judge_model"] = judge_model
        return parsed

    except json.JSONDecodeError:
        return {
            "error": "judge_json_parse_failed",
            "raw": raw,
            "_judge_mode": judge_mode,
            "_judge_model": judge_model,
        }


def build_setup_registry() -> dict[str, dict[str, Any]]:
    """
    Registry of all known generation setups.

    Generation phase chooses from this registry using --setups.
    Judge phase chooses setup ids from stored results using --judge-setups.

    Important:
    - Cloud API key is only required if cloud_rag_copilot is selected.
    - RL setup can exist here even if you do not run it yet.
    """

    return {
        "local_rag_copilot": {
            "id": "local_rag_copilot",
            "model": ORIGINAL_MODEL,
            "invocation": "copilot",
            "rag_enabled": True,
            "provider_base_url": LOCAL_BASE_URL,
            "provider_api_key": "dummy",
        },

        "cloud_rag_copilot": {
            "id": "cloud_rag_copilot",
            "model": CLOUD_CODE_MODEL,
            "invocation": "copilot",
            "rag_enabled": True,
            "provider_base_url": CLOUD_BASE_URL,
            "provider_api_key": CLOUD_API_KEY,
        },

        "rl_copilot": {
            "id": "rl_copilot",
            "model": RL_MODEL,
            "invocation": "copilot",
            "rag_enabled": False,
            "provider_base_url": LOCAL_BASE_URL,
            "provider_api_key": "dummy",
        },

        "rl_direct": {
            "id": "rl_direct",
            "model": RL_MODEL,
            "invocation": "direct",
            "rag_enabled": False,
        },
    }


def select_setups(setup_ids: list[str]) -> list[dict[str, Any]]:
    """
    Select generation setups by id.

    Example:
      --setups local_rag_copilot
      --setups cloud_rag_copilot
      --setups local_rag_copilot cloud_rag_copilot
    """

    registry = build_setup_registry()

    selected = []

    for setup_id in setup_ids:
        if setup_id not in registry:
            raise ValueError(
                f"Unknown setup id: {setup_id}. "
                f"Available setups: {sorted(registry.keys())}"
            )

        setup = registry[setup_id]

        if setup_id == "cloud_rag_copilot" and not CLOUD_API_KEY:
            raise RuntimeError(
                "cloud_rag_copilot selected, but no cloud API key was found. "
                "Set OPENAI_API_KEY or CLOUD_LLM_API_KEY."
            )

        selected.append(setup)

    return selected

def load_results_latest(results_path: Path) -> dict[str, dict[str, RunResult]]:
    """
    Load accumulated generation results.

    Returns:
      {
        task_id: {
          setup_id: latest RunResult for that task/setup
        }
      }

    If the same task_id + setup appears multiple times, the latest row wins.
    This lets you rerun failed generations and judge will naturally use the newest result.
    """

    latest: dict[str, dict[str, RunResult]] = defaultdict(dict)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file does not exist: {results_path}. "
            "Run generation phase first."
        )

    with open(results_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] Skipping invalid JSON line in {results_path} "
                    f"line={line_no}: {e}",
                    flush=True,
                )
                continue

            task_id = row.get("task_id")
            setup = row.get("setup")

            if not task_id or not setup:
                print(
                    f"[WARN] Skipping result row missing task_id/setup "
                    f"line={line_no}: {row}",
                    flush=True,
                )
                continue

            try:
                result = RunResult(**row)
            except TypeError as e:
                print(
                    f"[WARN] Skipping result row that does not match RunResult "
                    f"line={line_no}: {e}",
                    flush=True,
                )
                continue

            latest[task_id][setup] = result

    return latest


async def run_generation_phase(
    tasks_path: str,
    setup_ids: list[str],
    limit: Optional[int] = None,
):
    """
    Generation-only phase.

    Behavior:
    - Runs selected setup ids only.
    - Appends each RunResult to eval_runs/results.jsonl.
    - Does NOT run judge.
    """

    problems = load_tasks(tasks_path)

    if limit is not None:
        problems = problems[:limit]

    setups = select_setups(setup_ids)

    print("\n========== GENERATION PHASE ==========", flush=True)
    print(f"tasks_path={tasks_path}", flush=True)
    print(f"num_tasks={len(problems)}", flush=True)
    print(f"results_path={OUT_DIR / 'results.jsonl'}", flush=True)
    print("selected_setups:", flush=True)

    for s in setups:
        print(
            f"- id={s['id']} "
            f"model={s['model']} "
            f"invocation={s['invocation']} "
            f"rag_enabled={s.get('rag_enabled', False)} "
            f"base_url={s.get('provider_base_url', 'n/a')}",
            flush=True,
        )

    print("======================================\n", flush=True)

    # Print MCP tools once for debugging.
    # If this fails, do not stop the whole eval.
    try:
        await print_mcp_tools_direct()
    except Exception as e:
        print(f"[WARN] Failed to print MCP tools directly: {e}", flush=True)

    all_results_path = OUT_DIR / "results.jsonl"

    # IMPORTANT: append mode.
    with open(all_results_path, "a", encoding="utf-8") as rf:
        for problem in problems:
            print(f"\n========== TASK {problem.id} ==========", flush=True)

            for setup in setups:
                print(
                    f"[GENERATE] task={problem.id} "
                    f"setup={setup['id']} "
                    f"model={setup['model']} "
                    f"invocation={setup['invocation']} "
                    f"rag_enabled={setup.get('rag_enabled', False)}",
                    flush=True,
                )

                try:
                    result = await run_one(problem, setup)

                except Exception as e:
                    # This should rarely trigger because run_one already catches
                    # generation errors and returns a failure RunResult.
                    print(
                        f"[ERROR] setup={setup['id']} task={problem.id} "
                        f"unexpected failure: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    raise

                rf.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                rf.flush()

                print(
                    f"[SAVED] task={result.task_id} "
                    f"setup={result.setup} "
                    f"required_pass={result.required_pass} "
                    f"compile_pass={result.compile_pass} "
                    f"hard_pass={result.hard_pass} "
                    f"latency_ms={result.latency_ms} "
                    f"tokens={result.total_tokens}",
                    flush=True,
                )

    print(f"\nSaved appended generation results to: {all_results_path}", flush=True)


async def run_judge_phase(
    tasks_path: str,
    judge_setup_ids: list[str],
    dev: bool,
    limit: Optional[int] = None,
):
    """
    Judge-only phase.

    Behavior:
    - Reads eval_runs/results.jsonl.
    - Uses latest row for each task_id + setup.
    - Judges only tasks where all requested judge_setup_ids exist.
    - Appends judge outputs to eval_runs/judge.jsonl.
    """

    problems = load_tasks(tasks_path)

    if limit is not None:
        problems = problems[:limit]

    results_path = OUT_DIR / "results.jsonl"
    judge_path = OUT_DIR / "judge.jsonl"

    latest = load_results_latest(results_path)

    print("\n========== JUDGE PHASE ==========", flush=True)
    print(f"tasks_path={tasks_path}", flush=True)
    print(f"num_tasks={len(problems)}", flush=True)
    print(f"results_path={results_path}", flush=True)
    print(f"judge_path={judge_path}", flush=True)
    print(f"dev={dev}", flush=True)
    print(f"judge_setup_ids={judge_setup_ids}", flush=True)

    if dev:
        print("judge_mode=DEV/local judge", flush=True)
        print(f"local_base_url={LOCAL_BASE_URL}", flush=True)
        print(f"local_judge_model={LOCAL_JUDGE_MODEL}", flush=True)
    else:
        print("judge_mode=FULL/cloud judge", flush=True)
        print(f"cloud_base_url={CLOUD_BASE_URL}", flush=True)
        print(f"cloud_judge_model={CLOUD_JUDGE_MODEL}", flush=True)

        if not CLOUD_API_KEY:
            raise RuntimeError(
                "Cloud judge requested, but no API key was found. "
                "Set OPENAI_API_KEY or CLOUD_LLM_API_KEY, or run judge with --dev."
            )

    print("=================================\n", flush=True)

    judged_count = 0
    skipped_count = 0

    # IMPORTANT: append mode.
    with open(judge_path, "a", encoding="utf-8") as jf:
        for problem in problems:
            by_setup = latest.get(problem.id, {})

            missing = [
                setup_id
                for setup_id in judge_setup_ids
                if setup_id not in by_setup
            ]

            if missing:
                skipped_count += 1

                print(
                    f"[SKIP JUDGE] task={problem.id} "
                    f"missing_setups={missing}",
                    flush=True,
                )

                # Keep a record that this task was skipped.
                jf.write(
                    json.dumps(
                        {
                            "task_id": problem.id,
                            "dev": dev,
                            "judge_setup_ids": judge_setup_ids,
                            "error": "missing_setups",
                            "missing": missing,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                jf.flush()

                continue

            task_results = [
                by_setup[setup_id]
                for setup_id in judge_setup_ids
            ]

            print(
                f"[JUDGE] task={problem.id} "
                f"setups={judge_setup_ids}",
                flush=True,
            )

            try:
                judge = await judge_task(problem, task_results, dev=dev)

            except Exception as e:
                print(
                    f"[WARN] Judge failed for task={problem.id}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                judge = {
                    "error": "judge_failed",
                    "message": str(e),
                    "dev": dev,
                }

            jf.write(
                json.dumps(
                    {
                        "task_id": problem.id,
                        "dev": dev,
                        "judge_setup_ids": judge_setup_ids,
                        "judge": judge,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            jf.flush()

            judged_count += 1

    print("\n========== JUDGE SUMMARY ==========", flush=True)
    print(f"judged_count={judged_count}", flush=True)
    print(f"skipped_count={skipped_count}", flush=True)
    print(f"judge_path={judge_path}", flush=True)
    print("===================================\n", flush=True)

async def main(args: argparse.Namespace):
    if args.phase == "generate":
        await run_generation_phase(
            tasks_path=args.tasks_path,
            setup_ids=args.setups,
            limit=args.limit,
        )
        return

    if args.phase == "judge":
        await run_judge_phase(
            tasks_path=args.tasks_path,
            judge_setup_ids=args.judge_setups,
            dev=args.dev,
            limit=args.limit,
        )
        return

    raise ValueError(f"Unknown phase: {args.phase}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate C code-generation models in separated phases: "
            "generation and judge."
        )
    )

    parser.add_argument(
        "tasks_path",
        help="Path to tasks.jsonl",
    )

    parser.add_argument(
        "--phase",
        choices=["generate", "judge"],
        required=True,
        help=(
            "Phase to run. "
            "'generate' appends model outputs to results.jsonl. "
            "'judge' reads results.jsonl and appends judge outputs to judge.jsonl."
        ),
    )

    parser.add_argument(
        "--setups",
        nargs="+",
        default=["local_rag_copilot"],
        help=(
            "Generation setup ids to run. Used only with --phase generate. "
            "Examples: local_rag_copilot, cloud_rag_copilot, rl_copilot, rl_direct."
        ),
    )

    parser.add_argument(
        "--judge-setups",
        nargs="+",
        default=[
            "local_rag_copilot",
            "cloud_rag_copilot",
            "rl_copilot",
        ],
        help=(
            "Setup ids to compare during judge phase. "
            "Used only with --phase judge. "
            "Judge uses latest result for each task_id + setup."
        ),
    )

    # parser.add_argument(
    #     "--dev",
    #     action="store_true",
    #     help=(
    #         "For judge phase only: use local judge instead of cloud judge. "
    #         "Generation setup selection is controlled by --setups, not by --dev."
    #     ),
    # )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of tasks to run. Useful for smoke tests. "
            "Example: --limit 3"
        ),
    )

    args = parser.parse_args()

    if args.phase == "generate" and not args.setups:
        raise ValueError("--phase generate requires at least one --setups value")

    if args.phase == "judge" and not args.judge_setups:
        raise ValueError("--phase judge requires at least one --judge-setups value")

    return args

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
