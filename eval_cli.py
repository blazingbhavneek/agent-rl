import os
import re
import json
import time
import asyncio
import argparse
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import requests
from openai import OpenAI
from collections import defaultdict
from tree_sitter import Language, Parser
from tree_sitter_c import language as c_language
import subprocess

# region 1: Configuration

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
OUT_DIR = Path("eval_runs_gpt")
OUT_DIR.mkdir(exist_ok=True)

# endregion


# region 2: Data structures

@dataclass
class Problem:
    id: str
    question: str
    metadata: dict[str, Any]
    # The reference answer is used only by the absolute judge, never generation.
    answer: str = ""


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

    # Defaults keep older results.jsonl rows readable.
    created_at: str = ""


# endregion


# region 3: Local C call analysis

class FunctionCallAnalyzer:
    def __init__(self):
        self.parser = Parser(Language(c_language()))

    def extract_called_functions(self, code: str) -> set[str]:
        tree = self.parser.parse(code.encode("utf8"))
        called = set()

        def visit(node):
            if node.type == "call_expression":
                fn = node.child_by_field_name("function")
                # A member/function-pointer expression (for example s.fn())
                # is not evidence of a direct call to the required library API.
                if fn and fn.type == "identifier":
                    called.add(fn.text.decode())

            for c in node.children:
                visit(c)

        visit(tree.root_node)
        return called

    def extract_defined_functions(self, code: str) -> set[str]:
        """Return function names implemented in the generated source."""
        tree = self.parser.parse(code.encode("utf8"))
        defined = set()

        def declarator_name(node):
            if node.type == "identifier":
                return node.text.decode()
            for child in node.children:
                name = declarator_name(child)
                if name:
                    return name
            return None

        def visit(node):
            if node.type == "function_definition":
                declarator = node.child_by_field_name("declarator")
                if declarator:
                    name = declarator_name(declarator)
                    if name:
                        defined.add(name)
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return defined


analyzer = FunctionCallAnalyzer()


# endregion


# region 4: Task and prompt helpers

def extract_c_code(text: str) -> str:
    m = re.search(r"```(?:c)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # fallback: accept whole answer as code
    return text.strip()


def approx_tokens(text: str) -> int:
    # Development fallback only.
    # Later replace with provider/proxy usage logs for cloud.
    return max(1, len(text) // 4)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_prompt(
    question: str,
    required: list[str],
    main_c_path: Optional[Path] = None,
    rag_enabled: bool = False,
) -> str:
    required_text = (
        "\n".join(f"- {name}" for name in required)
        if required
        else "- No specific library function is mandated."
    )

    if main_c_path is not None:
        path = main_c_path.resolve()
        target_file_text = (
            "Write the code here:\n"
            f"{path}\n"
        )
        write_rule = (
            "- Create the file if it does not exist.\n"
            f"- Write/overwrite the complete C solution at:\n  {path} if it already exists\n"
        )
    else:
        target_file_text = "You are responding through a chat-completions API.\n"
        write_rule = (
            "- Return the complete C solution in one fenced ```c code block.\n"
        )

    research_rule = (
        "- Use the configured MCP tools to research required library functions before coding.\n"
        if rag_enabled
        else "- No MCP tools are available for this run.\n"
    )

    return f"""Solve this C code-generation task.

Code rules:
CPP Style comments are not allowed: Use /* <Comment Body> */ instead of // <Comment Body>
Dont declare variable inside for loops
Always have a main function in the code.
Do not forward declare library functions using extern, declared functions will not be compiled. Just including their headers is enough.
Dont try to run gcc commands or try to compile yourself, the written code will be tested in another location, dont try to find its headers, use the search tools to read about library and only then write code. Make no mistakes.

{target_file_text}

Rules:
{write_rule}{research_rule}- Do not create or edit any other files.
- Do not ask the user questions.
- Do not include explanations.
- The following required library functions must be called directly when they are listed:
{required_text}

Task:
{question}
"""

def load_tasks(path: str) -> list[Problem]:
    problems = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e

            if not isinstance(row, dict):
                raise ValueError(f"Task row {line_no} must be a JSON object")

            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Task row {line_no} must contain a non-empty question")

            required = row.get("required", [])
            if not isinstance(required, list) or not all(
                isinstance(name, str) and name.strip() for name in required
            ):
                raise ValueError(
                    f"Task row {line_no} field 'required' must be a list of non-empty strings"
                )

            answer = row.get("answer", "")
            if answer is None:
                answer = ""
            if not isinstance(answer, str):
                raise ValueError(f"Task row {line_no} field 'answer' must be a string")

            task_id = row.get("id") or f"task_{len(problems):05d}"
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"Task row {line_no} field 'id' must be a non-empty string")

            problems.append(
                Problem(
                    id=task_id,
                    question=question,
                    metadata={"required": required},
                    answer=answer,
                )
            )

    return problems


# endregion


# region 5: Verification

def verify_code(
    problem: Problem,
    code: str,
    apl_variant: int = 2,
    timeout: float = 60.0,
) -> VerifyResult:
    required = problem.metadata.get("required", [])

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

    try:
        data = r.json()
    except ValueError as e:
        return VerifyResult(
            compiled=False,
            passed=0,
            total=len(required),
            error=f"Verifier returned invalid JSON: {e}",
            details={
                "required": required,
                "missing": required,
                "compile_logs": r.text,
            },
        )

    if not isinstance(data, dict):
        return VerifyResult(
            compiled=False,
            passed=0,
            total=len(required),
            error="Verifier returned a JSON value other than an object",
            details={
                "required": required,
                "missing": required,
                "compile_logs": r.text,
            },
        )

    # Be strict here: a malformed value such as the string "false" must not
    # be treated as a successful compile.
    compiled = data.get("compiled") is True

    compile_logs = (
        data.get("compile_logs")
        or data.get("logs")
        or data.get("error")
        or ""
    )

    try:
        called = analyzer.extract_called_functions(code)
        defined = analyzer.extract_defined_functions(code)
    except Exception as e:
        return VerifyResult(
            compiled=compiled,
            passed=0,
            total=len(required),
            error=f"Local required-call analysis failed: {e}",
            details={
                "required": required,
                "called": [],
                "missing": required,
                "compile_logs": compile_logs,
                "required_pass": False,
                "external_pass": False,
                "hard_pass": False,
            },
        )

    # Do not count calls that can be satisfied by generated local definitions,
    # generated macros, or declared callbacks instead of the intended library API.
    macro_names = set(
        re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)", code, re.MULTILINE)
    )
    function_pointer_names = set(
        re.findall(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\(", code)
    )
    shadowed = defined | macro_names | function_pointer_names
    shadowed_required = [fn for fn in required if fn in shadowed]

    missing = [fn for fn in required if fn not in called or fn in shadowed]
    present = [fn for fn in required if fn in called and fn not in shadowed]

    total_external = data.get("total_external_functions_executed")
    total_correct = data.get("total_correct_functions_executed")

    details = {
        "required": required,
        "called": sorted(called),
        "present": present,
        "missing": missing,
        "shadowed_required": shadowed_required,
        "compile_logs": compile_logs,
        "total_external_functions_executed": total_external,
        "total_correct_functions_executed": total_correct,
    }

    # Your rule:
    # if any required function missing => fail
    required_pass = len(missing) == 0

    # Also keep your external-function rule if server provides it.
    external_ok = True
    if required and total_external is not None and total_correct is not None:
        if total_external == 0 or total_correct != total_external:
            external_ok = False

    hard_pass = compiled and required_pass and external_ok

    if hard_pass:
        error = None
    else:
        error = (
            f"compiled={compiled}; missing={missing}; "
            f"shadowed_required={shadowed_required}; "
            f"external_pass={external_ok}; compile_logs={compile_logs}"
        )

    details["required_pass"] = required_pass
    details["external_pass"] = external_ok
    details["hard_pass"] = hard_pass

    return VerifyResult(
        # This is strictly the compiler result. Requirement and external checks
        # are reported separately so judge inputs remain truthful.
        compiled=compiled,
        passed=len(present),
        total=len(required),
        error=error,
        details=details,
    )


# endregion


# region 6: Copilot CLI invocation and MCP trace parsing

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
            timeout=30.0,
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
        r"●\s+([^\n]+?)\s+\(MCP:\s*([^)]+)\)([^\n]*)"
    )

    for m in pattern.finditer(raw):
        tool = (m.group(1) or "").strip()
        server = (m.group(2) or "").strip()
        tail = m.group(3) or ""
        name_match = re.search(r"\bname:\s*\"([^\"]+)\"", tail)
        name = name_match.group(1).strip() if name_match else ""

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
    failed_attempt_input_tokens = 0
    failed_attempt_output_tokens = 0
    failed_attempt_rag_queries: list[str] = []

    # Use caller-provided workspace if available.
    # Do NOT overwrite this with OUT_DIR / f"copilot_workspace_{setup}".
    if workspace_dir is None:
        workspace = (OUT_DIR / f"copilot_workspace_{setup}").resolve()
    else:
        workspace = Path(workspace_dir).resolve()

    workspace.mkdir(parents=True, exist_ok=True)

    main_c_path = (workspace / "main.c").resolve()
    # Each invocation gets its own COPILOT_HOME, preventing stale MCP entries
    # from a previous task or resumed run from colliding with `mcp add`.
    run_token = f"{setup}_{workspace.name}_{os.getpid()}_{time.time_ns()}"

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

        attempt_transcript = ""
        prompt_for_copilot = prompt
        copilot_started = False

        try:
            # Critical:
            # Pre-create main.c as a blank file.
            # Copilot is much more reliable when the target file exists.
            initial_main_c = ""

            # TODO: Run it back
            # main_c_path.write_text(initial_main_c, encoding="utf-8")

            exact_main_c_path = main_c_path.resolve()

            mcp_instruction = (
                "You must use the MCP server tools for library/function research before coding."
                if rag_enabled
                else "No MCP server is configured for this run; do not attempt tool research."
            )

            prompt_for_copilot = f"""{prompt}

{mcp_instruction}

IMPORTANT WORKSPACE INSTRUCTION:
- Your current working directory is:
  {workspace}
- The exact file you must create is:
  {exact_main_c_path}
- Do not create any other files.
- Do not edit any other files.
- Do not only describe the solution in chat.
- Do not stop after research.
- Before finishing, make sure {exact_main_c_path} contains the final C code.
- The final chat response can be brief, but the actual solution must be written into that file.
"""

            copilot_home = prepare_copilot_home(
                f"{run_token}_attempt_{attempt}",
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

            if rag_enabled:
                # Confirm that the unique home actually exposes MCP before
                # launching a RAG evaluation; a non-zero list result is not a
                # valid RAG run.
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
                if mcp_proc.returncode != 0:
                    raise RuntimeError(
                        f"copilot mcp list failed with returncode={mcp_proc.returncode}"
                    )
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
            print(f"setup = {setup}", flush=True)
            print(f"model = {model}", flush=True)
            print(f"attempt = {attempt}/{attempts}", flush=True)
            print(f"timeout_sec = {int(timeout)}", flush=True)
            print(f"workspace = {workspace}", flush=True)
            print(f"main_c_path = {exact_main_c_path}", flush=True)
            print("command = ", " ".join(cmd[:1] + ["-p", "<PROMPT>"] + cmd[3:]), flush=True)
            print("=======================================\n", flush=True)

            copilot_started = True
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

            stdout_text = "".join(stdout_buf)
            stderr_text = "".join(stderr_buf)

            transcript_raw = stdout_text

            if timed_out:
                transcript_raw += "\n\n[TIMEOUT]"

            if stderr_text.strip():
                transcript_raw += "\n\n[stderr]\n" + stderr_text

            attempt_transcript = transcript_raw

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

            # If main.c is still blank, it means Copilot didn't write to the file.
            # Check if it outputted the code in stdout instead.
            if not generated_code:
                stdout_code = extract_c_code(transcript_raw)
                if not stdout_code:
                    raise RuntimeError(
                        f"Copilot did not write generated code to main.c and did not output code in stdout at {exact_main_c_path}"
                    )

            # Put main.c first so existing extract_c_code(raw) picks this code,
            # but only if it actually contains code.
            if generated_code:
                raw = (
                    "```c\n"
                    f"{generated_code}\n"
                    "```\n\n"
                    "[COPILOT_TRANSCRIPT]\n"
                    f"{transcript_raw}"
                )
            else:
                # If main.c is blank, just use the transcript so extract_c_code 
                # can find the code in stdout.
                raw = transcript_raw

            last_raw = raw

            parsed_input_tokens, parsed_output_tokens = parse_copilot_token_line(
                transcript_raw
            )

            if parsed_input_tokens is not None and parsed_output_tokens is not None:
                input_tokens = parsed_input_tokens + failed_attempt_input_tokens
                output_tokens = parsed_output_tokens + failed_attempt_output_tokens
                token_source = (
                    "copilot_cli_stdout"
                    if not failed_attempt_input_tokens and not failed_attempt_output_tokens
                    else "copilot_cli_stdout_plus_estimated_retries"
                )
            else:
                input_tokens = failed_attempt_input_tokens + approx_tokens(prompt_for_copilot)
                output_tokens = failed_attempt_output_tokens + approx_tokens(transcript_raw)
                token_source = "estimated_all_attempts"

            rag_queries = failed_attempt_rag_queries + extract_mcp_queries(transcript_raw)
            latency_ms = int((time.time() - overall_t0) * 1000)

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
            if copilot_started:
                failed_attempt_input_tokens += approx_tokens(prompt_for_copilot)
                failed_attempt_output_tokens += approx_tokens(attempt_transcript)
                failed_attempt_rag_queries.extend(extract_mcp_queries(attempt_transcript))

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

# endregion


# region 7: Direct OpenAI-compatible invocation

async def call_direct_local(
    model: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant. <|think|>",
    timeout: float = 180.0,
    max_retries: int = 2,
    retry_delay_sec: float = 5.0,
    provider_base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
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
    failed_attempt_input_tokens = 0

    if provider_base_url is None:
        provider_base_url = LOCAL_BASE_URL
    if provider_api_key is None:
        provider_api_key = "dummy"

    for attempt in range(1, attempts + 1):
        try:
            print(
                f"[direct_local] attempt={attempt}/{attempts} model={model}",
                flush=True,
            )

            client = OpenAI(
                base_url=provider_base_url,
                api_key=provider_api_key,
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                timeout=timeout,
            )

            latency_ms = int((time.time() - overall_t0) * 1000)

            raw = resp.choices[0].message.content or ""

            usage = getattr(resp, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                if input_tokens is not None:
                    input_tokens += failed_attempt_input_tokens
                token_source = (
                    "provider_usage"
                    if not failed_attempt_input_tokens
                    else "provider_usage_plus_estimated_retries"
                )
            else:
                input_tokens = (
                    failed_attempt_input_tokens
                    + approx_tokens(system_prompt + "\n" + prompt)
                )
                output_tokens = approx_tokens(raw)
                token_source = "estimated_all_attempts"

            return raw, latency_ms, input_tokens, output_tokens, token_source

        except Exception as e:
            last_exc = e
            failed_attempt_input_tokens += approx_tokens(system_prompt + "\n" + prompt)

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


# endregion


# region 8: Run one task/setup

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

        # Create a unique folder for each run to ensure a fresh blank main.c
        run_id = f"{int(time.time() * 1000)}_{os.getpid()}"
        workspace_dir = (
            OUT_DIR
            / f"copilot_workspace_{safe_setup}"
            / safe_task_id
            / run_id
        ).resolve()

        workspace_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Run it back
        main_c_path = (workspace_dir / "main.c").resolve()
        # Critical: Copilot is much more reliable if the file already exists.
        # Create a blank main.c file.
        # main_c_path.write_text("", encoding="utf-8")

        prompt = make_prompt(
            problem.question,
            required,
            main_c_path=main_c_path,
            rag_enabled=rag_enabled,
        )
    else:
        prompt = make_prompt(problem.question, required, rag_enabled=False)

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
                provider_base_url=setup_cfg.get("provider_base_url", LOCAL_BASE_URL),
                provider_api_key=setup_cfg.get("provider_api_key", "dummy"),
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
            created_at=now_iso(),
        )

    if invocation == "copilot" and main_c_path is not None and main_c_path.exists():
        code = main_c_path.read_text(encoding="utf-8", errors="replace").strip()

        # TODO: run it back
        if not code:
            code = ""
        #     code = extract_c_code(raw)
    else:
        code = extract_c_code(raw)

    try:
        verify = verify_code(problem, code)
    except Exception as e:
        verification_error = (
            f"[VERIFICATION_FAILED]\n"
            f"error_type={type(e).__name__}\n"
            f"error={e}\n"
        )
        print(
            f"[ERROR] setup={setup} task={problem.id} verification failed, "
            f"marking only this result as failed: {type(e).__name__}: {e}",
            flush=True,
        )
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
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
            called=[],
            missing=required,
            required_pass=False,
            compile_pass=False,
            compile_logs=verification_error,
            hard_pass=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_source=token_source,
            latency_ms=latency_ms,
            rag_query_count=len(rag_queries),
            rag_queries=rag_queries,
            created_at=now_iso(),
        )

    called = verify.details.get("called", [])
    missing = verify.details.get("missing", required)
    compile_logs = verify.details.get("compile_logs", "") or ""

    required_pass = bool(verify.details.get("required_pass", len(missing) == 0))
    compile_pass = bool(verify.compiled)
    hard_pass = bool(
        verify.details.get("hard_pass", required_pass and compile_pass)
    )

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
        created_at=now_iso(),
    )

# endregion


# region 9: Absolute judging

JUDGE_SCORE_KEYS = (
    "task_correctness",
    "reference_similarity",
    "code_quality",
)


def result_fingerprint(result: RunResult) -> str:
    """Stable identity for deciding whether an existing judgment is current."""
    payload = {
        "task_id": result.task_id,
        "setup": result.setup,
        "code": result.extracted_code,
        "raw_response": result.raw_response,
        "required": result.required,
        "called": result.called,
        "missing": result.missing,
        "compile_pass": result.compile_pass,
        "hard_pass": result.hard_pass,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
        "rag_queries": result.rag_queries,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_judge_prompt(problem: Problem, result: RunResult) -> str:
    candidate = {
        "code": result.extracted_code,
        "required": result.required,
        "called": result.called,
        "missing": result.missing,
        "required_pass": result.required_pass,
        "compile_pass": result.compile_pass,
        "hard_pass": result.hard_pass,
        "compile_logs": result.compile_logs,
    }

    return f"""You are an absolute evaluator for one C code-generation result.

Task:
{problem.question}

Reference answer / expected behavior:
{problem.answer or "No reference answer was supplied; rely on the task and verifier facts."}

Required library functions:
{problem.metadata.get("required", [])}

Evaluate only this result. Do not compare it to other models or setups. Compiler and
required-call facts are authoritative. A non-compiling candidate receives 0 for task
correctness and code quality. Reference similarity means behavioral agreement, not
textual or implementation similarity.

Scoring rubric (all scores are numbers from 0 to 1):
- task_correctness: fulfillment of the task and required library calls.
- reference_similarity: semantic agreement with the reference answer's behavior.
- code_quality: safety, clarity, and unnecessary complexity in the C implementation.

Return only valid JSON with exactly these keys:
{{
  "task_correctness": 0.0,
  "reference_similarity": 0.0,
  "code_quality": 0.0,
  "reasoning": "brief evidence-based explanation"
}}

Result:
{json.dumps(candidate, ensure_ascii=False, indent=2)}
"""


def validate_absolute_judgment(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("judge output must be a JSON object")

    expected_keys = set(JUDGE_SCORE_KEYS) | {"reasoning"}
    if set(parsed) != expected_keys:
        raise ValueError(
            f"judge output keys must be exactly {sorted(expected_keys)}, got {sorted(parsed)}"
        )

    normalized: dict[str, Any] = {}
    for key in JUDGE_SCORE_KEYS:
        value = parsed[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"judge score {key!r} must be a number")
        if not 0 <= value <= 1:
            raise ValueError(f"judge score {key!r} must be between 0 and 1")
        normalized[key] = value

    if not isinstance(parsed["reasoning"], str):
        raise ValueError("judge reasoning must be a string")

    normalized["reasoning"] = parsed["reasoning"]
    normalized["overall"] = (
        0.45 * normalized["task_correctness"]
        + 0.40 * normalized["reference_similarity"]
        + 0.15 * normalized["code_quality"]
    )
    return normalized


async def judge_task(
    problem: Problem,
    result: RunResult,
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

    prompt = make_judge_prompt(problem, result)

    print("\n========== JUDGE START ==========", flush=True)
    print(f"judge_mode={judge_mode}", flush=True)
    print(f"judge_model={judge_model}", flush=True)
    print(f"judge_base_url={judge_base_url}", flush=True)
    print("=================================\n", flush=True)

    attempt = 0
    while True:
        attempt += 1
        try:
            client = OpenAI(base_url=judge_base_url, api_key=judge_api_key)
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful evaluation judge. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=180.0,
            )
            parsed = validate_absolute_judgment(
                json.loads(resp.choices[0].message.content or "")
            )
            parsed["_judge_mode"] = judge_mode
            parsed["_judge_model"] = judge_model
            return parsed
        except Exception as e:
            delay = min(60.0, float(2 ** min(attempt - 1, 6)))
            print(
                f"[WARN] judge attempt={attempt} failed: {type(e).__name__}: {e}; "
                f"retrying in {delay:.0f}s",
                flush=True,
            )
            await asyncio.sleep(delay)


# endregion


# region 10: Setup registry and persisted result loading

def build_setup_registry() -> dict[str, dict[str, Any]]:
    """
    Registry of all known generation setups.

    Generation phase chooses from this registry using --setups.
    The judge phase discovers setup ids directly from its stored results.

    Important:
    - Cloud API key is only required if cloud_rag_copilot is selected.
    - The RL baseline is direct-only; there is intentionally no RL Copilot setup.
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

        "rl_direct": {
            "id": "rl_direct",
            "model": RL_MODEL,
            "invocation": "direct",
            "rag_enabled": False,
            "provider_base_url": LOCAL_BASE_URL,
            "provider_api_key": "dummy",
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


def load_judgments_latest(judge_path: Path) -> dict[tuple[str, str], str]:
    """Return the latest valid result fingerprint for each judged task/setup."""
    latest: dict[tuple[str, str], str] = {}

    if not judge_path.exists():
        return latest

    with open(judge_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] Skipping invalid JSON line in {judge_path} "
                    f"line={line_no}: {e}",
                    flush=True,
                )
                continue

            task_id = row.get("task_id")
            setup = row.get("setup")
            fingerprint = row.get("judge_fingerprint")
            judge = row.get("judge")
            if (
                isinstance(task_id, str)
                and isinstance(setup, str)
                and isinstance(fingerprint, str)
                and isinstance(judge, dict)
                and "error" not in judge
            ):
                latest[(task_id, setup)] = fingerprint

    return latest


def judge_fingerprint(problem: Problem, result: RunResult, dev: bool) -> str:
    """Include the task/reference so edited JSONL rows are re-judged."""
    judge_model = LOCAL_JUDGE_MODEL if dev else CLOUD_JUDGE_MODEL
    payload = {
        "result": result_fingerprint(result),
        "question": problem.question,
        "answer": problem.answer,
        "required": problem.metadata.get("required", []),
        "dev": dev,
        "judge_model": judge_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# endregion


# region 10b: Coherence-check rounds

COHERENCE_CLASSIFICATIONS = {
    "coherent_c",
    "empty_output",
    "provider_failure",
    "stdout_transcript",
    "non_code",
}


def results_round_path(round_number: int) -> Path:
    if round_number == 0:
        return OUT_DIR / "results.jsonl"
    return OUT_DIR / f"results_{round_number}.jsonl"


def check_round_path(round_number: int) -> Path:
    if round_number == 0:
        return OUT_DIR / "check.jsonl"
    return OUT_DIR / f"check_{round_number}.jsonl"


def available_result_rounds() -> list[int]:
    rounds = []
    if results_round_path(0).exists():
        rounds.append(0)
    for path in OUT_DIR.glob("results_*.jsonl"):
        match = re.fullmatch(r"results_(\d+)\.jsonl", path.name)
        if match:
            rounds.append(int(match.group(1)))
    return sorted(set(rounds))


def latest_checked_round() -> Optional[int]:
    checked = [
        round_number
        for round_number in available_result_rounds()
        if check_round_path(round_number).exists()
    ]
    return max(checked) if checked else None


def next_unchecked_round() -> Optional[int]:
    pending = [
        round_number
        for round_number in available_result_rounds()
        if not check_round_path(round_number).exists()
    ]
    return max(pending) if pending else None


def make_coherence_prompt(problem: Problem, result: RunResult) -> str:
    return f"""You validate whether a model extraction is usable C source text.

Task:
{problem.question}

Extracted output:
{result.extracted_code}

Compiler log (context only; a compile error alone is not a reason to retry):
{result.compile_logs}

Do not evaluate task correctness, required APIs, quality, or whether the code compiles.
Decide only whether the extracted output is recognizably coherent C source, rather than
empty output, an API/provider/budget failure, a Copilot/stdout transcript, or arbitrary
non-code text.

Return only valid JSON with exactly these keys:
{{
  "retry": false,
  "classification": "coherent_c|empty_output|provider_failure|stdout_transcript|non_code",
  "reasoning": "brief explanation"
}}
"""


def validate_coherence_check(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("coherence check output must be a JSON object")
    if set(parsed) != {"retry", "classification", "reasoning"}:
        raise ValueError("coherence check output has unexpected keys")
    if not isinstance(parsed["retry"], bool):
        raise ValueError("coherence check retry must be a boolean")
    if parsed["classification"] not in COHERENCE_CLASSIFICATIONS:
        raise ValueError("coherence check classification is invalid")
    if not isinstance(parsed["reasoning"], str):
        raise ValueError("coherence check reasoning must be a string")
    if parsed["retry"] == (parsed["classification"] == "coherent_c"):
        raise ValueError("coherence retry must be false exactly for coherent_c")
    return {
        "retry": parsed["retry"],
        "classification": parsed["classification"],
        "reasoning": parsed["reasoning"],
    }


async def check_result_coherence(problem: Problem, result: RunResult) -> dict[str, Any]:
    """Use the local model only to reject unusable extraction fallbacks."""
    prompt = make_coherence_prompt(problem, result)
    attempt = 0
    while True:
        attempt += 1
        try:
            client = OpenAI(base_url=LOCAL_BASE_URL, api_key="dummy")
            response = client.chat.completions.create(
                model=LOCAL_JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON. Do not execute code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=180.0,
            )
            return validate_coherence_check(
                json.loads(response.choices[0].message.content or "")
            )
        except Exception as e:
            delay = min(60.0, float(2 ** min(attempt - 1, 6)))
            print(
                f"[WARN] coherence check attempt={attempt} failed: "
                f"{type(e).__name__}: {e}; retrying in {delay:.0f}s",
                flush=True,
            )
            await asyncio.sleep(delay)


def load_checks(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the latest valid check row for each task/setup from one check file."""
    checks: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return checks

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row["task_id"]
                setup = row["setup"]
                result_hash = row["result_fingerprint"]
                decision = validate_coherence_check(row["check"])
                if not all(isinstance(value, str) for value in (task_id, setup, result_hash)):
                    raise ValueError("task_id, setup, and result_fingerprint must be strings")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                print(
                    f"[WARN] Skipping invalid check row {path} line={line_no}: {e}",
                    flush=True,
                )
                continue
            checks[(task_id, setup)] = {
                "result_fingerprint": result_hash,
                "check": decision,
                "round": row.get("round"),
            }
    return checks


def coherent_candidates_from_round(round_number: int) -> list[tuple[int, RunResult]]:
    results_path = results_round_path(round_number)
    checks = load_checks(check_round_path(round_number))
    candidates: list[tuple[int, RunResult]] = []
    for task_id, by_setup in load_results_latest(results_path).items():
        for setup, result in by_setup.items():
            decision = checks.get((task_id, setup))
            if (
                decision
                and decision["result_fingerprint"] == result_fingerprint(result)
                and not decision["check"]["retry"]
            ):
                candidates.append((round_number, result))
    return candidates


def outstanding_retry_pairs() -> set[tuple[str, str]]:
    """Return task/setup pairs whose newest checked candidate still needs a retry."""
    latest: dict[tuple[str, str], tuple[int, bool]] = {}
    for round_number in available_result_rounds():
        check_path = check_round_path(round_number)
        if not check_path.exists():
            continue
        checks = load_checks(check_path)
        results = load_results_latest(results_round_path(round_number))
        for task_id, by_setup in results.items():
            for setup, result in by_setup.items():
                decision = checks.get((task_id, setup))
                if decision and decision["result_fingerprint"] == result_fingerprint(result):
                    latest[(task_id, setup)] = (
                        round_number,
                        decision["check"]["retry"],
                    )
    return {pair for pair, (_round, retry) in latest.items() if retry}


# endregion


# region 11: Generation phase

async def run_generation_phase(
    tasks_path: str,
    setup_ids: list[str],
    limit: Optional[int] = None,
    resume: bool = False,
    max_concurrency: int = 5,
):
    """
    Generation-only phase.

    Behavior:
    - Runs selected setup ids only.
    - Appends each RunResult to OUT_DIR/results.jsonl.
    - Does NOT run judge.
    - With resume=True, skips any existing task_id + setup row. Limit then
      applies to the next pending tasks for each selected setup.
    - Runs up to `max_concurrency` tasks in parallel using asyncio.
    """

    problems = load_tasks(tasks_path)
    setups = select_setups(setup_ids)
    all_results_path = OUT_DIR / "results.jsonl"
    latest = (
        load_results_latest(all_results_path)
        if resume and all_results_path.exists()
        else defaultdict(dict)
    )

    scheduled: set[tuple[str, str]] = set()
    for setup in setups:
        pending = [
            problem
            for problem in problems
            if not resume or setup["id"] not in latest.get(problem.id, {})
        ]
        if limit is not None:
            pending = pending[:limit]
        scheduled.update((problem.id, setup["id"]) for problem in pending)

    print("\n========== GENERATION PHASE ==========", flush=True)
    print(f"tasks_path={tasks_path}", flush=True)
    print(f"num_tasks={len(problems)}", flush=True)
    print(f"scheduled_task_setup_pairs={len(scheduled)}", flush=True)
    print(f"resume={resume}", flush=True)
    print(f"limit_per_setup={limit}", flush=True)
    print(f"max_concurrency={max_concurrency}", flush=True)
    print(f"results_path={all_results_path}", flush=True)
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

    # 1. Semaphore to limit concurrent Copilot CLI instances
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # 2. Lock to prevent concurrent file writes from corrupting the JSONL
    file_lock = asyncio.Lock()

    # 3. Define the async worker for a single task/setup pair
    async def process_task(problem: Problem, setup: dict[str, Any]):
        if (problem.id, setup["id"]) not in scheduled:
            if resume:
                reason = (
                    "already has a result"
                    if setup["id"] in latest.get(problem.id, {})
                    else "outside the current pending-task limit"
                )
                print(
                    f"[SKIP GENERATE] task={problem.id} setup={setup['id']} "
                    f"{reason}",
                    flush=True,
                )
            return

        async with semaphore:
            print(f"\n========== TASK {problem.id} ==========", flush=True)
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
                print(
                    f"[ERROR] setup={setup['id']} task={problem.id} "
                    f"unexpected failure: {type(e).__name__}: {e}",
                    flush=True,
                )
                # NOTE: In concurrent execution, raising an exception here would crash 
                # the entire asyncio.gather and kill all other running tasks.
                # We log it and return to allow the rest of the batch to finish safely.
                return

            # 4. Safely append to the JSONL file
            async with file_lock:
                with open(all_results_path, "a", encoding="utf-8") as rf:
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

    # 5. Gather all the scheduled tasks and run them concurrently
    tasks_to_run = [
        process_task(problem, setup)
        for problem in problems
        for setup in setups
    ]

    # Run them all concurrently (up to the semaphore limit)
    await asyncio.gather(*tasks_to_run)

    print(f"\nSaved appended generation results to: {all_results_path}", flush=True)

# endregion


# region 12: Coherence checks, retries, merging, and source-aware judging

async def run_check_phase(tasks_path: str) -> None:
    round_number = next_unchecked_round()
    if round_number is None:
        print("[CHECK] No unchecked results round found.", flush=True)
        return

    results_path = results_round_path(round_number)
    check_path = check_round_path(round_number)
    problems = {problem.id: problem for problem in load_tasks(tasks_path)}
    latest = load_results_latest(results_path)
    bad_count = 0
    checked_count = 0

    print("\n========== COHERENCE CHECK PHASE ==========", flush=True)
    print(f"results_path={results_path}", flush=True)
    print(f"check_path={check_path}", flush=True)
    print(f"round={round_number}", flush=True)
    print(f"local_judge_model={LOCAL_JUDGE_MODEL}", flush=True)
    print("===========================================\n", flush=True)

    with open(check_path, "x", encoding="utf-8") as check_file:
        for task_id, by_setup in latest.items():
            problem = problems.get(task_id)
            if problem is None:
                print(
                    f"[WARN] Skipping result task={task_id}; it is absent from {tasks_path}",
                    flush=True,
                )
                continue
            for setup, result in by_setup.items():
                decision = await check_result_coherence(problem, result)
                row = {
                    "task_id": task_id,
                    "setup": setup,
                    "round": round_number,
                    "source_results": results_path.name,
                    "result_fingerprint": result_fingerprint(result),
                    "check": decision,
                    "created_at": now_iso(),
                }
                check_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                check_file.flush()
                checked_count += 1
                bad_count += int(decision["retry"])
                print(
                    f"[CHECK] task={task_id} setup={setup} "
                    f"classification={decision['classification']} retry={decision['retry']}",
                    flush=True,
                )

    print(
        f"[CHECK SUMMARY] round={round_number} checked={checked_count} bad={bad_count}",
        flush=True,
    )
    outstanding = outstanding_retry_pairs()
    if checked_count and not outstanding:
        await run_merge_phase(tasks_path)
    elif outstanding:
        print(
            f"[CHECK] outstanding_retry_pairs={len(outstanding)}; run --phase retry.",
            flush=True,
        )


async def _generate_pairs_to_path(
    problems: list[Problem],
    setups: list[dict[str, Any]],
    scheduled: set[tuple[str, str]],
    output_path: Path,
    max_concurrency: int = 5,
) -> None:
    semaphore = asyncio.Semaphore(max_concurrency)
    file_lock = asyncio.Lock()

    async def process_task(problem: Problem, setup: dict[str, Any]) -> None:
        if (problem.id, setup["id"]) not in scheduled:
            return
        async with semaphore:
            print(
                f"[GENERATE] task={problem.id} setup={setup['id']} "
                f"output={output_path.name}",
                flush=True,
            )
            try:
                result = await run_one(problem, setup)
            except Exception as e:
                # run_one normally converts provider failures into a RunResult. Keep
                # an unexpected exception visible instead of guessing its category.
                print(
                    f"[ERROR] task={problem.id} setup={setup['id']} "
                    f"unexpected generation failure: {type(e).__name__}: {e}",
                    flush=True,
                )
                return
            async with file_lock:
                with open(output_path, "a", encoding="utf-8") as results_file:
                    results_file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    results_file.flush()

    await asyncio.gather(
        *[
            process_task(problem, setup)
            for problem in problems
            for setup in setups
            if (problem.id, setup["id"]) in scheduled
        ]
    )


async def run_retry_phase(
    tasks_path: str,
    limit: Optional[int] = None,
    max_concurrency: int = 5,
) -> None:
    round_number = latest_checked_round()
    if round_number is None:
        raise FileNotFoundError("No check.jsonl exists. Run --phase check first.")

    retry_pairs = outstanding_retry_pairs()

    if not retry_pairs:
        print("[RETRY] There are no retryable task IDs.", flush=True)
        return

    next_round = round_number + 1
    output_path = results_round_path(next_round)
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Run --phase check for that round before retrying again."
        )

    problems = load_tasks(tasks_path)
    setup_ids = sorted({setup for _, setup in retry_pairs})
    setups = select_setups(setup_ids)
    scheduled: set[tuple[str, str]] = set()
    scheduled_by_setup: defaultdict[str, int] = defaultdict(int)
    for problem in problems:
        for setup in setups:
            pair = (problem.id, setup["id"])
            if pair not in retry_pairs:
                continue
            if limit is not None and scheduled_by_setup[setup["id"]] >= limit:
                continue
            scheduled.add(pair)
            scheduled_by_setup[setup["id"]] += 1

    print("\n========== RETRY PHASE ==========", flush=True)
    print(f"source_check={check_round_path(round_number)}", flush=True)
    print(f"output_results={output_path}", flush=True)
    print(f"retryable_pairs={len(retry_pairs)}", flush=True)
    print(f"scheduled_pairs={len(scheduled)}", flush=True)
    print("=================================\n", flush=True)
    await _generate_pairs_to_path(
        problems, setups, scheduled, output_path, max_concurrency=max_concurrency
    )


def merge_candidate_key(round_number: int, result: RunResult) -> tuple[int, int, int, int]:
    return (
        int(result.hard_pass),
        int(result.compile_pass),
        int(result.required_pass),
        round_number,
    )


async def run_merge_phase(tasks_path: str) -> None:
    problems = load_tasks(tasks_path)
    candidates: dict[tuple[str, str], tuple[int, RunResult]] = {}
    observed_setups: set[str] = set()
    for round_number in available_result_rounds():
        round_results = load_results_latest(results_round_path(round_number))
        observed_setups.update(
            setup for by_setup in round_results.values() for setup in by_setup
        )
        if not check_round_path(round_number).exists():
            continue
        for candidate_round, result in coherent_candidates_from_round(round_number):
            key = (result.task_id, result.setup)
            previous = candidates.get(key)
            if previous is None or merge_candidate_key(candidate_round, result) > merge_candidate_key(
                previous[0], previous[1]
            ):
                candidates[key] = (candidate_round, result)

    setups = sorted(observed_setups)
    expected = {(problem.id, setup) for problem in problems for setup in setups}
    selected = []
    selection = []
    for problem in problems:
        for setup in setups:
            chosen = candidates.get((problem.id, setup))
            if chosen is None:
                continue
            round_number, result = chosen
            selected.append(result)
            selection.append(
                {
                    "task_id": result.task_id,
                    "setup": result.setup,
                    "source_round": round_number,
                    "hard_pass": result.hard_pass,
                    "compile_pass": result.compile_pass,
                    "required_pass": result.required_pass,
                }
            )

    merged_path = OUT_DIR / "result_merged.jsonl"
    with open(merged_path, "w", encoding="utf-8") as merged_file:
        for result in selected:
            merged_file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    missing_pairs = sorted(expected - set(candidates))
    missing = sorted({task_id for task_id, _setup in missing_pairs})
    summary = {
        "created_at": now_iso(),
        "merged_path": merged_path.name,
        "selected_count": len(selected),
        "missing_task_ids": missing,
        "missing_task_setup_pairs": [
            {"task_id": task_id, "setup": setup}
            for task_id, setup in missing_pairs
        ],
        "selection": selection,
    }
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[MERGE] selected={len(selected)} missing={len(missing)} path={merged_path}",
        flush=True,
    )


async def run_judge_phase(
    tasks_path: str,
    judge_setup_ids: Optional[list[str]] = None,
    dev: bool = False,
    limit: Optional[int] = None,
    resume: bool = False,
    source: str = "results",
) -> None:
    if source not in {"results", "merged"}:
        raise ValueError("judge source must be 'results' or 'merged'")
    results_path = OUT_DIR / (
        "results.jsonl" if source == "results" else "result_merged.jsonl"
    )
    judge_path = OUT_DIR / "judge.jsonl"
    problems = load_tasks(tasks_path)
    latest = load_results_latest(results_path)
    stored_setups = sorted(
        {setup for by_setup in latest.values() for setup in by_setup}
    )
    selected_setups = (
        [setup for setup in judge_setup_ids if setup in stored_setups]
        if judge_setup_ids is not None
        else stored_setups
    )
    latest_judgments = load_judgments_latest(judge_path) if resume else {}
    scheduled_by_setup: defaultdict[str, int] = defaultdict(int)
    judged_count = 0
    skipped_count = 0

    print("\n========== JUDGE PHASE ==========", flush=True)
    print(f"results_path={results_path}", flush=True)
    print(f"judge_path={judge_path}", flush=True)
    print(f"discovered_setups={selected_setups}", flush=True)
    print(f"source={source}", flush=True)
    print("=================================\n", flush=True)

    with open(judge_path, "a", encoding="utf-8") as judge_file:
        for problem in problems:
            for setup in selected_setups:
                result = latest.get(problem.id, {}).get(setup)
                if result is None:
                    continue
                fingerprint = judge_fingerprint(problem, result, dev)
                if resume and latest_judgments.get((problem.id, setup)) == fingerprint:
                    skipped_count += 1
                    continue
                if limit is not None and scheduled_by_setup[setup] >= limit:
                    continue
                scheduled_by_setup[setup] += 1
                judge = await judge_task(problem, result, dev=dev)
                row = {
                    "task_id": problem.id,
                    "setup": setup,
                    "source_results": results_path.name,
                    "result_fingerprint": result_fingerprint(result),
                    "judge_fingerprint": fingerprint,
                    "dev": dev,
                    "judge": judge,
                    "created_at": now_iso(),
                }
                judge_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                judge_file.flush()
                judged_count += 1

    print(
        f"[JUDGE SUMMARY] judged={judged_count} skipped={skipped_count} path={judge_path}",
        flush=True,
    )


# endregion


# region 13: CLI

async def main(args: argparse.Namespace):
    if args.phase == "generate":
        await run_generation_phase(
            tasks_path=args.tasks_path,
            setup_ids=args.setups,
            limit=args.limit,
            resume=args.resume,
        )
        return

    if args.phase == "check":
        await run_check_phase(args.tasks_path)
        return

    if args.phase == "retry":
        await run_retry_phase(
            tasks_path=args.tasks_path,
            limit=args.limit,
        )
        return

    if args.phase == "merge":
        await run_merge_phase(args.tasks_path)
        return

    if args.phase == "judge":
        await run_judge_phase(
            tasks_path=args.tasks_path,
            dev=args.dev,
            limit=args.limit,
            resume=args.resume,
            source=args.judge_source,
        )
        return

    raise ValueError(f"Unknown phase: {args.phase}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate C code-generation models in separated phases: "
            "generation, coherence checks, retries, merging, and judging."
        )
    )

    parser.add_argument(
        "tasks_path",
        help="Path to tasks.jsonl",
    )

    parser.add_argument(
        "--phase",
        choices=["generate", "check", "retry", "merge", "judge"],
        required=True,
        help=(
            "Phase to run. "
            "'generate' writes results.jsonl; 'check' validates extracted output; "
            "'retry' writes the next results_N.jsonl; 'merge' writes result_merged.jsonl; "
            "'judge' appends judge.jsonl."
        ),
    )

    parser.add_argument(
        "--setups",
        nargs="+",
        default=["local_rag_copilot"],
        help=(
            "Generation setup ids to run. Used only with --phase generate. "
            "Examples: local_rag_copilot, cloud_rag_copilot, rl_direct."
        ),
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help=(
            "For judge phase only: use the local judge instead of the cloud judge."
        ),
    )

    parser.add_argument(
        "--judge-source",
        choices=["results", "merged"],
        default="results",
        help=(
            "Judge source. The default reads results.jsonl; 'merged' reads "
            "result_merged.jsonl after coherence-check retry rounds."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue this phase from JSONL progress. Generation skips existing "
            "task/setup results; judging skips only current valid judgments."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of pending tasks per selected setup. Useful for "
            "incremental runs; with --resume it selects the next unfinished rows."
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    if args.phase == "generate" and not args.setups:
        raise ValueError("--phase generate requires at least one --setups value")

    return args


# endregion

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
