"""Dependency-isolated regression tests for eval_cli.py.

Run with:
    python -m unittest -v test_eval_cli.py

The test imports eval_cli.py with lightweight dependency stubs, so it can test
CLI orchestration without a live Copilot binary, MCP server, compiler server,
or installed tree-sitter package. `test_copilot_cli_is_stubbed...` is the
reference fake Copilot interaction and documents the expected transcript.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parent
EVAL_CLI_PATH = ROOT / "eval_cli.py"


def load_eval_cli_module():
    """Import eval_cli.py without requiring its optional runtime services."""
    requests = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    requests.RequestException = RequestException
    requests.post = lambda *args, **kwargs: None

    openai = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - calls are patched in tests.
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI

    tree_sitter = types.ModuleType("tree_sitter")

    class Language:
        def __init__(self, value):
            self.value = value

    class Parser:
        def __init__(self, language):
            self.language = language

    tree_sitter.Language = Language
    tree_sitter.Parser = Parser

    tree_sitter_c = types.ModuleType("tree_sitter_c")
    tree_sitter_c.language = lambda: object()

    module_name = "eval_cli_under_test"
    spec = importlib.util.spec_from_file_location(module_name, EVAL_CLI_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    fake_modules = {
        "requests": requests,
        "openai": openai,
        "tree_sitter": tree_sitter,
        "tree_sitter_c": tree_sitter_c,
    }
    with patch.dict(sys.modules, fake_modules):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class FakeAnalyzer:
    def __init__(self, called: set[str], defined: set[str] | None = None):
        self.called = called
        self.defined = defined or set()

    def extract_called_functions(self, code: str) -> set[str]:
        return self.called

    def extract_defined_functions(self, code: str) -> set[str]:
        return self.defined


class FakeVerifyResponse:
    status_code = 200
    text = "compiler output"

    def __init__(self, data: dict):
        self.data = data

    def json(self):
        return self.data


class FakeMcpProcess:
    returncode = 0

    async def communicate(self):
        return b"moove configured\n", b""


class FakeCopilotProcess:
    returncode = 0

    def __init__(self, main_c_path: Path):
        self.main_c_path = main_c_path
        self.stdout = types.SimpleNamespace(
            text=(
                '● function_info (MCP: moove) · name: "library_call"\n'
                "Tokens     ↑ 14.4k • ↓ 2.5k\n"
            )
        )
        self.stderr = types.SimpleNamespace(text="")

    async def wait(self):
        self.main_c_path.write_text(
            "void solve(void) { library_call(); }\n",
            encoding="utf-8",
        )
        return self.returncode


class EvalCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_eval_cli_module()

    def make_result(self, task_id: str, setup: str = "local_rag_copilot"):
        return self.mod.RunResult(
            task_id=task_id,
            setup=setup,
            model="model",
            invocation="copilot",
            rag_enabled=True,
            prompt="prompt",
            raw_response="```c\nvoid solve(void) {}\n```",
            extracted_code="void solve(void) {}",
            required=[],
            called=[],
            missing=[],
            required_pass=True,
            compile_pass=True,
            compile_logs="",
            hard_pass=True,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            token_source="provider_usage",
            latency_ms=20,
            rag_query_count=0,
            rag_queries=[],
        )

    def test_registry_and_prompts_match_the_three_setup_design(self):
        registry = self.mod.build_setup_registry()
        self.assertEqual(
            set(registry),
            {"local_rag_copilot", "cloud_rag_copilot", "rl_direct"},
        )

        direct_prompt = self.mod.make_prompt("Do work", ["library_call"])
        self.assertIn("chat-completions API", direct_prompt)
        self.assertIn("```c code block", direct_prompt)
        self.assertIn("library_call", direct_prompt)
        self.assertNotIn("main.c", direct_prompt)

        rag_prompt = self.mod.make_prompt(
            "Do work",
            ["library_call"],
            main_c_path=Path("/tmp/main.c"),
            rag_enabled=True,
        )
        self.assertIn("configured MCP tools", rag_prompt)
        self.assertIn("library_call", rag_prompt)

        with patch.object(
            sys,
            "argv",
            ["eval_cli.py", "tasks.jsonl", "--phase", "judge", "--dev"],
        ):
            args = self.mod.parse_args()
        self.assertTrue(args.dev)
        self.assertEqual(args.judge_source, "results")

    def test_query_parser_keeps_the_mcp_function_name(self):
        raw = '● function_info (MCP: moove) · name: "pmf_preexit"\n'
        self.assertEqual(
            self.mod.extract_mcp_queries(raw),
            ["moove.function_info(name=pmf_preexit)"],
        )

    def test_verifier_keeps_compile_and_required_results_separate(self):
        problem = self.mod.Problem(
            id="task-1",
            question="Call both APIs",
            metadata={"required": ["library_call", "other_call"]},
        )
        response = FakeVerifyResponse(
            {
                "compiled": True,
                "compile_logs": "ok",
                "total_external_functions_executed": 2,
                "total_correct_functions_executed": 2,
            }
        )

        with patch.object(self.mod.requests, "post", return_value=response), patch.object(
            self.mod, "analyzer", FakeAnalyzer({"library_call"})
        ):
            verified = self.mod.verify_code(problem, "void solve(void) {}")

        self.assertTrue(verified.compiled)
        self.assertEqual(verified.details["missing"], ["other_call"])
        self.assertFalse(verified.details["required_pass"])
        self.assertFalse(verified.details["hard_pass"])

    def test_verifier_rejects_generated_shadow_calls_and_accepts_compile_only_tasks(self):
        response = FakeVerifyResponse({"compiled": True, "compile_logs": "ok"})
        shadowed_problem = self.mod.Problem(
            id="task-shadowed",
            question="Call the library API",
            metadata={"required": ["library_call"]},
        )
        with patch.object(self.mod.requests, "post", return_value=response), patch.object(
            self.mod, "analyzer", FakeAnalyzer({"library_call"}, {"library_call"})
        ):
            shadowed = self.mod.verify_code(shadowed_problem, "void library_call(void) {}")
        self.assertFalse(shadowed.details["required_pass"])
        self.assertEqual(shadowed.details["shadowed_required"], ["library_call"])

        compile_only_problem = self.mod.Problem(
            id="task-compile-only",
            question="Compile this",
            metadata={"required": []},
        )
        with patch.object(self.mod.requests, "post", return_value=response), patch.object(
            self.mod, "analyzer", FakeAnalyzer(set())
        ):
            compile_only = self.mod.verify_code(compile_only_problem, "int main(void) {}")
        self.assertTrue(compile_only.compiled)
        self.assertTrue(compile_only.details["hard_pass"])

    def test_copilot_cli_is_stubbed_and_returns_expected_artifacts(self):
        """The subprocess stub documents the expected Copilot/MCP transcript."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            commands: list[tuple[str, ...]] = []

            def fake_prepare(name: str, rag_enabled: bool) -> Path:
                self.assertTrue(rag_enabled)
                home = tmp_path / name
                home.mkdir()
                return home

            async def fake_subprocess(*args, **kwargs):
                commands.append(tuple(args))
                if args[1:3] == ("mcp", "list"):
                    return FakeMcpProcess()
                return FakeCopilotProcess(workspace / "main.c")

            async def fake_stream(stream, prefix, buf):
                if stream.text:
                    buf.append(stream.text)

            with patch.object(self.mod, "prepare_copilot_home", side_effect=fake_prepare), patch.object(
                self.mod.asyncio,
                "create_subprocess_exec",
                side_effect=fake_subprocess,
            ), patch.object(self.mod, "_stream_live", side_effect=fake_stream):
                raw, latency, input_tokens, output_tokens, source, queries = asyncio.run(
                    self.mod.call_copilot_cli(
                        setup="local_rag_copilot",
                        model="fake-model",
                        prompt="Generate C.",
                        rag_enabled=True,
                        workspace_dir=workspace,
                        max_retries=0,
                        timeout=1.0,
                    )
                )

        self.assertIn("void solve(void) { library_call(); }", raw)
        self.assertGreaterEqual(latency, 0)
        self.assertEqual((input_tokens, output_tokens), (14400, 2500))
        self.assertEqual(source, "copilot_cli_stdout")
        self.assertEqual(queries, ["moove.function_info(name=library_call)"])
        self.assertEqual(commands[0], ("copilot", "mcp", "list"))
        self.assertIn("--allow-all-tools", commands[1])

    def test_generation_resume_uses_the_next_unfinished_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.mod.OUT_DIR = tmp_path / "eval_runs"
            self.mod.OUT_DIR.mkdir()
            tasks_path = tmp_path / "tasks.jsonl"
            tasks_path.write_text(
                "\n".join(
                    json.dumps(
                        {"id": f"t{i}", "question": f"q{i}", "answer": "a"}
                    )
                    for i in range(1, 4)
                )
                + "\n",
                encoding="utf-8",
            )
            results_path = self.mod.OUT_DIR / "results.jsonl"
            results_path.write_text(
                json.dumps(self.mod.asdict(self.make_result("t1"))) + "\n",
                encoding="utf-8",
            )

            async def fake_run_one(problem, setup):
                return self.make_result(problem.id, setup["id"])

            with patch.object(self.mod, "run_one", side_effect=fake_run_one) as run_one:
                asyncio.run(
                    self.mod.run_generation_phase(
                        str(tasks_path),
                        ["local_rag_copilot"],
                        limit=1,
                        resume=True,
                    )
                )
                self.assertEqual(run_one.await_args.args[0].id, "t2")

                asyncio.run(
                    self.mod.run_generation_phase(
                        str(tasks_path),
                        ["local_rag_copilot"],
                        resume=True,
                    )
                )
                self.assertEqual(run_one.await_args.args[0].id, "t3")

    def test_absolute_judge_uses_reference_answer_and_resumes_by_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.mod.OUT_DIR = tmp_path / "eval_runs"
            self.mod.OUT_DIR.mkdir()
            tasks_path = tmp_path / "tasks.jsonl"
            tasks_path.write_text(
                json.dumps(
                    {
                        "id": "t1",
                        "question": "Return the value.",
                        "answer": "The function returns 7.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = self.make_result("t1")
            (self.mod.OUT_DIR / "results.jsonl").write_text(
                json.dumps(self.mod.asdict(result)) + "\n",
                encoding="utf-8",
            )
            prompt = self.mod.make_judge_prompt(
                self.mod.load_tasks(str(tasks_path))[0], result
            )
            self.assertIn("The function returns 7.", prompt)
            self.assertIn("Do not compare", prompt)
            self.assertIn("compile_logs", prompt)
            self.assertNotIn("latency_ms", prompt)

            judgment = {
                "task_correctness": 0.8,
                "reference_similarity": 0.9,
                "code_quality": 0.8,
                "overall": 0.84,
                "reasoning": "Compiled and matches the reference behavior.",
            }
            with patch.object(
                self.mod, "judge_task", new=AsyncMock(return_value=judgment)
            ) as judge_task:
                asyncio.run(
                    self.mod.run_judge_phase(
                        str(tasks_path),
                        ["local_rag_copilot"],
                        dev=True,
                        resume=True,
                    )
                )
                self.assertEqual(judge_task.await_count, 1)

                asyncio.run(
                    self.mod.run_judge_phase(
                        str(tasks_path),
                        dev=True,
                        resume=True,
                    )
                )
                self.assertEqual(judge_task.await_count, 1)

    def test_coherence_check_validation_and_round_files(self):
        scores = self.mod.validate_absolute_judgment(
            {
                "task_correctness": 0.8,
                "reference_similarity": 0.9,
                "code_quality": 0.8,
                "reasoning": "Evidence based.",
            }
        )
        self.assertAlmostEqual(scores["overall"], 0.84)
        coherent = self.mod.validate_coherence_check(
            {
                "retry": False,
                "classification": "coherent_c",
                "reasoning": "C source is present.",
            }
        )
        self.assertFalse(coherent["retry"])
        retry = self.mod.validate_coherence_check(
            {
                "retry": True,
                "classification": "provider_failure",
                "reasoning": "The extraction is an API error.",
            }
        )
        self.assertTrue(retry["retry"])
        with self.assertRaises(ValueError):
            self.mod.validate_coherence_check(
                {
                    "retry": True,
                    "classification": "coherent_c",
                    "reasoning": "Contradiction.",
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            self.mod.OUT_DIR = Path(tmp)
            self.assertEqual(self.mod.results_round_path(0).name, "results.jsonl")
            self.assertEqual(self.mod.check_round_path(0).name, "check.jsonl")
            self.assertEqual(self.mod.results_round_path(2).name, "results_2.jsonl")
            self.assertEqual(self.mod.check_round_path(2).name, "check_2.jsonl")

    def test_check_retry_and_merge_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.mod.OUT_DIR = tmp_path / "eval_runs"
            self.mod.OUT_DIR.mkdir()
            tasks_path = tmp_path / "tasks.jsonl"
            tasks_path.write_text(
                "\n".join(
                    json.dumps({"id": task_id, "question": task_id, "answer": "answer"})
                    for task_id in ("t1", "t2")
                ) + "\n",
                encoding="utf-8",
            )
            first = self.make_result("t1")
            failed = self.make_result("t2")
            failed.extracted_code = "[GENERATION_FAILED] quota exhausted"
            failed.compile_pass = False
            failed.hard_pass = False
            (self.mod.OUT_DIR / "results.jsonl").write_text(
                "\n".join(json.dumps(self.mod.asdict(row)) for row in (first, failed)) + "\n",
                encoding="utf-8",
            )

            initial_decisions = [
                {"retry": False, "classification": "coherent_c", "reasoning": "C"},
                {"retry": True, "classification": "provider_failure", "reasoning": "quota"},
            ]
            with patch.object(
                self.mod,
                "check_result_coherence",
                new=AsyncMock(side_effect=initial_decisions),
            ):
                asyncio.run(self.mod.run_check_phase(str(tasks_path)))
            self.assertTrue((self.mod.OUT_DIR / "check.jsonl").exists())

            retried = self.make_result("t2")
            async def fake_run_one(problem, setup):
                self.assertEqual(problem.id, "t2")
                return retried

            with patch.object(self.mod, "run_one", side_effect=fake_run_one):
                asyncio.run(self.mod.run_retry_phase(str(tasks_path)))
            self.assertTrue((self.mod.OUT_DIR / "results_1.jsonl").exists())

            with patch.object(
                self.mod,
                "check_result_coherence",
                new=AsyncMock(return_value=initial_decisions[0]),
            ):
                asyncio.run(self.mod.run_check_phase(str(tasks_path)))
            merged = self.mod.load_results_latest(self.mod.OUT_DIR / "result_merged.jsonl")
            self.assertEqual(set(merged), {"t1", "t2"})
            summary = json.loads((self.mod.OUT_DIR / "merge_summary.json").read_text())
            self.assertEqual(summary["missing_task_ids"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
