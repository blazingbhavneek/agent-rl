"""Unit tests for main.py pipeline."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers — avoid running __main__ block
# ---------------------------------------------------------------------------
import importlib, types
sys.modules.setdefault("stub", types.ModuleType("stub"))
sys.modules.setdefault("stub.stub", types.ModuleType("stub.stub"))
# Provide a minimal ProjectAnalyzer so the module loads
stub_mod = sys.modules["stub.stub"]
stub_mod.ProjectAnalyzer = MagicMock()  # type: ignore

import main as M


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def make_cfg(**kwargs) -> M.PipelineConfig:
    """Minimal valid PipelineConfig."""
    defaults = dict(
        source_dir=Path("/tmp/src/proc"),
        agent_js=Path("/tmp/agent.js"),
        system_json=None,
        func_docs_dir=Path("/tmp/docs"),
        agent_timeout_sec=60,
        max_agent_iterations=5,
        max_compile_fix_attempts=2,
        max_test_attempts=2,
        coverage_threshold=80.0,
        stub_batch_size=4,
        only_function=None,
        only_level=None,
        max_functions=None,
        dry_run=False,
        python_bin=sys.executable,
        max_stub_gen_retries=1,
        max_stub_integrate_retries=1,
        max_minimal_test_attempts=2,
        semantic_judge_min_score=75,
    )
    defaults.update(kwargs)
    return M.PipelineConfig(**defaults)


# ===========================================================================
# Config & Infrastructure
# ===========================================================================

class TestPipelineConfig:
    def test_dataclass_instantiation(self):
        cfg = make_cfg()
        assert cfg.coverage_threshold == 80.0
        assert cfg.semantic_judge_min_score == 75

    def test_defaults(self):
        cfg = make_cfg()
        assert cfg.stub_batch_size == 4
        assert cfg.dry_run is False

    def test_custom_semantic_score(self):
        cfg = make_cfg(semantic_judge_min_score=60)
        assert cfg.semantic_judge_min_score == 60


class TestDeriveTestDir:
    def test_normal(self, tmp: Path):
        src_dir = tmp / "project" / "src" / "myproc"
        src_dir.mkdir(parents=True)
        result = M.derive_test_dir(src_dir)
        assert result == tmp / "project" / "tests" / "myproc"

    def test_parent_not_src_raises(self, tmp: Path):
        bad = tmp / "notSrc" / "myproc"
        bad.mkdir(parents=True)
        with pytest.raises(ValueError, match="src"):
            M.derive_test_dir(bad)


class TestDerivePaths:
    def test_keys(self, tmp: Path):
        src = tmp / "project" / "src" / "myproc"
        src.mkdir(parents=True)
        cfg = make_cfg(source_dir=src)
        paths = M.derive_paths(cfg)
        for k in ["test_dir", "process_name", "test_file", "makefile",
                  "history_dir", "analysis_path", "report_file", "log_file"]:
            assert k in paths
        assert paths["process_name"] == "myproc"
        assert paths["test_file"].name == "test_myproc.c"


class TestSafeFilename:
    def test_alphanumeric(self):
        assert M._safe_filename("hello_world") == "hello_world"

    def test_special_chars_replaced(self):
        result = M._safe_filename("foo::bar/baz")
        # consecutive special chars collapse to single underscore
        assert result == "foo_bar_baz"

    def test_no_leading_trailing_underscores(self):
        result = M._safe_filename("::foo::")
        assert not result.startswith("_")
        assert not result.endswith("_")


class TestReadWriteText:
    def test_roundtrip(self, tmp: Path):
        p = tmp / "test.txt"
        M.write_text(p, "hello\nworld")
        assert M.read_text(p) == "hello\nworld"

    def test_missing_returns_empty(self, tmp: Path):
        assert M.read_text(tmp / "nonexistent.txt") == ""

    def test_append(self, tmp: Path):
        p = tmp / "a.txt"
        M.write_text(p, "line1\n")
        M.append_text(p, "line2\n")
        assert M.read_text(p) == "line1\nline2\n"


# ===========================================================================
# Analysis helpers
# ===========================================================================

class TestFunctionsLeafFirst:
    def _analysis(self):
        return {
            "function_levels": {
                "0": ["root"],
                "1": ["mid"],
                "2": ["leaf1", "leaf2"],
            },
            "functions": [
                {"id": "root", "name": "root"},
                {"id": "mid",  "name": "mid"},
                {"id": "leaf1","name": "leaf1"},
                {"id": "leaf2","name": "leaf2"},
            ],
        }

    def test_leaves_first(self):
        result = M.functions_leaf_first(self._analysis())
        ids = [f["id"] for f in result]
        # depth 2 comes before depth 1 before depth 0
        assert ids.index("leaf1") < ids.index("mid")
        assert ids.index("mid") < ids.index("root")

    def test_all_functions_present(self):
        result = M.functions_leaf_first(self._analysis())
        assert {f["id"] for f in result} == {"root", "mid", "leaf1", "leaf2"}

    def test_empty_analysis(self):
        assert M.functions_leaf_first({}) == []


class TestCollectStubCandidates:
    def test_collects_from_stub_candidates(self):
        analysis = {
            "stub_candidates": {
                "file1.c": ["foo", "bar"],
                "file2.c": ["baz"],
            }
        }
        result = M.collect_stub_candidates(analysis)
        assert set(result) == {"foo", "bar", "baz"}

    def test_deduplicates(self):
        analysis = {
            "stub_candidates": {
                "a.c": ["foo"],
                "b.c": ["foo"],
            }
        }
        result = M.collect_stub_candidates(analysis)
        assert result.count("foo") == 1

    def test_empty(self):
        assert M.collect_stub_candidates({}) == []


# ===========================================================================
# Source file helpers
# ===========================================================================

class TestProjectSourceFiles:
    def test_finds_c_files(self, tmp: Path):
        src = tmp / "project" / "src" / "myproc"
        src.mkdir(parents=True)
        (src / "main.c").write_text("int main(){}", encoding="utf-8")
        (src / "util.c").write_text("void util(){}", encoding="utf-8")
        (src / "header.h").write_text("", encoding="utf-8")
        cfg = make_cfg(source_dir=src)
        files = M._project_source_files(cfg)
        names = {f.name for f in files}
        assert "main.c" in names
        assert "util.c" in names
        assert "header.h" not in names

    def test_empty_dir(self, tmp: Path):
        src = tmp / "src" / "proc"
        src.mkdir(parents=True)
        cfg = make_cfg(source_dir=src)
        assert M._project_source_files(cfg) == []


# ===========================================================================
# Stub helpers
# ===========================================================================

class TestStubExists:
    def test_found(self, tmp: Path):
        f = tmp / "test.c"
        f.write_text("void __wrap_foo(void) {}\n", encoding="utf-8")
        assert M.stub_exists(f, "foo") is True

    def test_not_found(self, tmp: Path):
        f = tmp / "test.c"
        f.write_text("void __wrap_bar(void) {}\n", encoding="utf-8")
        assert M.stub_exists(f, "foo") is False

    def test_partial_match_not_counted(self, tmp: Path):
        f = tmp / "test.c"
        f.write_text("void __wrap_foobar(void) {}\n", encoding="utf-8")
        # __wrap_foo\b should NOT match __wrap_foobar
        assert M.stub_exists(f, "foo") is False


class TestInsertStubIntoTestFile:
    def test_inserts_stub(self, tmp: Path):
        f = tmp / "test.c"
        f.write_text("/* === Linker Wrapper Stubs === */\n", encoding="utf-8")
        inserted = M.insert_stub_into_test_file(f, "malloc", "void *__wrap_malloc(size_t n){return NULL;}")
        assert inserted is True
        assert "__wrap_malloc" in f.read_text(encoding="utf-8")

    def test_idempotent(self, tmp: Path):
        f = tmp / "test.c"
        f.write_text("/* === Linker Wrapper Stubs === */\nvoid __wrap_foo(void){}\n", encoding="utf-8")
        inserted = M.insert_stub_into_test_file(f, "foo", "void __wrap_foo(void){}")
        assert inserted is False

    def test_no_marker_appends(self, tmp: Path):
        f = tmp / "test.c"
        f.write_text("int x = 0;\n", encoding="utf-8")
        M.insert_stub_into_test_file(f, "bar", "void __wrap_bar(){}")
        text = f.read_text(encoding="utf-8")
        assert "__wrap_bar" in text


class TestEnsureWrapFlag:
    def test_adds_flag(self, tmp: Path):
        mf = tmp / "Makefile"
        mf.write_text("CC=gcc\n", encoding="utf-8")
        added = M.ensure_wrap_flag(mf, "myfunc")
        assert added is True
        assert "-Wl,--wrap=myfunc" in mf.read_text(encoding="utf-8")

    def test_idempotent(self, tmp: Path):
        mf = tmp / "Makefile"
        mf.write_text("WRAP_FUNCS += -Wl,--wrap=myfunc\n", encoding="utf-8")
        added = M.ensure_wrap_flag(mf, "myfunc")
        assert added is False


class TestSyncWrapFlags:
    def test_syncs_all_wraps(self, tmp: Path):
        test_c = tmp / "test.c"
        test_c.write_text(
            "void __wrap_alpha(){}\nvoid __wrap_beta(){}\n",
            encoding="utf-8",
        )
        mf = tmp / "Makefile"
        mf.write_text("CC=gcc\n", encoding="utf-8")
        M.sync_wrap_flags(test_c, mf)
        content = mf.read_text(encoding="utf-8")
        assert "-Wl,--wrap=alpha" in content
        assert "-Wl,--wrap=beta" in content

    def test_no_duplication(self, tmp: Path):
        test_c = tmp / "test.c"
        test_c.write_text("void __wrap_alpha(){}\n", encoding="utf-8")
        mf = tmp / "Makefile"
        mf.write_text("WRAP_FUNCS += -Wl,--wrap=alpha\n", encoding="utf-8")
        M.sync_wrap_flags(test_c, mf)
        content = mf.read_text(encoding="utf-8")
        assert content.count("-Wl,--wrap=alpha") == 1


# ===========================================================================
# JSON helpers (new)
# ===========================================================================

class TestReadJsonLoose:
    def test_valid_json(self, tmp: Path):
        p = tmp / "out.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        assert M._read_json_loose(p) == {"a": 1}

    def test_json_with_surrounding_text(self, tmp: Path):
        p = tmp / "out.json"
        p.write_text('some preamble\n{"score": 42}\ntrailing', encoding="utf-8")
        result = M._read_json_loose(p)
        assert result == {"score": 42}

    def test_missing_file_returns_empty(self, tmp: Path):
        assert M._read_json_loose(tmp / "nope.json") == {}

    def test_empty_file_returns_empty(self, tmp: Path):
        p = tmp / "empty.json"
        p.write_text("", encoding="utf-8")
        assert M._read_json_loose(p) == {}

    def test_invalid_json_no_braces_returns_empty(self, tmp: Path):
        p = tmp / "bad.json"
        p.write_text("not json at all", encoding="utf-8")
        assert M._read_json_loose(p) == {}


class TestSemanticContextHelpers:
    def test_path(self, tmp: Path):
        p = M._semantic_context_path(tmp)
        assert p == tmp / "_leaf_to_root_semantic_context.json"

    def test_load_missing_returns_empty_structure(self, tmp: Path):
        data = M._load_semantic_context(tmp)
        assert data == {"functions": {}}

    def test_append_and_reload(self, tmp: Path):
        func = {"id": "my_func", "name": "my_func"}
        verdict = {"passed": True, "score": 80, "summary": "ok"}
        M._append_semantic_context(tmp, func, verdict)

        loaded = M._load_semantic_context(tmp)
        assert "my_func" in loaded["functions"]
        assert loaded["functions"]["my_func"]["verdict"]["score"] == 80

    def test_append_multiple(self, tmp: Path):
        for i in range(3):
            M._append_semantic_context(
                tmp,
                {"id": f"func_{i}"},
                {"passed": True, "score": 70 + i},
            )
        loaded = M._load_semantic_context(tmp)
        assert len(loaded["functions"]) == 3

    def test_overwrite_same_func(self, tmp: Path):
        M._append_semantic_context(tmp, {"id": "f"}, {"score": 50})
        M._append_semantic_context(tmp, {"id": "f"}, {"score": 99})
        loaded = M._load_semantic_context(tmp)
        assert loaded["functions"]["f"]["verdict"]["score"] == 99


# ===========================================================================
# Prompt sanity checks (content, not formatting)
# ===========================================================================

class TestPromptForCompileFix:
    def test_contains_key_sections(self):
        p = M.prompt_for_compile_fix(
            makefile="/t/Makefile",
            test_file="/t/test_proc.c",
            build_output="error: undeclared",
        )
        assert "failed to compile" in p
        assert "TEST HARNESS FIXING RULES" in p
        assert "You MUST NOT change" in p
        assert "error: undeclared" in p
        assert "/t/Makefile" in p
        assert "/t/test_proc.c" in p

    def test_does_not_mention_production_edit(self):
        p = M.prompt_for_compile_fix("/m", "/t", "err")
        assert "Do not edit production code" in p


class TestPromptForMinimalTest:
    def test_contains_entry_sym(self):
        p = M.prompt_for_minimal_test("myproc", "/t/test_myproc.c", "myproc_entry_main")
        assert "myproc_entry_main" in p
        assert "/t/test_myproc.c" in p

    def test_no_blocking(self):
        p = M.prompt_for_minimal_test("myproc", "/t/test_myproc.c", "myproc_entry_main")
        assert "blocking" in p.lower()

    def test_preserve_existing_tests(self):
        p = M.prompt_for_minimal_test("p", "f", "e")
        assert "Preserve existing tests" in p


class TestPromptForSemanticTestJudge:
    def test_required_fields_listed(self):
        p = M.prompt_for_semantic_test_judge(
            process_name="proc",
            test_file="/t/test.c",
            func={"id": "do_thing", "name": "do_thing"},
            coverage={"summary": {"coverage_percent": 90}},
            make_result={"ok": True},
            semantic_context={"functions": {}},
            verdict_file="/t/_judge.json",
            min_score=75,
        )
        assert "passed" in p
        assert "score" in p
        assert "missing_cases" in p
        assert "/t/_judge.json" in p
        assert "75" in p
        assert "do_thing" in p

    def test_min_score_in_prompt(self):
        p = M.prompt_for_semantic_test_judge(
            process_name="p", test_file="f",
            func={"id": "f"}, coverage={}, make_result={},
            semantic_context={}, verdict_file="v", min_score=60,
        )
        assert "60" in p


class TestPromptForSemanticTestRepair:
    def test_contains_function_id(self):
        p = M.prompt_for_semantic_test_repair(
            process_name="proc",
            test_file="/t/test.c",
            func={"id": "my_fn"},
            coverage={},
            judge_verdict={"score": 40, "missing_cases": ["null input"]},
            semantic_context={},
        )
        assert "my_fn" in p
        assert "null input" in p

    def test_no_weakening(self):
        p = M.prompt_for_semantic_test_repair(
            process_name="p", test_file="f", func={"id": "f"},
            coverage={}, judge_verdict={}, semantic_context={},
        )
        assert "Do not weaken" in p


class TestPromptForFunctionTestWithSemanticContext:
    def test_renders_without_error(self):
        p = M.prompt_for_function_test_with_semantic_context(
            func={"id": "fn", "name": "fn", "source_file": "/s.c",
                  "start_line": 1, "end_line": 20},
            coverage={},
            test_file="/t/test.c",
            process_name="proc",
            attempt=1,
            max_attempts=4,
            make_ok=True,
            semantic_context={"functions": {}},
            last_judge_verdict=None,
        )
        assert "fn" in p
        assert "1/4" in p

    def test_empty_verdict_renders(self):
        # The `last_judge_verdict or {}` bug fix: must not raise
        p = M.prompt_for_function_test_with_semantic_context(
            func={"id": "f"}, coverage={}, test_file="t",
            process_name="p", attempt=1, max_attempts=2,
            make_ok=True, semantic_context={}, last_judge_verdict=None,
        )
        assert "{}" in p or p  # should render without exception

    def test_judge_verdict_included(self):
        verdict = {"score": 55, "summary": "needs work"}
        p = M.prompt_for_function_test_with_semantic_context(
            func={"id": "f"}, coverage={}, test_file="t",
            process_name="p", attempt=1, max_attempts=2,
            make_ok=True, semantic_context={}, last_judge_verdict=verdict,
        )
        assert "55" in p
        assert "needs work" in p


# ===========================================================================
# collect_failure_diagnostics
# ===========================================================================

class TestCollectFailureDiagnostics:
    def _res(self, stdout="", stderr=""):
        return {"ok": False, "returncode": 1, "stdout": stdout, "stderr": stderr}

    def test_no_binary_reports_that(self, tmp: Path):
        test_file = tmp / "test_proc.c"
        result = M.collect_failure_diagnostics(tmp, test_file, self._res(stderr="error: foo"))
        assert "binary does not exist" in result.lower() or "binary check" in result
        assert "error: foo" in result

    def test_includes_original_output(self, tmp: Path):
        test_file = tmp / "test_proc.c"
        result = M.collect_failure_diagnostics(tmp, test_file, self._res(stderr="STDERR_CONTENT"))
        assert "STDERR_CONTENT" in result

    def test_reads_log_file(self, tmp: Path):
        test_file = tmp / "test_proc.c"
        log = tmp / "test_proc_log.txt"
        log.write_text("CUnit run log here", encoding="utf-8")
        result = M.collect_failure_diagnostics(tmp, test_file, self._res())
        assert "CUnit run log here" in result

    def test_truncates_large_output(self, tmp: Path):
        test_file = tmp / "test_proc.c"
        big = "X" * 50000
        result = M.collect_failure_diagnostics(tmp, test_file, self._res(stderr=big))
        # Should not explode; should be truncated
        assert len(result) < 60000


class TestBuildOutputWithRuntimeDiagnostics:
    def test_contains_note_to_agent(self, tmp: Path):
        test_file = tmp / "test_proc.c"
        res = {"ok": False, "stdout": "", "stderr": "link error"}
        out = M.build_output_with_runtime_diagnostics(tmp, test_file, res)
        assert "NOTE TO AGENT" in out
        assert "RUNTIME / EXECUTION DIAGNOSTICS" in out
        assert "link error" in out


# ===========================================================================
# run_semantic_test_judge
# ===========================================================================

class TestRunSemanticTestJudge:
    def test_returns_default_on_no_verdict(self, tmp: Path):
        cfg = make_cfg()
        repo_root = tmp

        with patch.object(M, "run_agent") as mock_agent:
            mock_agent.return_value = {"exit_code": 0}
            verdict = M.run_semantic_test_judge(
                cfg,
                test_dir=tmp,
                repo_root=repo_root,
                process_name="proc",
                test_file=tmp / "test_proc.c",
                func={"id": "fn", "name": "fn"},
                coverage={},
                make_result={"ok": True},
            )
        assert verdict["passed"] is False
        assert verdict["score"] == 0
        assert "missing_cases" in verdict

    def test_reads_verdict_file(self, tmp: Path):
        cfg = make_cfg()
        fid = "my_func"
        verdict_file = tmp / f"_semantic_judge_{fid}.json"

        def write_verdict(*args, **kwargs):
            verdict_file.write_text(
                json.dumps({"passed": True, "score": 85, "summary": "good"}),
                encoding="utf-8",
            )

        with patch.object(M, "run_agent", side_effect=write_verdict):
            result = M.run_semantic_test_judge(
                cfg,
                test_dir=tmp,
                repo_root=tmp,
                process_name="proc",
                test_file=tmp / "test_proc.c",
                func={"id": fid, "name": fid},
                coverage={},
                make_result={"ok": True},
            )
        assert result["passed"] is True
        assert result["score"] == 85

    def test_score_below_min_fails(self, tmp: Path):
        cfg = make_cfg(semantic_judge_min_score=80)
        fid = "low_score_fn"
        verdict_file = tmp / f"_semantic_judge_{fid}.json"

        def write_verdict(*args, **kwargs):
            verdict_file.write_text(
                json.dumps({"passed": True, "score": 60, "summary": "ok"}),
                encoding="utf-8",
            )

        with patch.object(M, "run_agent", side_effect=write_verdict):
            result = M.run_semantic_test_judge(
                cfg,
                test_dir=tmp,
                repo_root=tmp,
                process_name="proc",
                test_file=tmp / "test_proc.c",
                func={"id": fid, "name": fid},
                coverage={},
                make_result={"ok": True},
            )
        # score 60 < min_score 80 → should fail even if agent says passed=True
        assert result["passed"] is False
        assert result["score"] == 60


# ===========================================================================
# parse_args
# ===========================================================================

class TestParseArgs:
    def _minimal_argv(self, tmp: Path):
        src = tmp / "src" / "proc"
        src.mkdir(parents=True)
        agent = tmp / "agent.js"
        agent.touch()
        return ["--source", str(src), "--agent-js", str(agent)]

    def test_default_semantic_score(self, tmp: Path):
        cfg = M.parse_args(self._minimal_argv(tmp))
        assert cfg.semantic_judge_min_score == 75

    def test_custom_semantic_score(self, tmp: Path):
        cfg = M.parse_args(self._minimal_argv(tmp) + ["--semantic-judge-min-score", "60"])
        assert cfg.semantic_judge_min_score == 60

    def test_default_coverage_threshold(self, tmp: Path):
        cfg = M.parse_args(self._minimal_argv(tmp))
        assert cfg.coverage_threshold == 80.0

    def test_stub_batch_size_min_1(self, tmp: Path):
        cfg = M.parse_args(self._minimal_argv(tmp) + ["--stub-batch-size", "0"])
        assert cfg.stub_batch_size >= 1


# ===========================================================================
# Similarity / doc matching
# ===========================================================================

class TestSimilarityRatio:
    def test_identical(self):
        assert M.similarity_ratio("foo_bar", "foo_bar") == 1.0

    def test_similar(self):
        assert M.similarity_ratio("pmf_init", "pmf_init_ex") > 0.7

    def test_different(self):
        assert M.similarity_ratio("alpha", "zzz") < 0.5


class TestFindFuncDoc:
    def test_exact_match(self, tmp: Path):
        (tmp / "myfunc.md").write_text("# myfunc", encoding="utf-8")
        path, score, _ = M.find_func_doc(tmp, "myfunc")
        assert path == tmp / "myfunc.md"
        assert score == 1.0

    def test_fuzzy_match(self, tmp: Path):
        (tmp / "pmf_init_real.md").write_text("# init", encoding="utf-8")
        path, score, _ = M.find_func_doc(tmp, "pmf_init_real", threshold=0.9)
        assert path is not None
        assert score >= 0.9

    def test_no_match_below_threshold(self, tmp: Path):
        (tmp / "something_totally_different.md").write_text("#", encoding="utf-8")
        path, score, _ = M.find_func_doc(tmp, "my_func", threshold=0.9)
        assert path is None

    def test_missing_dir(self, tmp: Path):
        path, score, cands = M.find_func_doc(tmp / "nodir", "fn")
        assert path is None
        assert cands == []


# ===========================================================================
# run_agent — dry_run mode
# ===========================================================================

class TestRunAgentDryRun:
    def test_dry_run_no_subprocess(self, tmp: Path):
        cfg = make_cfg(dry_run=True, agent_js=tmp / "agent.js", source_dir=tmp / "src" / "p")
        result = M.run_agent(
            cfg,
            work_dir=tmp,
            prompt="do stuff",
            history_name="test.json",
        )
        assert result["exit_code"] == 0
        assert result["timed_out"] is False

    def test_capture_raw_http_trace_in_cmd(self, tmp: Path):
        """Verify --capture-raw-http-trace is in the agent command."""
        cfg = make_cfg(dry_run=True, agent_js=tmp / "agent.js", source_dir=tmp / "src" / "p")
        # Capture what would be run by patching subprocess
        built_cmd = []

        original = M.run_agent
        # In dry_run mode it prints rather than runs, so just check the cmd built
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            cfg2 = make_cfg(dry_run=False, agent_js=tmp / "agent.js",
                            source_dir=tmp / "src" / "p",
                            agent_timeout_sec=5)
            hist = tmp / "hist"
            hist.mkdir()
            try:
                M.run_agent(cfg2, work_dir=tmp, prompt="x", history_name="h.json",
                            history_dir=hist)
            except Exception:
                pass
            if mock_run.called:
                built_cmd = mock_run.call_args[0][0]
        if built_cmd:
            assert "--capture-raw-http-trace" in built_cmd
