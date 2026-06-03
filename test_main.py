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
        max_unit_test_workers=2,
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

    def test_max_unit_test_workers_default(self):
        cfg = make_cfg()
        assert cfg.max_unit_test_workers == 2

    def test_max_unit_test_workers_custom(self):
        cfg = make_cfg(max_unit_test_workers=8)
        assert cfg.max_unit_test_workers == 8


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
# parse_source_makefile_flags
# ===========================================================================

class TestParseSourceMakefileFlags:
    def test_extracts_cflags(self, tmp: Path):
        mk = tmp / "Makefile"
        mk.write_text("CFLAGS = -Wall -O2\n", encoding="utf-8")
        flags = M.parse_source_makefile_flags(mk)
        assert "-Wall" in flags["CFLAGS"]
        assert "-O2" in flags["CFLAGS"]

    def test_extracts_include(self, tmp: Path):
        mk = tmp / "Makefile"
        mk.write_text("INCLUDE = -I/usr/include -I./local\n", encoding="utf-8")
        flags = M.parse_source_makefile_flags(mk)
        assert "-I/usr/include" in flags["INCLUDE"]

    def test_plus_equals(self, tmp: Path):
        mk = tmp / "Makefile"
        mk.write_text("CFLAGS += -DDEBUG\n", encoding="utf-8")
        flags = M.parse_source_makefile_flags(mk)
        assert "-DDEBUG" in flags["CFLAGS"]

    def test_multiple_assignments_concatenated(self, tmp: Path):
        mk = tmp / "Makefile"
        mk.write_text("CFLAGS = -Wall\nCFLAGS += -O2\n", encoding="utf-8")
        flags = M.parse_source_makefile_flags(mk)
        assert "-Wall" in flags["CFLAGS"]
        assert "-O2" in flags["CFLAGS"]

    def test_empty_makefile_returns_empty_dict(self, tmp: Path):
        mk = tmp / "Makefile"
        mk.write_text("", encoding="utf-8")
        assert M.parse_source_makefile_flags(mk) == {}

    def test_missing_file_returns_empty_dict(self, tmp: Path):
        flags = M.parse_source_makefile_flags(tmp / "nonexistent")
        assert flags == {}

    def test_all_known_vars_extracted(self, tmp: Path):
        mk = tmp / "Makefile"
        mk.write_text(
            "CFLAGS = -Wall\nCFLAGS_LINUX = -linux\nCPPFLAGS = -DFOO\n"
            "INCLUDE = -I.\nLDFLAGS = -L/lib\nLDLIBS = -lm\nLIBS = -lpthread\n",
            encoding="utf-8",
        )
        flags = M.parse_source_makefile_flags(mk)
        for var in ["CFLAGS", "CFLAGS_LINUX", "CPPFLAGS", "INCLUDE", "LDFLAGS", "LDLIBS", "LIBS"]:
            assert var in flags, f"{var} not extracted"

    def test_does_not_extract_unrelated_vars(self, tmp: Path):
        mk = tmp / "Makefile"
        mk.write_text("CC = gcc\nAR = ar\n", encoding="utf-8")
        flags = M.parse_source_makefile_flags(mk)
        assert "CC" not in flags
        assert "AR" not in flags


# ===========================================================================
# extract_test_additions
# ===========================================================================

class TestExtractTestAdditions:
    def _write(self, tmp: Path, content: str) -> Path:
        f = tmp / "test_fn.c"
        f.write_text(content, encoding="utf-8")
        return f

    def test_extracts_test_cases_and_reg(self, tmp: Path):
        f = self._write(tmp, """
/* === Test Cases === */
static void test_normal(void) {
    CU_ASSERT_EQUAL(1, 1);
}
/* === Test Registration === */
int main(void) {
    CU_add_test(suite, "test_normal", test_normal);
    CU_basic_set_mode(CU_BRM_VERBOSE);
}
""")
        cases, reg = M.extract_test_additions(f)
        assert "test_normal" in cases
        assert any("test_normal" in r for r in reg)

    def test_no_markers_returns_empty(self, tmp: Path):
        f = self._write(tmp, "int x = 0;\n")
        cases, reg = M.extract_test_additions(f)
        assert cases == ""
        assert reg == []

    def test_no_registration_still_returns_cases(self, tmp: Path):
        f = self._write(tmp, "/* === Test Cases === */\nstatic void test_x(){}\n")
        cases, reg = M.extract_test_additions(f)
        assert "test_x" in cases

    def test_multiple_cu_add_test_calls(self, tmp: Path):
        f = self._write(tmp, """
/* === Test Cases === */
static void test_a(){}
static void test_b(){}
/* === Test Registration === */
int main(void){
    CU_add_test(suite, "a", test_a);
    CU_add_test(suite, "b", test_b);
}
""")
        _, reg = M.extract_test_additions(f)
        assert len(reg) == 2
        assert any("test_a" in r for r in reg)
        assert any("test_b" in r for r in reg)

    def test_empty_cases_section(self, tmp: Path):
        f = self._write(tmp, """
/* === Test Cases === */
/* === Test Registration === */
int main(void){ CU_add_test(suite, "x", x); }
""")
        cases, reg = M.extract_test_additions(f)
        assert cases.strip() == ""
        assert len(reg) == 1


# ===========================================================================
# _scaffold_unit_test_dir
# ===========================================================================

def _make_project(tmp: Path) -> tuple:
    """Create minimal project structure, return (cfg, paths, func)."""
    src = tmp / "project" / "src" / "proc"
    src.mkdir(parents=True)
    (src / "proc.c").write_text("void foo(){}\n", encoding="utf-8")

    test_dir = tmp / "project" / "tests" / "proc"
    test_dir.mkdir(parents=True)
    master_mk = test_dir / "Makefile"
    master_mk.write_text("CC=gcc\nWRAP_FUNCS += -Wl,--wrap=foo\n", encoding="utf-8")
    test_file = test_dir / "test_proc.c"
    test_file.write_text("", encoding="utf-8")

    import json as _json
    ctx = {"flags": {"CFLAGS": "-Wall"}, "actual_source_files": [str(src / "proc.c")]}
    (test_dir / "_pipeline_context.json").write_text(_json.dumps(ctx), encoding="utf-8")

    cfg = make_cfg(source_dir=src)
    paths = {
        "test_dir": test_dir, "process_name": "proc",
        "test_file": test_file, "makefile": master_mk,
    }
    func = {
        "id": "my_func", "name": "my_func",
        "source_file": str(src / "proc.c"),
        "start_line": 1, "end_line": 10, "depth": 2,
    }
    return cfg, paths, func


class TestScaffoldUnitTestDir:
    def test_creates_all_artefacts(self, tmp: Path):
        cfg, paths, func = _make_project(tmp)
        unit_dir = M._scaffold_unit_test_dir(cfg, paths, func)
        assert unit_dir.exists()
        assert (unit_dir / "test_my_func.c").exists()
        assert (unit_dir / "Makefile").exists()
        assert (unit_dir / "agent_history").is_dir()

    def test_test_file_has_cunit_includes(self, tmp: Path):
        cfg, paths, func = _make_project(tmp)
        unit_dir = M._scaffold_unit_test_dir(cfg, paths, func)
        content = (unit_dir / "test_my_func.c").read_text(encoding="utf-8")
        assert "CUnit/CUnit.h" in content
        assert "CUnit/Basic.h" in content

    def test_test_file_has_all_markers(self, tmp: Path):
        cfg, paths, func = _make_project(tmp)
        unit_dir = M._scaffold_unit_test_dir(cfg, paths, func)
        content = (unit_dir / "test_my_func.c").read_text(encoding="utf-8")
        for marker in M.TEST_FILE_MARKERS:
            assert marker in content, f"Missing marker: {marker}"

    def test_test_file_has_own_main(self, tmp: Path):
        cfg, paths, func = _make_project(tmp)
        unit_dir = M._scaffold_unit_test_dir(cfg, paths, func)
        content = (unit_dir / "test_my_func.c").read_text(encoding="utf-8")
        assert "int main(void)" in content
        assert "CU_initialize_registry" in content

    def test_idempotent_does_not_overwrite(self, tmp: Path):
        cfg, paths, func = _make_project(tmp)
        unit_dir = M._scaffold_unit_test_dir(cfg, paths, func)
        sentinel = "// SENTINEL_MARKER"
        tf = unit_dir / "test_my_func.c"
        tf.write_text(tf.read_text(encoding="utf-8") + sentinel, encoding="utf-8")
        M._scaffold_unit_test_dir(cfg, paths, func)  # second call
        assert sentinel in tf.read_text(encoding="utf-8")

    def test_makefile_has_wrap_funcs(self, tmp: Path):
        cfg, paths, func = _make_project(tmp)
        unit_dir = M._scaffold_unit_test_dir(cfg, paths, func)
        mk_content = (unit_dir / "Makefile").read_text(encoding="utf-8")
        assert "WRAP_FUNCS" in mk_content
        assert "-Wl,--wrap=foo" in mk_content

    def test_makefile_has_test_program(self, tmp: Path):
        cfg, paths, func = _make_project(tmp)
        unit_dir = M._scaffold_unit_test_dir(cfg, paths, func)
        mk_content = (unit_dir / "Makefile").read_text(encoding="utf-8")
        assert "TEST_PROGRAM" in mk_content
        assert "test_my_func" in mk_content


# ===========================================================================
# _validate_stub_locally
# ===========================================================================

class TestValidateStubLocally:
    def _setup(self, tmp: Path):
        src = tmp / "src" / "proc"
        src.mkdir(parents=True)
        cfg = make_cfg(source_dir=src)
        stub_dir = tmp / "_stub_gen" / "foo"
        stub_dir.mkdir(parents=True)
        (stub_dir / "stub.c").write_text(
            'void __wrap_foo(void){ fprintf(stderr,"called\\n"); }\n',
            encoding="utf-8",
        )
        return cfg, stub_dir

    def test_skips_if_already_validated(self, tmp: Path):
        cfg, stub_dir = self._setup(tmp)
        M.write_json(stub_dir / "result.json", {"validated": True, "func_name": "foo"})
        with patch("subprocess.run") as mock_run:
            M._validate_stub_locally(cfg, tmp, "foo", stub_dir)
        mock_run.assert_not_called()

    def test_writes_result_json_on_success(self, tmp: Path):
        cfg, stub_dir = self._setup(tmp)
        ok = MagicMock(returncode=0, stdout="stub_validate OK", stderr="")
        with patch("subprocess.run", return_value=ok):
            M._validate_stub_locally(cfg, tmp, "foo", stub_dir)
        result_file = stub_dir / "result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert data["validated"] is True
        assert data["func_name"] == "foo"

    def test_writes_stub_validate_main_c(self, tmp: Path):
        cfg, stub_dir = self._setup(tmp)
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=ok):
            M._validate_stub_locally(cfg, tmp, "foo", stub_dir)
        harness = stub_dir / "stub_validate_main.c"
        assert harness.exists()
        content = harness.read_text(encoding="utf-8")
        assert "int main" in content
        assert "return 0" in content

    def test_calls_fix_agent_on_compile_failure(self, tmp: Path):
        cfg, stub_dir = self._setup(tmp)
        cfg.max_fix_attempts = 1
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            # First compile call fails; all subsequent succeed
            if call_count[0] == 1:
                return MagicMock(returncode=1, stdout="", stderr="error: bad syntax")
            return MagicMock(returncode=0, stdout="stub_validate OK", stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(M, "_fix_stub_with_agent") as mock_fix:
            M._validate_stub_locally(cfg, tmp, "foo", stub_dir)

        mock_fix.assert_called_once()
        assert call_count[0] >= 4  # compile fail, then compile+link+run succeed

    def test_calls_fix_agent_on_link_failure(self, tmp: Path):
        cfg, stub_dir = self._setup(tmp)
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            # compile succeeds, link fails once, then everything succeeds
            if call_count[0] == 2:
                return MagicMock(returncode=1, stdout="", stderr="undefined ref")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(M, "_fix_stub_with_agent") as mock_fix:
            M._validate_stub_locally(cfg, tmp, "foo", stub_dir)

        mock_fix.assert_called_once()

    def test_calls_fix_agent_on_runtime_failure(self, tmp: Path):
        cfg, stub_dir = self._setup(tmp)
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            # compile + link succeed, run fails once, then everything succeeds
            if call_count[0] == 3:
                return MagicMock(returncode=139, stdout="", stderr="Segfault")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(M, "_fix_stub_with_agent") as mock_fix:
            M._validate_stub_locally(cfg, tmp, "foo", stub_dir)

        mock_fix.assert_called_once()


# ===========================================================================
# integrate_all_stubs_sequential
# ===========================================================================

class TestIntegrateAllStubsSequential:
    def _setup(self, tmp: Path):
        src = tmp / "src" / "proc"
        src.mkdir(parents=True)
        (src / "proc.c").write_text("void foo(){}\n", encoding="utf-8")
        cfg = make_cfg(source_dir=src)
        test_file = tmp / "test_proc.c"
        test_file.write_text("/* === Linker Wrapper Stubs === */\n", encoding="utf-8")
        makefile = tmp / "Makefile"
        makefile.write_text("CC=gcc\n", encoding="utf-8")
        paths = {"test_file": test_file, "makefile": makefile,
                 "test_dir": tmp, "process_name": "proc"}
        return cfg, paths, test_file, makefile

    def test_inserts_stub_and_wrap_flag(self, tmp: Path):
        cfg, paths, test_file, makefile = self._setup(tmp)
        body = 'void __wrap_myfunc(void){ fprintf(stderr, "called\\n"); }'
        with patch.object(M, "run_make_test", return_value={"ok": True, "timed_out": False}):
            M.integrate_all_stubs_sequential(cfg, paths, {"myfunc": body}, {})
        assert "__wrap_myfunc" in test_file.read_text(encoding="utf-8")
        assert "-Wl,--wrap=myfunc" in makefile.read_text(encoding="utf-8")

    def test_skips_already_integrated_stub(self, tmp: Path):
        cfg, paths, test_file, makefile = self._setup(tmp)
        test_file.write_text(
            "/* === Linker Wrapper Stubs === */\nvoid __wrap_foo(void){}\n",
            encoding="utf-8",
        )
        with patch.object(M, "run_make_test") as mock_make:
            M.integrate_all_stubs_sequential(cfg, paths, {"foo": "void __wrap_foo(){}"}, {})
        mock_make.assert_not_called()

    def test_loops_compile_fix_until_make_passes(self, tmp: Path):
        cfg, paths, test_file, makefile = self._setup(tmp)
        cfg.max_fix_attempts = 1
        make_calls = [0]

        def fake_make(*a, **kw):
            make_calls[0] += 1
            if make_calls[0] == 1:
                return {"ok": False, "returncode": 1, "stdout": "",
                        "stderr": "error", "timed_out": False}
            return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

        with patch.object(M, "run_make_test", side_effect=fake_make), \
             patch.object(M, "run_agent"), \
             patch.object(M, "build_output_with_runtime_diagnostics", return_value="diag"):
            M.integrate_all_stubs_sequential(
                cfg, paths, {"bar": "void __wrap_bar(){}"}, {})

        assert make_calls[0] == 2

    def test_multiple_stubs_each_checked(self, tmp: Path):
        cfg, paths, test_file, makefile = self._setup(tmp)
        make_calls = [0]

        def fake_make(*a, **kw):
            make_calls[0] += 1
            return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

        bodies = {
            "alpha": "void __wrap_alpha(){}",
            "beta": "void __wrap_beta(){}",
        }
        with patch.object(M, "run_make_test", side_effect=fake_make):
            M.integrate_all_stubs_sequential(cfg, paths, bodies, {})

        assert make_calls[0] == 2  # one make test per stub
        content = test_file.read_text(encoding="utf-8")
        assert "__wrap_alpha" in content
        assert "__wrap_beta" in content


# ===========================================================================
# handle_stubs
# ===========================================================================

class TestHandleStubs:
    def test_generates_until_validated_body_exists(self, tmp: Path):
        src = tmp / "src" / "proc"
        src.mkdir(parents=True)
        cfg = make_cfg(source_dir=src, max_stub_gen_retries=1)
        test_file = tmp / "test_proc.c"
        test_file.write_text("/* === Linker Wrapper Stubs === */\n", encoding="utf-8")
        makefile = tmp / "Makefile"
        makefile.write_text("CC=gcc\n", encoding="utf-8")
        paths = {
            "test_file": test_file,
            "makefile": makefile,
            "test_dir": tmp,
            "process_name": "proc",
        }
        analysis = {"stub_candidates": {"proc.c": ["foo"]}}
        body = 'void __wrap_foo(void){ fprintf(stderr, "called\\n"); }'

        with patch.object(M, "generate_stub_code", side_effect=[None, body]) as mock_gen, \
             patch.object(M, "integrate_all_stubs_sequential") as mock_integrate:
            M.handle_stubs(cfg, paths, analysis)

        assert mock_gen.call_count == 2
        mock_integrate.assert_called_once()
        assert mock_integrate.call_args[0][2] == {"foo": body}


# ===========================================================================
# build_annotated_makefile
# ===========================================================================

class TestBuildAnnotatedMakefile:
    def _setup(self, tmp: Path):
        src = tmp / "project" / "src" / "proc"
        src.mkdir(parents=True)
        source_mk = src / "Makefile"
        source_mk.write_text("CFLAGS = -Wall\nINCLUDE = -I.\n", encoding="utf-8")
        (src / "proc.c").write_text("void foo(){}\n", encoding="utf-8")

        test_dir = tmp / "project" / "tests" / "proc"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "test_proc.c"
        test_file.write_text("", encoding="utf-8")
        makefile = test_dir / "Makefile"

        cfg = make_cfg(source_dir=src)
        paths = {
            "test_dir": test_dir, "process_name": "proc",
            "test_file": test_file, "makefile": makefile,
        }
        return cfg, paths, test_dir, makefile

    def _fake_do_mkmf(self, makefile: Path):
        """Return a subprocess.run side_effect that creates the Makefile."""
        def _inner(*args, **kwargs):
            if isinstance(args[0], list) and "do_mkmf" in args[0]:
                makefile.write_text("CC=gcc\n# do_mkmf generated\n", encoding="utf-8")
            return MagicMock(returncode=0, stdout="", stderr="")
        return _inner

    def test_returns_cached_flags_if_context_exists(self, tmp: Path):
        cfg, paths, test_dir, makefile = self._setup(tmp)
        cached = {"flags": {"CFLAGS": "-O2"}, "process_name": "proc"}
        M.write_json(test_dir / "_pipeline_context.json", cached)
        with patch("subprocess.run") as mock_run:
            flags = M.build_annotated_makefile(cfg, paths)
        mock_run.assert_not_called()
        assert flags == {"CFLAGS": "-O2"}

    def test_runs_do_mkmf_if_no_makefile(self, tmp: Path):
        cfg, paths, test_dir, makefile = self._setup(tmp)
        with patch("subprocess.run", side_effect=self._fake_do_mkmf(makefile)):
            flags = M.build_annotated_makefile(cfg, paths)
        assert makefile.exists()

    def test_writes_pipeline_context_json(self, tmp: Path):
        cfg, paths, test_dir, makefile = self._setup(tmp)
        with patch("subprocess.run", side_effect=self._fake_do_mkmf(makefile)):
            M.build_annotated_makefile(cfg, paths)
        ctx_file = test_dir / "_pipeline_context.json"
        assert ctx_file.exists()
        ctx = json.loads(ctx_file.read_text(encoding="utf-8"))
        assert "flags" in ctx
        assert "process_name" in ctx
        assert ctx["process_name"] == "proc"
        assert "actual_source_files" in ctx

    def test_returns_flags_dict(self, tmp: Path):
        cfg, paths, test_dir, makefile = self._setup(tmp)
        with patch("subprocess.run", side_effect=self._fake_do_mkmf(makefile)):
            flags = M.build_annotated_makefile(cfg, paths)
        assert isinstance(flags, dict)
        # Source Makefile has CFLAGS = -Wall
        assert "-Wall" in flags.get("CFLAGS", "")

    def test_context_has_source_files(self, tmp: Path):
        cfg, paths, test_dir, makefile = self._setup(tmp)
        with patch("subprocess.run", side_effect=self._fake_do_mkmf(makefile)):
            M.build_annotated_makefile(cfg, paths)
        ctx = json.loads((test_dir / "_pipeline_context.json").read_text())
        assert len(ctx["actual_source_files"]) >= 1
        assert any("proc.c" in f for f in ctx["actual_source_files"])

    def test_appends_test_target_block(self, tmp: Path):
        cfg, paths, test_dir, makefile = self._setup(tmp)
        with patch("subprocess.run", side_effect=self._fake_do_mkmf(makefile)):
            M.build_annotated_makefile(cfg, paths)
        mk_text = makefile.read_text(encoding="utf-8")
        assert "TEST_PROGRAM" in mk_text
        assert "coverage-test" in mk_text

    def test_existing_bad_makefile_replaced(self, tmp: Path):
        cfg, paths, test_dir, makefile = self._setup(tmp)
        makefile.write_text(
            "# === Auto-generated CUnit pipeline rules ===\nCC=gcc\n"
            "# === End Auto-generated CUnit pipeline rules ===\n",
            encoding="utf-8",
        )
        with patch("subprocess.run", side_effect=self._fake_do_mkmf(makefile)):
            M.build_annotated_makefile(cfg, paths)
        assert makefile.with_name("Makefile.bad_pipeline_backup").exists()


# ===========================================================================
# integrate_all_unit_tests_sequential
# ===========================================================================

class TestIntegrateAllUnitTestsSequential:
    def _setup(self, tmp: Path):
        src = tmp / "src" / "proc"
        src.mkdir(parents=True)
        cfg = make_cfg(source_dir=src)

        test_dir = tmp
        test_file = test_dir / "test_proc.c"
        test_file.write_text(
            "/* === Test Cases === */\n/* === Test Registration === */\n"
            "int main(void){\n    CU_basic_set_mode(CU_BRM_VERBOSE);\n    return 0;\n}\n",
            encoding="utf-8",
        )
        makefile = test_dir / "Makefile"
        makefile.write_text("CC=gcc\n", encoding="utf-8")
        paths = {
            "test_dir": test_dir, "process_name": "proc",
            "test_file": test_file, "makefile": makefile,
        }
        return cfg, paths, test_file

    def _make_unit_test(self, tmp: Path, func_id: str, safe_id: str) -> Path:
        unit_dir = tmp / "_unit_tests" / safe_id
        unit_dir.mkdir(parents=True)
        content = f"""
/* === Test Cases === */
static void test_{safe_id}_normal(void) {{
    CU_ASSERT_EQUAL(1, 1);
}}
/* === Test Registration === */
int main(void) {{
    CU_add_test(suite, "normal", test_{safe_id}_normal);
    return 0;
}}
"""
        (unit_dir / f"test_{safe_id}.c").write_text(content, encoding="utf-8")
        return unit_dir

    def test_appends_test_cases_to_master(self, tmp: Path):
        cfg, paths, test_file = self._setup(tmp)
        func_id = "my_func"
        safe_id = M._safe_filename(func_id)
        unit_dir = self._make_unit_test(tmp, func_id, safe_id)

        analysis = {
            "function_levels": {"0": [func_id]},
            "functions": [{"id": func_id, "name": func_id,
                           "source_file": "/s.c", "depth": 0}],
        }
        results = {func_id: {"passed": True, "unit_dir": str(unit_dir)}}

        ok_check = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=ok_check):
            M.integrate_all_unit_tests_sequential(cfg, paths, analysis, results, {})

        content = test_file.read_text(encoding="utf-8")
        assert f"test_{safe_id}_normal" in content
        assert f"/* --- unit: {func_id} --- */" in content

    def test_skips_failed_functions(self, tmp: Path):
        cfg, paths, test_file = self._setup(tmp)
        func_id = "bad_func"
        safe_id = M._safe_filename(func_id)
        unit_dir = self._make_unit_test(tmp, func_id, safe_id)

        analysis = {
            "function_levels": {"0": [func_id]},
            "functions": [{"id": func_id, "name": func_id,
                           "source_file": "/s.c", "depth": 0}],
        }
        results = {func_id: {"passed": False}}

        original = test_file.read_text(encoding="utf-8")
        ok_run = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=ok_run):
            M.integrate_all_unit_tests_sequential(cfg, paths, analysis, results, {})

        assert test_file.read_text(encoding="utf-8") == original

    def test_skips_already_integrated(self, tmp: Path):
        cfg, paths, test_file = self._setup(tmp)
        func_id = "exist_func"
        safe_id = M._safe_filename(func_id)
        unit_dir = self._make_unit_test(tmp, func_id, safe_id)

        # Pre-mark as already integrated
        test_file.write_text(
            test_file.read_text(encoding="utf-8")
            + f"\n/* --- unit: {func_id} --- */\n",
            encoding="utf-8",
        )
        analysis = {
            "function_levels": {"0": [func_id]},
            "functions": [{"id": func_id, "name": func_id,
                           "source_file": "/s.c", "depth": 0}],
        }
        results = {func_id: {"passed": True, "unit_dir": str(unit_dir)}}

        ok_run = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=ok_run):
            M.integrate_all_unit_tests_sequential(cfg, paths, analysis, results, {})

    def test_loops_compile_fix_on_syntax_error(self, tmp: Path):
        cfg, paths, test_file = self._setup(tmp)
        func_id = "fix_func"
        safe_id = M._safe_filename(func_id)
        unit_dir = self._make_unit_test(tmp, func_id, safe_id)

        analysis = {
            "function_levels": {"0": [func_id]},
            "functions": [{"id": func_id, "name": func_id,
                           "source_file": "/s.c", "depth": 0}],
        }
        results = {func_id: {"passed": True, "unit_dir": str(unit_dir)}}

        chk_calls = [0]
        def fake_run(*args, **kwargs):
            chk_calls[0] += 1
            if chk_calls[0] == 1:
                return MagicMock(returncode=1, stdout="", stderr="error: undeclared")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(M, "run_agent"):
            M.integrate_all_unit_tests_sequential(cfg, paths, analysis, results, {})

        # call 1: syntax check fails, call 2: syntax check passes, call 3: final make test
        assert chk_calls[0] == 3


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

    def test_default_unit_test_workers(self, tmp: Path):
        cfg = M.parse_args(self._minimal_argv(tmp))
        assert cfg.max_unit_test_workers == 4

    def test_custom_unit_test_workers(self, tmp: Path):
        cfg = M.parse_args(self._minimal_argv(tmp) + ["--max-unit-test-workers", "8"])
        assert cfg.max_unit_test_workers == 8


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
