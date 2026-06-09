from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("stub", types.ModuleType("stub"))
sys.modules.setdefault("stub.stub", types.ModuleType("stub.stub"))
sys.modules["stub.stub"].ProjectAnalyzer = MagicMock()  # type: ignore[attr-defined]

import main2 as M
from pipeline import data_collection as DC
from pipeline.common import write_json
from pipeline.execution import run_command


def make_source(tmp_path: Path) -> Path:
    src = tmp_path / "project" / "src" / "proc"
    src.mkdir(parents=True)
    return src


def make_cfg(tmp_path: Path, **kwargs) -> M.PipelineConfig:
    defaults = dict(
        source_dir=make_source(tmp_path),
        agent_js=Path("/home/seigyo/rl/agent.js"),
        system_json=Path("/home/seigyo/rl/system_functions.json"),
    )
    defaults.update(kwargs)
    return M.PipelineConfig(**defaults)


def test_parse_args_combined_flags(tmp_path: Path):
    src = make_source(tmp_path)
    with patch.object(sys, "argv", [
        "main2.py",
        str(src),
        "--execution-mode",
        "docker",
        "--container-name",
        "attempt_1",
        "--container-profile",
        "/profile",
        "--forbid-host-prefix",
        "/host",
        "--stage",
        "collect-unit-tests",
        "--episodes-per-item",
        "3",
    ]):
        cfg = M.parse_args()

    assert cfg.execution_mode == "docker"
    assert cfg.container_name == "attempt_1"
    assert cfg.container_profile == Path("/profile")
    assert cfg.forbidden_host_prefixes == ("/host",)
    assert cfg.stage == "collect-unit-tests"
    assert cfg.episodes_per_item == 3


def test_run_command_local_uses_subprocess_cwd(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    completed = subprocess.CompletedProcess(["echo", "ok"], 0, "ok\n", "")
    with patch("pipeline.execution.subprocess.run", return_value=completed) as run:
        result = run_command(cfg, ["echo", "ok"], cwd=tmp_path, timeout=5)

    assert result is completed
    assert run.call_args.kwargs["cwd"] == str(tmp_path)
    assert run.call_args.kwargs["capture_output"] is True
    assert run.call_args.kwargs["text"] is True


def test_run_command_docker_builds_exec_command(tmp_path: Path):
    cfg = make_cfg(
        tmp_path,
        execution_mode="docker",
        container_name="attempt_1",
        container_profile=Path("/home/seigyo/.bash_profile"),
    )
    completed = subprocess.CompletedProcess(["docker"], 0, "ok\n", "")
    with patch("pipeline.execution.subprocess.run", return_value=completed) as run:
        result = run_command(
            cfg,
            ["make", "test"],
            cwd=Path("/home/seigyo/proj/tests/proc"),
            timeout=30,
            env={"MAX_ITERATIONS": "2"},
        )

    assert result is completed
    cmd = run.call_args.args[0]
    assert cmd[:2] == ["docker", "exec"]
    assert "-e" in cmd
    assert "MAX_ITERATIONS=2" in cmd
    assert "attempt_1" in cmd
    assert "source /home/seigyo/.bash_profile" in cmd[-1]
    assert "cd /home/seigyo/proj/tests/proc" in cmd[-1]
    assert "exec make test" in cmd[-1]


def test_run_command_docker_requires_container(tmp_path: Path):
    cfg = make_cfg(tmp_path, execution_mode="docker", container_name=None)
    with pytest.raises(ValueError, match="container-name"):
        run_command(cfg, ["make", "test"], cwd=tmp_path, timeout=5)


def test_main2_default_path_preserves_stage_order(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    calls: list[str] = []

    def record(name, value=None):
        def _inner(*_args, **_kwargs):
            calls.append(name)
            return value
        return _inner

    with patch.object(M, "ensure_test_file", side_effect=record("ensure_test_file")), \
        patch.object(M, "build_annotated_makefile", side_effect=record("build_annotated_makefile", {})), \
        patch.object(M, "run_or_load_analysis", side_effect=record("run_or_load_analysis", {"functions": []})), \
        patch.object(M, "handle_stubs", side_effect=record("handle_stubs")), \
        patch.object(M, "ensure_minimal_test_runs", side_effect=record("ensure_minimal_test_runs", True)), \
        patch.object(M, "parallel_generate_unit_tests", side_effect=record("parallel_generate_unit_tests", {})), \
        patch.object(M, "integrate_all_unit_tests_sequential", side_effect=record("integrate_all_unit_tests_sequential", True)):
        M.run(cfg)

    assert calls == [
        "ensure_test_file",
        "build_annotated_makefile",
        "run_or_load_analysis",
        "handle_stubs",
        "ensure_minimal_test_runs",
        "parallel_generate_unit_tests",
        "integrate_all_unit_tests_sequential",
    ]


def test_main2_stage_dispatch_skips_default_path(tmp_path: Path):
    cfg = make_cfg(tmp_path, stage="prepare")
    with patch("pipeline.data_collection.run_selected_stage") as selected, \
        patch.object(M, "ensure_test_file") as normal_stage:
        M.run(cfg)

    selected.assert_called_once()
    normal_stage.assert_not_called()


def test_collect_stubs_writes_episode_not_active_stub_dir(tmp_path: Path):
    cfg = make_cfg(tmp_path, stage="collect-stubs")
    paths = M.derive_paths(cfg)
    analysis = {"stub_candidates": {"a.c": ["foo"]}}

    def fake_generate(_cfg, _test_dir, func_name, *, stub_dir_override, history_dir_override):
        assert func_name == "foo"
        stub_dir_override.mkdir(parents=True)
        history_dir_override.mkdir(parents=True)
        (stub_dir_override / "stub.c").write_text("void __wrap_foo(void) {}\n", encoding="utf-8")
        write_json(stub_dir_override / "result.json", {"validated": True, "func_name": "foo"})
        return "void __wrap_foo(void) {}"

    with patch.object(DC, "_prepare", return_value=({}, analysis)), \
        patch.object(DC, "generate_stub_code", side_effect=fake_generate):
        DC.run_selected_stage(cfg, paths)

    episodes = list((Path(paths["test_dir"]) / "_trace_dataset" / "episodes" / "stubs" / "foo").iterdir())
    assert len(episodes) == 1
    assert (episodes[0] / "workspace" / "stub.c").exists()
    assert not (Path(paths["test_dir"]) / "_stub_gen" / "foo").exists()


def test_select_and_materialize_stub(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    paths = M.derive_paths(cfg)
    episode = Path(paths["test_dir"]) / "_trace_dataset" / "episodes" / "stubs" / "foo" / "ep1"
    workspace = episode / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "stub.c").write_text("void __wrap_foo(void) {}\n", encoding="utf-8")
    write_json(workspace / "result.json", {"validated": True, "func_name": "foo"})
    write_json(episode / "metadata.json", {"func_name": "foo", "episode_id": "ep1", "passed": True})

    DC.run_selected_stage(M.PipelineConfig(**{**cfg.__dict__, "stage": "select-stubs"}), paths)
    DC.run_selected_stage(M.PipelineConfig(**{**cfg.__dict__, "stage": "materialize-stubs"}), paths)

    active = Path(paths["test_dir"]) / "_stub_gen" / "foo"
    assert (active / "stub.c").read_text(encoding="utf-8") == "void __wrap_foo(void) {}\n"
    assert (active / "result.json").exists()


def test_materialize_unit_tests_sets_completed_only_for_full_frontier(tmp_path: Path):
    cfg = make_cfg(tmp_path, stage="materialize-unit-tests")
    paths = M.derive_paths(cfg)
    frontier = Path(paths["test_dir"]) / "_trace_dataset" / "frontiers" / "unit_tests" / "leaf"
    frontier.mkdir(parents=True)
    (frontier / "test_leaf.c").write_text("void test_leaf(void) {}\n", encoding="utf-8")
    (frontier / "Makefile").write_text("TEST_PROGRAM = test_leaf\n", encoding="utf-8")
    write_json(frontier / "coverage.json", {"summary": {"coverage_percent": 90.0}})
    write_json(frontier / "judge_verdict.json", {"passed": True, "score": 90})
    write_json(frontier / "selected.json", {
        "func_id": "leaf",
        "safe_id": "leaf",
        "passed": True,
        "coverage_pct": 90.0,
        "semantic_score": 90,
        "result": {"passed": True, "coverage_pct": 90.0, "semantic_score": 90},
    })
    analysis = {
        "function_levels": {"1": ["leaf", "other"]},
        "functions": [
            {"id": "leaf", "depth": 1},
            {"id": "other", "depth": 1},
        ],
    }

    with patch.object(DC, "_prepare", return_value=({}, analysis)):
        DC.run_selected_stage(cfg, paths)

    ctx = M.load_json(Path(paths["test_dir"]) / "_pipeline_context.json")
    assert "leaf" in ctx["unit_test_results"]
    assert "unit_tests_completed" not in ctx

    cfg_one = M.PipelineConfig(**{**cfg.__dict__, "only_function": "leaf"})
    with patch.object(DC, "_prepare", return_value=({}, analysis)):
        DC.run_selected_stage(cfg_one, paths)

    ctx = M.load_json(Path(paths["test_dir"]) / "_pipeline_context.json")
    assert ctx["unit_tests_completed"] is True
