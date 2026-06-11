"""Integration test for the data collection pipeline.

REAL CODE: all pipeline/ modules -- generate_stub_code, _validate_stub_locally,
  run_agent, _collect_one_stub, _harvest, _prepare_episode_testdir, etc.

MOCK BOUNDARY (subprocess wall only):
  - run_command              -- all 3 module aliases
  - container.create/teardown -- no real docker
  - ensure_test_file, build_annotated_makefile (scaffold writers)
  - run_or_load_analysis     -- returns FAKE_ANALYSIS
  - stages not under test (integrate, minimal-master)
  - _load_semantic_context

CONTAINER MODE: container.create/teardown are no-ops (no docker). The agent
  prompt is containerized (scratch H -> canonical) by run_agent, so the
  run_command mock reverse-maps the OUTPUT FILE path (canonical -> H) to emulate
  the bind mount: an in-container write at the canonical path lands in scratch H
  on the host. ONE stub/unit per episode via generate_stub_code /
  _generate_unit_test_for_func -- no stage dispatch, no _inner_trace_dataset.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Inject stub.stub BEFORE any pipeline import
# ---------------------------------------------------------------------------

def _install_stub_modules() -> None:
    if "stub" not in sys.modules:
        stub_pkg = types.ModuleType("stub")
        stub_mod = types.ModuleType("stub.stub")
        stub_mod.ProjectAnalyzer = MagicMock()
        sys.modules["stub"] = stub_pkg
        sys.modules["stub.stub"] = stub_mod

_install_stub_modules()

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wj(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def print_tree(directory: Path, prefix: str = "") -> None:
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
    except (PermissionError, FileNotFoundError):
        return
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            print(f"{prefix}{connector}{entry.name}/")
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension)
        else:
            print(f"{prefix}{connector}{entry.name}")


# ---------------------------------------------------------------------------
# Fake analysis
# ---------------------------------------------------------------------------

STUB_CANDIDATES = ["FuncA", "FuncB", "FuncC"]

FUNCTIONS = [
    {"id": "do_thing",  "depth": 0, "source_file": "main.c", "start_line": 1, "end_line": 3},
    {"id": "helper_x", "depth": 1, "source_file": "main.c", "start_line": 4, "end_line": 6},
    {"id": "util_y",   "depth": 2, "source_file": "main.c", "start_line": 7, "end_line": 9},
]

FAKE_ANALYSIS = {
    "stub_candidates": {"group": STUB_CANDIDATES},
    "function_levels": {"0": ["do_thing"], "1": ["helper_x"], "2": ["util_y"]},
    "functions": FUNCTIONS,
}

# FuncA/FuncC validate; FuncB agent writes bad stub (no __wrap_)
STUB_VALID = {"FuncA": True, "FuncB": False, "FuncC": True}


# ---------------------------------------------------------------------------
# run_command mock -- the subprocess boundary
# ---------------------------------------------------------------------------

_FUNC_NAME_RE = re.compile(
    r"generating ONE C linker-wrapper stub.*?TARGET FUNCTION\s+(\S+)",
    re.DOTALL,
)
_OUTPUT_FILE_RE = re.compile(r"OUTPUT FILE\s*\n\s*(\S+)")
_FIX_STUB_RE = re.compile(r"Fix the __wrap_(\S+?) stub")


def _reverse_map(cfg, path_str: str) -> str:
    """Emulate the per-episode bind mount: rewrite canonical -> scratch host dir.

    In docker mode run_agent containerizes prompt paths (H -> canonical); a real
    container write at the canonical path lands in H via the mount. With no real
    docker we reproduce that here by mapping canonical back to H.
    """
    for host_prefix, canon_prefix in (getattr(cfg, "path_map", ()) or ()):
        if canon_prefix and path_str.startswith(canon_prefix):
            return host_prefix + path_str[len(canon_prefix):]
    return path_str


def _make_run_command_mock(stub_valid_map: dict):
    def _run_command(cfg, cmd, *, cwd, timeout, env=None, shell=False):
        if isinstance(cmd, list):
            cmd_str = " ".join(str(c) for c in cmd)
        else:
            cmd_str = str(cmd)

        if "node" in cmd_str:
            prompt_file = None
            if isinstance(cmd, list):
                for i, part in enumerate(cmd):
                    if str(part) == "--prompt-file" and i + 1 < len(cmd):
                        prompt_file = Path(cmd[i + 1])
                        break
            if prompt_file and prompt_file.exists():
                prompt_text = prompt_file.read_text(errors="replace")
                m_stub = _FUNC_NAME_RE.search(prompt_text)
                m_fix = _FIX_STUB_RE.search(prompt_text)
                func_name = None
                if m_stub:
                    func_name = m_stub.group(1).strip()
                elif m_fix:
                    func_name = m_fix.group(1).strip()
                m_out = _OUTPUT_FILE_RE.search(prompt_text)
                if m_out:
                    out_path = Path(_reverse_map(cfg, m_out.group(1).strip()))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if func_name:
                        if stub_valid_map.get(func_name, True):
                            body = (
                                "#include <stdio.h>\n"
                                f"void __wrap_{func_name}(void) {{\n"
                                f'    fprintf(stderr, "__wrap_{func_name} called\\n");\n'
                                "}\n"
                            )
                        else:
                            body = (
                                "#include <stdio.h>\n"
                                f"/* invalid -- no __wrap_ for {func_name} */\n"
                                f"void bad_stub_{func_name}(void) {{}}\n"
                            )
                        out_path.write_text(body, encoding="utf-8")
                    else:
                        out_path.write_text("/* agent output */\n", encoding="utf-8")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        if "stub_validate" in cmd_str and "gcc" not in cmd_str:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return _run_command


# ---------------------------------------------------------------------------
# Scaffold mocks
# ---------------------------------------------------------------------------

_TEST_FILE_MARKERS = [
    "/* === Includes === */",
    "/* === Compatibility Definitions === */",
    "/* === Test Globals === */",
    "/* === Test Helpers === */",
    "/* === Linker Wrapper Stubs === */",
    "/* === Test Cases === */",
    "/* === Test Registration === */",
]


def _mock_ensure_test_file(cfg, paths):
    tf = Path(paths["test_file"])
    tf.parent.mkdir(parents=True, exist_ok=True)
    if not tf.exists():
        tf.write_text(
            "/* minimal test file */\n"
            + "\n".join(_TEST_FILE_MARKERS)
            + "\n#include <CUnit/CUnit.h>\n"
            "#include <CUnit/Basic.h>\n"
            "int main(void) { return 0; }\n",
            encoding="utf-8",
        )


def _mock_build_annotated_makefile(cfg, paths):
    mf = Path(paths["makefile"])
    mf.parent.mkdir(parents=True, exist_ok=True)
    if not mf.exists():
        mf.write_text("# minimal test makefile\ntest:\n\t@echo ok\n", encoding="utf-8")
    ctx = Path(paths["test_dir"]) / "_pipeline_context.json"
    if not ctx.exists():
        _wj(ctx, {
            "flags": {},
            "process_name": paths["process_name"],
            "source_dir": str(cfg.source_dir),
            "actual_source_files": [],
        })
    return {}


def _fake_integrate_all_unit_tests_sequential(cfg, paths, analysis, unit_test_results, flags):
    test_dir = Path(paths["test_dir"]).resolve()
    assert (test_dir / "_stub_gen").is_dir()
    assert (test_dir / "_unit_tests").is_dir()

    for result in unit_test_results.values():
        unit_dir = result.get("unit_dir")
        if unit_dir:
            assert str(unit_dir).startswith(str(test_dir)), f"unit_dir not rebased into workspace: {unit_dir}"

    test_file = Path(paths["test_file"])
    marker = "/* final integration marker */\n"
    current = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
    if "final integration marker" not in current:
        if current and not current.endswith("\n"):
            current += "\n"
        test_file.write_text(current + marker, encoding="utf-8")

    makefile = Path(paths["makefile"])
    mk_marker = "# final integration marker\n"
    mk_current = makefile.read_text(encoding="utf-8") if makefile.exists() else ""
    if "final integration marker" not in mk_current:
        if mk_current and not mk_current.endswith("\n"):
            mk_current += "\n"
        makefile.write_text(mk_current + mk_marker, encoding="utf-8")

    return True


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _base_patches():
    """Common patches for both local and container modes."""
    run_cmd = _make_run_command_mock(STUB_VALID)
    return [
        patch("pipeline.data_collection.ensure_test_file", side_effect=_mock_ensure_test_file),
        patch("pipeline.data_collection.build_annotated_makefile", side_effect=_mock_build_annotated_makefile),
        patch("pipeline.data_collection.run_or_load_analysis", return_value=FAKE_ANALYSIS),
        patch("pipeline.execution.run_command", side_effect=run_cmd),
        patch("pipeline.common.run_command", side_effect=run_cmd),
        patch("pipeline.stage3_stubs.run_command", side_effect=run_cmd),
        patch("pipeline.container.create", return_value=None),
        patch("pipeline.container.teardown", return_value=None),
        patch("pipeline.data_collection.ensure_minimal_test_runs", return_value=True),
        patch("pipeline.data_collection.integrate_all_stubs_sequential", return_value=None),
        patch(
            "pipeline.data_collection.integrate_all_unit_tests_sequential",
            side_effect=_fake_integrate_all_unit_tests_sequential,
        ),
        patch("pipeline.data_collection._load_semantic_context", return_value={"functions": {}}),
        patch("pipeline.stage5_unit_tests._load_semantic_context", return_value={"functions": {}}),
    ]


def _full_flow_patches():
    """Extra patches for full flow (unit test stages)."""
    fake_make = {"ok": False, "returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "errors": []}
    return _base_patches() + [
        patch("pipeline.common.run_make_test", return_value=fake_make),
        patch("pipeline.stage5_unit_tests.run_make_test", return_value=fake_make),
        patch("pipeline.stage5_unit_tests.check_function_coverage", return_value=None),
        patch("pipeline.stage5_unit_tests.run_semantic_test_judge",
              return_value={"score": 0, "passed": False, "reason": "mocked"}),
        patch("pipeline.stage5_unit_tests.backup_good_cunit_if_best", return_value=None),
        patch("pipeline.stage5_unit_tests._append_semantic_context", return_value=None),
    ]


def _container_patches(outer_paths):
    """Container mode: subprocess + container lifecycle mocks only.

    Real generate_stub_code / _generate_unit_test_for_func run for ONE item per
    episode, writing into the scratch test dir. No inner-stage dispatch.
    """
    return _base_patches()


class _ApplyPatches:
    def __init__(self, ctx_managers):
        self._cms = ctx_managers
        self._stack = ExitStack()

    def __enter__(self):
        for cm in self._cms:
            self._stack.enter_context(cm)
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source_dir(tmp_path):
    src_dir = tmp_path / "project" / "src" / "myproc"
    src_dir.mkdir(parents=True)
    (src_dir / "main.c").write_text(
        "int FuncA(void) { return 0; }\n"
        "int FuncB(void) { return 0; }\n"
        "int FuncC(void) { return 0; }\n"
        "int do_thing(void) { return 0; }\n"
        "int helper_x(void) { return 0; }\n"
        "int util_y(void) { return 0; }\n",
        encoding="utf-8",
    )
    (src_dir / "Makefile").write_text(
        "CFLAGS = -Wall\ntest:\n\t@echo ok\n", encoding="utf-8"
    )
    return src_dir


def _make_cfg(tmp_path, src_dir, *, execution_mode="local", per_episode_container=False,
              container_image=None):
    from pipeline.config import PipelineConfig, derive_paths

    test_dir = tmp_path / "project" / "tests" / "myproc"
    test_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent.js").write_text("// fake\n", encoding="utf-8")
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        source_dir=src_dir,
        agent_js=tmp_path / "agent.js",
        execution_mode=execution_mode,
        per_episode_container=per_episode_container,
        container_image=container_image,
        episodes_per_item=2,
        trace_dataset_dirname="_trace_dataset",
        func_docs_dir=tmp_path / "docs",
        max_stub_gen_retries=1,
    )
    paths = derive_paths(cfg)
    return cfg, paths, test_dir


@pytest.fixture()
def local_env(tmp_path):
    src_dir = _make_source_dir(tmp_path)
    cfg, paths, test_dir = _make_cfg(tmp_path, src_dir)
    return cfg, paths, test_dir


@pytest.fixture()
def container_env(tmp_path):
    src_dir = _make_source_dir(tmp_path)
    cfg, paths, test_dir = _make_cfg(
        tmp_path, src_dir,
        execution_mode="docker",
        per_episode_container=True,
        container_image="fake-image:latest",
    )
    return cfg, paths, test_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_stages(cfg, paths, stages, patches):
    from dataclasses import replace
    from pipeline.data_collection import run_selected_stage

    with _ApplyPatches(patches):
        for stage in stages:
            run_selected_stage(replace(cfg, stage=stage), paths)


STUB_STAGES = ["prepare", "collect-stubs", "select-stubs", "materialize-stubs"]
ALL_STAGES = STUB_STAGES + ["collect-unit-tests", "select-unit-tests", "materialize-unit-tests"]


# ===========================================================================
# Tests
# ===========================================================================

class TestLocalStubs:
    """Local mode stub pipeline: real generate_stub_code + validate."""

    def test_local_stubs(self, local_env):
        cfg, paths, test_dir = local_env
        _run_stages(cfg, paths, STUB_STAGES, _base_patches())

        eps_root = test_dir / "_trace_dataset" / "episodes" / "stubs"
        frontiers = test_dir / "_trace_dataset" / "frontiers" / "stubs"
        stub_gen = test_dir / "_stub_gen"

        # 6 episodes: 2 per candidate x 3 candidates
        episode_count = sum(
            len([p for p in sd.iterdir() if p.is_dir()])
            for sd in eps_root.iterdir() if sd.is_dir()
        )
        assert episode_count == 6, f"expected 6 episodes, got {episode_count}"

        # Every episode: metadata.json + workspace/
        validated_count = 0
        for safe_dir in sorted(p for p in eps_root.iterdir() if p.is_dir()):
            for ep in sorted(p for p in safe_dir.iterdir() if p.is_dir()):
                assert (ep / "metadata.json").exists()
                assert (ep / "workspace").is_dir()
                meta = json.loads((ep / "metadata.json").read_text())
                assert meta["func_name"] == safe_dir.name
                # No testdir or nested datasets in local mode
                assert not (ep / "testdir").exists()
                ws = ep / "workspace"
                if meta.get("validated"):
                    validated_count += 1
                    assert (ws / "stub.c").exists()
                    assert (ws / "result.json").exists()

        assert validated_count == 4  # FuncA x2 + FuncC x2

        # No nested datasets
        assert not list(test_dir.rglob("_inner_trace_dataset"))
        for p in (test_dir / "_trace_dataset").rglob("*"):
            if p.is_dir() and p.name == "_trace_dataset":
                pytest.fail(f"nested _trace_dataset at {p}")

        # Frontiers: FuncA + FuncC, not FuncB
        present = {p.name for p in frontiers.iterdir() if p.is_dir()}
        assert present == {"FuncA", "FuncC"}
        for name in ("FuncA", "FuncC"):
            body = (frontiers / name / "stub.c").read_text()
            assert f"__wrap_{name}" in body
            assert f"__real_{name}" not in body

        # _stub_gen materialized
        assert (stub_gen / "FuncA" / "stub.c").exists()
        assert (stub_gen / "FuncC" / "stub.c").exists()
        assert not (stub_gen / "FuncB" / "stub.c").exists()

        # Manifest
        manifest = json.loads((test_dir / "_trace_dataset" / "manifest.json").read_text())
        assert isinstance(manifest.get("runs"), list)

        print()
        print("=" * 70)
        print(f"[local stubs] tests/{paths['process_name']}/")
        print("=" * 70)
        print_tree(test_dir)


class TestContainerStubs:
    """Container mode: ONE stub per episode via generate_stub_code in scratch dir.

    Reproduces the FIXED structure (the bug was testdir/_inner_trace_dataset
    recursion). Asserts the invariants from file_system_change.md.
    """

    def test_container_stubs(self, container_env):
        cfg, paths, test_dir = container_env
        _run_stages(cfg, paths, STUB_STAGES, _container_patches(paths))

        ds = test_dir / "_trace_dataset"
        eps_root = ds / "episodes" / "stubs"
        frontiers = ds / "frontiers" / "stubs"
        stub_gen = test_dir / "_stub_gen"

        # 6 episodes: 2 per candidate x 3 candidates
        episode_count = sum(
            len([p for p in sd.iterdir() if p.is_dir()])
            for sd in eps_root.iterdir() if sd.is_dir()
        )
        assert episode_count == 6, f"expected 6 episodes, got {episode_count}"

        # Invariant #1: episode dir = metadata.json + workspace/ + agent_history/.
        # #2/#3: NO testdir/, NO _inner_trace_dataset, NO nested _trace_dataset.
        validated_count = 0
        for safe_dir in sorted(p for p in eps_root.iterdir() if p.is_dir()):
            for ep in sorted(p for p in safe_dir.iterdir() if p.is_dir()):
                assert (ep / "metadata.json").exists()
                assert (ep / "workspace").is_dir()
                meta = json.loads((ep / "metadata.json").read_text())
                assert meta["func_name"] == safe_dir.name

                assert not (ep / "testdir").exists(), f"testdir/ leaked into episode {ep}"
                assert not list(ep.rglob("_inner_trace_dataset")), f"inner dataset in {ep}"
                assert not list(ep.rglob("_trace_dataset")), f"nested _trace_dataset in {ep}"

                allowed = {"metadata.json", "workspace", "agent_history"}
                extra = {c.name for c in ep.iterdir()} - allowed
                assert not extra, f"unexpected entries in episode {ep}: {extra}"

                ws = ep / "workspace"
                if meta.get("validated"):
                    validated_count += 1
                    assert (ws / "stub.c").exists()
                    assert (ws / "result.json").exists()

        assert validated_count == 4  # FuncA x2 + FuncC x2

        # Invariant: scratch is visible at _trace_dataset/scratch/<eid>, sibling
        # to episodes/, and is a pristine canonical test dir (NOT a dataset).
        scratch = ds / "scratch"
        assert scratch.is_dir(), "scratch dir missing"
        scratch_dirs = [p for p in scratch.iterdir() if p.is_dir()]
        assert len(scratch_dirs) == 6, f"expected 6 scratch dirs, got {len(scratch_dirs)}"
        for sc in scratch_dirs:
            assert (sc / f"test_{paths['process_name']}.c").exists()
            assert not (sc / "_inner_trace_dataset").exists()
            assert not (sc / "_trace_dataset").exists()

        # Global negatives: no _inner_trace_dataset / testdir anywhere under the tree
        assert not list(test_dir.rglob("_inner_trace_dataset"))
        assert not list(eps_root.rglob("testdir"))

        # Frontiers + materialize identical to local mode
        present = {p.name for p in frontiers.iterdir() if p.is_dir()}
        assert present == {"FuncA", "FuncC"}
        assert (stub_gen / "FuncA" / "stub.c").exists()
        assert (stub_gen / "FuncC" / "stub.c").exists()
        assert not (stub_gen / "FuncB" / "stub.c").exists()

        print()
        print("=" * 70)
        print(f"[container stubs] tests/{paths['process_name']}/")
        print("=" * 70)
        print_tree(test_dir)


class TestFullLocalFlow:
    """Full local pipeline: stubs + unit tests."""

    def test_full_flow(self, local_env):
        cfg, paths, test_dir = local_env
        _run_stages(cfg, paths, ALL_STAGES, _full_flow_patches())

        stub_gen = test_dir / "_stub_gen"
        assert (stub_gen / "FuncA" / "stub.c").exists()
        assert (stub_gen / "FuncC" / "stub.c").exists()
        assert not (stub_gen / "FuncB" / "stub.c").exists()

        eps_unit = test_dir / "_trace_dataset" / "episodes" / "unit_tests"
        assert eps_unit.exists()
        ut_count = sum(
            len([p for p in sd.iterdir() if p.is_dir()])
            for sd in eps_unit.iterdir() if sd.is_dir()
        )
        assert ut_count == 6  # 3 funcs x 2 episodes

        # No nested datasets in local mode
        assert not list(test_dir.rglob("_inner_trace_dataset"))

        print()
        print("=" * 70)
        print(f"[full local flow] tests/{paths['process_name']}/")
        print("=" * 70)
        print_tree(test_dir)


class TestAllCollectContainer:
    """--stage all-collect chains every collection stage in ONE run (container mode).

    Verifies the fix holds end-to-end, not just per individual stage.
    """

    def test_all_collect(self, container_env):
        from dataclasses import replace
        from pipeline.data_collection import run_selected_stage

        cfg, paths, test_dir = container_env
        with _ApplyPatches(_full_flow_patches()):
            run_selected_stage(replace(cfg, stage="all-collect"), paths)

        ds = test_dir / "_trace_dataset"
        stub_eps = ds / "episodes" / "stubs"
        unit_eps = ds / "episodes" / "unit_tests"
        assert stub_eps.is_dir() and unit_eps.is_dir()

        stub_count = sum(len([p for p in sd.iterdir() if p.is_dir()])
                         for sd in stub_eps.iterdir() if sd.is_dir())
        unit_count = sum(len([p for p in sd.iterdir() if p.is_dir()])
                         for sd in unit_eps.iterdir() if sd.is_dir())
        assert stub_count == 6, f"expected 6 stub episodes, got {stub_count}"
        assert unit_count == 6, f"expected 6 unit episodes, got {unit_count}"

        # Invariants hold across the whole tree, end-to-end
        assert not list(test_dir.rglob("_inner_trace_dataset"))
        assert not list((ds / "episodes").rglob("testdir"))

        # Episode dirs stay clean (no scratch leak)
        for kind in (stub_eps, unit_eps):
            for sd in kind.iterdir():
                for ep in sd.iterdir():
                    assert not (ep / "testdir").exists()

        scratch = ds / "scratch"
        assert scratch.is_dir()
        scratch_dirs = [p for p in scratch.iterdir() if p.is_dir()]
        assert len(scratch_dirs) == 6, f"expected 6 stub scratch dirs, got {len(scratch_dirs)}"

        unit_workspaces = ds / "unit_test_episodes"
        assert unit_workspaces.is_dir()
        unit_workspace_count = sum(
            len([p for p in func_dir.iterdir() if p.is_dir()])
            for func_dir in unit_workspaces.iterdir() if func_dir.is_dir()
        )
        assert unit_workspace_count == 6, f"expected 6 unit-test workspaces, got {unit_workspace_count}"
        for func_dir in sorted(p for p in unit_workspaces.iterdir() if p.is_dir()):
            for ep in sorted(p for p in func_dir.iterdir() if p.is_dir()):
                assert (ep / f"test_{paths['process_name']}.c").exists()
                assert (ep / "Makefile").exists()
                assert (ep / "_pipeline_context.json").exists()
                assert (ep / "_stub_gen").is_dir()
                assert (ep / "_unit_tests" / func_dir.name).is_dir()

        integrate_eps = ds / "integrate_episodes"
        assert integrate_eps.is_dir()
        assert len([p for p in integrate_eps.iterdir() if p.is_dir()]) == 2

        minimal_eps = ds / "minimal_episodes"
        assert minimal_eps.is_dir()
        assert len([p for p in minimal_eps.iterdir() if p.is_dir()]) == 2

        final_integration_eps = ds / "final_integration_episodes"
        assert final_integration_eps.is_dir()
        final_eps = [p for p in final_integration_eps.iterdir() if p.is_dir()]
        assert len(final_eps) == 2, f"expected 2 final integration workspaces, got {len(final_eps)}"
        for ep in final_eps:
            assert (ep / f"test_{paths['process_name']}.c").exists()
            assert (ep / "Makefile").exists()
            assert (ep / "_pipeline_context.json").exists()
            assert (ep / "_stub_gen").is_dir()
            assert (ep / "_unit_tests").is_dir()

        assert (test_dir / "_stub_gen" / "FuncA" / "stub.c").exists()
        assert (test_dir / "_stub_gen" / "FuncC" / "stub.c").exists()
        assert "final integration marker" in Path(paths["test_file"]).read_text()
        assert "final integration marker" in Path(paths["makefile"]).read_text()

        print()
        print("=" * 70)
        print(f"[all-collect container] tests/{paths['process_name']}/")
        print("=" * 70)
        print_tree(test_dir)
