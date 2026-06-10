from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Optional

from . import container as container_mod
from .analysis import collect_stub_candidates, functions_leaf_first, run_or_load_analysis
from .common import _safe_filename, load_json, read_text, write_json
from .config import PipelineConfig
from .execution import run_command_to_files
from .semantic import _load_semantic_context
from .stage1_scaffold import ensure_test_file
from .stage2_makefile import build_annotated_makefile
from .stage3_stubs import generate_stub_code, integrate_all_stubs_sequential
from .stage4_minimal import ensure_minimal_test_runs
from .stage5_unit_tests import _generate_unit_test_for_func
from .stage6_integrate import integrate_all_unit_tests_sequential


def dataset_root(cfg: PipelineConfig, paths: dict) -> Path:
    return Path(paths["test_dir"]) / cfg.trace_dataset_dirname


def _episode_id() -> str:
    return f"{int(time.time() * 1000)}_{time.monotonic_ns()}"


def _prepare(cfg: PipelineConfig, paths: dict) -> tuple[dict, dict]:
    ensure_test_file(cfg, paths)
    flags = build_annotated_makefile(cfg, paths)
    analysis = run_or_load_analysis(cfg, paths["analysis_path"])
    return flags, analysis


def _write_stage_event(cfg: PipelineConfig, paths: dict, stage: str) -> None:
    root = dataset_root(cfg, paths)
    manifest = root / "manifest.json"
    try:
        data = load_json(manifest) if manifest.exists() else {}
    except Exception:
        data = {}
    runs = data.get("runs")
    if not isinstance(runs, list):
        runs = []
    runs.append({
        "stage": stage,
        "time": int(time.time()),
        "source_dir": str(cfg.source_dir),
        "test_dir": str(paths["test_dir"]),
        "execution_mode": cfg.execution_mode,
        "container_name": cfg.container_name,
    })
    data["runs"] = runs
    write_json(manifest, data)


def _metadata_base(cfg: PipelineConfig, paths: dict, stage: str, episode_id: str) -> dict:
    return {
        "stage": stage,
        "episode_id": episode_id,
        "process_name": paths["process_name"],
        "source_dir": str(cfg.source_dir),
        "test_dir": str(paths["test_dir"]),
        "execution_mode": cfg.execution_mode,
        "container_name": cfg.container_name,
    }


def _target_functions(cfg: PipelineConfig, analysis: dict) -> list[dict]:
    funcs = []
    for func in functions_leaf_first(analysis):
        if cfg.only_function and func.get("id") != cfg.only_function:
            continue
        if cfg.only_level is not None and func.get("depth") != cfg.only_level:
            continue
        funcs.append(func)
        if cfg.max_functions is not None and len(funcs) >= cfg.max_functions:
            break
    return funcs


def _reset_dir(p: Path) -> None:
    """Remove an active item dir so the next episode starts pristine."""
    shutil.rmtree(p, ignore_errors=True)


def _harvest(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    """Copy each named file/dir from src_dir into dst_dir if it exists."""
    for name in names:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _harvest_glob(src_dir: Path, dst_dir: Path, pattern: str) -> None:
    """Copy files matching glob (shared agent_history entries) into dst_dir."""
    if not src_dir.exists():
        return
    matches = list(src_dir.glob(pattern))
    if not matches:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in matches:
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)


def _prepare_episode_testdir(
    base: Path,
    dest: Path,
    *,
    include_stub_gen: bool,
    seed_from_base: bool,
) -> None:
    """Copy the prepared canonical test dir into a per-episode host dir.

    Excludes the dataset archive and the active per-function dirs so the agent
    sees a pristine layout. Stub episodes start without _stub_gen; unit episodes
    keep the selected/materialized stubs they depend on.
    """
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not seed_from_base:
        dest.mkdir(parents=True, exist_ok=True)
        return

    ignore_names = {"_trace_dataset", "_unit_tests"}
    if not include_stub_gen:
        ignore_names.add("_stub_gen")
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns(*ignore_names))


def _episode_paths(paths: dict, host_test_dir: Path) -> dict:
    episode_paths = dict(paths)
    episode_paths["test_dir"] = host_test_dir
    episode_paths["test_file"] = host_test_dir / Path(paths["test_file"]).name
    episode_paths["makefile"] = host_test_dir / "Makefile"
    episode_paths["history_dir"] = host_test_dir / "agent_history"
    episode_paths["analysis_path"] = host_test_dir / "analysis.json"
    episode_paths["report_file"] = host_test_dir / Path(paths["report_file"]).name
    episode_paths["log_file"] = host_test_dir / Path(paths["log_file"]).name
    return episode_paths


def _inner_dataset_root(host_test_dir: Path) -> Path:
    return host_test_dir / "_inner_trace_dataset"


def _run_inner_stage(
    run_cfg: PipelineConfig,
    *,
    source_dir: Path,
    stage: str,
    log_dir: Path,
    extra_args: list[str],
) -> None:
    repo_main = Path(__file__).resolve().parent.parent / "main2.py"
    cmd = [
        "python3",
        "-u",
        str(repo_main),
        str(source_dir),
        "--stage",
        stage,
        "--execution-mode",
        "local",
        "--trace-dataset-dirname",
        "_inner_trace_dataset",
        *extra_args,
    ]
    print(
        f"[pipeline] episode inner stage -> {stage} cwd={source_dir.parent.parent.resolve()}",
        file=sys.stderr,
    )
    stdout_log = log_dir / "inner_stage.stdout.log"
    stderr_log = log_dir / "inner_stage.stderr.log"
    print(
        f"[pipeline] inner logs -> stdout={stdout_log} stderr={stderr_log}",
        file=sys.stderr,
    )
    res = run_command_to_files(
        run_cfg,
        cmd,
        cwd=source_dir.parent.parent.resolve(),
        stdout_path=stdout_log,
        stderr_path=stderr_log,
        env={
            "PYTHONIOENCODING": "utf-8",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        },
        timeout=max(3600, int(getattr(run_cfg, "agent_timeout_sec", 1800)) * 2),
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"inner stage failed: {stage}\n"
            f"cmd={' '.join(cmd)}\n"
            f"exit={res.returncode}\n"
            f"stdout_log={stdout_log}\n"
            f"stderr_log={stderr_log}\n"
        )


def _run_jobs(cfg: PipelineConfig, jobs: list) -> None:
    """Run episode jobs, in parallel only for per-episode-container mode."""
    workers = max(1, int(cfg.episode_concurrency)) if cfg.per_episode_container else 1
    if workers == 1:
        for job in jobs:
            job()
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(job) for job in jobs]
        for fut in as_completed(futures):
            fut.result()


def _episode_container(
    cfg: PipelineConfig,
    paths: dict,
    episode_root: Path,
    *,
    include_stub_gen: bool,
    seed_from_base: bool,
):
    """Set up one fresh container for an episode.

    Returns (run_cfg, host_test_dir, container_name). The host test dir is bound
    at the canonical path; run_cfg carries the docker target + the host->canonical
    path_map so the agent only sees canonical paths.
    """
    canonical = Path(paths["test_dir"])
    repo_root = cfg.source_dir.parent.parent.resolve()
    host_test_dir = episode_root / "testdir"
    print(
        f"[pipeline] starting episode container root={episode_root}",
        file=sys.stderr,
    )
    _prepare_episode_testdir(
        canonical,
        host_test_dir,
        include_stub_gen=include_stub_gen,
        seed_from_base=seed_from_base,
    )
    name = container_mod.episode_container_name()
    container_mod.create(
        cfg,
        name,
        host_test_dir=host_test_dir,
        canonical_test_dir=canonical,
        repo_root=repo_root,
    )
    print(
        f"[pipeline] episode container ready name={name} mount={host_test_dir} -> {canonical}",
        file=sys.stderr,
    )
    run_cfg = replace(
        cfg,
        execution_mode="docker",
        container_name=name,
        per_episode_container=False,
        path_map=((str(host_test_dir), str(canonical)),),
    )
    return run_cfg, host_test_dir, name


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _load_selected_json(frontier_dir: Path) -> dict:
    selected = frontier_dir / "selected.json"
    if not selected.exists():
        return {}
    try:
        data = load_json(selected)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _stage_prepare(cfg: PipelineConfig, paths: dict) -> None:
    _prepare(cfg, paths)


def _collect_one_stub(cfg: PipelineConfig, paths: dict, func_name: str, safe: str) -> None:
    test_dir = Path(paths["test_dir"])
    root = dataset_root(cfg, paths)
    eid = _episode_id()
    episode_root = root / "episodes" / "stubs" / safe / eid
    workspace = episode_root / "workspace"
    ep_history = episode_root / "agent_history"
    body: Optional[str]

    if cfg.per_episode_container:
        run_cfg, H, name = _episode_container(
            cfg,
            paths,
            episode_root,
            include_stub_gen=False,
            seed_from_base=False,
        )
        try:
            _run_inner_stage(
                run_cfg,
                source_dir=cfg.source_dir,
                stage="collect-stubs",
                log_dir=episode_root,
                extra_args=[
                    "--only-function",
                    func_name,
                    "--episodes-per-item",
                    "1",
                ],
            )
            inner_root = _inner_dataset_root(H) / "episodes" / "stubs" / safe
            inner_episodes = sorted(p for p in inner_root.iterdir() if p.is_dir()) if inner_root.exists() else []
            inner_episode = inner_episodes[-1] if inner_episodes else None
            if inner_episode is None:
                raise RuntimeError(f"inner collect-stubs produced no episode for {func_name}")
            workspace_src = inner_episode / "workspace"
            result_src = workspace_src / "result.json"
            if result_src.exists():
                try:
                    result = load_json(result_src)
                    body = read_text(workspace_src / "stub.c") if (workspace_src / "stub.c").exists() else None
                except Exception:
                    body = None
        finally:
            container_mod.teardown(name)
        if inner_episode is not None:
            _harvest(workspace_src, workspace, ["stub.c", "result.json", "stub_validate_main.c"])
            _harvest_glob(inner_episode / "agent_history", ep_history, f"_gen_stub_{safe}.*")
            _harvest_glob(inner_episode / "agent_history", ep_history, f"_stub_fix_{safe}_*")
    else:
        active = test_dir / "_stub_gen" / safe
        history_dir = test_dir / "agent_history"
        # Pristine start: drop cached stub so the agent reruns and only ever sees
        # the canonical _stub_gen path.
        _reset_dir(active)
        body = generate_stub_code(cfg, test_dir, func_name)
        _harvest(active, workspace, ["stub.c", "result.json", "stub_validate_main.c"])
        _harvest_glob(history_dir, ep_history, f"_gen_stub_{safe}.*")
        _harvest_glob(history_dir, ep_history, f"_stub_fix_{safe}_*")
        _reset_dir(active)  # leave pristine for the next episode/stage

    result = {}
    result_path = workspace / "result.json"
    if result_path.exists():
        try:
            result = load_json(result_path)
        except Exception:
            result = {}
    metadata = {
        **_metadata_base(cfg, paths, "collect-stubs", eid),
        "func_name": func_name,
        "safe_name": safe,
        "validated": bool(result.get("validated")),
        "body_chars": len(body or ""),
        "workspace": str(workspace),
        "agent_history": str(ep_history),
        "result": result,
    }
    write_json(episode_root / "metadata.json", metadata)


def _stage_collect_stubs(cfg: PipelineConfig, paths: dict) -> None:
    if cfg.per_episode_container:
        analysis = run_or_load_analysis(cfg, paths["analysis_path"])
    else:
        _flags, analysis = _prepare(cfg, paths)
    candidates = collect_stub_candidates(analysis)
    if cfg.only_function:
        candidates = [name for name in candidates if name == cfg.only_function]
    eps = max(1, int(cfg.episodes_per_item))
    jobs = [
        partial(_collect_one_stub, cfg, paths, func_name, _safe_filename(func_name))
        for func_name in candidates
        for _ in range(eps)
    ]
    _run_jobs(cfg, jobs)


def _stage_select_stubs(cfg: PipelineConfig, paths: dict) -> None:
    root = dataset_root(cfg, paths)
    episodes_root = root / "episodes" / "stubs"
    if not episodes_root.exists():
        print("[pipeline] no stub episodes to select", file=sys.stderr)
        return

    for safe_dir in sorted(p for p in episodes_root.iterdir() if p.is_dir()):
        chosen: Optional[tuple[Path, dict]] = None
        for episode in sorted(p for p in safe_dir.iterdir() if p.is_dir()):
            metadata_path = episode / "metadata.json"
            workspace = episode / "workspace"
            result_path = workspace / "result.json"
            stub_path = workspace / "stub.c"
            if not metadata_path.exists() or not result_path.exists() or not stub_path.exists():
                continue
            try:
                result = load_json(result_path)
                metadata = load_json(metadata_path)
            except Exception:
                continue
            if result.get("validated"):
                chosen = (episode, metadata)
                break

        if chosen is None:
            print(f"[pipeline] no validated stub episode for {safe_dir.name}", file=sys.stderr)
            continue

        episode, metadata = chosen
        workspace = episode / "workspace"
        frontier = root / "frontiers" / "stubs" / safe_dir.name
        frontier.mkdir(parents=True, exist_ok=True)
        _copy_if_exists(workspace / "stub.c", frontier / "stub.c")
        _copy_if_exists(workspace / "result.json", frontier / "result.json")
        write_json(frontier / "selected.json", {
            **metadata,
            "selected_episode": str(episode),
            "frontier_dir": str(frontier),
        })
        print(f"[pipeline] selected stub {safe_dir.name}: {episode.name}", file=sys.stderr)


def _stage_materialize_stubs(cfg: PipelineConfig, paths: dict) -> None:
    root = dataset_root(cfg, paths)
    frontier_root = root / "frontiers" / "stubs"
    test_dir = Path(paths["test_dir"])
    if not frontier_root.exists():
        print("[pipeline] no selected stubs to materialize", file=sys.stderr)
        return

    for frontier in sorted(p for p in frontier_root.iterdir() if p.is_dir()):
        selected = _load_selected_json(frontier)
        if not selected:
            continue
        active = test_dir / "_stub_gen" / frontier.name
        _copy_if_exists(frontier / "stub.c", active / "stub.c")
        _copy_if_exists(frontier / "result.json", active / "result.json")
        print(f"[pipeline] materialized stub {frontier.name}", file=sys.stderr)


def _validated_stub_bodies(paths: dict, analysis: dict) -> dict[str, str]:
    test_dir = Path(paths["test_dir"])
    bodies: dict[str, str] = {}
    for name in collect_stub_candidates(analysis):
        safe = _safe_filename(name)
        stub_dir = test_dir / "_stub_gen" / safe
        stub_file = stub_dir / "stub.c"
        result_file = stub_dir / "result.json"
        if not stub_file.exists() or not result_file.exists():
            continue
        try:
            if not load_json(result_file).get("validated"):
                continue
        except Exception:
            continue
        body = read_text(stub_file).strip()
        if body and f"__wrap_{name}" in body and f"__real_{name}" not in body:
            bodies[name] = body
    return bodies


def _stage_integrate_stubs(cfg: PipelineConfig, paths: dict) -> None:
    flags, analysis = _prepare(cfg, paths)
    bodies = _validated_stub_bodies(paths, analysis)
    integrate_all_stubs_sequential(cfg, paths, bodies, flags)


def _stage_minimal_master(cfg: PipelineConfig, paths: dict) -> None:
    _prepare(cfg, paths)
    ensure_minimal_test_runs(cfg, paths)
    root = dataset_root(cfg, paths)
    frontier = root / "frontiers" / "minimal_master"
    frontier.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(Path(paths["test_file"]), frontier / Path(paths["test_file"]).name)
    _copy_if_exists(Path(paths["makefile"]), frontier / "Makefile")
    write_json(frontier / "selected.json", {
        "stage": "minimal-master",
        "process_name": paths["process_name"],
        "source_dir": str(cfg.source_dir),
        "test_file": str(paths["test_file"]),
        "makefile": str(paths["makefile"]),
    })


def _collect_one_unit(
    cfg: PipelineConfig,
    paths: dict,
    func: dict,
    safe: str,
    flags: dict,
    semantic_context: dict,
) -> None:
    test_dir = Path(paths["test_dir"])
    root = dataset_root(cfg, paths)
    eid = _episode_id()
    episode_root = root / "episodes" / "unit_tests" / safe / eid
    workspace = episode_root / "workspace"
    harvest_names = [
        f"test_{safe}.c",
        "Makefile",
        "coverage.json",
        "judge_verdict.json",
        "unit_test_failed.json",
        "agent_history",  # includes good_cunit_backups/
    ]

    if cfg.per_episode_container:
        run_cfg, H, name = _episode_container(
            cfg,
            paths,
            episode_root,
            include_stub_gen=True,
            seed_from_base=True,
        )
        try:
            _run_inner_stage(
                run_cfg,
                source_dir=cfg.source_dir,
                stage="collect-unit-tests",
                log_dir=episode_root,
                extra_args=[
                    "--only-function",
                    func["id"],
                    "--episodes-per-item",
                    "1",
                ],
            )
            inner_root = _inner_dataset_root(H) / "episodes" / "unit_tests" / safe
            inner_episodes = sorted(p for p in inner_root.iterdir() if p.is_dir()) if inner_root.exists() else []
            inner_episode = inner_episodes[-1] if inner_episodes else None
            if inner_episode is None:
                raise RuntimeError(f"inner collect-unit-tests produced no episode for {func['id']}")
            metadata = load_json(inner_episode / "metadata.json")
            fid = metadata.get("func_id", func["id"])
            result = metadata.get("result", {})
        finally:
            container_mod.teardown(name)
        if inner_episode is not None:
            _harvest(inner_episode / "workspace", workspace, harvest_names)
    else:
        active = test_dir / "_unit_tests" / safe
        # Pristine start: clear only this function's active unit dir. Inputs
        # (selected _stub_gen, minimal-master baseline, context) survive.
        _reset_dir(active)
        fid, result = _generate_unit_test_for_func(cfg, paths, func, flags, semantic_context)
        _harvest(active, workspace, harvest_names)
        _reset_dir(active)  # leave pristine for the next episode/stage

    metadata = {
        **_metadata_base(cfg, paths, "collect-unit-tests", eid),
        "func_id": fid,
        "safe_id": safe,
        "func": func,
        "workspace": str(workspace),
        "result": result,
        "passed": bool(result.get("passed")),
        "coverage_pct": result.get("coverage_pct"),
        "semantic_score": result.get("semantic_score"),
    }
    write_json(episode_root / "metadata.json", metadata)


def _stage_collect_unit_tests(cfg: PipelineConfig, paths: dict) -> None:
    if cfg.per_episode_container:
        flags = {}
        analysis = run_or_load_analysis(cfg, paths["analysis_path"])
    else:
        flags, analysis = _prepare(cfg, paths)
    semantic_context = _load_semantic_context(Path(paths["test_dir"]))
    eps = max(1, int(cfg.episodes_per_item))
    jobs = [
        partial(_collect_one_unit, cfg, paths, func, _safe_filename(func["id"]), flags, semantic_context)
        for func in _target_functions(cfg, analysis)
        for _ in range(eps)
    ]
    _run_jobs(cfg, jobs)


def _unit_sort_key(metadata: dict) -> tuple[int, float, float, str]:
    result = metadata.get("result") if isinstance(metadata.get("result"), dict) else {}
    passed = bool(metadata.get("passed") or result.get("passed"))
    coverage = metadata.get("coverage_pct", result.get("coverage_pct"))
    score = metadata.get("semantic_score", result.get("semantic_score"))
    try:
        coverage_f = float(coverage)
    except Exception:
        coverage_f = -1.0
    try:
        score_f = float(score)
    except Exception:
        score_f = -1.0
    return (1 if passed else 0, coverage_f, score_f, str(metadata.get("episode_id", "")))


def _stage_select_unit_tests(cfg: PipelineConfig, paths: dict) -> None:
    root = dataset_root(cfg, paths)
    episodes_root = root / "episodes" / "unit_tests"
    if not episodes_root.exists():
        print("[pipeline] no unit-test episodes to select", file=sys.stderr)
        return

    for safe_dir in sorted(p for p in episodes_root.iterdir() if p.is_dir()):
        candidates: list[tuple[dict, Path]] = []
        for episode in sorted(p for p in safe_dir.iterdir() if p.is_dir()):
            metadata_path = episode / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                metadata = load_json(metadata_path)
            except Exception:
                continue
            candidates.append((metadata, episode))
        if not candidates:
            continue

        metadata, episode = max(candidates, key=lambda item: _unit_sort_key(item[0]))
        workspace = episode / "workspace"
        frontier = root / "frontiers" / "unit_tests" / safe_dir.name
        frontier.mkdir(parents=True, exist_ok=True)
        _copy_if_exists(workspace / f"test_{safe_dir.name}.c", frontier / f"test_{safe_dir.name}.c")
        _copy_if_exists(workspace / "Makefile", frontier / "Makefile")
        _copy_if_exists(workspace / "coverage.json", frontier / "coverage.json")
        _copy_if_exists(workspace / "judge_verdict.json", frontier / "judge_verdict.json")
        _copy_if_exists(workspace / "unit_test_failed.json", frontier / "unit_test_failed.json")
        write_json(frontier / "selected.json", {
            **metadata,
            "selected_episode": str(episode),
            "frontier_dir": str(frontier),
        })
        print(f"[pipeline] selected unit test {safe_dir.name}: {episode.name}", file=sys.stderr)


def _selected_unit_results(
    cfg: PipelineConfig,
    paths: dict,
    *,
    prefer_active: bool,
) -> dict[str, dict]:
    root = dataset_root(cfg, paths)
    frontier_root = root / "frontiers" / "unit_tests"
    test_dir = Path(paths["test_dir"])
    results: dict[str, dict] = {}
    if not frontier_root.exists():
        return results

    for frontier in sorted(p for p in frontier_root.iterdir() if p.is_dir()):
        selected = _load_selected_json(frontier)
        if not selected:
            continue
        result = selected.get("result") if isinstance(selected.get("result"), dict) else {}
        func_id = selected.get("func_id") or result.get("func_id") or frontier.name
        active_dir = test_dir / "_unit_tests" / frontier.name
        unit_dir = active_dir if prefer_active and (active_dir / f"test_{frontier.name}.c").exists() else frontier
        results[str(func_id)] = {
            **result,
            "passed": bool(selected.get("passed") or result.get("passed")),
            "coverage_pct": selected.get("coverage_pct", result.get("coverage_pct")),
            "semantic_score": selected.get("semantic_score", result.get("semantic_score")),
            "verdict": result.get("verdict"),
            "unit_dir": str(unit_dir),
        }
    return results


def _stage_materialize_unit_tests(cfg: PipelineConfig, paths: dict) -> None:
    _flags, analysis = _prepare(cfg, paths)
    root = dataset_root(cfg, paths)
    frontier_root = root / "frontiers" / "unit_tests"
    test_dir = Path(paths["test_dir"])
    if not frontier_root.exists():
        print("[pipeline] no selected unit tests to materialize", file=sys.stderr)
        return

    for frontier in sorted(p for p in frontier_root.iterdir() if p.is_dir()):
        selected = _load_selected_json(frontier)
        if not selected:
            continue
        active = test_dir / "_unit_tests" / frontier.name
        _copy_if_exists(frontier / f"test_{frontier.name}.c", active / f"test_{frontier.name}.c")
        _copy_if_exists(frontier / "Makefile", active / "Makefile")
        _copy_if_exists(frontier / "coverage.json", active / "coverage.json")
        _copy_if_exists(frontier / "judge_verdict.json", active / "judge_verdict.json")
        _copy_if_exists(frontier / "unit_test_failed.json", active / "unit_test_failed.json")
        print(f"[pipeline] materialized unit test {frontier.name}", file=sys.stderr)

    unit_results = _selected_unit_results(cfg, paths, prefer_active=True)
    context_file = test_dir / "_pipeline_context.json"
    ctx: dict[str, Any] = {}
    if context_file.exists():
        try:
            ctx = load_json(context_file)
        except Exception:
            ctx = {}
    ctx["unit_test_results"] = unit_results
    targeted_ids = [func["id"] for func in _target_functions(cfg, analysis)]
    all_targeted_passed = bool(targeted_ids) and all(
        unit_results.get(fid, {}).get("passed") for fid in targeted_ids
    )
    if all_targeted_passed:
        ctx["unit_tests_completed"] = True
    else:
        ctx.pop("unit_tests_completed", None)
    write_json(context_file, ctx)


def _stage_integrate(cfg: PipelineConfig, paths: dict) -> None:
    flags, analysis = _prepare(cfg, paths)
    context_file = Path(paths["test_dir"]) / "_pipeline_context.json"
    unit_results = {}
    if context_file.exists():
        try:
            unit_results = load_json(context_file).get("unit_test_results", {})
        except Exception:
            unit_results = {}
    if not unit_results:
        unit_results = _selected_unit_results(cfg, paths, prefer_active=True)
    integrate_all_unit_tests_sequential(cfg, paths, analysis, unit_results, flags)


def run_selected_stage(cfg: PipelineConfig, paths: dict) -> None:
    stages = {
        "prepare": _stage_prepare,
        "collect-stubs": _stage_collect_stubs,
        "select-stubs": _stage_select_stubs,
        "materialize-stubs": _stage_materialize_stubs,
        "integrate-stubs": _stage_integrate_stubs,
        "minimal-master": _stage_minimal_master,
        "collect-unit-tests": _stage_collect_unit_tests,
        "select-unit-tests": _stage_select_unit_tests,
        "materialize-unit-tests": _stage_materialize_unit_tests,
        "integrate": _stage_integrate,
    }
    handler = stages.get(cfg.stage)
    if handler is None:
        raise ValueError(f"Unsupported stage: {cfg.stage}")
    dataset_root(cfg, paths).mkdir(parents=True, exist_ok=True)
    _write_stage_event(cfg, paths, cfg.stage)
    handler(cfg, paths)
