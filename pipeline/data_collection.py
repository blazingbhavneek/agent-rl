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
from .common import _function_artifact_key, _safe_filename, load_json, read_text, write_json
from .config import PipelineConfig
from .semantic import _load_semantic_context
from . import scoring
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


def _run_jobs(cfg: PipelineConfig, jobs: list) -> None:
    """Run episode jobs, in parallel only for per-episode-container mode."""
    if not jobs:
        return

    workers = (
        min(len(jobs), max(1, int(cfg.episode_concurrency)))
        if cfg.per_episode_container
        else 1
    )

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
    eid: str,
    *,
    include_stub_gen: bool,
    seed_from_base: bool,
):
    """Set up one fresh container for an episode.

    Returns (run_cfg, host_test_dir, container_name). The per-episode scratch
    test dir lives at _trace_dataset/scratch/<eid> (visible, outside the episode
    dir) and is bound at the canonical path; run_cfg carries the docker target +
    the host->canonical path_map so the agent only sees canonical paths.
    """
    canonical = Path(paths["test_dir"])
    repo_root = cfg.source_dir.parent.parent.resolve()
    host_test_dir = dataset_root(cfg, paths) / "scratch" / eid
    print(
        f"[pipeline] starting episode container scratch={host_test_dir}",
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


def _score_workspace_results(stage: str, cfg: PipelineConfig, paths: dict, results: list) -> None:
    """Write score.json for episodic workspace stages (minimal / integrate)."""
    for r in results:
        ws = Path(r["workspace"])
        meta = {
            "episode_id": r.get("eid"),
            "process_name": paths["process_name"],
            "ok": bool(r.get("ok")),
        }
        try:
            scoring.write_episode_score(
                stage, ws, meta,
                coverage_threshold=cfg.coverage_threshold,
                semantic_min=cfg.semantic_judge_min_score,
                agent_history=ws / "agent_history",
            )
        except Exception as exc:
            print(f"[pipeline] WARN score failed {ws}: {exc}", file=sys.stderr)


def _stage_prepare(cfg: PipelineConfig, paths: dict) -> None:
    _prepare(cfg, paths)


def _collect_one_stub(cfg: PipelineConfig, paths: dict, func_name: str, safe: str) -> None:
    test_dir = Path(paths["test_dir"])
    root = dataset_root(cfg, paths)
    eid = _episode_id()
    episode_root = root / "episodes" / "stubs" / safe / eid
    workspace = episode_root / "workspace"
    ep_history = episode_root / "agent_history"
    body: Optional[str] = None

    if cfg.per_episode_container:
        # Container = isolated execution env for ONE stub. Generate directly into
        # the scratch test dir (bound at the canonical path); no stage dispatch,
        # so no recursive collection and no _trace_dataset/testdir leak.
        run_cfg, H, name = _episode_container(
            cfg,
            paths,
            eid,
            include_stub_gen=False,
            seed_from_base=True,
        )
        try:
            body = generate_stub_code(
                run_cfg,
                test_dir,
                func_name,
                stub_dir_override=H / "_stub_gen" / safe,
                history_dir_override=H / "agent_history",
            )
        finally:
            container_mod.teardown(name)
        _harvest(H / "_stub_gen" / safe, workspace, ["stub.c", "result.json", "stub_validate_main.c"])
        _harvest_glob(H / "agent_history", ep_history, f"_gen_stub_{safe}.*")
        _harvest_glob(H / "agent_history", ep_history, f"_stub_fix_{safe}_*")
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
    scoring.write_episode_score(
        "collect-stubs", episode_root, metadata,
        coverage_threshold=cfg.coverage_threshold,
        semantic_min=cfg.semantic_judge_min_score,
        agent_history=ep_history,
    )


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

import os
import shutil
import uuid
from pathlib import Path


def _rebase_paths_for_workspace(paths: dict, old_root: Path, new_root: Path) -> dict:
    """
    Copy paths dict and rewrite anything under old_root to the matching path
    under new_root.

    Example:
      /repo/tests/dio100d/test_dio100d.c
    becomes:
      /repo/tests/dio100d/_trace_dataset/integrate_episodes/eid/test_dio100d.c
    """
    old_root = old_root.resolve()
    new_root = new_root.resolve()

    return {
        key: _rebase_value_for_workspace(value, old_root, new_root)
        for key, value in paths.items()
    }


def _rebase_value_for_workspace(value: Any, old_root: Path, new_root: Path) -> Any:
    old_root_str = str(old_root)

    if isinstance(value, Path):
        s = str(value.resolve())
        if s == old_root_str or s.startswith(old_root_str + os.sep):
            rel = Path(s).relative_to(old_root)
            return new_root / rel
        return value

    if isinstance(value, str):
        s = str(Path(value).resolve())
        if s == old_root_str or s.startswith(old_root_str + os.sep):
            rel = Path(s).relative_to(old_root)
            return str(new_root / rel)
        return value

    if isinstance(value, dict):
        return {
            key: _rebase_value_for_workspace(item, old_root, new_root)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_rebase_value_for_workspace(item, old_root, new_root) for item in value]

    if isinstance(value, tuple):
        return tuple(_rebase_value_for_workspace(item, old_root, new_root) for item in value)

    return value


def _rebase_unit_test_results_for_workspace(
    unit_results: dict[str, dict],
    old_root: Path,
    new_root: Path,
) -> dict[str, dict]:
    return _rebase_value_for_workspace(unit_results, old_root, new_root)


def _make_integrate_workspace(cfg: PipelineConfig, paths: dict, eid: str) -> Path:
    """
    Create a clean workspace for one integrate-stubs episode.

    Copies only the minimal files needed:
      - Makefile
      - main test file
      - _stub_gen
      - _pipeline_context.json
      - analysis.json

    Does NOT copy:
      - _trace_dataset
      - old agent_history
      - stdout/stderr logs
      - object files
    """
    canonical_test_dir = Path(paths["test_dir"]).resolve()

    dataset_root_dir = dataset_root(cfg, paths)
    workspace = dataset_root_dir / "integrate_episodes" / eid

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)

    # Copy main test file.
    test_file = Path(paths["test_file"]).resolve()
    if test_file.exists():
        shutil.copy2(test_file, workspace / test_file.name)

    # Copy Makefile.
    makefile = canonical_test_dir / "Makefile"
    if makefile.exists():
        shutil.copy2(makefile, workspace / "Makefile")

    # Copy cached pipeline context if available.
    context_file = canonical_test_dir / "_pipeline_context.json"
    if context_file.exists():
        shutil.copy2(context_file, workspace / "_pipeline_context.json")

    # Copy analysis if available.
    analysis_file = canonical_test_dir / "analysis.json"
    if analysis_file.exists():
        shutil.copy2(analysis_file, workspace / "analysis.json")

    # Copy generated stubs.
    stub_gen = canonical_test_dir / "_stub_gen"
    if stub_gen.exists():
        shutil.copytree(stub_gen, workspace / "_stub_gen")

    # Fresh isolated agent history for this episode.
    (workspace / "agent_history").mkdir(parents=True, exist_ok=True)

    return workspace

def _run_one_integrate_episode(
    cfg: PipelineConfig,
    paths: dict,
    bodies: list,
    flags,
    episode_index: int,
):
    canonical_test_dir = Path(paths["test_dir"]).resolve()
    repo_root = cfg.source_dir.parent.parent.resolve()

    eid = f"integrate_{episode_index:06d}_{uuid.uuid4().hex[:8]}"
    workspace = _make_integrate_workspace(cfg, paths, eid)

    episode_paths = _rebase_paths_for_workspace(
        paths,
        old_root=canonical_test_dir,
        new_root=workspace,
    )

    name = container_mod.episode_container_name()

    print(
        f"[pipeline] starting integrate episode {episode_index} "
        f"name={name} workspace={workspace}",
        file=sys.stderr,
    )

    container_mod.create(
        cfg,
        name,
        host_test_dir=workspace,
        canonical_test_dir=canonical_test_dir,
        repo_root=repo_root,
    )

    run_cfg = replace(
        cfg,
        execution_mode="docker",
        container_name=name,
        per_episode_container=False,
        # Host workspace paths must become canonical container test paths.
        path_map=((str(workspace), str(canonical_test_dir)),),
    )

    try:
        integrate_all_stubs_sequential(
            run_cfg,
            episode_paths,
            bodies,
            flags,
        )

        print(
            f"[pipeline] integrate episode success index={episode_index} "
            f"workspace={workspace}",
            file=sys.stderr,
        )

        return {
            "ok": True,
            "episode_index": episode_index,
            "eid": eid,
            "workspace": workspace,
        }

    except Exception as exc:
        print(
            f"[pipeline] integrate episode failed index={episode_index} "
            f"workspace={workspace}: {exc}",
            file=sys.stderr,
        )

        return {
            "ok": False,
            "episode_index": episode_index,
            "eid": eid,
            "workspace": workspace,
            "error": repr(exc),
        }

    finally:
        container_mod.teardown(name)
        print(
            f"[pipeline] integrate episode container removed name={name}",
            file=sys.stderr,
        )

def _promote_integrate_workspace(paths: dict, workspace: Path) -> None:
    """
    Copy selected episode result back into the real canonical test dir.

    Only promote files that integration is expected to modify.
    Do NOT copy trace logs, agent_history, object files, etc.
    """
    canonical_test_dir = Path(paths["test_dir"]).resolve()

    test_file = Path(paths["test_file"]).resolve()
    workspace_test_file = workspace / test_file.name

    if workspace_test_file.exists():
        shutil.copy2(workspace_test_file, test_file)

    workspace_makefile = workspace / "Makefile"
    canonical_makefile = canonical_test_dir / "Makefile"

    if workspace_makefile.exists():
        shutil.copy2(workspace_makefile, canonical_makefile)

def _stage_integrate_stubs(cfg: PipelineConfig, paths: dict) -> None:
    """
    Integrate validated stubs using multiple clean workspaces.

    Each episode gets:
      - copied Makefile
      - copied main test file
      - copied _stub_gen
      - copied cached context/analysis
      - fresh agent_history
      - fresh container
    """
    flags, analysis = _prepare(cfg, paths)
    bodies = _validated_stub_bodies(paths, analysis)

    print(
        f"[pipeline] integrate-stubs: {len(bodies)} validated stub bodies found",
        file=sys.stderr,
    )

    if not cfg.per_episode_container:
        integrate_all_stubs_sequential(cfg, paths, bodies, flags)
        return

    if cfg.execution_mode != "docker":
        raise ValueError("--per-episode-container for integrate-stubs requires --execution-mode docker")
    if not cfg.container_image:
        raise ValueError("--container-image is required with --per-episode-container")

    episode_count = int(getattr(cfg, "episodes_per_item", 1) or 1)

    episode_concurrency = int(getattr(cfg, "episode_concurrency", 1) or 1)
    max_workers = max(1, min(episode_concurrency, episode_count))

    results = []

    print(
        f"[pipeline] integrate-stubs running {episode_count} episodes "
        f"with concurrency={max_workers}",
        file=sys.stderr,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_one_integrate_episode,
                cfg,
                paths,
                bodies,
                flags,
                i,
            )
            for i in range(episode_count)
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    _score_workspace_results("integrate-stubs", cfg, paths, results)
    successes = [r for r in results if r.get("ok")]

    print(
        f"[pipeline] integrate-stubs episodes complete: "
        f"{len(successes)}/{len(results)} succeeded",
        file=sys.stderr,
    )

    if not successes:
        raise RuntimeError("all integrate-stubs episodes failed")

    # For now, promote the first successful workspace.
    best = successes[0]
    _promote_integrate_workspace(paths, best["workspace"])

    print(
        f"[pipeline] promoted integrate workspace: {best['workspace']}",
        file=sys.stderr,
    )


def _make_minimal_workspace(cfg: PipelineConfig, paths: dict, eid: str) -> Path:
    canonical_test_dir = Path(paths["test_dir"]).resolve()
    dataset_root_dir = dataset_root(cfg, paths)

    workspace = dataset_root_dir / "minimal_episodes" / eid

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)

    test_file = Path(paths["test_file"]).resolve()
    if test_file.exists():
        shutil.copy2(test_file, workspace / test_file.name)

    makefile = Path(paths["makefile"]).resolve()
    if makefile.exists():
        shutil.copy2(makefile, workspace / "Makefile")

    context_file = canonical_test_dir / "_pipeline_context.json"
    if context_file.exists():
        shutil.copy2(context_file, workspace / "_pipeline_context.json")

    analysis_file = canonical_test_dir / "analysis.json"
    if analysis_file.exists():
        shutil.copy2(analysis_file, workspace / "analysis.json")

    stub_gen = canonical_test_dir / "_stub_gen"
    if stub_gen.exists():
        shutil.copytree(stub_gen, workspace / "_stub_gen")

    (workspace / "agent_history").mkdir(parents=True, exist_ok=True)

    return workspace

def _run_one_minimal_episode(
    cfg: PipelineConfig,
    paths: dict,
    episode_index: int,
):
    canonical_test_dir = Path(paths["test_dir"]).resolve()
    repo_root = cfg.source_dir.parent.parent.resolve()

    eid = f"minimal_{episode_index:06d}_{uuid.uuid4().hex[:8]}"
    workspace = _make_minimal_workspace(cfg, paths, eid)

    episode_paths = _rebase_paths_for_workspace(
        paths,
        old_root=canonical_test_dir,
        new_root=workspace,
    )

    name = container_mod.episode_container_name()

    print(
        f"[pipeline] starting minimal episode {episode_index} "
        f"name={name} workspace={workspace}",
        file=sys.stderr,
    )

    container_mod.create(
        cfg,
        name,
        host_test_dir=workspace,
        canonical_test_dir=canonical_test_dir,
        repo_root=repo_root,
    )

    run_cfg = replace(
        cfg,
        execution_mode="docker",
        container_name=name,
        per_episode_container=False,
        path_map=((str(workspace), str(canonical_test_dir)),),
    )

    try:
        ensure_minimal_test_runs(run_cfg, episode_paths)

        print(
            f"[pipeline] minimal episode success index={episode_index} "
            f"workspace={workspace}",
            file=sys.stderr,
        )

        return {
            "ok": True,
            "episode_index": episode_index,
            "eid": eid,
            "workspace": workspace,
        }

    except Exception as exc:
        print(
            f"[pipeline] minimal episode failed index={episode_index} "
            f"workspace={workspace}: {exc}",
            file=sys.stderr,
        )

        return {
            "ok": False,
            "episode_index": episode_index,
            "eid": eid,
            "workspace": workspace,
            "error": repr(exc),
        }

    finally:
        container_mod.teardown(name)
        print(
            f"[pipeline] minimal episode container removed name={name}",
            file=sys.stderr,
        )

def _promote_minimal_workspace(paths: dict, workspace: Path) -> None:
    canonical_test_dir = Path(paths["test_dir"]).resolve()

    test_file = Path(paths["test_file"]).resolve()
    workspace_test_file = workspace / test_file.name

    if workspace_test_file.exists():
        shutil.copy2(workspace_test_file, test_file)

    workspace_makefile = workspace / "Makefile"
    canonical_makefile = canonical_test_dir / "Makefile"

    if workspace_makefile.exists():
        shutil.copy2(workspace_makefile, canonical_makefile)

def _stage_minimal_master(cfg: PipelineConfig, paths: dict) -> None:
    _prepare(cfg, paths)

    if not cfg.per_episode_container:
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
        return

    if cfg.execution_mode != "docker":
        raise ValueError("--per-episode-container for minimal-master requires --execution-mode docker")

    if not cfg.container_image:
        raise ValueError("--container-image is required with --per-episode-container")

    episode_count = int(getattr(cfg, "episodes_per_item", 1) or 1)
    episode_concurrency = int(getattr(cfg, "episode_concurrency", 1) or 1)
    max_workers = max(1, min(episode_concurrency, episode_count))

    print(
        f"[pipeline] minimal-master running {episode_count} episodes "
        f"with concurrency={max_workers}",
        file=sys.stderr,
    )

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_one_minimal_episode,
                cfg,
                paths,
                i,
            )
            for i in range(episode_count)
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    _score_workspace_results("minimal-master", cfg, paths, results)
    successes = [r for r in results if r.get("ok")]

    print(
        f"[pipeline] minimal-master episodes complete: "
        f"{len(successes)}/{len(results)} succeeded",
        file=sys.stderr,
    )

    if not successes:
        raise RuntimeError("all minimal-master episodes failed")

    # Deterministic choice: lowest episode_index wins.
    best = sorted(successes, key=lambda r: r["episode_index"])[0]

    _promote_minimal_workspace(paths, best["workspace"])

    print(
        f"[pipeline] promoted minimal workspace: {best['workspace']}",
        file=sys.stderr,
    )

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
        "selected_workspace": str(best["workspace"]),
        "episode_index": best["episode_index"],
        "episodes_total": len(results),
        "episodes_succeeded": len(successes),
    })


def _make_unit_test_workspace(cfg: PipelineConfig, paths: dict, safe: str, eid: str) -> Path:
    canonical_test_dir = Path(paths["test_dir"]).resolve()
    dataset_root_dir = dataset_root(cfg, paths)

    workspace = dataset_root_dir / "unit_test_episodes" / safe / eid

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)

    test_file = Path(paths["test_file"]).resolve()
    if test_file.exists():
        shutil.copy2(test_file, workspace / test_file.name)

    makefile = Path(paths["makefile"]).resolve()
    if makefile.exists():
        shutil.copy2(makefile, workspace / "Makefile")

    context_file = canonical_test_dir / "_pipeline_context.json"
    if context_file.exists():
        shutil.copy2(context_file, workspace / "_pipeline_context.json")

    analysis_file = canonical_test_dir / "analysis.json"
    if analysis_file.exists():
        shutil.copy2(analysis_file, workspace / "analysis.json")

    stub_gen = canonical_test_dir / "_stub_gen"
    if stub_gen.exists():
        shutil.copytree(stub_gen, workspace / "_stub_gen")

    (workspace / "agent_history").mkdir(parents=True, exist_ok=True)

    return workspace

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
    episode_workspace: Optional[Path] = None
    harvest_names = [
        f"test_{safe}.c",
        "Makefile",
        "coverage.json",
        "judge_verdict.json",
        "unit_test_failed.json",
        "agent_history",  # includes good_cunit_backups/
    ]

    if cfg.per_episode_container:
        # Container = isolated execution env for ONE function's unit test.
        # Each episode gets its own dedicated workspace under unit_test_episodes/.
        canonical_test_dir = Path(paths["test_dir"]).resolve()
        repo_root = cfg.source_dir.parent.parent.resolve()
        H = _make_unit_test_workspace(cfg, paths, safe, eid).resolve()
        episode_workspace = H
        name = container_mod.episode_container_name()

        print(
            f"[pipeline] starting unit-test episode {safe} workspace={H}",
            file=sys.stderr,
        )

        container_mod.create(
            cfg,
            name,
            host_test_dir=H,
            canonical_test_dir=canonical_test_dir,
            repo_root=repo_root,
        )

        run_cfg = replace(
            cfg,
            execution_mode="docker",
            container_name=name,
            per_episode_container=False,
            path_map=((str(H), str(canonical_test_dir)),),
        )

        episode_paths = _rebase_paths_for_workspace(
            paths,
            old_root=canonical_test_dir,
            new_root=H,
        )

        unit_dir = H / "_unit_tests" / safe

        try:
            result_key, result = _generate_unit_test_for_func(
                run_cfg,
                episode_paths,
                func,
                flags,
                semantic_context,
                unit_dir_override=unit_dir,
            )
        finally:
            container_mod.teardown(name)
            print(
                f"[pipeline] unit-test episode container removed name={name}",
                file=sys.stderr,
            )

        _harvest(unit_dir, workspace, harvest_names)
    else:
        active = test_dir / "_unit_tests" / safe
        # Pristine start: clear only this function's active unit dir. Inputs
        # (selected _stub_gen, minimal-master baseline, context) survive.
        _reset_dir(active)
        result_key, result = _generate_unit_test_for_func(cfg, paths, func, flags, semantic_context)
        _harvest(active, workspace, harvest_names)
        _reset_dir(active)  # leave pristine for the next episode/stage

    metadata = {
        **_metadata_base(cfg, paths, "collect-unit-tests", eid),
        "func_id": func.get("id"),
        "function_key": result_key,
        "safe_id": safe,
        "func": func,
        "workspace": str(workspace),
        "result": result,
        "passed": bool(result.get("passed")),
        "coverage_pct": result.get("coverage_pct"),
        "semantic_score": result.get("semantic_score"),
    }
    if episode_workspace is not None:
        metadata["episode_workspace"] = str(episode_workspace)
    write_json(episode_root / "metadata.json", metadata)
    scoring.write_episode_score(
        "collect-unit-tests", episode_root, metadata,
        coverage_threshold=cfg.coverage_threshold,
        semantic_min=cfg.semantic_judge_min_score,
        agent_history=workspace / "agent_history",
    )


def _stage_collect_unit_tests(cfg: PipelineConfig, paths: dict) -> None:
    flags, analysis = _prepare(cfg, paths)
    semantic_context = _load_semantic_context(Path(paths["test_dir"]))
    eps = max(1, int(cfg.episodes_per_item))
    jobs = [
        partial(_collect_one_unit, cfg, paths, func, _function_artifact_key(cfg, func), flags, semantic_context)
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
        function_key = selected.get("function_key") or result.get("function_key") or frontier.name
        active_dir = test_dir / "_unit_tests" / frontier.name
        unit_dir = active_dir if prefer_active and (active_dir / f"test_{frontier.name}.c").exists() else frontier
        results[str(function_key)] = {
            **result,
            "func_id": str(func_id),
            "function_key": str(function_key),
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
    targeted_keys = [_function_artifact_key(cfg, func) for func in _target_functions(cfg, analysis)]
    all_targeted_passed = bool(targeted_keys) and all(
        unit_results.get(key, {}).get("passed") for key in targeted_keys
    )
    if all_targeted_passed:
        ctx["unit_tests_completed"] = True
    else:
        ctx.pop("unit_tests_completed", None)
    write_json(context_file, ctx)


def _make_final_integration_workspace(cfg: PipelineConfig, paths: dict, eid: str) -> Path:
    canonical_test_dir = Path(paths["test_dir"]).resolve()
    dataset_root_dir = dataset_root(cfg, paths)

    workspace = dataset_root_dir / "final_integration_episodes" / eid

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)

    test_file = Path(paths["test_file"]).resolve()
    if test_file.exists():
        shutil.copy2(test_file, workspace / test_file.name)

    makefile = Path(paths["makefile"]).resolve()
    if makefile.exists():
        shutil.copy2(makefile, workspace / "Makefile")

    context_file = canonical_test_dir / "_pipeline_context.json"
    if context_file.exists():
        shutil.copy2(context_file, workspace / "_pipeline_context.json")

    analysis_file = canonical_test_dir / "analysis.json"
    if analysis_file.exists():
        shutil.copy2(analysis_file, workspace / "analysis.json")

    stub_gen = canonical_test_dir / "_stub_gen"
    if stub_gen.exists():
        shutil.copytree(stub_gen, workspace / "_stub_gen")

    unit_tests = canonical_test_dir / "_unit_tests"
    if unit_tests.exists():
        shutil.copytree(unit_tests, workspace / "_unit_tests")

    # Selected unit-test frontiers live under _trace_dataset and may be the
    # target of unit_dir when the active _unit_tests copy is absent. Copy them
    # to the rebased location so frontier-based unit_dir paths resolve inside
    # the workspace.
    frontier_unit_tests = dataset_root(cfg, paths) / "frontiers" / "unit_tests"
    if frontier_unit_tests.exists():
        dst = workspace / cfg.trace_dataset_dirname / "frontiers" / "unit_tests"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(frontier_unit_tests, dst)

    (workspace / "agent_history").mkdir(parents=True, exist_ok=True)

    return workspace


def _run_one_final_integration_episode(
    cfg: PipelineConfig,
    paths: dict,
    analysis: dict,
    unit_results: dict[str, dict],
    flags: dict,
    episode_index: int,
) -> dict[str, Any]:
    canonical_test_dir = Path(paths["test_dir"]).resolve()
    repo_root = cfg.source_dir.parent.parent.resolve()

    eid = f"final_integration_{episode_index:06d}_{uuid.uuid4().hex[:8]}"
    workspace = _make_final_integration_workspace(cfg, paths, eid).resolve()

    episode_paths = _rebase_paths_for_workspace(
        paths,
        old_root=canonical_test_dir,
        new_root=workspace,
    )
    episode_unit_results = _rebase_unit_test_results_for_workspace(
        unit_results,
        old_root=canonical_test_dir,
        new_root=workspace,
    )

    name = container_mod.episode_container_name()

    print(
        f"[pipeline] starting final integration episode {episode_index} "
        f"name={name} workspace={workspace}",
        file=sys.stderr,
    )

    container_mod.create(
        cfg,
        name,
        host_test_dir=workspace,
        canonical_test_dir=canonical_test_dir,
        repo_root=repo_root,
    )

    run_cfg = replace(
        cfg,
        execution_mode="docker",
        container_name=name,
        per_episode_container=False,
        path_map=((str(workspace), str(canonical_test_dir)),),
    )

    try:
        ok = bool(
            integrate_all_unit_tests_sequential(
                run_cfg,
                episode_paths,
                analysis,
                episode_unit_results,
                flags,
            )
        )
        print(
            f"[pipeline] final integration episode "
            f"{'success' if ok else 'failed'} index={episode_index} workspace={workspace}",
            file=sys.stderr,
        )
        return {
            "ok": ok,
            "episode_index": episode_index,
            "eid": eid,
            "workspace": workspace,
        }
    except Exception as exc:
        print(
            f"[pipeline] final integration episode failed index={episode_index} "
            f"workspace={workspace}: {exc}",
            file=sys.stderr,
        )
        return {
            "ok": False,
            "episode_index": episode_index,
            "eid": eid,
            "workspace": workspace,
            "error": repr(exc),
        }
    finally:
        container_mod.teardown(name)
        print(
            f"[pipeline] final integration episode container removed name={name}",
            file=sys.stderr,
        )


def _promote_final_integration_workspace(paths: dict, workspace: Path) -> None:
    canonical_test_dir = Path(paths["test_dir"]).resolve()

    test_file = Path(paths["test_file"]).resolve()
    workspace_test_file = workspace / test_file.name
    if workspace_test_file.exists():
        shutil.copy2(workspace_test_file, test_file)

    workspace_makefile = workspace / "Makefile"
    canonical_makefile = canonical_test_dir / "Makefile"
    if workspace_makefile.exists():
        shutil.copy2(workspace_makefile, canonical_makefile)


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

    if not cfg.per_episode_container:
        integrate_all_unit_tests_sequential(cfg, paths, analysis, unit_results, flags)
        return

    if cfg.execution_mode != "docker":
        raise ValueError("--per-episode-container for integrate requires --execution-mode docker")
    if not cfg.container_image:
        raise ValueError("--container-image is required with --per-episode-container")

    episode_count = int(getattr(cfg, "episodes_per_item", 1) or 1)
    episode_concurrency = int(getattr(cfg, "episode_concurrency", 1) or 1)
    max_workers = max(1, min(episode_concurrency, episode_count))

    print(
        f"[pipeline] final integration running {episode_count} episodes "
        f"with concurrency={max_workers}",
        file=sys.stderr,
    )

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_one_final_integration_episode,
                cfg,
                paths,
                analysis,
                unit_results,
                flags,
                i,
            )
            for i in range(episode_count)
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    _score_workspace_results("integrate", cfg, paths, results)
    successes = [r for r in results if r.get("ok")]

    print(
        f"[pipeline] final integration episodes complete: "
        f"{len(successes)}/{len(results)} succeeded",
        file=sys.stderr,
    )

    if not successes:
        raise RuntimeError("all final integration episodes failed")

    best = sorted(successes, key=lambda r: r["episode_index"])[0]
    _promote_final_integration_workspace(paths, best["workspace"])

    print(
        f"[pipeline] promoted final integration workspace: {best['workspace']}",
        file=sys.stderr,
    )


def _stage_score(cfg: PipelineConfig, paths: dict) -> None:
    """Backfill base score.json over the dataset, then normalize within item groups."""
    root = dataset_root(cfg, paths)
    for meta_path in sorted(root.rglob("metadata.json")):
        episode_dir = meta_path.parent
        if (episode_dir / "score.json").exists():
            continue
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        stage = meta.get("stage")
        if not stage:
            continue
        # Prefer the history path recorded in metadata; fall back to layout guesses.
        hist = None
        if meta.get("agent_history"):
            hist = Path(meta["agent_history"])
        elif meta.get("workspace"):
            hist = Path(meta["workspace"]) / "agent_history"
        if hist is None or not hist.exists():
            hist = episode_dir / "agent_history"
        if not hist.exists():
            hist = episode_dir / "workspace" / "agent_history"
        try:
            scoring.write_episode_score(
                stage, episode_dir, meta,
                coverage_threshold=cfg.coverage_threshold,
                semantic_min=cfg.semantic_judge_min_score,
                agent_history=hist,
            )
        except Exception as exc:
            print(f"[pipeline] WARN score backfill {episode_dir}: {exc}", file=sys.stderr)
    scoring.finalize_scores(root)


def _stage_build_dpo(cfg: PipelineConfig, paths: dict) -> None:
    root = dataset_root(cfg, paths)
    scoring.finalize_scores(root)
    scoring.build_dpo_pairs(root)


_STAGE_HANDLERS = {
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
    "score": _stage_score,
    "build-dpo": _stage_build_dpo,
}

# Full collection sequence run by --stage all-collect, in order.
COLLECTION_STAGE_ORDER = [
    "prepare",
    "collect-stubs",
    "select-stubs",
    "materialize-stubs",
    "integrate-stubs",
    "minimal-master",
    "collect-unit-tests",
    "select-unit-tests",
    "materialize-unit-tests",
    "integrate",
    "score",
]


def _stage_all_collect(cfg: PipelineConfig, paths: dict) -> None:
    """Run the whole collection pipeline stage-by-stage in one process.

    Each stage reads/writes the canonical test dir + _trace_dataset on disk, so
    chaining is just calling them in order. Per-episode container behavior is
    identical to running each stage individually.
    """
    for stage in COLLECTION_STAGE_ORDER:
        print(f"[pipeline] === all-collect: {stage} ===", file=sys.stderr)
        _write_stage_event(cfg, paths, stage)
        _STAGE_HANDLERS[stage](cfg, paths)


def run_selected_stage(cfg: PipelineConfig, paths: dict) -> None:
    dataset_root(cfg, paths).mkdir(parents=True, exist_ok=True)
    if cfg.stage == "all-collect":
        _stage_all_collect(cfg, paths)
        return
    handler = _STAGE_HANDLERS.get(cfg.stage)
    if handler is None:
        raise ValueError(f"Unsupported stage: {cfg.stage}")
    _write_stage_event(cfg, paths, cfg.stage)
    handler(cfg, paths)
