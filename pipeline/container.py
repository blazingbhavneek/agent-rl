from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .config import PipelineConfig

"""
Per-episode docker container lifecycle for parallel trace collection.

Each episode gets a fresh container with:
  - the immutable project root bind-mounted READ-ONLY at its own canonical path
    (shared by every container; source is never copied or mutated), and
  - a per-episode host test dir bind-mounted READ-WRITE over the canonical
    tests/<proc> path.

So every agent sees the identical canonical pristine filesystem; only the host
dir behind the tests mount differs, which is what makes the traces differ and
lets episodes run in parallel without colliding. The host episode dir lives
under _trace_dataset, but the mount remaps it to the canonical path, so the
agent never sees _trace_dataset. Path rewriting for commands/prompts is handled
by pipeline.execution.containerize_text via cfg.path_map.
"""


def episode_container_name() -> str:
    return f"attempt_{int(time.time() * 1000)}_{time.monotonic_ns()}"


def create(
    cfg: PipelineConfig,
    name: str,
    *,
    host_test_dir: Path,
    canonical_test_dir: Path,
    repo_root: Path,
) -> None:
    """Create and start a detached container named `name` with the mounts above.

    The container is kept alive with `sleep infinity` so run_command's
    `docker exec` can target it, mirroring the single-container mode.
    Extra mounts (e.g. /home/seigyo/rl) come from cfg.container_run_args.
    """
    if not cfg.container_image:
        raise ValueError("--container-image is required with --per-episode-container")
    repo = str(Path(repo_root))
    cmd = ["docker", "run", "-d", "--name", name]
    # Immutable source/project tree, read-only, at its canonical path.
    cmd += ["-v", f"{repo}:{repo}:ro"]
    # Per-episode writable test dir, overlaid at the canonical tests path.
    cmd += ["-v", f"{Path(host_test_dir)}:{Path(canonical_test_dir)}"]
    cmd += [str(a) for a in (cfg.container_run_args or ())]
    cmd += [str(cfg.container_image), "sleep", "infinity"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker run failed for {name}: {proc.stderr.strip() or proc.stdout.strip()}"
        )


def teardown(name: str) -> None:
    """Force-remove the container. Best-effort; never raises."""
    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # pragma: no cover - cleanup best effort
        print(f"[pipeline] WARN: container teardown failed for {name}: {exc}", file=sys.stderr)
