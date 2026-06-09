from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .config import PipelineConfig


def assert_no_forbidden_host_paths(
    cfg: PipelineConfig,
    text: str,
    label: str,
) -> None:
    """Fail fast if docker-mode text leaks a configured host-only prefix."""
    if getattr(cfg, "execution_mode", "local") != "docker":
        return
    for prefix in getattr(cfg, "forbidden_host_prefixes", ()) or ():
        if prefix and prefix in text:
            raise RuntimeError(
                f"Forbidden host path prefix leaked into {label}: {prefix}"
            )


def _command_text(cmd: list[str] | str, *, shell: bool) -> str:
    if isinstance(cmd, str):
        return cmd
    if shell:
        return " ".join(str(part) for part in cmd)
    return shlex.join(str(part) for part in cmd)


def run_command(
    cfg: PipelineConfig,
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command locally or inside an already-mounted docker container."""
    mode = getattr(cfg, "execution_mode", "local")
    env_updates = {str(k): str(v) for k, v in (env or {}).items()}

    if mode == "local":
        run_env = os.environ.copy()
        run_env.update(env_updates)
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            env=run_env,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    if mode != "docker":
        raise ValueError(f"Unsupported execution_mode: {mode}")

    container = getattr(cfg, "container_name", None)
    if not container:
        raise ValueError("--container-name is required when --execution-mode=docker")

    inner_cmd = _command_text(cmd, shell=shell)
    profile = shlex.quote(str(getattr(cfg, "container_profile")))
    cwd_text = shlex.quote(str(Path(cwd)))
    script = f"source {profile}; cd {cwd_text}; exec {inner_cmd}"

    docker_cmd = ["docker", "exec"]
    for key, value in env_updates.items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    docker_cmd.extend([str(container), "bash", "-lc", script])

    return subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
