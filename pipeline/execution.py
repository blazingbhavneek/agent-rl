from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from .config import PipelineConfig

# region Forwarded environment variables

_FORWARDED_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
)

_FORWARDED_ENV_PREFIXES = (
    "http_",
    "https_",
    "HTTP_",
    "HTTPS_",
)

_MERGED_ENV_KEYS = (
    # Keep empty to preserve your original behavior.
    # If you later re-enable PATH merging, it should be done in the shell prologue.
)

# endregion Forwarded environment variables

_STATE_ROOT = Path("/tmp/pseudo_containers")


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


def containerize_text(cfg: PipelineConfig, text: str) -> str:
    """
    Rewrite host-side per-episode prefixes to canonical container paths.

    No-op unless docker mode with a configured path_map. Longest prefixes are
    applied first so a host dir nested under the canonical root maps cleanly.
    """
    if getattr(cfg, "execution_mode", "local") != "docker":
        return text

    pairs = sorted(
        (getattr(cfg, "path_map", ()) or ()),
        key=lambda p: len(p[0]),
        reverse=True,
    )

    for host_prefix, container_prefix in pairs:
        if host_prefix:
            text = text.replace(host_prefix, container_prefix)

    return text


def _containerize_cmd(cfg: PipelineConfig, cmd: list[str] | str) -> list[str] | str:
    if isinstance(cmd, str):
        return containerize_text(cfg, cmd)
    return [containerize_text(cfg, str(part)) for part in cmd]


def _command_text(cmd: list[str] | str, *, shell: bool) -> str:
    if isinstance(cmd, str):
        return cmd
    if shell:
        return " ".join(str(part) for part in cmd)
    return shlex.join(str(part) for part in cmd)


def _with_forwarded_env(env: dict[str, str] | None) -> dict[str, str]:
    env_updates = {str(k): str(v) for k, v in (env or {}).items()}

    for key in _FORWARDED_ENV_KEYS:
        if key not in env_updates and key in os.environ:
            env_updates[key] = os.environ[key]

    for key, value in os.environ.items():
        if key in env_updates:
            continue
        if any(key.startswith(prefix) for prefix in _FORWARDED_ENV_PREFIXES):
            env_updates[key] = value

    return env_updates


def forwarded_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the env payload that should always follow docker/proot execution."""
    return _with_forwarded_env(env)


def _load_pseudo_container_metadata(container_name: str) -> dict:
    metadata_path = _STATE_ROOT / container_name / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"pseudo-container metadata not found for {container_name}: {metadata_path}"
        )

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"failed to read pseudo-container metadata for {container_name}: {exc}"
        ) from exc


def _build_runtime_env(
    metadata: dict,
    passthrough_env: dict[str, str],
) -> dict[str, str]:
    """
    Reconstruct the environment that Docker would have provided.

    Original Docker behavior:
      - image ENV exists in container
      - docker run forwarded env exists in container
      - docker exec adds HOME=/home/seigyo
      - docker exec adds passthrough env from run_command env/proxy forwarding

    We mimic that ordering.
    """
    runtime_env: dict[str, str] = {}

    image_env = metadata.get("image_env")
    if isinstance(image_env, dict):
        runtime_env.update({str(k): str(v) for k, v in image_env.items()})

    container_env = metadata.get("container_env")
    if isinstance(container_env, dict):
        runtime_env.update({str(k): str(v) for k, v in container_env.items()})

    # Original docker exec always passed HOME=/home/seigyo.
    runtime_env["HOME"] = "/home/seigyo"

    # Useful defaults for profile scripts. These do not override explicit env.
    runtime_env.setdefault("USER", "seigyo")
    runtime_env.setdefault("LOGNAME", "seigyo")
    runtime_env.setdefault("SHELL", "/bin/bash")

    # If the image config did not have PATH, provide a normal base path.
    runtime_env.setdefault(
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    )

    # docker exec -e values override container env.
    runtime_env.update({str(k): str(v) for k, v in passthrough_env.items()})

    return runtime_env


def _proot_script(
    cfg: PipelineConfig,
    *,
    cwd: Path | str,
    cmd: list[str] | str,
    env_updates: dict[str, str],
    shell: bool,
) -> tuple[list[str], str]:
    """
    Build a proot command that preserves the original _docker_script semantics.

    Original Docker:
      docker exec -u seigyo -e HOME=/home/seigyo ... CONTAINER bash -lc SCRIPT

    PRoot:
      proot -r rootfs -b mounts... /usr/bin/env KEY=VAL ... bash -lc SCRIPT
    """
    container = getattr(cfg, "container_name", None)
    if not container:
        raise ValueError("--container-name is required when --execution-mode=docker")

    metadata = _load_pseudo_container_metadata(str(container))

    rootfs = metadata.get("rootfs")
    if not rootfs:
        raise RuntimeError(f"pseudo-container {container} metadata has no rootfs")

    rootfs_path = Path(str(rootfs))
    if not rootfs_path.exists():
        raise RuntimeError(f"pseudo-container rootfs does not exist: {rootfs_path}")

    # Translate host paths/commands into container equivalents.
    cmd = _containerize_cmd(cfg, cmd)
    cwd = containerize_text(cfg, str(Path(cwd)))

    inner_cmd = _command_text(cmd, shell=shell)
    cwd_text = shlex.quote(str(cwd))

    prologue_lines: list[str] = []
    passthrough_env: dict[str, str] = {}

    for key, value in env_updates.items():
        if key in _MERGED_ENV_KEYS:
            quoted = shlex.quote(str(value))
            prologue_lines.append(f"export {key}={quoted}${{{key}:+:${key}}}")
        else:
            passthrough_env[key] = value

    # Preserve original behavior:
    #   if cfg.container_profile is set, source it inside bash -lc.
    #
    # Do not replace this with --noprofile/--norc.
    # Your do_mkmf environment depends on this profile.
    profile = getattr(cfg, "container_profile", None)
    if profile:
        profile_text = shlex.quote(str(profile))
        prologue_lines.append(f"source {profile_text}")

    prologue_lines.append(f"cd {cwd_text}")

    prologue = "; ".join(prologue_lines)
    script = f"{prologue}; exec {inner_cmd}"

    runtime_env = _build_runtime_env(metadata, passthrough_env)

    proot_cmd: list[str] = [
        "proot",
        "-r",
        str(rootfs_path),
    ]

    # Standard pseudo-filesystems.
    # Bind only if they exist on host.
    for p in ("/dev", "/proc", "/sys"):
        if Path(p).exists():
            proot_cmd.extend(["-b", p])

    # User-defined/container lifecycle binds.
    #
    # Order matters:
    #   repo root first
    #   narrower test-dir overlay second
    for bind in metadata.get("binds", []) or []:
        if not isinstance(bind, dict):
            continue

        src = bind.get("src")
        dst = bind.get("dst")
        if not src or not dst:
            continue

        src_path = Path(str(src))
        if not src_path.exists():
            raise RuntimeError(
                f"pseudo-container bind source does not exist: {src_path} -> {dst}"
            )

        proot_cmd.extend(["-b", f"{src_path}:{dst}"])

    # Match Docker behavior: docker exec did not set -w; script does cd.
    proot_cmd.extend(["-w", "/"])

    # Clear host env and reconstruct container-ish env.
    proot_cmd.append("/usr/bin/env")
    proot_cmd.append("-i")

    for key in sorted(runtime_env):
        proot_cmd.append(f"{key}={runtime_env[key]}")

    proot_cmd.extend(["bash", "-lc", script])

    return proot_cmd, script


def run_command(
    cfg: PipelineConfig,
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command locally or inside a pseudo-container via proot."""
    mode = getattr(cfg, "execution_mode", "local")
    env_updates = _with_forwarded_env(env)

    if mode == "local":
        run_env = os.environ.copy()
        run_env.update(env_updates)

        print(cmd if isinstance(cmd, str) else shlex.join(cmd))

        return subprocess.run(
            cmd,
            cwd=str(cwd),
            env=run_env,
            shell=shell,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )

    if mode != "docker":
        raise ValueError(f"Unsupported execution_mode: {mode}")

    proot_cmd, script = _proot_script(
        cfg,
        cwd=cwd,
        cmd=cmd,
        env_updates=env_updates,
        shell=shell,
    )

    print(proot_cmd)

    return subprocess.run(
        proot_cmd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def run_command_live(
    cfg: PipelineConfig,
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command and stream stdout/stderr live to the parent process."""
    mode = getattr(cfg, "execution_mode", "local")
    env_updates = _with_forwarded_env(env)

    if mode == "local":
        run_env = os.environ.copy()
        run_env.update(env_updates)

        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=run_env,
            shell=shell,
            timeout=timeout,
        )
    else:
        if mode != "docker":
            raise ValueError(f"Unsupported execution_mode: {mode}")

        proot_cmd, _script = _proot_script(
            cfg,
            cwd=cwd,
            cmd=cmd,
            env_updates=env_updates,
            shell=shell,
        )

        proc = subprocess.run(
            proot_cmd,
            timeout=timeout,
        )

    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout="",
        stderr="",
    )


def run_command_to_files(
    cfg: PipelineConfig,
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command and stream stdout/stderr directly to files."""
    mode = getattr(cfg, "execution_mode", "local")
    env_updates = _with_forwarded_env(env)

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w", encoding="utf-8", errors="replace") as out_f, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err_f:
        if mode == "local":
            run_env = os.environ.copy()
            run_env.update(env_updates)

            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=run_env,
                shell=shell,
                stdout=out_f,
                stderr=err_f,
                timeout=timeout,
                text=True,
            )
        else:
            if mode != "docker":
                raise ValueError(f"Unsupported execution_mode: {mode}")

            proot_cmd, _script = _proot_script(
                cfg,
                cwd=cwd,
                cmd=cmd,
                env_updates=env_updates,
                shell=shell,
            )

            print(proot_cmd)

            proc = subprocess.run(
                proot_cmd,
                stdout=out_f,
                stderr=err_f,
                timeout=timeout,
                text=True,
            )

    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout="",
        stderr="",
    )
