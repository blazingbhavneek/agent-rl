from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .config import PipelineConfig

# TODO: remove unnecessary ones

# region Forwarded environement variables

_FORWARDED_ENV_KEYS = (
    # "PATH",
    # "HOME",
    # "USER",
    # "LOGNAME",
    # "SHELL",
    # "TERM",
    # "TMPDIR",
    # "LANG",
    # "LC_ALL",
    # "LC_CTYPE",
    # "LD_LIBRARY_PATH",
    # "LIBRARY_PATH",
    # "CPATH",
    # "C_INCLUDE_PATH",
    # "CPLUS_INCLUDE_PATH",
    # "PROVIDER_ID",
    # "MODEL_NAME",
    # "OPENAI_BASE_URL",
    # "OPENAI_API_KEY",
    # "RAG_SERVICE_URL",
    # "MAX_WAIT_MS",
    # "IDLE_WAIT_MS",
    # "HEARTBEAT_MS",
    # "CAPTURE_RAW_HTTP_TRACE",
    # "DISABLE_STREAMING",
    # "CC",
    # "CXX",
    # "CUDA_HOME",
    # "CUDA_PATH",
    # "NVM_DIR",
    # "BUN_INSTALL",
    # "CONDA_EXE",
    # "CONDA_PREFIX",
    # "CONDA_DEFAULT_ENV",
    # "CONDA_PYTHON_EXE",
    # "CONDA_SHLVL",
    # "MAMBA_ROOT_PREFIX",
    # "PYTHONPATH",
    # "HF_HUB_ENABLE_HF_TRANSFER",
    # "PYTORCH_CUDA_ALLOC_CONF",
    # "VLLM_USE_FLASHINFER_SAMPLER",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
)

_FORWARDED_ENV_PREFIXES = (
    # "OPENAI_",
    # "RAG_",
    # "VLLM_",
    # "CUDA",
    # "CONDA",
    # "HF_",
    # "PYTHON",
    # "PIP_",
    # "UV_",
    # "NVM_",
    # "BUN_",
    "http_",
    "https_",
    # "no_",
    "HTTP_",
    "HTTPS_",
    # "NO_",
)

_MERGED_ENV_KEYS = (
    # "PATH",
    # "LD_LIBRARY_PATH",
    # "LIBRARY_PATH",
    # "CPATH",
    # "C_INCLUDE_PATH",
    # "CPLUS_INCLUDE_PATH",
)

# endregion Forwarded environement variables

# making sure the host paths are not seen by the container?
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

# Change host paths to container paths by changing prefixes, so that for container it looks like that we are in the container itself.
# TODO: We shouldn't need to do that? no matter how our host side paths look due to bind mount we should know its container side path already?
# The stages should be seperate enough at generatio time that we can move on manually do file management?
def containerize_text(cfg: PipelineConfig, text: str) -> str:
    """Rewrite host-side per-episode prefixes to canonical container paths.

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

# change each part of command as container friendly
def _containerize_cmd(cfg: PipelineConfig, cmd: list[str] | str) -> list[str] | str:
    if isinstance(cmd, str):
        return containerize_text(cfg, cmd)
    return [containerize_text(cfg, str(part)) for part in cmd]

# Seperate elements of a command (bin args etc)
def _command_text(cmd: list[str] | str, *, shell: bool) -> str:
    if isinstance(cmd, str):
        return cmd
    if shell:
        return " ".join(str(part) for part in cmd)
    return shlex.join(str(part) for part in cmd)

# Take environment variables from this environment and add it to the dictionary for forwarding to the environment. 
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
    """Return the env payload that should always follow docker execution."""
    return _with_forwarded_env(env)

# Build a docker exec command that prepares the container environment
# (merged env vars, profile sourcing, cwd) and then executes the target command.
def _docker_script(
    cfg: PipelineConfig,
    *,
    cwd: Path | str,
    cmd: list[str] | str,
    env_updates: dict[str, str],
    shell: bool,
) -> tuple[list[str], str]:
    # Require a target container since we'll be using `docker exec`.
    container = getattr(cfg, "container_name", None)
    if not container:
        raise ValueError("--container-name is required when --execution-mode=docker")

    # Translate host paths/commands into their container equivalents.
    cmd = _containerize_cmd(cfg, cmd)
    cwd = containerize_text(cfg, str(Path(cwd)))

    # Convert the command into shell-safe text for execution via bash -lc.
    inner_cmd = _command_text(cmd, shell=shell)
    cwd_text = shlex.quote(str(cwd))

    # Build shell setup commands and collect env vars that can be passed
    # directly through `docker exec -e`.
    prologue_lines: list[str] = []
    passthrough_env: dict[str, str] = {}

    for key, value in env_updates.items():
        if key in _MERGED_ENV_KEYS:
            # Merge with the container's existing value (e.g. PATH).
            quoted = shlex.quote(str(value))
            prologue_lines.append(f"export {key}={quoted}${{{key}:+:${key}}}")
        else:
            # Pass ordinary env vars via docker exec.
            passthrough_env[key] = value

    # Optionally source a shell profile before running the command.
    profile = getattr(cfg, "container_profile", None)
    if profile:
        profile_text = shlex.quote(str(profile))
        prologue_lines.append(f"source {profile_text}")

    # Run the command from the requested working directory.
    prologue_lines.append(f"cd {cwd_text}")

    # Build the shell script executed inside the container.
    prologue = "; ".join(prologue_lines)
    script = f"{prologue}; exec {inner_cmd}"

    # Build the final docker exec command.
    docker_cmd = [
        "docker",
        "exec",
        "-u",
        "seigyo",
        "-e",
        "HOME=/home/seigyo",
    ]

    # Forward non-merged environment variables.
    for key, value in passthrough_env.items():
        docker_cmd.extend(["-e", f"{key}={value}"])

    # Execute the generated script inside the running container.
    docker_cmd.extend([str(container), "bash", "-lc", script])

    return docker_cmd, script

# runs the command based on either the host (source code is being generated in same host) or docker (actual generation/checking is happening in seperate docker container)
import os
import shlex
import subprocess
from pathlib import Path


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
    env_updates = _with_forwarded_env(env)

    if mode == "local":
        run_env = os.environ.copy()
        run_env.update(env_updates)

        print(
            cmd if isinstance(cmd, str) else shlex.join(cmd)
        )

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

    docker_cmd, script = _docker_script(
        cfg,
        cwd=cwd,
        cmd=cmd,
        env_updates=env_updates,
        shell=shell,
    )

    print(docker_cmd)

    return subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )

# same as above but logs are visible to parent caller
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

        docker_cmd, _script = _docker_script(
            cfg,
            cwd=cwd,
            cmd=cmd,
            env_updates=env_updates,
            shell=shell,
        )

        proc = subprocess.run(
            docker_cmd,
            timeout=timeout,
        )
    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout="",
        stderr="",
    )

# same as above but this time we are saving logs to a file
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

            docker_cmd, _script = _docker_script(
                cfg,
                cwd=cwd,
                cmd=cmd,
                env_updates=env_updates,
                shell=shell,
            )

            proc = subprocess.run(
                docker_cmd,
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
