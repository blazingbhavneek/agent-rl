from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .config import PipelineConfig
from .execution import forwarded_env

"""
Pseudo per-episode container lifecycle using proot instead of Docker.

This preserves the original Docker semantics as closely as possible:

  Docker original:
    docker run -d --name NAME \
      -v repo_root:repo_root:ro \
      -v host_test_dir:canonical_test_dir \
      IMAGE sleep infinity

  PRoot replacement:
    - Extract IMAGE tar/tar.gz once into /tmp/pseudo_container_rootfs_cache/<hash>/rootfs
    - Create an episode metadata dir under /tmp/pseudo_containers/<name>
    - Store the intended binds:
        repo_root -> repo_root
        host_test_dir -> canonical_test_dir
    - execution.py later reads that metadata and runs:
        proot -r rootfs -b repo_root:repo_root -b host_test_dir:canonical_test_dir ...

Important:
  - We do NOT bind /home/seigyo from the host.
    The image rootfs already contains /home/seigyo/.bash_profile and /home/seigyo/MNG.
  - PRoot does not reliably enforce read-only binds the way Docker does.
    The repo_root bind is kept semantically, but not made truly read-only.
"""

_STATE_ROOT = Path("/tmp/pseudo_containers")
_CACHE_ROOT = Path("/tmp/pseudo_container_rootfs_cache")


def episode_container_name() -> str:
    return f"attempt_{int(time.time() * 1000)}_{time.monotonic_ns()}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _member_relpath(name: str) -> str | None:
    """
    Convert a tar member name to a safe relative POSIX path.
    Returns None if the path is unsafe or empty.
    """
    name = str(name).replace("\\", "/")
    name = name.lstrip("/")

    # Normalize ./a/../b forms.
    p = PurePosixPath(name)
    parts = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            return None
        parts.append(part)

    if not parts:
        return None

    return "/".join(parts)


def _target_path(root: Path, rel: str) -> Path:
    return root.joinpath(*PurePosixPath(rel).parts)


def _remove_path(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _apply_whiteout(root: Path, rel: str) -> bool:
    """
    Apply Docker overlay whiteout files.

    Returns True if this member was a whiteout and should not be extracted.
    """
    p = PurePosixPath(rel)
    name = p.name

    if name == ".wh..wh..opq":
        # Opaque directory marker: remove current contents of parent dir.
        parent_rel = str(p.parent)
        parent = root if parent_rel in ("", ".") else _target_path(root, parent_rel)
        if parent.exists() and parent.is_dir():
            for child in parent.iterdir():
                _remove_path(child)
        return True

    if name.startswith(".wh."):
        target_name = name[len(".wh.") :]
        parent_rel = str(p.parent)
        parent = root if parent_rel in ("", ".") else _target_path(root, parent_rel)
        _remove_path(parent / target_name)
        return True

    return False


def _ensure_dir_user_writable(path: Path) -> None:
    """
    Ensure a directory is writable/traversable by the current user.

    During Docker layer extraction, later layers may need to create/replace
    files under directories whose final image mode is restrictive.
    Since we extract as a non-root user, keep dirs owner-rwx during extraction.
    """
    try:
        if path.exists() and path.is_dir() and not path.is_symlink():
            st = path.stat()
            os.chmod(path, st.st_mode | 0o700)
    except FileNotFoundError:
        pass
    except PermissionError:
        pass


def _ensure_parent_dirs_writable(root: Path, target: Path) -> None:
    """
    Make all existing directories from root to target.parent owner-rwx.

    This prevents extraction failures when a previous layer made /etc, /var,
    etc. too restrictive for the extracting user.
    """
    try:
        root_resolved = root.resolve()
    except FileNotFoundError:
        root_resolved = root

    cur = root
    _ensure_dir_user_writable(cur)

    try:
        rel = target.parent.relative_to(root)
    except ValueError:
        return

    for part in rel.parts:
        cur = cur / part
        _ensure_dir_user_writable(cur)


def _prepare_target_for_replace(target: Path) -> None:
    """
    Prepare an existing path so tarfile can replace it as a non-root user.

    Python tarfile extracts regular files by opening target with "wb".
    If an earlier layer left that file mode 000/0400, open("wb") fails.
    Removing the old path first matches Docker layer replacement better.
    """
    try:
        if not target.exists() and not target.is_symlink():
            return

        if target.is_dir() and not target.is_symlink():
            # Do not remove directories for normal directory members.
            # File-vs-directory conflicts are handled below by _remove_path.
            return

        try:
            if not target.is_symlink():
                st = target.stat()
                os.chmod(target, st.st_mode | 0o600)
        except FileNotFoundError:
            return
        except PermissionError:
            pass

        target.unlink()
    except FileNotFoundError:
        pass


def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, root: Path) -> None:
    """
    Extract a tar member safely without chowning, while preserving executable bits.

    Important details:
      - Docker layers may overwrite protected files like /etc/gshadow.
      - We extract as a normal user, not root.
      - Therefore existing files from earlier layers must be removable/writable
        before later layers replace them.
      - We still restore member.mode after extraction so /usr/bin/env, bash, as,
        etc. stay executable.
    """
    rel = _member_relpath(member.name)
    if rel is None:
        return

    if _apply_whiteout(root, rel):
        return

    member.name = rel

    target = _target_path(root, rel)

    # Device files cannot be created as an ordinary user and are not needed here.
    if member.ischr() or member.isblk() or member.isfifo():
        return

    # Ensure the containing directories are usable during extraction.
    target.parent.mkdir(parents=True, exist_ok=True)
    _ensure_parent_dirs_writable(root, target)

    # Docker layer conflict handling.
    #
    # If the existing target is a file/symlink and this layer writes a file/link,
    # remove the old path first. This avoids PermissionError for paths such as:
    #   /etc/gshadow
    # that may have been chmodded to a restrictive mode by an earlier layer.
    if member.isfile() or member.islnk() or member.issym():
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                _remove_path(target)
            else:
                _prepare_target_for_replace(target)

    # If this layer wants a directory but a file/symlink already exists there,
    # remove the file/symlink.
    if member.isdir():
        if target.exists() or target.is_symlink():
            if not target.is_dir() or target.is_symlink():
                _remove_path(target)

    try:
        tar.extract(member, path=root, set_attrs=False)
    except FileExistsError:
        pass
    except PermissionError:
        # One more recovery attempt:
        # make the target and parent writable/removable, then retry once.
        _ensure_parent_dirs_writable(root, target)
        if target.exists() or target.is_symlink():
            if target.is_dir() and not member.isdir():
                _remove_path(target)
            elif not target.is_dir() or target.is_symlink():
                _prepare_target_for_replace(target)

        tar.extract(member, path=root, set_attrs=False)

    # Restore final mode from the Docker layer.
    #
    # This is required. Without it, executables can become non-executable and
    # proot later fails with:
    #   proot error: '/usr/bin/env' is not executable
    try:
        if member.isfile():
            os.chmod(target, member.mode)
        elif member.isdir():
            # Keep directories owner-traversable during the rest of extraction.
            os.chmod(target, member.mode | 0o700)
        # Do not chmod symlinks.
    except FileNotFoundError:
        pass
    except PermissionError:
        pass

def _extract_tar_into_rootfs(layer_tar: tarfile.TarFile, rootfs: Path) -> None:
    for member in layer_tar:
        _safe_extract_member(layer_tar, member, rootfs)


def _read_json_member(tar: tarfile.TarFile, name: str) -> Any:
    f = tar.extractfile(name)
    if f is None:
        return None
    with f:
        return json.loads(f.read().decode("utf-8", errors="replace"))


def _parse_env_list(env_list: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in env_list or []:
        if "=" in item:
            k, v = item.split("=", 1)
            env[str(k)] = str(v)
    return env


def _extract_docker_save_image(image_path: Path, rootfs: Path) -> dict[str, str]:
    """
    Extract a docker-save tar/tar.gz image into rootfs.

    Expected format:
      manifest.json
      <config>.json
      <layer>/layer.tar
      ...
    """
    image_env: dict[str, str] = {}

    with tarfile.open(image_path, mode="r:*") as outer:
        names = set(outer.getnames())

        if "manifest.json" not in names:
            # Not a docker-save archive. Treat the file itself as a rootfs tar.
            outer.members = []
            outer.close()
            with tarfile.open(image_path, mode="r:*") as root_tar:
                _extract_tar_into_rootfs(root_tar, rootfs)
            return image_env

        manifest = _read_json_member(outer, "manifest.json")
        if not isinstance(manifest, list) or not manifest:
            raise RuntimeError(f"Invalid docker save manifest in {image_path}")

        entry = manifest[0]

        # Read image config ENV, if available.
        config_name = entry.get("Config")
        if config_name and config_name in names:
            try:
                config = _read_json_member(outer, config_name)
                image_env = _parse_env_list(
                    ((config or {}).get("config") or {}).get("Env")
                )
            except Exception:
                image_env = {}

        layers = entry.get("Layers") or []
        if not isinstance(layers, list) or not layers:
            raise RuntimeError(f"No layers found in docker save image {image_path}")

        for layer_name in layers:
            if layer_name not in names:
                raise RuntimeError(f"Layer missing from image: {layer_name}")

            f = outer.extractfile(layer_name)
            if f is None:
                raise RuntimeError(f"Could not read layer: {layer_name}")

            with f:
                # Stream mode works for non-seekable ExFileObject.
                with tarfile.open(fileobj=f, mode="r|*") as layer_tar:
                    _extract_tar_into_rootfs(layer_tar, rootfs)

    return image_env


def _normalize_rootfs_after_extract(rootfs: Path) -> None:
    """
    Final pass:
      - ensure all dirs are owner-traversable
      - do not blindly chmod files; file modes came from tar members
    """
    for dirpath, dirnames, filenames in os.walk(rootfs):
        p = Path(dirpath)
        try:
            st = p.stat()
            os.chmod(p, st.st_mode | 0o700)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass


def _extract_image_cached(image: Path) -> tuple[Path, dict[str, str]]:
    """
    Extract image into a shared cache and return:
      (rootfs_path, image_env)

    Cache key is the image file SHA256, so stale cache is avoided.
    """
    if not image.exists():
        raise FileNotFoundError(f"container_image does not exist: {image}")

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    key = _sha256_file(image)
    cache_dir = _CACHE_ROOT / key
    rootfs = cache_dir / "rootfs"
    env_json = cache_dir / "image_env.json"
    lock_path = _CACHE_ROOT / f"{key}.lock"

    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        if rootfs.exists():
            image_env: dict[str, str] = {}
            if env_json.exists():
                try:
                    image_env = json.loads(env_json.read_text(encoding="utf-8"))
                except Exception:
                    image_env = {}
            return rootfs, image_env

        tmp = _CACHE_ROOT / f"{key}.tmp.{os.getpid()}.{time.monotonic_ns()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp_rootfs = tmp / "rootfs"
        tmp_rootfs.mkdir(parents=True, exist_ok=True)

        try:
            image_env = _extract_docker_save_image(image, tmp_rootfs)
            _normalize_rootfs_after_extract(tmp_rootfs)

            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / "image_env.json").write_text(
                json.dumps(image_env, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            os.rename(tmp, cache_dir)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        return rootfs, image_env


def _parse_extra_volume_arg(value: str) -> tuple[str, str] | None:
    """
    Parse Docker-style -v/--volume value:
      host:container
      host:container:ro
      host:container:rw

    PRoot ignores ro/rw enforcement.
    """
    parts = value.split(":")
    if len(parts) < 2:
        return None
    src = parts[0]
    dst = parts[1]
    if not src or not dst:
        return None
    return src, dst


def _parse_extra_mount_arg(value: str) -> tuple[str, str] | None:
    """
    Parse simple Docker --mount syntax:
      type=bind,source=/host,target=/container
      type=bind,src=/host,dst=/container
    """
    fields: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        fields[k.strip()] = v.strip()

    if fields.get("type") not in (None, "bind"):
        return None

    src = fields.get("source") or fields.get("src")
    dst = fields.get("target") or fields.get("destination") or fields.get("dst")
    if src and dst:
        return src, dst
    return None


def _extract_extra_binds(args: tuple | list | None) -> list[tuple[str, str]]:
    """
    Best-effort support for volume args from cfg.container_run_args.

    Supported:
      -v /host:/container[:ro]
      --volume /host:/container[:ro]
      --mount type=bind,source=/host,target=/container

    Unsupported Docker-only args are ignored.
    """
    result: list[tuple[str, str]] = []
    items = [str(a) for a in (args or ())]
    i = 0

    while i < len(items):
        item = items[i]

        if item in ("-v", "--volume") and i + 1 < len(items):
            parsed = _parse_extra_volume_arg(items[i + 1])
            if parsed:
                result.append(parsed)
            i += 2
            continue

        if item.startswith("-v") and item != "-v":
            parsed = _parse_extra_volume_arg(item[2:])
            if parsed:
                result.append(parsed)
            i += 1
            continue

        if item.startswith("--volume="):
            parsed = _parse_extra_volume_arg(item.split("=", 1)[1])
            if parsed:
                result.append(parsed)
            i += 1
            continue

        if item == "--mount" and i + 1 < len(items):
            parsed = _parse_extra_mount_arg(items[i + 1])
            if parsed:
                result.append(parsed)
            i += 2
            continue

        if item.startswith("--mount="):
            parsed = _parse_extra_mount_arg(item.split("=", 1)[1])
            if parsed:
                result.append(parsed)
            i += 1
            continue

        i += 1

    return result


def create(
    cfg: PipelineConfig,
    name: str,
    *,
    host_test_dir: Path,
    canonical_test_dir: Path,
    repo_root: Path,
) -> None:
    """
    Create a pseudo-container state directory.

    This replaces:
      docker run -d --name NAME ...

    No process is kept alive. execution.py will use the stored metadata to run
    commands through proot.
    """
    if not cfg.container_image:
        raise ValueError("--container-image is required with --per-episode-container")

    image_path = Path(str(cfg.container_image)).expanduser().resolve()
    rootfs, image_env = _extract_image_cached(image_path)

    _STATE_ROOT.mkdir(parents=True, exist_ok=True)

    state_dir = _STATE_ROOT / name
    shutil.rmtree(state_dir, ignore_errors=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    repo = str(Path(repo_root).resolve())
    host_test = str(Path(host_test_dir).resolve())
    canonical_test = str(Path(canonical_test_dir).resolve())

    # Bind order matters:
    #   broad repo bind first
    #   narrow per-episode test-dir overlay second
    binds: list[dict[str, str]] = [
        {
            "src": repo,
            "dst": repo,
            "readonly": "true",
        },
        {
            "src": host_test,
            "dst": canonical_test,
            "readonly": "false",
        },
    ]

    for src, dst in _extract_extra_binds(cfg.container_run_args):
        binds.append(
            {
                "src": str(Path(src).expanduser()),
                "dst": str(dst),
                "readonly": "false",
            }
        )

    # Docker run forwarded env lived in the container environment.
    container_env = dict(image_env)
    container_env.update(forwarded_env())

    metadata = {
        "name": name,
        "created_at": time.time(),
        "image": str(image_path),
        "rootfs": str(rootfs),
        "image_env": image_env,
        "container_env": container_env,
        "binds": binds,
        "host_test_dir": host_test,
        "canonical_test_dir": canonical_test,
        "repo_root": repo,
    }

    (state_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"[pipeline] pseudo-container ready name={name} rootfs={rootfs}",
        file=sys.stderr,
    )


def teardown(name: str) -> None:
    """
    Remove pseudo-container state.

    The shared extracted rootfs cache is intentionally kept.
    """
    try:
        shutil.rmtree(_STATE_ROOT / name, ignore_errors=True)
    except Exception as exc:  # pragma: no cover - cleanup best effort
        print(
            f"[pipeline] WARN: pseudo-container teardown failed for {name}: {exc}",
            file=sys.stderr,
        )
