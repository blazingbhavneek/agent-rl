from __future__ import annotations

import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .common import (
    _project_source_files,
    _read_json_loose,
    _safe_filename,
    build_output_with_runtime_diagnostics,
    ensure_wrap_flag,
    load_json,
    prompt_for_compile_fix,
    read_text,
    run_agent,
    run_make_test,
    sync_wrap_flags,
    write_json,
    write_text,
)


# region Doc matching

def _normalize_doc_name(s: str) -> str:
    s = Path(s).stem
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        _normalize_doc_name(a),
        _normalize_doc_name(b),
    ).ratio()


def find_func_doc(
    docs_dir: Path,
    func_name: str,
    threshold: float = 0.90,
) -> tuple[Optional[Path], float, list[tuple[float, Path]]]:
    """Find the best markdown doc for a function.

    Matching policy:
    1. Prefer exact `<func_name>.md`.
    2. Otherwise fuzzy-match against markdown filename stems.
    3. Accept only if best score >= threshold.

    Returns:
        (matched_path, score, top_candidates)

    matched_path is None when no acceptable match exists.
    """
    docs_dir = docs_dir.resolve()
    if not docs_dir.exists():
        return None, 0.0, []

    exact = docs_dir / f"{func_name}.md"
    if exact.exists():
        return exact, 1.0, [(1.0, exact)]

    scored: list[tuple[float, Path]] = []
    for p in docs_dir.glob("*.md"):
        score = similarity_ratio(func_name, p.stem)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None, 0.0, []

    best_score, best_path = scored[0]
    if best_score >= threshold:
        return best_path, best_score, scored[:5]
    return None, best_score, scored[:5]

# endregion Doc matching


# region Stub helpers

def insert_stub_into_test_file(test_file: Path, func_name: str, body: str) -> bool:
    """
    Insert `body` after the `/* === Linker Wrapper Stubs === */` marker.
    Idempotent: if __wrap_<name> already exists, do nothing.
    """
    text = read_text(test_file)
    marker = "/* === Linker Wrapper Stubs === */"
    if re.search(rf"__wrap_{re.escape(func_name)}\b", text):
        return False
    if marker not in text:
        text += f"\n{marker}\n"
    insertion = f"\n/* --- stub: {func_name} --- */\n{body.strip()}\n"
    text = text.replace(marker, marker + insertion, 1)
    write_text(test_file, text)
    return True


def stub_exists(test_file: Path, name: str) -> bool:
    text = read_text(test_file)
    return bool(re.search(rf"__wrap_{re.escape(name)}\b", text))

# endregion Stub helpers


# region Stub generation

def generate_stub_code(
    cfg: PipelineConfig,
    test_dir: Path,
    func_name: str,
) -> Optional[str]:
    """
    Generate or reuse one wrapper stub.

    Continuation behavior:
    - If _stub_gen/<func>/stub.c already exists and contains a valid
      __wrap_<func> implementation, reuse it.
    - Do not call the agent again for already-generated valid stub files.
    - If the cached stub is invalid, regenerate it.

    The agent must not edit source or final test file.
    It only writes _stub_gen/<func>/stub.c.
    """
    safe_name = _safe_filename(func_name)
    stub_dir = test_dir / "_stub_gen" / safe_name
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_out = stub_dir / "stub.c"

    def _clean_stub_body(raw: str) -> str:
        body = raw.strip()
        body = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n", "", body)
        body = re.sub(r"\n```\s*$", "", body).strip()
        return body

    def _is_valid_stub_body(body: str) -> bool:
        if not body:
            return False
        if f"__wrap_{func_name}" not in body:
            return False
        if f"__real_{func_name}" in body:
            return False
        return True

    result_file = stub_dir / "result.json"
    if stub_out.exists():
        cached_body = _clean_stub_body(read_text(stub_out))
        if _is_valid_stub_body(cached_body):
            already_validated = False
            if result_file.exists():
                try:
                    already_validated = load_json(result_file).get("validated", False)
                except Exception:
                    pass
            if already_validated:
                print(
                    f"[pipeline] reuse cached validated stub: {func_name}",
                    file=sys.stderr,
                )
                return cached_body
            print(
                f"[pipeline] stub exists unvalidated for {func_name}, validating",
                file=sys.stderr,
            )
            if not _validate_stub_locally(cfg, test_dir, func_name, stub_dir):
                print(
                    f"[pipeline] existing stub {func_name} failed validation, regenerating",
                    file=sys.stderr,
                )
                stub_out.unlink(missing_ok=True)
                result_file.unlink(missing_ok=True)
            else:
                return cached_body
        print(
            f"[pipeline] cached stub invalid for {func_name}, regenerating: {stub_out}",
            file=sys.stderr,
        )
        try:
            stub_out.unlink()
        except FileNotFoundError:
            pass

    repo_root = cfg.source_dir.parent.parent.resolve()
    docs_dir = cfg.func_docs_dir.resolve()
    func_doc, func_doc_score, func_doc_candidates = find_func_doc(
        docs_dir=docs_dir,
        func_name=func_name,
        threshold=0.90,
    )
    doc_candidates_text = "\n".join(
        f" score={score:.3f} path={path}"
        for score, path in func_doc_candidates
    ) or " (no markdown candidates found)"

    prompt = f"""You are generating ONE C linker-wrapper stub for unit testing.

TARGET FUNCTION
{func_name}

REQUIRED WRAPPER NAME
__wrap_{func_name}

REPOSITORY ROOT
{repo_root}

PRODUCTION SOURCE FOLDER
{cfg.source_dir}

FUNCTION DOCS DIRECTORY
{docs_dir}

MATCHED FUNCTION DOC FILE
{func_doc if func_doc else '(none matched with similarity >= 0.90)'}

MATCH SCORE
{func_doc_score:.3f}

TOP DOC CANDIDATES
{doc_candidates_text}

TEST FOLDER
{test_dir}

OUTPUT FILE
{stub_out}

TASK
Generate a deterministic fake/stub implementation for `{func_name}`.

Before writing the stub:
1. If a matched function doc file is listed above, read it first.
   Only trust it as the target docs if the match score is >= 0.90.
2. Search the production source folder for occurrences of `{func_name}`.
3. Read the call sites where production code uses `{func_name}`.
4. Read nearby headers/declarations to determine the correct signature.
5. Use docs + source occurrences to infer:
   - return type
   - output pointer behavior
   - callback registration/invocation behavior
   - required dummy values
   - whether NULL arguments are allowed

STRICT FILE RULES
- You may read anything under the repository.
- Do NOT modify production source files.
- Do NOT modify headers.
- Do NOT modify the final test file.
- Do NOT modify the Makefile.
- Write only this file:
  `{stub_out}`

STUB RULES
- Write raw C code only.
- No markdown fences.
- No explanations outside C comments.
- Generate exactly one wrapper function unless helper static state is required.
- The wrapper function must be named exactly:
  `__wrap_{func_name}`
- Include this log at the top of the wrapper:
  `fprintf(stderr, "__wrap_{func_name} called\\n");`
- This is a unit-test fake, not a pass-through wrapper.
- Do NOT call `__real_{func_name}`.
- Do NOT forward to the real implementation.
- Keep behavior deterministic and safe.
- If docs/source show success is `0`, return `0`.
- If docs/source show failure/success conventions, follow them.
- If output pointers exist, guard NULL and fill minimal safe values.
- If callbacks are registered, store callback pointers in static globals or invoke them with safe dummy values when appropriate.

When done, call submit_and_exit.
"""

    run_agent(
        cfg,
        work_dir=repo_root,
        folder=repo_root,
        history_dir=test_dir / "agent_history",
        prompt=prompt,
        history_name=f"_gen_stub_{safe_name}.json",
        max_iterations=cfg.max_agent_iterations,
        timeout_sec=cfg.agent_timeout_sec,
    )

    if not stub_out.exists():
        return None

    body = _clean_stub_body(read_text(stub_out))
    if not body:
        return None
    if f"__wrap_{func_name}" not in body:
        print(
            f"[pipeline] rejecting stub for {func_name}: missing __wrap_{func_name}",
            file=sys.stderr,
        )
        return None
    if f"__real_{func_name}" in body:
        print(
            f"[pipeline] rejecting stub for {func_name}: calls __real_{func_name}",
            file=sys.stderr,
        )
        return None

    _validate_stub_locally(cfg, test_dir, func_name, stub_dir)
    return body


def _validate_stub_locally(
    cfg: PipelineConfig,
    test_dir: Path,
    func_name: str,
    stub_dir: Path,
) -> bool:
    """
    Compile-validate and runtime-validate a stub.
    Loops with fresh micro-fix agent until both pass.
    Writes result.json {validated:true} on success.
    """
    stub_c = stub_dir / "stub.c"
    validate_main = stub_dir / "stub_validate_main.c"
    validate_bin = stub_dir / "stub_validate"
    result_file = stub_dir / "result.json"

    if result_file.exists():
        try:
            if load_json(result_file).get("validated"):
                return True
        except Exception:
            pass

    # The harness is intentionally shallow: a weak no-arg dummy satisfies the
    # linker so --wrap redirects the call to __wrap_<func>. Real argument-type
    # compatibility is checked in Stage 2 when the stub is linked against
    # production code that includes the real headers.
    harness = f"""#include <stdio.h>
/* Weak dummy — linker needs a real {func_name} to wrap; --wrap redirects calls to __wrap */
__attribute__((weak)) int {func_name}() {{ return 0; }}
int main(void) {{
    (void){func_name}();  /* exercised via --wrap_{func_name} */
    fprintf(stderr, "stub_validate OK\\n");
    return 0;
}}
"""
    write_text(validate_main, harness)

    repo_root = cfg.source_dir.parent.parent.resolve()
    context_file = test_dir / "_pipeline_context.json"
    flags: dict = {}
    if context_file.exists():
        try:
            flags = load_json(context_file).get("flags", {})
        except Exception:
            pass

    cflags = " ".join(filter(None, [
        flags.get("CFLAGS", ""),
        flags.get("CFLAGS_LINUX", ""),
        flags.get("CPPFLAGS", ""),
        flags.get("INCLUDE", ""),
    ]))

    attempt = 1
    while True:
        compile_res = subprocess.run(
            f"gcc -c {cflags} {stub_c} -o {stub_dir}/stub.o",
            shell=True, cwd=str(stub_dir),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60,
        )
        if compile_res.returncode != 0:
            err = (compile_res.stderr + compile_res.stdout)[:3000]
            print(f"[pipeline] stub compile error {func_name} attempt {attempt}: {err[:200]}", file=sys.stderr)
            _fix_stub_with_agent(cfg, stub_dir, func_name, err, repo_root, test_dir)
            attempt += 1
            continue

        link_res = subprocess.run(
            f"gcc {cflags} {stub_c} {validate_main} -Wl,--wrap={func_name} -o {validate_bin}",
            shell=True, cwd=str(stub_dir),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60,
        )
        if link_res.returncode != 0:
            err = (link_res.stderr + link_res.stdout)[:3000]
            print(f"[pipeline] stub link error {func_name} attempt {attempt}: {err[:200]}", file=sys.stderr)
            _fix_stub_with_agent(cfg, stub_dir, func_name, err, repo_root, test_dir)
            attempt += 1
            continue

        run_res = subprocess.run(
            [str(validate_bin)],
            cwd=str(stub_dir),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15,
        )
        if run_res.returncode != 0:
            err = f"runtime exit={run_res.returncode}\n{run_res.stderr}\n{run_res.stdout}"
            print(f"[pipeline] stub runtime error {func_name} attempt {attempt}: {err[:200]}", file=sys.stderr)
            _fix_stub_with_agent(cfg, stub_dir, func_name, err, repo_root, test_dir)
            attempt += 1
            continue

        write_json(result_file, {"validated": True, "func_name": func_name})
        print(f"[pipeline] stub validated: {func_name}", file=sys.stderr)
        return True


def _fix_stub_with_agent(
    cfg: PipelineConfig,
    stub_dir: Path,
    func_name: str,
    error: str,
    repo_root: Path,
    history_dir: Path,
) -> None:
    stub_c = stub_dir / "stub.c"
    prompt = f"""Fix the __wrap_{func_name} stub.

FILE: {stub_c}

ERROR:
{error[:3000]}

RULES:
- Edit ONLY {stub_c}
- Function must be named exactly __wrap_{func_name}
- Do NOT call __real_{func_name}
- Raw C only, no markdown
- Keep: fprintf(stderr, "__wrap_{func_name} called\\n");

When done, call submit_and_exit.
"""
    run_agent(
        cfg,
        work_dir=stub_dir,
        prompt=prompt,
        history_name=f"_stub_fix_{_safe_filename(func_name)}_{int(time.time())}.json",
        folder=repo_root,
        history_dir=history_dir / "agent_history",
        protect_source=True,
    )

# endregion Stub generation


# region Stub integration

def integrate_all_stubs_sequential(
    cfg: PipelineConfig,
    paths: dict,
    validated_bodies: dict[str, str],
    flags: dict,
) -> None:
    """
    Integrate validated stubs one at a time into master test file.
    After each insertion: make test on master. Loop compile-fix until pass.
    Master always passes make test after each stub.
    """
    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    test_dir: Path = paths["test_dir"]
    source_dir = cfg.source_dir.resolve()
    repo_root = cfg.source_dir.parent.parent.resolve()

    for func_name, body in validated_bodies.items():
        if stub_exists(test_file, func_name):
            print(f"[pipeline] stub already integrated: {func_name}", file=sys.stderr)
            continue

        print(f"[pipeline] integrating stub: {func_name}", file=sys.stderr)
        insert_stub_into_test_file(test_file, func_name, body)
        ensure_wrap_flag(makefile, func_name)

        attempt = 1
        while True:
            sync_wrap_flags(test_file, makefile)
            res = run_make_test(test_dir)
            if res["ok"]:
                print(f"[pipeline] master OK after stub: {func_name}", file=sys.stderr)
                break
            print(f"[pipeline] make test failed after stub {func_name} attempt {attempt}, fixing", file=sys.stderr)
            diag = build_output_with_runtime_diagnostics(test_dir, test_file, res)
            run_agent(
                cfg,
                test_dir,
                prompt_for_compile_fix(
                    str(makefile), str(test_file), diag,
                    source_dir=str(source_dir),
                    source_makefile=str(source_dir / "Makefile"),
                    actual_source_files=[str(p) for p in _project_source_files(cfg)],
                ),
                f"_stub_integrate_fix_{_safe_filename(func_name)}_{int(time.time())}.json",
                folder=repo_root,
            )
            attempt += 1

# endregion Stub integration


# region Main stub handler

def handle_stubs(cfg: PipelineConfig, paths: dict, analysis: dict) -> None:
    from .analysis import collect_stub_candidates

    test_file: Path = paths["test_file"]
    makefile: Path = paths["makefile"]
    test_dir: Path = paths["test_dir"]
    batch_size = max(1, int(cfg.stub_batch_size))

    candidates = collect_stub_candidates(analysis)
    print(f"[pipeline] {len(candidates)} stub candidates total", file=sys.stderr)

    def _validated_stub_body(name: str) -> Optional[str]:
        stub_dir = test_dir / "_stub_gen" / _safe_filename(name)
        stub_out = stub_dir / "stub.c"
        result_file = stub_dir / "result.json"
        if not stub_out.exists() or not result_file.exists():
            return None
        try:
            if not load_json(result_file).get("validated"):
                return None
        except Exception:
            return None
        body = read_text(stub_out).strip()
        if not body or f"__wrap_{name}" not in body or f"__real_{name}" in body:
            return None
        return body

    bodies: dict[str, str] = {}
    for n in candidates:
        cached = _validated_stub_body(n)
        if cached:
            bodies[n] = cached

    # Phase 1: generate stub bodies in parallel until every candidate has a
    # validated body. Each round runs up to 6 concurrent agent calls.
    # Already-cached stubs are skipped (loaded above).
    gen_round = 1
    while True:
        to_gen = [
            n for n in candidates
            if n not in bodies
        ]
        if not to_gen:
            break
        print(
            f"[pipeline] stub gen round {gen_round}: "
            f"{len(to_gen)} need bodies",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=min(6, len(to_gen))) as pool:
            futs = {pool.submit(generate_stub_code, cfg, test_dir, n): n for n in to_gen}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    body = fut.result()
                except Exception as e:
                    print(f"[pipeline] stub gen error {name}: {e}", file=sys.stderr)
                    body = None
                if body:
                    bodies[name] = body
                    print(f"[pipeline] body ready: {name} ({len(body)} chars)", file=sys.stderr)
                else:
                    print(f"[pipeline] NO body: {name} round {gen_round}", file=sys.stderr)
        gen_round += 1

    # Phase 2: integrate validated stubs one-at-a-time into the master test file.
    # make test is run after each stub is inserted; failures are fixed by agent.
    context_file = test_dir / "_pipeline_context.json"
    flags: dict = {}
    if context_file.exists():
        try:
            flags = load_json(context_file).get("flags", {})
        except Exception:
            pass
    validated = {n: bodies[n] for n in candidates if n in bodies}
    integrate_all_stubs_sequential(cfg, paths, validated, flags)

# endregion Main stub handler
