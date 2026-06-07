from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .common import (
    _read_json_loose,
    _safe_filename,
    load_json,
    read_text,
    run_agent,
    write_json,
)


# region Semantic context

def _semantic_context_path(test_dir: Path) -> Path:
    return test_dir / "_leaf_to_root_semantic_context.json"


def _load_semantic_context(test_dir: Path) -> dict:
    path = _semantic_context_path(test_dir)
    if not path.exists():
        return {"functions": {}}

    data = _read_json_loose(path)
    if not isinstance(data, dict):
        return {"functions": {}}

    if "functions" not in data or not isinstance(data["functions"], dict):
        data["functions"] = {}

    return data


def _append_semantic_context(test_dir: Path, func: dict, verdict: dict) -> None:
    path = _semantic_context_path(test_dir)
    data = _load_semantic_context(test_dir)
    fid = func.get("id") or func.get("name") or "unknown"
    data["functions"][fid] = {
        "func": func,
        "verdict": verdict,
    }
    write_json(path, data)

# endregion Semantic context


# region Judge verdict helpers

def _normalize_simple_judge_verdict(
    verdict: dict,
    *,
    min_score: int,
) -> Optional[dict]:
    """
    Required judge output:
      {"score": 85, "reason": "..."}

    Pipeline-compatible returned object:
      {
        "parse_ok": True,
        "score": 85,
        "reason": "...",
        "summary": "...",
        "passed": True/False
      }
    """
    if not isinstance(verdict, dict):
        return None

    if "score" not in verdict:
        return None

    try:
        score = int(verdict.get("score"))
    except Exception:
        return None

    score = max(0, min(100, score))
    reason = str(verdict.get("reason") or "").strip()

    return {
        "parse_ok": True,
        "score": score,
        "reason": reason,
        "summary": reason,
        "passed": score >= min_score,
    }


def _extract_coverage_pct(coverage: dict) -> Optional[float]:
    if not isinstance(coverage, dict):
        return None

    for key in ["percent", "coverage", "pct", "line_percent"]:
        if key in coverage and coverage.get(key) is not None:
            try:
                return float(coverage.get(key))
            except Exception:
                pass

    return None

# endregion Judge verdict helpers


# region Semantic judge prompts

def prompt_for_semantic_test_judge(
    *,
    process_name: str,
    test_file: str,
    func: dict,
    coverage: dict,
    make_result: dict,
    semantic_context: dict,
    verdict_file: str,
    min_score: int,
) -> str:
    fid = func.get("id") or func.get("name") or "unknown_function"

    return f"""
You are a semantic unit-test judge for generated CUnit tests.

You are NOT fixing code in this step.

Judge whether the existing CUnit tests for the target function are meaningful.

Important:
- Do NOT automatically fail only because make_result says CUnit failed.
- If coverage is valid, decide whether the failing assertion is a meaningful
  production behavior check or just a bad/generated test.
- Fail smoke/coverage-only tests even if coverage is high.

Process:
{process_name}

Target function:
{fid}

Target function metadata:
{json.dumps(func, indent=2, default=str)}

Current coverage info:
{json.dumps(coverage, indent=2, default=str)}

Latest make/test result:
{json.dumps(make_result, indent=2, default=str)}

Existing leaf-to-root semantic context:
{json.dumps(semantic_context, indent=2, default=str)}

Test file:
{test_file}

Verdict output file:
{verdict_file}

Minimum acceptable semantic score:
{min_score}

Inspect:
1. target function implementation,
2. real headers/macros/globals,
3. callees/wrappers/mocks,
4. current CUnit tests,
5. parent/caller functions if useful.

Score high only if tests would fail when important production behavior is broken.

Evaluate:
- return values,
- state/global changes,
- struct field changes,
- timer/counter behavior,
- dependency call count,
- dependency arguments,
- payload structs passed to dependencies,
- error/cleanup behavior,
- boundary/null/error paths where relevant.

You MUST write ONLY valid JSON to:
{verdict_file}

Required JSON format, exactly two fields:
{{
 "score": 0,
 "reason": "short reason explaining semantic strength or weakness"
}}

Rules:
- No markdown.
- No code fence.
- No prose outside JSON.
- No extra fields.
- score must be integer 0 to 100.
- reason must be a string.
"""


def prompt_for_semantic_test_repair(
    *,
    process_name: str,
    test_file: str,
    func: dict,
    coverage: dict,
    judge_verdict: dict,
    semantic_context: dict,
) -> str:
    fid = func.get("id") or func.get("name") or "unknown_function"

    return f"""
You are editing an existing CUnit test file.

The tests already reached coverage threshold, but the semantic judge says
they are not meaningful enough.

Process:
{process_name}

Target function:
{fid}

Target function metadata:
{json.dumps(func, indent=2, default=str)}

Current coverage:
{json.dumps(coverage, indent=2, default=str)}

Semantic judge verdict:
{json.dumps(judge_verdict, indent=2, default=str)}

The judge returns only:
- score
- reason

Use judge_verdict["reason"] as the main repair instruction.

Existing leaf-to-root semantic context:
{json.dumps(semantic_context, indent=2, default=str)}

Test file to modify:
{test_file}

You must inspect the real source before editing:
- target function implementation,
- structs/macros/globals it uses,
- external dependencies it calls,
- callers/parents if useful,
- existing mocks/wrappers in the test file.

Do NOT add generic template tests.
Do NOT add coverage-only tests.
Do NOT guess field names or signatures.
Do NOT create fake project headers/types/macros/functions.

If compilation fails because of missing headers/types/macros, inspect real
production headers and Makefiles, then fix INCLUDE/CPPFLAGS/LDFLAGS/LIBS in
the unit Makefile.

Fix the tests so they semantically prove behavior.

Required repair behavior:
1. Address the judge reason.
2. Add or strengthen assertions so tests fail when production behavior is wrong.
3. Assert exact return values/state/mocks/arguments where possible.
4. Capture dependency arguments in wrappers when behavior depends on them.
5. If a struct pointer is passed to a dependency, copy it and assert important fields.
6. Add comments near the tests explaining:
   - what {fid} does,
   - why each scenario matters,
   - what parent/root functions can rely on from this function.
7. Extend reset_mocks/reset helpers for any new mock state.
8. Preserve existing meaningful tests and assertions.
9. Keep the test binary non-blocking and terminating.

Do not weaken tests just to compile.
If something does not compile, inspect headers/source and correct it.

When done, call submit_and_exit.
"""


def prompt_for_function_test_with_semantic_context(
    *,
    func: dict,
    coverage: dict,
    test_file: str,
    process_name: str,
    attempt: int,
    max_attempts: int,
    make_ok: bool,
    semantic_context: dict,
    last_judge_verdict: Optional[dict],
) -> str:
    fid = func.get("id") or func.get("name") or "unknown_function"

    return f"""
You are writing or repairing a real CUnit unit test file for production C code.

Process:
{process_name}

Target function:
{fid}

Target metadata:
{json.dumps(func, indent=2, default=str)}

Test file:
{test_file}

You must explore the source yourself before editing.
Inspect:
1. target implementation,
2. used headers,
3. structs/enums/macros/constants,
4. globals read/written,
5. callees,
6. callers/parent functions,
7. current test file,
8. current mocks/wrappers,
9. Makefile wrapping style.

Add meaningful tests for realistic use cases of {fid}.

The tests must include comments/explanations that describe:
- what {fid} is responsible for,
- which lower-level behavior it depends on,
- what each scenario proves,
- why parent/root functions can rely on this behavior later.

Do not create generic template tests.
Do not create smoke tests.
Do not merely execute lines for gcov.

A meaningful test should verify observable behavior:
- return value,
- global state change,
- struct field change,
- timer/counter transition,
- dependency call count,
- dependency argument values,
- payload passed to dependency,
- error log,
- cleanup,
- exit behavior,
- early return/no-op behavior.

Create separate tests for distinct source behaviors when present:
- normal path,
- boundary path,
- null/invalid path,
- external dependency failure,
- state transition,
- cleanup/error branch,
- early return/no-op.

Mocks/wrappers:
- capture call count,
- capture important arguments,
- allow configurable returns,
- capture payload structs with memcpy when needed,
- never block or call real hardware/IPC/mainloop.

reset_mocks/reset helper:
- reset all new mock state,
- reset production globals touched by tests,
- reset arrays/pointers/null-injection flags,
- do not rely on test order.

CUnit main:
- must return non-zero when CUnit has failures.

Do not delete meaningful tests.
Do not weaken assertions to pass build.
If something does not compile, inspect source/header definitions and fix it.

When done, call submit_and_exit.
"""

# endregion Semantic judge prompts


# region Semantic judge runner

def run_semantic_test_judge(
    cfg,
    *,
    test_dir: Path,
    repo_root: Path,
    process_name: str,
    test_file: Path,
    func: dict,
    coverage: dict,
    make_result: dict,
) -> dict:
    """
    Retry semantic judge until readable simple JSON is produced.

    JSON parse failure is judge infrastructure failure — does NOT become score=0
    and does NOT trigger unit-test regeneration.
    """
    min_score = int(getattr(cfg, "semantic_judge_min_score", 75))
    max_parse_retries = int(getattr(cfg, "semantic_judge_parse_retries", 10))

    fid = func.get("id") or func.get("name") or "unknown_function"
    safe_fid = _safe_filename(fid)

    verdict_file = test_dir / f"_semantic_judge_{safe_fid}.json"
    semantic_context = _load_semantic_context(test_dir)

    last_raw = ""

    for i in range(1, max_parse_retries + 1):
        try:
            verdict_file.unlink()
        except FileNotFoundError:
            pass

        retry_note = ""
        if i > 1:
            retry_note = f"""

PREVIOUS ATTEMPT FAILED TO PRODUCE PARSEABLE JSON.

Retry:
{i}/{max_parse_retries}

You must write exactly this shape:

{{"score": 75, "reason": "your reason"}}

No markdown.
No prose.
No code fence.
No extra fields.
"""

        prompt = prompt_for_semantic_test_judge(
            process_name=process_name,
            test_file=str(test_file),
            func=func,
            coverage=coverage or {},
            make_result=make_result or {},
            semantic_context=semantic_context,
            verdict_file=str(verdict_file),
            min_score=min_score,
        ) + retry_note

        run_agent(
            cfg,
            test_dir,
            prompt,
            f"{safe_fid}_semantic_judge_try_{i}.json",
            folder=repo_root,
        )

        verdict_raw = {}
        if verdict_file.exists():
            last_raw = verdict_file.read_text(errors="ignore")[-4000:]
            verdict_raw = _read_json_loose(verdict_file)

        verdict = _normalize_simple_judge_verdict(
            verdict_raw,
            min_score=min_score,
        )

        if verdict is not None:
            verdict["judge_attempts"] = i
            return verdict

        print(
            f"[pipeline] judge JSON parse failed for {fid}, retry {i}/{max_parse_retries}",
            file=sys.stderr,
        )

    raise RuntimeError(
        "Semantic judge failed to produce parseable JSON after "
        f"{max_parse_retries} attempts for {fid}. "
        "Not regenerating tests because this is judge infrastructure failure.\n"
        f"Last judge output:\n{last_raw}"
    )

# endregion Semantic judge runner


# region Best backup

def _backup_root_for_func(unit_dir: Path, func: dict) -> Path:
    fid = func.get("id") or func.get("name") or "unknown_function"
    return unit_dir / "agent_history" / "good_cunit_backups" / _safe_filename(fid)


def _best_backup_meta_path(unit_dir: Path, func: dict) -> Path:
    return _backup_root_for_func(unit_dir, func) / "_best.json"


def _load_best_backup_meta(unit_dir: Path, func: dict) -> dict:
    p = _best_backup_meta_path(unit_dir, func)
    if not p.exists():
        return {}
    return _read_json_loose(p)


def backup_good_cunit_if_best(
    *,
    unit_dir: Path,
    unit_test_file: Path,
    unit_makefile: Path,
    func: dict,
    coverage_pct: Optional[float],
    coverage: dict,
    make_result: dict,
    judge_verdict: dict,
    cfg,
) -> None:
    """
    Save best known CUnit test and Makefile.

    Backup condition:
    - coverage exists
    - coverage >= cfg.coverage_threshold
    - judge score >= previous best score
    """
    if coverage_pct is None:
        return

    threshold = float(getattr(cfg, "coverage_threshold", 100.0))
    coverage_pct = float(coverage_pct)

    if coverage_pct < threshold:
        return

    score = int(judge_verdict.get("score") or 0)

    root = _backup_root_for_func(unit_dir, func)
    root.mkdir(parents=True, exist_ok=True)

    best_meta = _load_best_backup_meta(unit_dir, func)
    old_best = int(best_meta.get("score") or -1)

    if score < old_best:
        return
    ts = int(time.time())

    fid = func.get("id") or func.get("name") or "unknown_function"

    backup_dir = root / f"score_{score}_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if unit_test_file.exists():
        shutil.copy2(unit_test_file, backup_dir / unit_test_file.name)

    if unit_makefile.exists():
        shutil.copy2(unit_makefile, backup_dir / unit_makefile.name)

    meta = {
        "function": fid,
        "timestamp": ts,
        "score": score,
        "reason": judge_verdict.get("reason", ""),
        "passed": bool(judge_verdict.get("passed")),
        "coverage_pct": coverage_pct,
        "coverage_threshold": threshold,
        "make_ok": bool(make_result.get("ok")),
        "backup_dir": str(backup_dir),
        "test_file": str(backup_dir / unit_test_file.name),
        "makefile": str(backup_dir / unit_makefile.name),
        "coverage": coverage or {},
    }

    write_json(backup_dir / "meta.json", meta)
    write_json(_best_backup_meta_path(unit_dir, func), meta)

    latest = root / "latest_best"
    latest.mkdir(parents=True, exist_ok=True)

    if unit_test_file.exists():
        shutil.copy2(unit_test_file, latest / unit_test_file.name)

    if unit_makefile.exists():
        shutil.copy2(unit_makefile, latest / unit_makefile.name)

    write_json(latest / "meta.json", meta)

    print(
        f"[pipeline] backed up best CUnit for {fid}: score={score} coverage={coverage_pct}%",
        file=sys.stderr,
    )


def judge_and_backup_if_covered(
    cfg,
    *,
    test_dir: Path,
    unit_dir: Path,
    repo_root: Path,
    process_name: str,
    unit_test_file: Path,
    unit_makefile: Path,
    func: dict,
    coverage: dict,
    coverage_pct: Optional[float],
    make_result: dict,
) -> dict:
    if coverage_pct is None:
        return {
            "covered": False,
            "accepted": False,
            "judge_verdict": None,
            "reason": "coverage_pct is None",
        }

    coverage_pct = float(coverage_pct)
    threshold = float(getattr(cfg, "coverage_threshold", 100.0))
    if coverage_pct < threshold:
        return {
            "covered": False,
            "accepted": False,
            "judge_verdict": None,
            "reason": f"coverage {coverage_pct}% below threshold {threshold}%",
        }

    judge_verdict = run_semantic_test_judge(
        cfg,
        test_dir=test_dir,
        repo_root=repo_root,
        process_name=process_name,
        test_file=unit_test_file,
        func=func,
        coverage=coverage or {},
        make_result=make_result or {},
    )

    backup_good_cunit_if_best(
        unit_dir=unit_dir,
        unit_test_file=unit_test_file,
        unit_makefile=unit_makefile,
        func=func,
        coverage_pct=coverage_pct,
        coverage=coverage or {},
        make_result=make_result or {},
        judge_verdict=judge_verdict,
        cfg=cfg,
    )

    fid = func.get("id") or func.get("name") or "unknown_function"

    if judge_verdict.get("passed"):
        print(
            f"[pipeline] judge PASSED: {fid} score={judge_verdict.get('score')}",
            file=sys.stderr,
        )
        _append_semantic_context(test_dir, func, judge_verdict)
        return {
            "covered": True,
            "accepted": True,
            "judge_verdict": judge_verdict,
            "reason": "semantic judge passed",
        }

    print(
        f"[pipeline] judge FAILED: {fid} score={judge_verdict.get('score')}",
        file=sys.stderr,
    )
    return {
        "covered": True,
        "accepted": False,
        "judge_verdict": judge_verdict,
        "reason": "semantic judge failed",
    }

# endregion Best backup
