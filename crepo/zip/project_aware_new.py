from __future__ import annotations

# =============================================================================
# REGION 01 - CONTEXT, IMPORTS, AND COMPATIBILITY PATCHING
# =============================================================================
#
# Why this file exists:
#
#   project_aware.py is the original working tracer. It already knows how to:
#     1. discover project files,
#     2. build call graph paths,
#     3. parse C context for each path,
#     4. ask the LLM for one answer,
#     5. turn that answer into the legacy CSV/stats outputs.
#
#   This file keeps that original flow and adds the new DPO/data-collection
#   behavior at the narrowest boundary: run_with_retry(...).
#
# What changes here:
#
#   Original:
#       make_llm_calls_for_function -> run_with_retry(llm_calls, args)
#
#   New:
#       make_llm_calls_for_function -> run_with_retry(...)
#       run_with_retry runs multiple independent attempts, writes attempt
#       artifacts, selects one answer, and returns that selected answer in the
#       same shape the original code expects.
#
# Practical editing note:
#
#   This file is intentionally split into large, named regions. If you need to
#   paste code into a chat UI from a work machine, paste one whole region at a
#   time so the surrounding context is still visible.
#

import contextlib
import hashlib
import inspect
import json
import os
import random
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path as _Path
from typing import Any

import project_aware as _base
from project_aware import *  # noqa: F401,F403 - re-export original public names.


DPO_ATTEMPTS_PER_PATH = 5
DPO_DATA_ROOT = _Path("./dpo_llm_data")
DPO_SUPPRESS_AGENT_STDOUT = True
DPO_FAILED_SCORE = -1_000_000

_ORIGINAL_LLM_CALLS = _base.llm_calls

_DPO_PATH_INDEX_BY_IDENTITY: dict[tuple[str, str, str, str], int] = {}
_DPO_NEXT_PATH_INDEX_BY_FUNCTION: defaultdict[tuple[str, str], int] = defaultdict(int)


@contextlib.contextmanager
def _patched_base_runtime():
    """Temporarily make project_aware.py use this module's narrow overrides."""

    mirrored_names = [
        "Path",
        "return_project_mapping",
        "Preprocess",
        "get_local_function_definitions",
        "extract_all_macros",
        "extract_includes",
        "identify_funs_to_trace",
        "orchestrate",
        "make_graph",
        "parseFiles",
        "save_dict_csv",
        "llm_calls",
        "run_with_retry",
    ]

    old_values = {name: getattr(_base, name) for name in mirrored_names}
    old_file = getattr(_base, "__file__", None)

    try:
        for name in mirrored_names:
            setattr(_base, name, globals()[name])
        _base.__file__ = globals().get("__file__", _base.__file__)
        yield
    finally:
        for name, value in old_values.items():
            setattr(_base, name, value)
        if old_file is not None:
            _base.__file__ = old_file


def trace_variable(project_path):
    """Run the original tracer with DPO attempt collection enabled."""

    with _patched_base_runtime():
        return _base.trace_variable(project_path)


# =============================================================================
# REGION 02 - DPO JSON, SCORING, SELECTION, AND RETRY BOUNDARY
# =============================================================================
#
# This region is the new behavior. It is deliberately independent from the old
# parsing/call-graph code. The only contract it must preserve is:
#
#   run_with_retry(func, args, timeout, retries) -> (answer_model, stats_model)
#
# That lets the original make_llm_calls_for_function continue to build Combined
# rows, CSVs, and token stats exactly as before.
#


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, _Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, BaseException):
        return {
            "exception_type": type(value).__name__,
            "message": str(value),
        }
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    return str(value)


def _write_json(path: _Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, ensure_ascii=False, indent=2)


def _safe_name(value: Any, max_len: int = 120) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", str(value)).strip("_")
    return value[:max_len] or "unknown"


def _hash_text(value: Any, length: int = 12) -> str:
    raw = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _coerce_args(args=()) -> tuple:
    if isinstance(args, tuple):
        return args
    if isinstance(args, list):
        return tuple(args)
    return (args,)


def _current_project_name() -> str:
    try:
        state = State()
        return str(state.get("PROJECT_NAME") or "unknown_project")
    except Exception:
        return "unknown_project"


def _next_path_index(process_name: str, function_name: str, path: str, context: str) -> int:
    path_hash = _hash_text(path, length=10)
    context_hash = _hash_text(context, length=10)
    identity = (process_name, function_name, path_hash, context_hash)

    if identity not in _DPO_PATH_INDEX_BY_IDENTITY:
        counter_key = (process_name, function_name)
        _DPO_NEXT_PATH_INDEX_BY_FUNCTION[counter_key] += 1
        _DPO_PATH_INDEX_BY_IDENTITY[identity] = _DPO_NEXT_PATH_INDEX_BY_FUNCTION[
            counter_key
        ]

    return _DPO_PATH_INDEX_BY_IDENTITY[identity]


def _job_metadata_from_args(args: tuple) -> dict[str, Any]:
    process_name = _current_project_name()
    function_name = str(args[1]) if len(args) > 1 else "unknown_function"
    argument_numbers = args[2] if len(args) > 2 else []
    context = str(args[3]) if len(args) > 3 else ""
    path = str(args[4]) if len(args) > 4 else ""
    get_upper = bool(args[5]) if len(args) > 5 else True
    path_index = _next_path_index(process_name, function_name, path, context)

    return {
        "process_name": process_name,
        "function_name": function_name,
        "argument_numbers": argument_numbers,
        "context": context,
        "path": path,
        "get_upper": get_upper,
        "path_index": path_index,
        "path_hash": _hash_text(path, length=8),
        "context_hash": _hash_text(context, length=8),
    }


def _job_dir(meta: dict[str, Any]) -> _Path:
    folder_name = (
        f"{_safe_name(meta['function_name'])}"
        f"__path_{int(meta['path_index']):04d}"
        f"__p_{meta['path_hash']}"
        f"__c_{meta['context_hash']}"
    )
    return DPO_DATA_ROOT / _safe_name(meta["process_name"]) / folder_name


def _answer_key(answer: Any) -> str:
    safe = _json_safe(answer)
    if isinstance(safe, dict):
        safe = {
            "output": safe.get("output"),
            "call_number": safe.get("call_number"),
        }
    return json.dumps(safe, sort_keys=True, ensure_ascii=False)


def _score_attempt(answer: Any, stats: Any) -> dict[str, Any]:
    answer_dict = _json_safe(answer)
    stats_dict = _json_safe(stats)
    reasons: list[str] = []
    score = 0

    output = ""
    call_number = None
    if isinstance(answer_dict, dict):
        output = str(answer_dict.get("output", ""))
        call_number = answer_dict.get("call_number")

    if answer is not None:
        score += 100
        reasons.append("+100 valid_answer")
    else:
        score -= 500
        reasons.append("-500 missing_answer")

    unresolved_count = output.upper().count("UNRESOLVED")
    if not output:
        score -= 200
        reasons.append("-200 empty_output")
    elif unresolved_count == 0:
        score += 50
        reasons.append("+50 no_unresolved")
    else:
        penalty = unresolved_count * 25
        score -= penalty
        reasons.append(f"-{penalty} unresolved_count={unresolved_count}")

    if output in {"READF", "WRITEF"}:
        score += 40
        reasons.append(f"+40 resolved_return_usage={output}")

    if call_number not in [None, "None", "NONE", "UNRESOLVED", ""]:
        score += 10
        reasons.append(f"+10 resolved_call_number={call_number}")

    if isinstance(stats_dict, dict):
        random_tool_calls = int(stats_dict.get("Random_tool_calls", 0) or 0)
        other_tool_errors = int(stats_dict.get("Other_tool_errors", 0) or 0)
        iterations = int(stats_dict.get("Iterations", 0) or 0)
        tokens = stats_dict.get("Tokens", {}) or {}
        total_tokens = 0
        if isinstance(tokens, dict):
            total_tokens = int(tokens.get("Total_tokens", 0) or 0)

        if random_tool_calls:
            penalty = random_tool_calls * 30
            score -= penalty
            reasons.append(f"-{penalty} random_tool_calls={random_tool_calls}")
        if other_tool_errors:
            penalty = other_tool_errors * 25
            score -= penalty
            reasons.append(f"-{penalty} other_tool_errors={other_tool_errors}")
        if iterations:
            penalty = max(0, iterations - 1) * 2
            score -= penalty
            reasons.append(f"-{penalty} iterations={iterations}")
        if total_tokens:
            penalty = total_tokens // 20000
            score -= penalty
            reasons.append(f"-{penalty} token_penalty={total_tokens}")

    return {
        "score": score,
        "reasons": reasons,
        "answer": answer_dict,
        "stats": stats_dict,
    }


def _select_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        attempt
        for attempt in attempts
        if attempt.get("answer") is not None and attempt.get("error") is None
    ]

    if not successful:
        return random.choice(attempts)

    counts = Counter(_answer_key(attempt["answer"]) for attempt in successful)
    most_common_key, most_common_count = counts.most_common(1)[0]

    if most_common_count == 1:
        return max(successful, key=lambda attempt: attempt["score"])

    matching = [
        attempt
        for attempt in successful
        if _answer_key(attempt["answer"]) == most_common_key
    ]
    return max(matching, key=lambda attempt: attempt["score"])


def _call_with_optional_history(func, args: tuple):
    signature = inspect.signature(func)
    if "collect_history" in signature.parameters:
        return func(*args, collect_history=True)
    return func(*args)


def _original_llm_calls_with_history(*args):
    """
    Run the original project_aware.llm_calls prompt/build logic, but capture
    llm2's returned message history for DPO training artifacts.

    The original function still expects:

        answer, stats = client.start_tool_chain(...)

    So the adapter returns only those two values to the original function while
    storing the third value locally for this wrapper.
    """

    from client.llm2 import OllamaClient as HistoryOllamaClient

    captured_history: dict[str, Any] = {"messages": []}

    class CapturingOllamaClient(HistoryOllamaClient):
        def start_tool_chain(self, prompt_data):
            result = super().start_tool_chain(prompt_data)

            if isinstance(result, tuple) and len(result) == 3:
                answer, stats, messages = result
            elif isinstance(result, tuple) and len(result) == 2:
                answer, stats = result
                messages = getattr(self, "messages", [])
            else:
                raise TypeError(
                    f"llm2 start_tool_chain returned unsupported result: {type(result)}"
                )

            captured_history["messages"] = messages
            return answer, stats

    old_client = _base.OllamaClient
    try:
        _base.OllamaClient = CapturingOllamaClient
        answer, stats = _ORIGINAL_LLM_CALLS(*args)
    finally:
        _base.OllamaClient = old_client

    return answer, stats, captured_history["messages"]


def llm_calls(
    project_structure: dict[str, str],
    function_name_to_traced,
    argument_numbers: list[int],
    intial_context: str,
    path: str,
    get_upper: bool = True,
    collect_history: bool = False,
):
    if collect_history:
        return _original_llm_calls_with_history(
            project_structure,
            function_name_to_traced,
            argument_numbers,
            intial_context,
            path,
            get_upper,
        )

    result = _ORIGINAL_LLM_CALLS(
        project_structure,
        function_name_to_traced,
        argument_numbers,
        intial_context,
        path,
        get_upper,
    )

    return result


def run_with_retry(func, args=(), timeout=180, retries=2):
    args = _coerce_args(args)
    meta = _job_metadata_from_args(args)
    job_dir = _job_dir(meta)
    attempts: list[dict[str, Any]] = []

    for attempt_no in range(1, DPO_ATTEMPTS_PER_PATH + 1):
        attempt_dir = job_dir / f"attempt_{attempt_no:02d}"
        history_path = attempt_dir / "history.json"
        answer_path = attempt_dir / "answer.json"
        score_path = attempt_dir / "score.json"
        error_path = attempt_dir / "error.json"

        try:
            if DPO_SUPPRESS_AGENT_STDOUT:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull):
                        with contextlib.redirect_stderr(devnull):
                            result = _call_with_optional_history(func, args)
            else:
                result = _call_with_optional_history(func, args)

            if isinstance(result, tuple) and len(result) == 3:
                answer, stats, history = result
            elif isinstance(result, tuple) and len(result) == 2:
                answer, stats = result
                history = []
            else:
                raise TypeError(f"llm call returned unsupported result: {type(result)}")

            score_data = _score_attempt(answer=answer, stats=stats)
            attempt_record = {
                **meta,
                "attempt_no": attempt_no,
                "score": score_data["score"],
                "answer": answer,
                "stats": stats,
                "history_path": str(history_path),
                "answer_path": str(answer_path),
                "score_path": str(score_path),
                "error": None,
            }

            _write_json(
                history_path,
                {
                    **meta,
                    "attempt_no": attempt_no,
                    "messages": history,
                    "answer": answer,
                    "stats": stats,
                },
            )
            _write_json(answer_path, {**meta, "attempt_no": attempt_no, "answer": answer})
            _write_json(score_path, {**attempt_record, **score_data})
            attempts.append(attempt_record)

        except Exception as exc:
            attempt_record = {
                **meta,
                "attempt_no": attempt_no,
                "score": DPO_FAILED_SCORE,
                "answer": None,
                "stats": None,
                "history_path": None,
                "answer_path": None,
                "score_path": None,
                "error_path": str(error_path),
                "error": str(exc),
            }
            _write_json(
                error_path,
                {
                    **meta,
                    "attempt_no": attempt_no,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            attempts.append(attempt_record)

    selected = _select_attempt(attempts)
    _write_json(
        job_dir / "selected.json",
        {
            **meta,
            "selection_policy": "majority_answer_else_highest_score",
            "selected_attempt": selected["attempt_no"],
            "selected_score": selected["score"],
            "selected_answer": selected["answer"],
            "selected_stats": selected["stats"],
            "attempts": attempts,
        },
    )

    if selected.get("answer") is None or selected.get("stats") is None:
        return None

    return selected["answer"], selected["stats"]
