"""Episode scoring for weighted-SFT and DPO.

One score.json per episode (sibling to metadata.json). Two consumers:
  - weighted SFT: uses sft_weight scalar.
  - DPO: within-item preference pairs (correctness dominates, tokens tie-break).

An "episode" is a bag of agent traces (gen + every compile-fix + semantic-repair).
All traces land in the episode's agent_history/ dir; tokens are summed across all
of them. Token usage lives in the cline history json (per-turn usage); .result.json
carries only exit/elapsed/timed_out. Token extraction is schema-tolerant: it sums
any *token* field it finds, so it survives unknown work-PC trace layouts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from .common import load_json, write_json


# region tunables

# unit-test outcome weights: passed / coverage / semantic.
UNIT_W = (0.5, 0.3, 0.2)
# how hard token cost pulls sft_weight down (correctness still dominates).
LAMBDA = 0.2
# token keys to skip (budgets/limits, not usage). NOT "total" — total_tokens is
# real usage in many schemas; double-count is handled per-dict in _walk_tokens.
_TOKEN_SKIP = ("limit", "max", "window", "budget", "remaining", "left")
# keys that report a per-node total (taken as max vs summed parts, not added on top).
_TOTAL_KEYS = ("total_tokens", "totaltokens", "tokens_total")

# endregion


# region token aggregation

def _walk_tokens(node: Any) -> int:
    """Recursively sum usage-like *token* fields in a trace json.

    Per dict: take max(node_total, sum_of_parts) so a node reporting both
    total_tokens and its components is not double counted. Also tries to parse
    JSON embedded in string values (some trace schemas stash usage there).
    """
    total = 0
    if isinstance(node, dict):
        node_total = 0
        parts = 0
        for k, v in node.items():
            kl = str(k).lower()
            is_tok = ("token" in kl and isinstance(v, (int, float))
                      and not isinstance(v, bool)
                      and not any(bad in kl for bad in _TOKEN_SKIP))
            if is_tok:
                if any(t in kl for t in _TOTAL_KEYS):
                    node_total = max(node_total, int(v))
                else:
                    parts += int(v)
            elif isinstance(v, str) and "token" in v.lower():
                try:
                    total += _walk_tokens(json.loads(v))
                except Exception:
                    pass
            else:
                total += _walk_tokens(v)
        total += max(node_total, parts)
    elif isinstance(node, list):
        for item in node:
            total += _walk_tokens(item)
    return total


def _classify(stem: str) -> str:
    # Anchor on underscore-delimited markers so a function named e.g. judge_packet
    # or repair_conn is not misread as a fix trace.
    s = "_" + stem.lower() + "_"
    if any(t in s for t in (
        "_stub_fix_", "_compile_fix_", "_integrate_fix_",
        "_pre_integration_fix_", "_master_fix_", "_integrate_cleanup_",
    )):
        return "compile_fix"
    if "_semantic_repair_" in s:
        return "semantic_repair"
    return "gen"


def _iter_traces(agent_history: Path) -> Iterable[tuple[str, Path, Path]]:
    """Yield (stem, main_trace, result_json) for each agent invocation.

    An invocation is identified by its <stem>.result.json marker (written by
    run_agent); the matching <stem>.json holds the per-turn token usage.
    """
    if not agent_history.exists():
        return
    for result_json in sorted(agent_history.rglob("*.result.json")):
        stem = result_json.name[: -len(".result.json")]
        main = result_json.with_name(stem + ".json")
        yield stem, main, result_json


def aggregate_tokens(agent_history: Path) -> dict:
    """Sum tokens across all agent traces in one episode."""
    by_kind = {"gen": 0, "compile_fix": 0, "semantic_repair": 0}
    fix_loops = 0
    elapsed = 0.0
    timed_out = False
    traces_seen = 0
    for stem, main, result_json in _iter_traces(agent_history):
        traces_seen += 1
        kind = _classify(stem)
        if kind != "gen":
            fix_loops += 1
        if main.exists():
            try:
                by_kind[kind] += _walk_tokens(load_json(main))
            except Exception:
                pass
        try:
            res = load_json(result_json)
            elapsed += float(res.get("elapsed") or 0.0)
            timed_out = timed_out or bool(res.get("timed_out"))
        except Exception:
            pass
    tokens_total = sum(by_kind.values())
    # Traces existed but no tokens extracted => extraction failed (schema drift).
    # Flag it so finalize does not silently reward it with max sft_weight.
    extraction_failed = traces_seen > 0 and tokens_total == 0
    return {
        "tokens_total": tokens_total,
        "tokens_by_kind": by_kind,
        "fix_loops": fix_loops,
        "traces_seen": traces_seen,
        "token_extraction_failed": extraction_failed,
        "elapsed": round(elapsed, 2),
        "timed_out": timed_out,
    }

# endregion


# region per-stage outcome

def _f(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def outcome_for(stage: str, metadata: dict, *, coverage_threshold: float = 70.0,
                semantic_min: float = 75.0) -> dict:
    """Deterministic outcome score in [0,1] from metadata."""
    result = metadata.get("result") if isinstance(metadata.get("result"), dict) else {}

    if stage in ("collect-stubs", "stubs"):
        validated = bool(metadata.get("validated") or result.get("validated"))
        return {"score": 1.0 if validated else 0.0, "passed": validated,
                "components": {"validated": validated}}

    if stage in ("collect-unit-tests", "unit-tests"):
        passed = bool(metadata.get("passed") or result.get("passed"))
        cov = _f(metadata.get("coverage_pct") if metadata.get("coverage_pct") is not None
                 else result.get("coverage_pct")) or 0.0
        sem = _f(metadata.get("semantic_score") if metadata.get("semantic_score") is not None
                 else result.get("semantic_score")) or 0.0
        score = (UNIT_W[0] * (1.0 if passed else 0.0)
                 + UNIT_W[1] * (cov / 100.0)
                 + UNIT_W[2] * (sem / 100.0))
        gate = passed and cov >= coverage_threshold and sem >= semantic_min
        # Hard-cap explicit failures (cap-out / error) so a high-coverage-but-failed
        # episode can never rival a real pass. Derived from result, not a phantom key.
        failed = bool(result.get("error")) or bool(metadata.get("unit_test_failed"))
        if not gate and failed:
            score = min(score, 0.2)
        return {"score": round(score, 4), "passed": passed,
                "components": {"passed": passed, "coverage_pct": cov,
                               "semantic_score": sem, "gate": gate}}

    # minimal-master, integrate, integrate-stubs: binary ok.
    ok = bool(metadata.get("ok") or result.get("ok"))
    return {"score": 1.0 if ok else 0.0, "passed": ok, "components": {"ok": ok}}


def _hit_cap(metadata: dict) -> bool:
    """True when the episode exhausted its fix-attempt budget (clean hard-negative)."""
    if metadata.get("hit_cap"):
        return True
    result = metadata.get("result") if isinstance(metadata.get("result"), dict) else {}
    return str(result.get("error") or "").startswith("max attempts reached")


def _item_id(stage: str, metadata: dict) -> str:
    for key in ("safe_id", "safe_name", "func_id"):
        v = metadata.get(key)
        if v:
            return str(v)
    return str(metadata.get("process_name") or "unknown")


def _quality_bucket(stage: str, outcome: dict) -> str:
    c = outcome.get("components", {})
    if stage in ("collect-unit-tests", "unit-tests"):
        cov_band = int((_f(c.get("coverage_pct")) or 0.0) // 10)
        sem_band = int((_f(c.get("semantic_score")) or 0.0) // 10)
        return f"{int(bool(c.get('passed')))}.{cov_band}.{sem_band}"
    return str(int(bool(outcome.get("passed"))))

# endregion


# region base score (inline, per episode)

def write_episode_score(stage: str, episode_dir: Path, metadata: dict, *,
                        coverage_threshold: float = 70.0,
                        semantic_min: float = 75.0,
                        agent_history: Optional[Path] = None) -> dict:
    """Compute + write base score.json for one episode (Tier 0 + Tier 1)."""
    hist = agent_history or (episode_dir / "agent_history")
    outcome = outcome_for(stage, metadata, coverage_threshold=coverage_threshold,
                          semantic_min=semantic_min)
    eff = aggregate_tokens(hist)
    eff["hit_cap"] = _hit_cap(metadata)
    score = {
        "episode_id": metadata.get("episode_id"),
        "stage": stage,
        "item_id": _item_id(stage, metadata),
        "outcome": outcome,
        "efficiency": eff,
        "norm": {"tokens_norm": None, "advantage": None},
        "sft_weight": None,
        "dpo": {"bucket": _quality_bucket(stage, outcome), "role": None, "pair_id": None},
        "tier": 1,
    }
    write_json(episode_dir / "score.json", score)
    return score

# endregion


# region finalize (within-item normalization)

def _group_key(score: dict) -> tuple[str, str]:
    return (str(score.get("stage")), str(score.get("item_id")))


def finalize_scores(dataset_root: Path) -> int:
    """Walk every score.json, normalize tokens + advantage within item group."""
    paths = sorted(dataset_root.rglob("score.json"))
    groups: dict[tuple[str, str], list[tuple[Path, dict]]] = {}
    for p in paths:
        try:
            s = load_json(p)
        except Exception:
            continue
        groups.setdefault(_group_key(s), []).append((p, s))

    token_fail = 0
    for members in groups.values():
        outs = [s["outcome"]["score"] for _, s in members]
        mean_out = sum(outs) / len(outs) if outs else 0.0
        toks = [s["efficiency"]["tokens_total"] for _, s in members
                if s["efficiency"]["tokens_total"] > 0]
        lo, hi = (min(toks), max(toks)) if toks else (0, 0)
        span = (hi - lo) or 1

        for p, s in members:
            eff = s["efficiency"]
            tt = eff["tokens_total"]
            if eff.get("token_extraction_failed"):
                # Can't measure cost -> treat as worst, never reward the gap.
                tnorm = 1.0
                token_fail += 1
            elif tt > 0 and hi > lo:
                tnorm = (tt - lo) / span
            else:
                tnorm = 0.0
            s["norm"]["tokens_norm"] = round(tnorm, 4)
            s["norm"]["advantage"] = round(s["outcome"]["score"] - mean_out, 4)
            # hit_cap is terminal failure; a mid-run timeout that still passed is a
            # recovered success and is NOT zeroed.
            passed = bool(s["outcome"].get("passed"))
            blocked = eff.get("hit_cap") or (eff.get("timed_out") and not passed)
            s["sft_weight"] = 0.0 if blocked else round(
                s["outcome"]["score"] * (1.0 - LAMBDA * tnorm), 4)
            s["tier"] = max(int(s.get("tier", 0)), 2)
            write_json(p, s)

    msg = f"[scoring] finalized {len(paths)} episodes in {len(groups)} item groups"
    if token_fail:
        msg += f" ; WARNING {token_fail} episodes had traces but 0 tokens (extraction failed)"
    print(msg, file=sys.stderr)
    return len(paths)

# endregion


# region DPO pair builder

def build_dpo_pairs(dataset_root: Path, out_path: Optional[Path] = None) -> Path:
    """Emit within-item preference pairs as JSONL.

    Two pair types:
      - accuracy:   higher outcome bucket = chosen (teaches correctness).
      - efficiency: same bucket, fewer tokens = chosen, quality within EPS
                    (teaches brevity, correctness held fixed).
    Pairs reference episode dirs + prompt files; raw chosen/rejected completion
    text is extracted downstream (one plug point: depends on trace schema).
    """
    out_path = out_path or (dataset_root / "dpo_pairs.jsonl")
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in sorted(dataset_root.rglob("score.json")):
        try:
            s = load_json(p)
        except Exception:
            continue
        s["_dir"] = str(p.parent)
        groups.setdefault(_group_key(s), []).append(s)

    pairs: list[dict] = []
    for (stage, item), members in groups.items():
        if len(members) < 2:
            continue
        ranked = sorted(members, key=lambda s: s["outcome"]["score"], reverse=True)

        # accuracy pairs: best vs each strictly-worse-outcome episode.
        best = ranked[0]
        for other in ranked[1:]:
            if other["outcome"]["score"] < best["outcome"]["score"] - 1e-9:
                pairs.append(_pair(stage, item, "accuracy", best, other))

        # efficiency pairs: within same dpo bucket, cheapest vs costlier.
        # Only within a PASSING bucket — never teach "cheap failure" over costly
        # failure. Same bucket already means equal cov/sem decile, so quality is
        # held fixed by construction (no extra epsilon check needed).
        by_bucket: dict[str, list[dict]] = {}
        for s in members:
            by_bucket.setdefault(s["dpo"]["bucket"], []).append(s)
        for bucket, bm in by_bucket.items():
            if not _is_passing_bucket(bucket):
                continue
            bm = [s for s in bm
                  if s["efficiency"]["tokens_total"] > 0
                  and not s["efficiency"].get("token_extraction_failed")]
            if len(bm) < 2:
                continue
            bm.sort(key=lambda s: s["efficiency"]["tokens_total"])
            cheap = bm[0]
            for costly in bm[1:]:
                if costly["efficiency"]["tokens_total"] > cheap["efficiency"]["tokens_total"]:
                    pairs.append(_pair(stage, item, "efficiency", cheap, costly))

    with out_path.open("w", encoding="utf-8") as f:
        for pr in pairs:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    print(f"[scoring] wrote {len(pairs)} DPO pairs -> {out_path}", file=sys.stderr)
    return out_path


def _is_passing_bucket(bucket: str) -> bool:
    # binary stages: "1". unit stages: "1.<cov>.<sem>".
    return bucket.split(".")[0] == "1"


def _prompt_ref(episode_dir: str) -> Optional[str]:
    """Prompt that seeded the episode — the gen trace, not a later compile-fix."""
    hist = Path(episode_dir) / "agent_history"
    if not hist.exists():
        hist = Path(episode_dir) / "workspace" / "agent_history"
    prompts = list(hist.rglob("*.prompt.txt")) if hist.exists() else []
    if not prompts:
        return None
    gen = [p for p in prompts if _classify(p.name[: -len(".prompt.txt")]) == "gen"]
    pool = gen or prompts
    return str(min(pool, key=lambda p: p.stat().st_mtime))


def _pair(stage: str, item: str, kind: str, chosen: dict, rejected: dict) -> dict:
    return {
        "stage": stage,
        "item_id": item,
        "pair_type": kind,
        "chosen": {
            "episode_dir": chosen["_dir"],
            "prompt_ref": _prompt_ref(chosen["_dir"]),
            "outcome": chosen["outcome"]["score"],
            "tokens": chosen["efficiency"]["tokens_total"],
        },
        "rejected": {
            "episode_dir": rejected["_dir"],
            "prompt_ref": _prompt_ref(rejected["_dir"]),
            "outcome": rejected["outcome"]["score"],
            "tokens": rejected["efficiency"]["tokens_total"],
        },
    }

# endregion
