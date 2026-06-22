"""Create score datasets and plots from one or more eval judge JSONL files.

Example:
    python3 eval_plots.py \
      --judge-jsonl local/eval_runs_gpt/judge.jsonl \
      --judge-jsonl cloud/eval_runs_gpt/judge.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


SCORE_KEYS = (
    "task_correctness",
    "reference_similarity",
    "code_quality",
    "overall",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON {path} line={line_no}: {e}")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def setup_from_sibling_results(judge_path: Path) -> str:
    results_path = judge_path.parent / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Expected sibling results.jsonl for {judge_path}, but {results_path} does not exist"
        )
    setups = {
        row.get("setup")
        for row in read_jsonl(results_path)
        if isinstance(row.get("setup"), str) and row["setup"]
    }
    if len(setups) != 1:
        raise ValueError(
            f"{results_path} must identify exactly one setup, found {sorted(setups)}"
        )
    return next(iter(setups))


def valid_score_row(row: dict[str, Any]) -> dict[str, Any] | None:
    task_id = row.get("task_id")
    judge = row.get("judge")
    if not isinstance(task_id, str) or not isinstance(judge, dict):
        return None
    values: dict[str, float] = {}
    for key in SCORE_KEYS:
        value = judge.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not 0 <= value <= 1:
            return None
        values[key] = float(value)
    return {
        "task_id": task_id,
        "setup": row.get("setup"),
        "result_fingerprint": row.get("result_fingerprint"),
        **values,
    }


def latest_scores(judge_path: Path, setup: str) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(judge_path):
        score = valid_score_row(row)
        if score is None or row.get("setup") != setup:
            continue
        latest[score["task_id"]] = score
    return [latest[task_id] for task_id in sorted(latest)]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_score_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_plots(series: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [index / 20 for index in range(21)]
    for setup, rows in series.items():
        values = [row["overall"] for row in rows]
        if not values:
            continue
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.hist(values, bins=bins, weights=[1 / len(values)] * len(values), edgecolor="black")
        axis.set(
            title=f"Overall score distribution: {setup}",
            xlabel="Overall score",
            ylabel="Probability per bin",
            xlim=(0, 1),
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"overall_distribution_{safe_name(setup)}.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for setup, rows in series.items():
        values = [row["overall"] for row in rows]
        if values:
            axis.hist(
                values,
                bins=bins,
                weights=[1 / len(values)] * len(values),
                histtype="step",
                linewidth=2,
                label=setup,
            )
    axis.set(
        title="Overall score distributions",
        xlabel="Overall score",
        ylabel="Probability per bin",
        xlim=(0, 1),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overall_distribution_combined.png", dpi=160)
    plt.close(fig)

    labels = []
    means = []
    deviations = []
    for setup, rows in series.items():
        values = [row["overall"] for row in rows]
        if not values:
            continue
        labels.append(setup)
        means.append(statistics.fmean(values))
        deviations.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
    fig, axis = plt.subplots(figsize=(max(8, len(labels) * 2.5), 5))
    axis.bar(labels, means, yerr=deviations, capsize=5)
    axis.set(
        title="Mean overall score with population standard deviation",
        ylabel="Overall score",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_mean_std.png", dpi=160)
    plt.close(fig)


def build_summary(series: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for setup, rows in series.items():
        overall = [row["overall"] for row in rows]
        summary[setup] = {
            "count": len(rows),
            "mean_overall": statistics.fmean(overall) if overall else None,
            "std_overall": statistics.pstdev(overall) if len(overall) > 1 else 0.0 if overall else None,
            "mean_task_correctness": statistics.fmean(
                [row["task_correctness"] for row in rows]
            ) if rows else None,
            "mean_reference_similarity": statistics.fmean(
                [row["reference_similarity"] for row in rows]
            ) if rows else None,
            "mean_code_quality": statistics.fmean(
                [row["code_quality"] for row in rows]
            ) if rows else None,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot judge JSONL outputs.")
    parser.add_argument(
        "--judge-jsonl",
        action="append",
        required=True,
        type=Path,
        help="A judge.jsonl path; may be supplied multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval_plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series: dict[str, list[dict[str, Any]]] = {}
    for judge_path in args.judge_jsonl:
        setup = setup_from_sibling_results(judge_path)
        if setup in series:
            raise ValueError(f"Duplicate setup label {setup!r}; provide one run per setup")
        scores = latest_scores(judge_path, setup)
        series[setup] = scores
        score_path = args.output_dir / f"scores_{safe_name(setup)}.jsonl"
        write_score_jsonl(score_path, scores)
        print(f"[PLOTS] setup={setup} scores={len(scores)} data={score_path}")

    (args.output_dir / "summary.json").write_text(
        json.dumps(build_summary(series), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_plots(series, args.output_dir)


if __name__ == "__main__":
    main()
