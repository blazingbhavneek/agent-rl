"""Regression tests for the standalone judge-JSONL plot input pipeline."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLOTS_PATH = ROOT / "eval_plots.py"
spec = importlib.util.spec_from_file_location("eval_plots_under_test", PLOTS_PATH)
assert spec and spec.loader
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


class EvalPlotsTests(unittest.TestCase):
    def test_reads_setup_from_sibling_results_and_keeps_latest_valid_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "cloud"
            run_dir.mkdir()
            (run_dir / "results.jsonl").write_text(
                json.dumps({"task_id": "t1", "setup": "cloud_rag_copilot"}) + "\n",
                encoding="utf-8",
            )
            judge_path = run_dir / "judge.jsonl"
            judge_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "t1",
                                "setup": "cloud_rag_copilot",
                                "judge": {
                                    "task_correctness": 0.1,
                                    "reference_similarity": 0.2,
                                    "code_quality": 0.3,
                                    "overall": 0.17,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "t1",
                                "setup": "cloud_rag_copilot",
                                "judge": {
                                    "task_correctness": 0.8,
                                    "reference_similarity": 0.9,
                                    "code_quality": 0.7,
                                    "overall": 0.825,
                                },
                            }
                        ),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            setup = plots.setup_from_sibling_results(judge_path)
            scores = plots.latest_scores(judge_path, setup)
            self.assertEqual(setup, "cloud_rag_copilot")
            self.assertEqual(len(scores), 1)
            self.assertEqual(scores[0]["overall"], 0.825)
            summary = plots.build_summary({setup: scores})
            self.assertEqual(summary[setup]["mean_overall"], 0.825)
            self.assertEqual(summary[setup]["std_overall"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
