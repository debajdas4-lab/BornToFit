"""CLI / CI gate:  python scripts/run_evaluation.py --min-score 0.95 --benchmark"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.evaluation.evaluator import benchmark_parallelism, run_evaluation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--min-score", type=float, default=0.95)
ap.add_argument("--benchmark", action="store_true")
a = ap.parse_args()
st = get_settings()
st.fixed_now = st.fixed_now or "2026-10-05T10:00"
rep = run_evaluation(st)
print(rep["by_module"].to_string(index=False))
print(rep["by_check"].to_string(index=False))
fails = rep["cases"][~rep["cases"]["passed"]]
if len(fails):
    print("\nFAILED CASES:\n", fails[["case", "failed_checks", "detail"]].to_string(index=False))
print(f"\nOverall score {rep['overall']}  |  pass rate {rep['pass_rate']}  ({rep['n']} cases)")
if a.benchmark:
    print("Parallelism benchmark:", benchmark_parallelism(st))
sys.exit(0 if rep["overall"] >= a.min_score else 1)
