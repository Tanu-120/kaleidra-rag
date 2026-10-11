"""CLI benchmark: python scripts/run_eval.py [--n 30] [--k 5]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kaleidra import evals  # noqa: E402
from kaleidra.pipeline import Kaleidra  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--k", type=int, default=5)
a = ap.parse_args()

p = Kaleidra()
print("ingest:", p.ingest_dir())
b = evals.benchmark(p, a.n, a.k)
print(f"\n{'config':<18}{'hit':>6}{'mrr':>7}{'ndcg':>7}{'recall':>8}{'faith':>7}{'p50ms':>8}")
for r in b["results"]:
    print(f"{r['name']:<18}{r['hit']:>6}{r['mrr']:>7}{r['ndcg']:>7}{r['recall']:>8}{r['faithfulness']:>7}{r['latency_ms_p50']:>8}")
print("\nproduction pick:", b["selection"])
print("guardrails:", json.dumps(b["guardrails"], indent=2))
Path("eval_report.json").write_text(json.dumps(b, indent=2))
print("wrote eval_report.json")
