"""
SentinelForge — scripts/run_regression_eval.py

PURPOSE:
    CLI runner to execute Gate 1 Benchmark and Gate 2 Head-to-Head Regression Evaluation
    comparing a candidate model against the active production champion.

USAGE:
    python scripts/run_regression_eval.py --candidate candidate_v2 --production baseline_v1
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np

from ml.evaluation.benchmark_eval import BenchmarkEvaluator
from ml.evaluation.eval_report import generate_evaluation_report
from ml.evaluation.metrics import FraudMetricsCalculator
from ml.evaluation.regression_eval import RegressionEvaluator
from ml.training.data_loader import DataLoader
from ml.training.feature_builder import FeatureBuilder
from shared.logging.logger import configure_logging, get_logger

configure_logging(environment="development", log_level="INFO")
logger = get_logger("scripts.run_regression_eval")


def run_evaluation(
    candidate_tag: str = "candidate_v2",
    production_tag: str = "baseline_v1",
    model_dir: str = "model_cache/lightgbm",
    data_path: str = "data/historical_transactions_100k.parquet",
    report_output: str = "reports/regression_evaluation_report.md",
) -> None:
    """
    Run Gate 1 and Gate 2 evaluation suites.
    """
    m_dir = Path(model_dir)
    cand_file = m_dir / f"{candidate_tag}.txt"
    prod_file = m_dir / f"{production_tag}.txt"

    if not cand_file.exists():
        raise FileNotFoundError(f"Candidate model weights not found at {cand_file}")
    if not prod_file.exists():
        raise FileNotFoundError(f"Production model weights not found at {prod_file}")

    logger.info("Loading models for comparison", candidate=str(cand_file), production=str(prod_file))
    cand_booster = lgb.Booster(model_file=str(cand_file))
    prod_booster = lgb.Booster(model_file=str(prod_file))

    # Load test split
    data_loader = DataLoader()
    df_raw = data_loader.load_historical_parquet(data_path)
    feature_builder = FeatureBuilder()
    df_features = feature_builder.build_features(df_raw)

    _, _, df_test = data_loader.split_stratified(
        df_features, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    X_test, y_test = feature_builder.prepare_xy(df_test)
    logger.info("Prepared test slice", test_samples=len(y_test), fraud_samples=int(np.sum(y_test == 1)))

    # Compute predictions
    cand_preds = cand_booster.predict(X_test)
    prod_preds = prod_booster.predict(X_test)

    # Benchmark latencies
    cand_latencies = FraudMetricsCalculator.benchmark_inference_latency(
        cand_booster.predict, X_test[:1], iterations=50
    )
    prod_latencies = FraudMetricsCalculator.benchmark_inference_latency(
        prod_booster.predict, X_test[:1], iterations=50
    )

    # 1. Gate 1 Benchmark Eval
    benchmark_evaluator = BenchmarkEvaluator()
    bench_result = benchmark_evaluator.evaluate(
        y_true=y_test,
        y_pred_probs=cand_preds,
        latency_p99_ms=cand_latencies["latency_p99_ms"],
    )

    # 2. Gate 2 Head-to-Head Regression Eval
    regression_evaluator = RegressionEvaluator()
    reg_result = regression_evaluator.compare_models(
        y_true=y_test,
        candidate_probs=cand_preds,
        production_probs=prod_preds,
        candidate_latency_p99_ms=cand_latencies["latency_p99_ms"],
        production_latency_p99_ms=prod_latencies["latency_p99_ms"],
    )

    # 3. Generate Report
    report_md = generate_evaluation_report(
        candidate_name=candidate_tag,
        production_name=production_tag,
        benchmark_res=bench_result,
        regression_res=reg_result,
        output_path=report_output,
    )

    logger.info("Audit Report Generated", report_path=report_output)
    print("\n" + "="*80)
    print(f"VERDICT: {'APPROVED FOR PROMOTION' if reg_result.is_promotable else 'PROMOTION BLOCKED'}")
    print(f"Candidate PR-AUC: {reg_result.candidate_metrics['pr_auc']} | Prod: {reg_result.production_metrics['pr_auc']} | Delta: {reg_result.deltas['delta_pr_auc']:+.4f}")
    print(f"Candidate Recall@1% FPR: {reg_result.candidate_metrics['recall_at_1_fpr']} | Prod: {reg_result.production_metrics['recall_at_1_fpr']} | Delta: {reg_result.deltas['delta_recall_at_1_fpr']:+.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelForge Model Regression Evaluation")
    parser.add_argument("--candidate", type=str, default="candidate_v2", help="Candidate model tag")
    parser.add_argument("--production", type=str, default="baseline_v1", help="Production model tag")
    parser.add_argument("--report", type=str, default="reports/regression_evaluation_report.md", help="Report output path")
    args = parser.parse_args()

    run_evaluation(
        candidate_tag=args.candidate,
        production_tag=args.production,
        report_output=args.report,
    )
