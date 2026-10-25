"""
SentinelForge — ml/evaluation package
Provides multi-metric fraud scoring, Gate 1 absolute benchmark gates,
Gate 2 Head-to-Head Regression Testing against production champions,
and automated markdown/JSON evaluation reports.
"""

from ml.evaluation.benchmark_eval import BenchmarkEvaluator
from ml.evaluation.metrics import FraudMetricsCalculator
from ml.evaluation.regression_eval import RegressionEvaluator

__all__ = ["FraudMetricsCalculator", "BenchmarkEvaluator", "RegressionEvaluator"]
