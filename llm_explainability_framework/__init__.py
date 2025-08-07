"""
LLM Explainability Framework

An innovative framework for explaining failures in Large Language Model evaluation
across NL2NL, NL2CODE, and CODE2NL tasks with multi-dimensional analysis.
"""

__version__ = "1.0.0"
__author__ = "LLM Explainability Research Team"

from llm_explainability_framework.core.failure_classifier import FailureClassifier
from llm_explainability_framework.core.root_cause_analyzer import RootCauseAnalyzer
from llm_explainability_framework.core.recommendation_engine import RecommendationEngine, StakeholderType
from llm_explainability_framework.core.explainability_engine import ExplainabilityEngine
from llm_explainability_framework.models.llm_wrapper import LLMWrapper, create_default_llm_wrapper
from llm_explainability_framework.utils.metrics import ExplainabilityMetrics
from llm_explainability_framework.visualization.reporter import ExplainabilityReporter

__all__ = [
    "FailureClassifier",
    "RootCauseAnalyzer", 
    "RecommendationEngine",
    "ExplainabilityEngine",
    "LLMWrapper",
    "create_default_llm_wrapper",
    "ExplainabilityMetrics",
    "ExplainabilityReporter",
    "StakeholderType"
] 