"""
LLM Explainability Framework (MADF - Multi-dimensional Attention-based Diagnostic Framework)

An innovative framework for explaining failures in Large Language Model evaluation
across NL2NL, NL2CODE, and CODE2NL tasks with multi-dimensional analysis.

Enhanced with:
- Systematic Pattern Library for failure pattern management
- Surrogate Model Integration for improved cross-attention approximation
- Multi-source data collection and validation
"""

__version__ = "2.0.0"
__author__ = "LLM Explainability Research Team"

# Core framework components
from llm_explainability_framework.core.failure_classifier import FailureClassifier
from llm_explainability_framework.core.root_cause_analyzer import RootCauseAnalyzer
from llm_explainability_framework.core.recommendation_engine import RecommendationEngine, StakeholderType
from llm_explainability_framework.core.explainability_engine import ExplainabilityEngine
from llm_explainability_framework.models.llm_wrapper import LLMWrapper, create_default_llm_wrapper
from llm_explainability_framework.utils.metrics import ExplainabilityMetrics, AttentionAnalyzer, SemanticSimilarity
from llm_explainability_framework.visualization.reporter import ExplainabilityReporter

# Enhanced components
from llm_explainability_framework.core.pattern import (
    PatternLibrarySystem, FailurePattern, PatternExample, CounterfactualFix,
    PatternSeverity, PatternSource
)
from llm_explainability_framework.core.surrogate import (
    SurrogateModelSelector, SurrogateModelIntegration, SelectionCriteria,
    SurrogateModel, TaskComplexity, DomainType, EnhancedMADFFramework
)

__all__ = [
    # Core framework
    "FailureClassifier",
    "RootCauseAnalyzer", 
    "RecommendationEngine",
    "ExplainabilityEngine",
    "LLMWrapper",
    "create_default_llm_wrapper",
    "ExplainabilityMetrics",
    "AttentionAnalyzer",
    "SemanticSimilarity",
    "ExplainabilityReporter",
    "StakeholderType",
    
    # Enhanced components - Pattern Library
    "PatternLibrarySystem",
    "FailurePattern",
    "PatternExample", 
    "CounterfactualFix",
    "PatternSeverity",
    "PatternSource",
    
    # Enhanced components - Surrogate Models
    "SurrogateModelSelector",
    "SurrogateModelIntegration",
    "SelectionCriteria",
    "SurrogateModel",
    "TaskComplexity",
    "DomainType",
    "EnhancedMADFFramework"
] 