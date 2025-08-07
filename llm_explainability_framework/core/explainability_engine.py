"""
Explainability Engine - Main Orchestrator

This module implements the main explainability engine that orchestrates all components
of the LLM explainability framework to provide comprehensive failure analysis.
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .failure_classifier import FailureClassifier, FailureInstance, FailureClassification
from .root_cause_analyzer import RootCauseAnalyzer, RootCauseAnalysis
from .recommendation_engine import RecommendationEngine, RecommendationSuite, StakeholderType
from ..models.llm_wrapper import LLMWrapper
from ..utils.metrics import ExplainabilityMetrics


@dataclass
class ExplainabilityReport:
    """Complete explainability report for a failure instance"""
    instance_id: str
    timestamp: str
    task_type: str
    
    # Core analysis results
    failure_classification: FailureClassification
    root_cause_analysis: RootCauseAnalysis
    recommendation_suite: RecommendationSuite
    
    # Meta information
    processing_time: float
    confidence_score: float
    quality_metrics: Dict[str, float]
    
    # Generated report text
    markdown_report: str


class ExplainabilityEngine:
    """
    Main explainability engine that orchestrates all framework components
    
    Innovation: Unified pipeline that combines multiple AI techniques for
    comprehensive LLM failure analysis and explanation generation.
    """
    
    def __init__(self, llm_wrapper: LLMWrapper, config: Optional[Dict[str, Any]] = None):
        self.llm = llm_wrapper
        self.config = config or self._get_default_config()
        
        # Initialize core components
        self.failure_classifier = FailureClassifier(llm_wrapper)
        self.root_cause_analyzer = RootCauseAnalyzer(llm_wrapper)
        self.recommendation_engine = RecommendationEngine(llm_wrapper)
        self.metrics_evaluator = ExplainabilityMetrics()
        
        # Performance tracking
        self.analysis_history = []
        self.performance_stats = {
            'total_analyses': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'success_rate': 0.0
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the engine"""
        return {
            'enable_caching': True,
            'parallel_processing': False,
            'quality_threshold': 0.7,
            'max_processing_time': 300,  # 5 minutes
            'generate_visualizations': True,
            'stakeholder_optimization': True,
            'adaptive_learning': True
        }
    
    def analyze_failure(self, 
                       input_id: str,
                       task_type: str,
                       input_text: str,
                       model_output: str,
                       reference_output: str,
                       context_metadata: Optional[Dict[str, Any]] = None,
                       target_stakeholder: Optional[StakeholderType] = None) -> ExplainabilityReport:
        """
        Perform comprehensive failure analysis
        
        Args:
            input_id: Unique identifier for the input
            task_type: Type of task (NL2NL, NL2CODE, CODE2NL)
            input_text: Original input text
            model_output: Failed model output
            reference_output: Expected/correct output
            context_metadata: Optional context information
            target_stakeholder: Optional target stakeholder for optimization
            
        Returns:
            Complete explainability report
        """
        start_time = time.time()
        
        try:
            # Create failure instance
            instance = FailureInstance(
                input_id=input_id,
                task_type=task_type,
                input_text=input_text,
                model_output=model_output,
                reference_output=reference_output,
                context_metadata=context_metadata or {}
            )
            
            # Step 1: Classify failure
            print(f"🔍 Classifying failure for {input_id}...")
            classification = self.failure_classifier.classify(instance)
            
            # Step 2: Analyze root cause
            print(f"🧬 Analyzing root cause for {input_id}...")
            root_cause = self.root_cause_analyzer.analyze(instance, classification)
            
            # Step 3: Generate recommendations
            print(f"💡 Generating recommendations for {input_id}...")
            recommendations = self.recommendation_engine.generate_recommendations(
                instance, classification, root_cause, target_stakeholder
            )
            
            # Step 4: Generate markdown report
            print(f"📄 Generating report for {input_id}...")
            markdown_report = self._generate_markdown_report(
                instance, classification, root_cause, recommendations
            )
            
            # Step 5: Evaluate quality
            quality_metrics = self._evaluate_report_quality(
                markdown_report, classification, root_cause
            )
            
            # Step 6: Compute overall confidence
            overall_confidence = self._compute_overall_confidence(
                classification, root_cause, recommendations
            )
            
            # Create final report
            processing_time = time.time() - start_time
            
            report = ExplainabilityReport(
                instance_id=input_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                task_type=task_type,
                failure_classification=classification,
                root_cause_analysis=root_cause,
                recommendation_suite=recommendations,
                processing_time=processing_time,
                confidence_score=overall_confidence,
                quality_metrics=quality_metrics,
                markdown_report=markdown_report
            )
            
            # Update performance statistics
            self._update_performance_stats(processing_time, True)
            self.analysis_history.append(report)
            
            print(f"✅ Analysis completed for {input_id} in {processing_time:.2f}s")
            return report
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False)
            
            print(f"❌ Analysis failed for {input_id}: {str(e)}")
            
            # Return minimal error report
            return self._create_error_report(input_id, task_type, str(e), processing_time)
    
    def batch_analyze(self, instances: List[Dict[str, Any]], 
                     target_stakeholder: Optional[StakeholderType] = None) -> List[ExplainabilityReport]:
        """
        Analyze multiple failure instances in batch
        
        Args:
            instances: List of instance dictionaries with required fields
            target_stakeholder: Optional target stakeholder for optimization
            
        Returns:
            List of explainability reports
        """
        reports = []
        
        print(f"🚀 Starting batch analysis of {len(instances)} instances...")
        
        for i, instance_data in enumerate(instances):
            print(f"📊 Processing instance {i+1}/{len(instances)}")
            
            try:
                report = self.analyze_failure(
                    input_id=instance_data['input_id'],
                    task_type=instance_data['task_type'],
                    input_text=instance_data['input_text'],
                    model_output=instance_data['model_output'],
                    reference_output=instance_data['reference_output'],
                    context_metadata=instance_data.get('context_metadata'),
                    target_stakeholder=target_stakeholder
                )
                reports.append(report)
                
            except Exception as e:
                print(f"⚠️ Failed to process instance {i+1}: {str(e)}")
                error_report = self._create_error_report(
                    instance_data.get('input_id', f'unknown_{i}'),
                    instance_data.get('task_type', 'unknown'),
                    str(e),
                    0.0
                )
                reports.append(error_report)
        
        print(f"✅ Batch analysis completed. {len(reports)} reports generated.")
        return reports
    
    def _generate_markdown_report(self, 
                                 instance: FailureInstance,
                                 classification: FailureClassification,
                                 root_cause: RootCauseAnalysis,
                                 recommendations: RecommendationSuite) -> str:
        """Generate comprehensive markdown report"""
        
        # Load the template from the original prompt.md structure
        markdown_template = """# Explainability Report: {input_id}

## 1. Summary

- **Input ID:** `{input_id}`
- **Task Type:** `{task_type}`
- **Status:** **FAIL**
- **Failure Category:** `{failure_category}`
- **Confidence Score:** `{confidence_score:.3f}`
- **Analysis Timestamp:** `{timestamp}`

---

## 2. Detailed Analysis

### Input

```
{input_text}
```

### Model Output (Failed)

```
{model_output}
```

### Reference Output (Correct)

```
{reference_output}
```

---

## 3. Root Cause Analysis

{root_cause_analysis}

### Causal Factors

{causal_factors_section}

### Counterfactual Analysis

{counterfactual_section}

---

## 4. Actionable Recommendations

{recommendations_output}

---

## 5. Technical Analysis

### Classification Details
- **Primary Category:** {failure_category}
- **Sub-categories:** {sub_categories}
- **Semantic Features:** {semantic_features_summary}
- **Attention Patterns:** {attention_summary}

### Confidence Metrics
- **Classification Confidence:** {classification_confidence:.3f}
- **Root Cause Confidence:** {root_cause_confidence:.3f}
- **Overall Confidence:** {overall_confidence:.3f}

### Performance Metrics
- **Processing Time:** {processing_time:.2f} seconds
- **Quality Score:** {quality_score:.3f}

---

## 6. Implementation Roadmap

{implementation_roadmap}

---

*Report generated by LLM Explainability Framework v1.0.0*
"""
        
        # Format causal factors section
        causal_factors_section = self._format_causal_factors(root_cause.causal_factors)
        
        # Format counterfactual section
        counterfactual_section = self._format_counterfactuals(root_cause.counterfactual_scenarios)
        
        # Format recommendations section
        recommendations_output = self._format_recommendations(recommendations.recommendations)
        
        # Format implementation roadmap
        implementation_roadmap = self._format_roadmap(recommendations.implementation_roadmap)
        
        # Format semantic features summary
        semantic_features_summary = f"Vector length: {len(classification.semantic_features)}, " \
                                   f"Max value: {np.max(classification.semantic_features):.3f}"
        
        # Format attention summary
        attention_summary = f"Attention variance: {np.var(classification.attention_weights):.3f}, " \
                           f"Max attention: {np.max(classification.attention_weights):.3f}"
        
        # Fill template
        formatted_report = markdown_template.format(
            input_id=instance.input_id,
            task_type=instance.task_type,
            failure_category=classification.failure_category,
            confidence_score=classification.confidence_score,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            input_text=instance.input_text,
            model_output=instance.model_output,
            reference_output=instance.reference_output,
            root_cause_analysis=root_cause.explanation_text,
            causal_factors_section=causal_factors_section,
            counterfactual_section=counterfactual_section,
            recommendations_output=recommendations_output,
            sub_categories=", ".join(classification.sub_categories),
            semantic_features_summary=semantic_features_summary,
            attention_summary=attention_summary,
            classification_confidence=classification.confidence_score,
            root_cause_confidence=root_cause.confidence_score,
            overall_confidence=recommendations.overall_confidence,
            processing_time=0.0,  # Will be filled later
            quality_score=0.0,   # Will be filled later
            implementation_roadmap=implementation_roadmap
        )
        
        return formatted_report
    
    def _format_causal_factors(self, causal_factors: List[Any]) -> str:
        """Format causal factors for markdown display"""
        if not causal_factors:
            return "No significant causal factors identified."
        
        sections = []
        for i, factor in enumerate(causal_factors[:5], 1):  # Top 5 factors
            section = f"""
**Factor {i}: {factor.factor_name}**
- **Type:** {factor.factor_type}
- **Causal Strength:** {factor.causal_strength:.3f}
- **Confidence:** {factor.confidence:.3f}
- **Evidence:** {', '.join(factor.evidence)}
"""
            sections.append(section)
        
        return "\n".join(sections)
    
    def _format_counterfactuals(self, counterfactuals: List[Dict[str, Any]]) -> str:
        """Format counterfactual scenarios for markdown display"""
        if not counterfactuals:
            return "No counterfactual scenarios generated."
        
        sections = []
        for i, cf in enumerate(counterfactuals[:3], 1):  # Top 3 counterfactuals
            section = f"""
**Scenario {i}: {cf.get('intervention_type', 'Unknown')}**
- **Description:** {cf.get('description', 'No description available')}
- **Expected Impact:** {cf.get('expected_impact', 0.0):.3f}
"""
            sections.append(section)
        
        return "\n".join(sections)
    
    def _format_recommendations(self, recommendations: List[Any]) -> str:
        """Format recommendations for markdown display"""
        if not recommendations:
            return "No recommendations generated."
        
        # Group by recommendation type
        grouped_recs = {}
        for rec in recommendations:
            rec_type = rec.recommendation_type.value
            if rec_type not in grouped_recs:
                grouped_recs[rec_type] = []
            grouped_recs[rec_type].append(rec)
        
        sections = []
        
        for rec_type, recs in grouped_recs.items():
            section = f"\n### {rec_type.replace('_', ' ').title()}\n"
            
            for i, rec in enumerate(recs[:2], 1):  # Top 2 per type
                section += f"""
**{i}. {rec.title}**
- **Description:** {rec.description}
- **Expected Impact:** {rec.expected_impact:.2f}
- **Implementation Effort:** {rec.implementation_effort:.2f}
- **Confidence:** {rec.confidence:.2f}

*Implementation Steps:*
"""
                for step in rec.implementation_steps[:3]:  # Top 3 steps
                    section += f"- {step}\n"
                
                section += "\n"
            
            sections.append(section)
        
        return "\n".join(sections)
    
    def _format_roadmap(self, roadmap: List[Dict[str, Any]]) -> str:
        """Format implementation roadmap for markdown display"""
        if not roadmap:
            return "No implementation roadmap generated."
        
        sections = []
        for phase in roadmap:
            section = f"""
**Phase {phase['phase']}**
- **Recommendations:** {len(phase['recommendations'])} items
- **Total Effort:** {phase['total_effort']:.2f}
- **Expected Impact:** {phase['expected_impact']:.2f}
"""
            sections.append(section)
        
        return "\n".join(sections)
    
    def _evaluate_report_quality(self, 
                                markdown_report: str,
                                classification: FailureClassification,
                                root_cause: RootCauseAnalysis) -> Dict[str, float]:
        """Evaluate the quality of the generated report"""
        return self.metrics_evaluator.compute_explanation_quality(
            explanation=root_cause.explanation_text,
            ground_truth=None,  # No ground truth available
            user_feedback=None  # No user feedback yet
        )
    
    def _compute_overall_confidence(self, 
                                   classification: FailureClassification,
                                   root_cause: RootCauseAnalysis,
                                   recommendations: RecommendationSuite) -> float:
        """Compute overall confidence in the analysis"""
        confidences = [
            classification.confidence_score,
            root_cause.confidence_score,
            recommendations.overall_confidence
        ]
        
        # Weighted average with more weight on classification and root cause
        weights = [0.4, 0.4, 0.2]
        overall_confidence = sum(c * w for c, w in zip(confidences, weights))
        
        return overall_confidence
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics"""
        self.performance_stats['total_analyses'] += 1
        self.performance_stats['total_time'] += processing_time
        
        if self.performance_stats['total_analyses'] > 0:
            self.performance_stats['average_time'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_analyses']
            )
        
        # Update success rate (simplified)
        if success:
            current_successes = self.performance_stats['success_rate'] * (self.performance_stats['total_analyses'] - 1)
            self.performance_stats['success_rate'] = (current_successes + 1) / self.performance_stats['total_analyses']
        else:
            current_successes = self.performance_stats['success_rate'] * (self.performance_stats['total_analyses'] - 1)
            self.performance_stats['success_rate'] = current_successes / self.performance_stats['total_analyses']
    
    def _create_error_report(self, input_id: str, task_type: str, 
                           error_message: str, processing_time: float) -> ExplainabilityReport:
        """Create a minimal error report for failed analyses"""
        from .failure_classifier import FailureClassification
        from .root_cause_analyzer import RootCauseAnalysis
        from .recommendation_engine import RecommendationSuite
        
        # Create minimal objects for error case
        error_classification = FailureClassification(
            failure_category="analysis_error",
            confidence_score=0.0,
            sub_categories=["processing_failed"],
            attention_weights=np.array([]),
            semantic_features=np.array([]),
            explanation_vector=np.array([])
        )
        
        error_root_cause = RootCauseAnalysis(
            primary_cause="Analysis engine error",
            causal_factors=[],
            causal_graph=None,
            counterfactual_scenarios=[],
            confidence_score=0.0,
            explanation_text=f"Analysis failed with error: {error_message}",
            intervention_recommendations=[]
        )
        
        error_recommendations = RecommendationSuite(
            instance_id=input_id,
            failure_category="analysis_error",
            recommendations=[],
            optimization_strategy={},
            stakeholder_alignment={},
            overall_confidence=0.0,
            implementation_roadmap=[]
        )
        
        error_report = ExplainabilityReport(
            instance_id=input_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            task_type=task_type,
            failure_classification=error_classification,
            root_cause_analysis=error_root_cause,
            recommendation_suite=error_recommendations,
            processing_time=processing_time,
            confidence_score=0.0,
            quality_metrics={'overall_quality': 0.0},
            markdown_report=f"# Error Report\n\nAnalysis failed: {error_message}"
        )
        
        return error_report
    
    def save_report(self, report: ExplainabilityReport, output_dir: str = "reports"):
        """Save explainability report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save markdown report
        markdown_file = output_path / f"{report.instance_id}_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(report.markdown_report)
        
        # Save JSON data
        json_file = output_path / f"{report.instance_id}_data.json"
        report_dict = asdict(report)
        
        # Convert non-serializable objects to strings
        report_dict = self._serialize_report_data(report_dict)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Report saved to {markdown_file} and {json_file}")
    
    def _serialize_report_data(self, report_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert non-serializable objects to JSON-compatible format"""
        
        def convert_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif hasattr(value, '__dict__'):
                return str(value)
            elif isinstance(value, dict):
                return {str(k): convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif hasattr(value, 'value'):  # Handle enums
                return value.value
            else:
                return value
        
        return convert_value(report_dict)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return self.performance_stats.copy()
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed"""
        if not self.analysis_history:
            return {"message": "No analyses performed yet"}
        
        # Compute summary statistics
        failure_categories = [report.failure_classification.failure_category 
                            for report in self.analysis_history]
        task_types = [report.task_type for report in self.analysis_history]
        confidence_scores = [report.confidence_score for report in self.analysis_history]
        processing_times = [report.processing_time for report in self.analysis_history]
        
        summary = {
            "total_analyses": len(self.analysis_history),
            "failure_category_distribution": {
                category: failure_categories.count(category) 
                for category in set(failure_categories)
            },
            "task_type_distribution": {
                task_type: task_types.count(task_type) 
                for task_type in set(task_types)
            },
            "average_confidence": np.mean(confidence_scores),
            "average_processing_time": np.mean(processing_times),
            "confidence_range": {
                "min": np.min(confidence_scores),
                "max": np.max(confidence_scores)
            }
        }
        
        return summary 