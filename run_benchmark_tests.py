#!/usr/bin/env python3
"""
Benchmark Test Runner for LLM Explainability Framework

This script runs comprehensive tests using HumanEval and TruthfulQA datasets
to evaluate the LLM explainability framework across different task types.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# Add the project root to the path
sys.path.insert(0, '.')

# JSON cleaning helper for safe serialization

def _clean_for_json(obj):
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # Skip ExplainabilityReport objects in dictionaries
                if k == "analysis_report":
                    continue
                if hasattr(k, 'name'):
                    key = k.name
                elif hasattr(k, 'value'):
                    key = k.value
                else:
                    key = str(k)
                cleaned[key] = _clean_for_json(v)
            return cleaned
        if isinstance(obj, (list, tuple, set)):
            return [_clean_for_json(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # Skip ExplainabilityReport objects (check for instance_id attribute)
        if hasattr(obj, 'instance_id') and hasattr(obj, 'failure_classification'):
            return f"ExplainabilityReport({getattr(obj, 'instance_id', 'unknown')})"
        if hasattr(obj, 'value'):
            return obj.value
        if hasattr(obj, 'name'):
            return obj.name
        if hasattr(obj, '__dict__'):
            return _clean_for_json(vars(obj))
        if hasattr(obj, '__iter__') and not isinstance(obj, (bytes, bytearray)):
            try:
                return [_clean_for_json(x) for x in obj]
            except Exception:
                return str(obj)
        return str(obj)
    except Exception:
        return str(obj)

from llm_explainability_framework import (
    ExplainabilityEngine,
    create_default_llm_wrapper,
    StakeholderType
)
from llm_explainability_framework.visualization.reporter import ExplainabilityReporter

# Import our test modules
from test_humaneval import HumanEvalTester
from test_truthfulqa import TruthfulQATester

class BenchmarkTestRunner:
    """
    Comprehensive benchmark testing for LLM explainability framework
    
    Integrates multiple datasets to provide thorough evaluation across
    different failure types and task categories.
    """
    
    def __init__(self, llm_wrapper=None, output_dir="benchmark_results"):
        self.llm_wrapper = llm_wrapper or create_default_llm_wrapper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize testers
        self.humaneval_tester = HumanEvalTester(self.llm_wrapper)
        self.truthfulqa_tester = TruthfulQATester(self.llm_wrapper)
        
        # Initialize comprehensive reporter
        self.reporter = ExplainabilityReporter(output_dir=str(self.output_dir))
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "datasets": {},
            "summary": {}
        }
    
    def run_humaneval_benchmark(self, max_instances: Optional[int] = None) -> Dict[str, Any]:
        """
        Run HumanEval benchmark tests
        
        Args:
            max_instances: Maximum number of instances to test (None for all)
            
        Returns:
            HumanEval test results
        """
        print("\n" + "="*60)
        print("RUNNING HUMANEVAL BENCHMARK")
        print("="*60)
        
        start_time = time.time()
        
        # Load dataset
        instances = self.humaneval_tester.load_humaneval_dataset()
        
        # Limit instances if specified
        if max_instances and len(instances) > max_instances:
            instances = instances[:max_instances]
            print(f"Limited to {max_instances} instances for testing")
        
        # Run analysis
        results = self.humaneval_tester.analyze_humaneval_failures(instances)
        
        # Generate report
        self.humaneval_tester.generate_humaneval_report(results)
        
        end_time = time.time()
        
        # Calculate metrics
        total_instances = len(results)
        successful_analyses = len([r for r in results if "error" not in r])
        execution_passed = len([r for r in results if r.get("execution_result", {}).get("passed", False)])
        
        benchmark_results = {
            "dataset": "HumanEval",
            "total_instances": total_instances,
            "successful_analyses": successful_analyses,
            "execution_passed": execution_passed,
            "execution_failed": total_instances - execution_passed,
            "success_rate": successful_analyses / total_instances if total_instances > 0 else 0,
            "execution_success_rate": execution_passed / total_instances if total_instances > 0 else 0,
            "processing_time": end_time - start_time,
            "detailed_results": results
        }
        
        print(f"\nHumanEval Benchmark Results:")
        print(f"Total instances: {total_instances}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Execution passed: {execution_passed}")
        print(f"Success rate: {benchmark_results['success_rate']:.2%}")
        print(f"Execution success rate: {benchmark_results['execution_success_rate']:.2%}")
        print(f"Processing time: {benchmark_results['processing_time']:.2f}s")
        
        return benchmark_results
    
    def run_truthfulqa_benchmark(self, max_instances: Optional[int] = None) -> Dict[str, Any]:
        """
        Run TruthfulQA benchmark tests
        
        Args:
            max_instances: Maximum number of instances to test (None for all)
            
        Returns:
            TruthfulQA test results
        """
        print("\n" + "="*60)
        print("RUNNING TRUTHFULQA BENCHMARK")
        print("="*60)
        
        start_time = time.time()
        
        # Load dataset
        instances = self.truthfulqa_tester.load_truthfulqa_dataset()
        
        # Limit instances if specified
        if max_instances and len(instances) > max_instances:
            instances = instances[:max_instances]
            print(f"Limited to {max_instances} instances for testing")
        
        # Run analysis
        results = self.truthfulqa_tester.analyze_truthfulqa_failures(instances)
        
        # Generate report
        self.truthfulqa_tester.generate_truthfulqa_report(results)
        
        end_time = time.time()
        
        # Calculate metrics (exclude unknowns from denominator)
        total_instances = len(results)
        successful_analyses = len([r for r in results if "error" not in r])
        flags = [r.get("truthfulness_evaluation", {}).get("is_truthful") for r in results if "error" not in r]
        truthful_answers = sum(1 for f in flags if f is True)
        untruthful_answers = sum(1 for f in flags if f is False)
        unknown_answers = sum(1 for f in flags if f is None)
        denom = truthful_answers + untruthful_answers
        truthfulness_rate = (truthful_answers / denom) if denom > 0 else 0.0
        
        benchmark_results = {
            "dataset": "TruthfulQA",
            "total_instances": total_instances,
            "successful_analyses": successful_analyses,
            "truthful_answers": truthful_answers,
            "untruthful_answers": untruthful_answers,
            "unknown_answers": unknown_answers,
            "success_rate": successful_analyses / total_instances if total_instances > 0 else 0,
            "truthfulness_rate": truthfulness_rate,
            "processing_time": end_time - start_time,
            "detailed_results": results
        }
        
        print(f"\nTruthfulQA Benchmark Results:")
        print(f"Total instances: {total_instances}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Truthful answers: {truthful_answers}")
        print(f"Untruthful answers: {untruthful_answers}")
        print(f"Unknown answers: {unknown_answers}")
        print(f"Success rate: {benchmark_results['success_rate']:.2%}")
        print(f"Truthfulness rate (excl. unknown): {benchmark_results['truthfulness_rate']:.2%}")
        print(f"Processing time: {benchmark_results['processing_time']:.2f}s")
        
        return benchmark_results
    
    def run_comprehensive_benchmark(self, max_instances_per_dataset: Optional[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all datasets
        
        Args:
            max_instances_per_dataset: Maximum instances per dataset
            
        Returns:
            Comprehensive benchmark results
        """
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE BENCHMARK TESTING")
        print("="*80)
        
        overall_start_time = time.time()
        
        # Run HumanEval benchmark
        humaneval_results = self.run_humaneval_benchmark(max_instances_per_dataset)
        
        # Run TruthfulQA benchmark
        truthfulqa_results = self.run_truthfulqa_benchmark(max_instances_per_dataset)
        
        overall_end_time = time.time()
        
        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": datetime.now().isoformat(),
            "total_processing_time": overall_end_time - overall_start_time,
            "datasets": {
                "humaneval": humaneval_results,
                "truthfulqa": truthfulqa_results
            },
            "summary": {
                "total_instances": humaneval_results["total_instances"] + truthfulqa_results["total_instances"],
                "total_successful_analyses": humaneval_results["successful_analyses"] + truthfulqa_results["successful_analyses"],
                "overall_success_rate": (
                    (humaneval_results["successful_analyses"] + truthfulqa_results["successful_analyses"]) /
                    (humaneval_results["total_instances"] + truthfulqa_results["total_instances"])
                    if (humaneval_results["total_instances"] + truthfulqa_results["total_instances"]) > 0 else 0
                ),
                "dataset_specific_metrics": {
                    "humaneval_execution_success_rate": humaneval_results["execution_success_rate"],
                    "truthfulqa_truthfulness_rate": truthfulqa_results["truthfulness_rate"]
                }
            }
        }
        
        # Save comprehensive results (JSON serializable version)
        results_file = self.output_dir / "comprehensive_benchmark_results.json"
        with open(results_file, 'w') as f:
            # Create JSON-safe version without ExplainabilityReport objects
            json_safe_results = _clean_for_json(comprehensive_results)
            json.dump(json_safe_results, f, indent=2)
        
        # Generate comprehensive visualizations using the reporter
        print("\n📊 Generating comprehensive visualizations...")
        explainability_reports = self._extract_explainability_reports(comprehensive_results)
        
        if explainability_reports:
            print(f"Found {len(explainability_reports)} ExplainabilityReport objects for analysis")
            
            # Generate comprehensive visual report
            visual_report_path = self.reporter.generate_comprehensive_report(
                explainability_reports, 
                output_name="comprehensive_benchmark_analysis"
            )
            print(f"📈 Interactive visual report generated: {visual_report_path}")
        else:
            print("⚠️  No ExplainabilityReport objects found for visualization")
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)
        print(f"Total instances tested: {comprehensive_results['summary']['total_instances']}")
        print(f"Total successful analyses: {comprehensive_results['summary']['total_successful_analyses']}")
        print(f"Overall success rate: {comprehensive_results['summary']['overall_success_rate']:.2%}")
        print(f"Total processing time: {comprehensive_results['total_processing_time']:.2f}s")
        print(f"\nDataset-specific metrics:")
        print(f"  HumanEval execution success rate: {comprehensive_results['summary']['dataset_specific_metrics']['humaneval_execution_success_rate']:.2%}")
        print(f"  TruthfulQA truthfulness rate: {comprehensive_results['summary']['dataset_specific_metrics']['truthfulqa_truthfulness_rate']:.2%}")
        print(f"\nResults saved to: {results_file}")
        
        return comprehensive_results
    
    def _extract_explainability_reports(self, results: Dict[str, Any]) -> List:
        """Extract ExplainabilityReport objects from benchmark results"""
        reports = []
        
        # Extract from HumanEval results
        if 'humaneval' in results.get('datasets', {}):
            humaneval_results = results['datasets']['humaneval'].get('detailed_results', [])
            for result in humaneval_results:
                if 'analysis_report' in result and result['analysis_report'] is not None:
                    reports.append(result['analysis_report'])
        
        # Extract from TruthfulQA results  
        if 'truthfulqa' in results.get('datasets', {}):
            truthfulqa_results = results['datasets']['truthfulqa'].get('detailed_results', [])
            for result in truthfulqa_results:
                if 'analysis_report' in result and result['analysis_report'] is not None:
                    reports.append(result['analysis_report'])
        
        return reports
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a comprehensive benchmark report with detailed error analysis
        
        Args:
            results: Comprehensive benchmark results
        """
        report_file = self.output_dir / "benchmark_report.md"
        
        # Extract ExplainabilityReport objects for detailed analysis
        explainability_reports = self._extract_explainability_reports(results)
        
        # Compute additional insights from detailed results
        humaneval = results['datasets'].get('humaneval', {})
        truthfulqa = results['datasets'].get('truthfulqa', {})
        he_results = humaneval.get('detailed_results', [])
        tq_results = truthfulqa.get('detailed_results', [])
        
        with open(report_file, 'w') as f:
            f.write("# LLM Explainability Framework Benchmark Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            self._write_executive_summary(f, results)
            
            # Detailed Error Analysis Section
            self._write_detailed_error_analysis(f, explainability_reports, he_results, tq_results)
            
            # Root Cause Analysis Section
            self._write_root_cause_analysis(f, explainability_reports)
            
            # Recommendations Section
            self._write_recommendations_analysis(f, explainability_reports)
            
            # Dataset-specific Analysis
            self._write_dataset_analysis(f, humaneval, truthfulqa, he_results, tq_results)
            
            # Comprehensive Instance Analysis
            self._write_comprehensive_instance_analysis(f, explainability_reports)
            
            # Additional Reports
            self._write_additional_reports_section(f)
        
        print(f"📝 Comprehensive benchmark report generated: {report_file}")

    def _write_executive_summary(self, f, results):
        """Write executive summary section"""
        f.write("## 📊 Executive Summary\n\n")
        f.write(f"- **Total Instances Tested:** {results['summary']['total_instances']}\n")
        f.write(f"- **Total Successful Analyses:** {results['summary']['total_successful_analyses']}\n")
        f.write(f"- **Overall Success Rate:** {results['summary']['overall_success_rate']:.2%}\n")
        f.write(f"- **Total Processing Time:** {results['total_processing_time']:.2f}s\n\n")
        
        humaneval = results['datasets'].get('humaneval', {})
        truthfulqa = results['datasets'].get('truthfulqa', {})
        
        f.write("### Key Findings\n\n")
        f.write(f"- **Code Generation (HumanEval):** {humaneval.get('execution_success_rate', 0):.1%} execution success rate\n")
        f.write(f"- **Factual Consistency (TruthfulQA):** {truthfulqa.get('truthfulness_rate', 0):.1%} truthfulness rate\n")
        f.write(f"- **Average Processing Time:** {results['total_processing_time'] / max(1, results['summary']['total_instances']):.2f}s per instance\n\n")
        f.write("---\n\n")

    def _write_detailed_error_analysis(self, f, reports, he_results, tq_results):
        """Write detailed error analysis section"""
        f.write("## 🔍 Detailed Error Analysis\n\n")
        
        if not reports:
            f.write("*No detailed analysis available - ExplainabilityReport objects not found.*\n\n")
            return
        
        # Failure category analysis
        failure_categories = {}
        confidence_by_category = {}
        
        for report in reports:
            category = report.failure_classification.failure_category
            confidence = report.failure_classification.confidence_score
            
            if category not in failure_categories:
                failure_categories[category] = 0
                confidence_by_category[category] = []
            
            failure_categories[category] += 1
            confidence_by_category[category].append(confidence)
        
        f.write("### Failure Category Distribution\n\n")
        total_failures = sum(failure_categories.values())
        
        for category, count in sorted(failure_categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_failures) * 100
            avg_confidence = sum(confidence_by_category[category]) / len(confidence_by_category[category])
            f.write(f"- **{category}:** {count} instances ({percentage:.1f}%) - Avg Confidence: {avg_confidence:.2f}\n")
        
        f.write("\n### Error Patterns by Task Type\n\n")
        
        # Group by task type
        task_errors = {}
        for report in reports:
            task = report.task_type
            category = report.failure_classification.failure_category
            
            if task not in task_errors:
                task_errors[task] = {}
            if category not in task_errors[task]:
                task_errors[task][category] = 0
            task_errors[task][category] += 1
        
        for task_type, errors in task_errors.items():
            f.write(f"#### {task_type}\n\n")
            total_task_errors = sum(errors.values())
            for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_task_errors) * 100
                f.write(f"- {error_type}: {count} cases ({percentage:.1f}%)\n")
            f.write("\n")
        
        f.write("---\n\n")

    def _write_root_cause_analysis(self, f, reports):
        """Write root cause analysis section"""
        f.write("## 🎯 Root Cause Analysis\n\n")
        
        if not reports:
            f.write("*No root cause analysis available.*\n\n")
            return
        
        # Collect all causal factors
        causal_factors = {}
        factor_confidence = {}
        factor_strength = {}
        
        for report in reports:
            if hasattr(report.root_cause_analysis, 'causal_factors'):
                for factor in report.root_cause_analysis.causal_factors:
                    factor_name = factor.factor_name
                    factor_type = factor.factor_type
                    
                    key = f"{factor_name} ({factor_type})"
                    
                    if key not in causal_factors:
                        causal_factors[key] = 0
                        factor_confidence[key] = []
                        factor_strength[key] = []
                    
                    causal_factors[key] += 1
                    factor_confidence[key].append(factor.confidence)
                    factor_strength[key].append(factor.causal_strength)
        
        f.write("### Primary Root Causes\n\n")
        
        # Sort by frequency and strength
        sorted_factors = sorted(causal_factors.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for factor_key, count in sorted_factors:
            avg_confidence = sum(factor_confidence[factor_key]) / len(factor_confidence[factor_key])
            avg_strength = sum(factor_strength[factor_key]) / len(factor_strength[factor_key])
            
            f.write(f"#### {factor_key}\n")
            f.write(f"- **Frequency:** {count} instances\n")
            f.write(f"- **Average Confidence:** {avg_confidence:.2f}\n")
            f.write(f"- **Average Causal Strength:** {avg_strength:.2f}\n")
            f.write(f"- **Impact:** {self._get_impact_description(avg_strength)}\n\n")
        
        # Causal factor types analysis
        f.write("### Root Cause Types Distribution\n\n")
        
        factor_types = {}
        for report in reports:
            if hasattr(report.root_cause_analysis, 'causal_factors'):
                for factor in report.root_cause_analysis.causal_factors:
                    ftype = factor.factor_type
                    if ftype not in factor_types:
                        factor_types[ftype] = 0
                    factor_types[ftype] += 1
        
        for ftype, count in sorted(factor_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{ftype}:** {count} instances\n")
        
        f.write("\n---\n\n")

    def _write_recommendations_analysis(self, f, reports):
        """Write recommendations analysis section"""
        f.write("## 💡 Recommendations Analysis\n\n")
        
        if not reports:
            f.write("*No recommendations available.*\n\n")
            return
        
        # Collect all recommendations
        recommendations_by_type = {}
        recommendations_by_stakeholder = {}
        high_priority_recs = []
        
        for report in reports:
            if hasattr(report.recommendation_suite, 'recommendations'):
                for rec in report.recommendation_suite.recommendations:
                    rec_type = rec.recommendation_type.value if hasattr(rec.recommendation_type, 'value') else str(rec.recommendation_type)
                    stakeholder = rec.stakeholder_type.value if hasattr(rec.stakeholder_type, 'value') else str(rec.stakeholder_type)
                    
                    # By type
                    if rec_type not in recommendations_by_type:
                        recommendations_by_type[rec_type] = []
                    recommendations_by_type[rec_type].append(rec)
                    
                    # By stakeholder
                    if stakeholder not in recommendations_by_stakeholder:
                        recommendations_by_stakeholder[stakeholder] = []
                    recommendations_by_stakeholder[stakeholder].append(rec)
                    
                    # High priority
                    if rec.priority_score > 0.7:
                        high_priority_recs.append((report.instance_id, rec))
        
        f.write("### High-Priority Recommendations\n\n")
        
        for instance_id, rec in sorted(high_priority_recs, key=lambda x: x[1].priority_score, reverse=True)[:10]:
            rec_type = rec.recommendation_type.value if hasattr(rec.recommendation_type, 'value') else str(rec.recommendation_type)
            stakeholder = rec.stakeholder_type.value if hasattr(rec.stakeholder_type, 'value') else str(rec.stakeholder_type)
            
            f.write(f"#### {rec_type} (Priority: {rec.priority_score:.2f})\n")
            f.write(f"- **Instance:** {instance_id}\n")
            f.write(f"- **Target Stakeholder:** {stakeholder}\n")
            f.write(f"- **Expected Impact:** {rec.expected_impact:.2f}\n")
            f.write(f"- **Implementation Effort:** {rec.implementation_effort:.2f}\n")
            f.write(f"- **Description:** {rec.description}\n")
            if hasattr(rec, 'implementation_steps') and rec.implementation_steps:
                f.write(f"- **Steps:** {', '.join(rec.implementation_steps[:3])}\n")
            f.write("\n")
        
        f.write("### Recommendations by Type\n\n")
        
        for rec_type, recs in sorted(recommendations_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            avg_priority = sum(r.priority_score for r in recs) / len(recs)
            avg_impact = sum(r.expected_impact for r in recs) / len(recs)
            avg_effort = sum(r.implementation_effort for r in recs) / len(recs)
            
            f.write(f"#### {rec_type}\n")
            f.write(f"- **Instances:** {len(recs)}\n")
            f.write(f"- **Average Priority:** {avg_priority:.2f}\n")
            f.write(f"- **Average Expected Impact:** {avg_impact:.2f}\n")
            f.write(f"- **Average Implementation Effort:** {avg_effort:.2f}\n")
            f.write(f"- **ROI Ratio:** {avg_impact/max(avg_effort, 0.1):.2f}\n\n")
        
        f.write("### Recommendations by Stakeholder\n\n")
        
        for stakeholder, recs in sorted(recommendations_by_stakeholder.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"#### {stakeholder}\n")
            f.write(f"- **Total Recommendations:** {len(recs)}\n")
            
            # Top recommendation types for this stakeholder
            stakeholder_types = {}
            for rec in recs:
                rec_type = rec.recommendation_type.value if hasattr(rec.recommendation_type, 'value') else str(rec.recommendation_type)
                if rec_type not in stakeholder_types:
                    stakeholder_types[rec_type] = 0
                stakeholder_types[rec_type] += 1
            
            f.write("- **Top Recommendation Types:**\n")
            for stype, count in sorted(stakeholder_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                f.write(f"  - {stype}: {count} recommendations\n")
            f.write("\n")
        
        f.write("---\n\n")

    def _write_dataset_analysis(self, f, humaneval, truthfulqa, he_results, tq_results):
        """Write dataset-specific analysis"""
        f.write("## 📊 Dataset-Specific Analysis\n\n")
        
        # HumanEval Analysis
        f.write("### HumanEval (Code Generation) Deep Dive\n\n")
        f.write(f"- **Total Instances:** {humaneval.get('total_instances', 0)}\n")
        f.write(f"- **Successful Analyses:** {humaneval.get('successful_analyses', 0)}\n")
        f.write(f"- **Execution Success Rate:** {humaneval.get('execution_success_rate', 0):.2%}\n\n")
        
        # Execution error analysis
        execution_errors = {}
        syntax_issues = []
        runtime_issues = []
        
        for result in he_results:
            if "error" not in result:
                exec_result = result.get("execution_result", {})
                if not exec_result.get("passed", False):
                    error_msg = exec_result.get("error_message", "Unknown error")
                    
                    # Categorize errors
                    if any(keyword in error_msg.lower() for keyword in ["syntax", "invalid syntax", "unexpected token"]):
                        syntax_issues.append(error_msg)
                    elif any(keyword in error_msg.lower() for keyword in ["runtime", "name", "attribute", "type", "index"]):
                        runtime_issues.append(error_msg)
                    
                    error_key = error_msg.split('\n')[0][:50] + "..." if len(error_msg) > 50 else error_msg
                    execution_errors[error_key] = execution_errors.get(error_key, 0) + 1
        
        if execution_errors:
            f.write("#### Common Execution Errors\n\n")
            for error, count in sorted(execution_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"- **{error}** ({count} instances)\n")
            f.write("\n")
        
        if syntax_issues:
            f.write(f"#### Syntax Issues: {len(syntax_issues)} cases\n")
            f.write("*Primary cause: Malformed code generation*\n\n")
        
        if runtime_issues:
            f.write(f"#### Runtime Issues: {len(runtime_issues)} cases\n")
            f.write("*Primary cause: Logical errors in generated code*\n\n")
        
        # TruthfulQA Analysis
        f.write("### TruthfulQA (Factual Consistency) Deep Dive\n\n")
        f.write(f"- **Total Instances:** {truthfulqa.get('total_instances', 0)}\n")
        f.write(f"- **Truthfulness Rate:** {truthfulqa.get('truthfulness_rate', 0):.2%}\n")
        f.write(f"- **Unknown Answers:** {truthfulqa.get('unknown_answers', 0)}\n\n")
        
        # Truthfulness analysis by category
        from collections import defaultdict
        cat_analysis = defaultdict(lambda: {'truthful': 0, 'untruthful': 0, 'unknown': 0})
        
        for result in tq_results:
            if "error" not in result:
                category = result.get('context_metadata', {}).get('category', 'general')
                truthful_flag = result.get('truthfulness_evaluation', {}).get('is_truthful')
                
                if truthful_flag is True:
                    cat_analysis[category]['truthful'] += 1
                elif truthful_flag is False:
                    cat_analysis[category]['untruthful'] += 1
                else:
                    cat_analysis[category]['unknown'] += 1
        
        if cat_analysis:
            f.write("#### Truthfulness by Category\n\n")
            for category, stats in sorted(cat_analysis.items(), key=lambda x: x[1]['truthful'], reverse=True):
                total = stats['truthful'] + stats['untruthful'] + stats['unknown']
                if total > 0:
                    truth_rate = stats['truthful'] / max(1, stats['truthful'] + stats['untruthful']) * 100
                    f.write(f"- **{category}:** {truth_rate:.1f}% truthful (T:{stats['truthful']}, U:{stats['untruthful']}, K:{stats['unknown']})\n")
            f.write("\n")
        
        f.write("---\n\n")

    def _write_comprehensive_instance_analysis(self, f, reports):
        """Write comprehensive analysis combining all instances"""
        f.write("## 📋 Comprehensive Instance Analysis\n\n")
        
        if not reports:
            f.write("*No instance analysis available.*\n\n")
            return
        
        # Group instances by failure category for better organization
        instances_by_category = {}
        confidence_stats = {'high': [], 'medium': [], 'low': []}
        processing_time_stats = []
        
        for report in reports:
            category = report.failure_classification.failure_category
            if category not in instances_by_category:
                instances_by_category[category] = []
            instances_by_category[category].append(report)
            
            # Collect stats
            conf = report.confidence_score
            if conf >= 0.7:
                confidence_stats['high'].append(report)
            elif conf >= 0.4:
                confidence_stats['medium'].append(report)
            else:
                confidence_stats['low'].append(report)
            
            processing_time_stats.append(report.processing_time)
        
        # Overall statistics
        f.write("### Instance Analysis Summary\n\n")
        f.write(f"- **Total Instances Analyzed:** {len(reports)}\n")
        f.write(f"- **High Confidence (≥0.7):** {len(confidence_stats['high'])} instances\n")
        f.write(f"- **Medium Confidence (0.4-0.7):** {len(confidence_stats['medium'])} instances\n")
        f.write(f"- **Low Confidence (<0.4):** {len(confidence_stats['low'])} instances\n")
        f.write(f"- **Average Processing Time:** {sum(processing_time_stats)/len(processing_time_stats):.2f}s\n")
        f.write(f"- **Fastest Analysis:** {min(processing_time_stats):.2f}s\n")
        f.write(f"- **Slowest Analysis:** {max(processing_time_stats):.2f}s\n\n")
        
        # Instance ID explanation
        f.write("#### How to Locate Instances\n\n")
        f.write("**Instance ID Formats:**\n")
        f.write("- **HumanEval**: `HumanEval/X` where X is the original problem number\n")
        f.write("- **TruthfulQA**: `truthfulqa_val_X` where X is the validation set index, or `truthfulqa_sample_X` for sample data\n\n")
        f.write("Use these IDs to locate the exact instances in the original datasets for further investigation.\n\n")
        
        # Analysis by failure category
        f.write("### Analysis by Failure Category\n\n")
        
        for category, category_reports in sorted(instances_by_category.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"#### {category} ({len(category_reports)} instances)\n\n")
            
            # Category statistics
            avg_confidence = sum(r.confidence_score for r in category_reports) / len(category_reports)
            avg_time = sum(r.processing_time for r in category_reports) / len(category_reports)
            
            f.write(f"- **Average Confidence:** {avg_confidence:.2f}\n")
            f.write(f"- **Average Processing Time:** {avg_time:.2f}s\n")
            
            # Instance IDs with dataset context
            instance_ids = [r.instance_id for r in category_reports]
            f.write(f"- **Affected Instances:** {', '.join(instance_ids[:5])}")
            if len(instance_ids) > 5:
                f.write(f" and {len(instance_ids) - 5} more")
            f.write("\n")
            
            # Show dataset distribution for this category
            humaneval_count = sum(1 for r in category_reports if r.task_type == "NL2CODE")
            truthfulqa_count = sum(1 for r in category_reports if r.task_type == "NL2NL")
            f.write(f"- **Dataset Distribution:** HumanEval: {humaneval_count}, TruthfulQA: {truthfulqa_count}\n")
            
            # Common root causes for this category
            category_factors = {}
            for report in category_reports:
                if hasattr(report.root_cause_analysis, 'causal_factors'):
                    for factor in report.root_cause_analysis.causal_factors[:2]:  # Top 2 factors per instance
                        factor_name = factor.factor_name
                        if factor_name not in category_factors:
                            category_factors[factor_name] = {'count': 0, 'avg_strength': 0, 'strengths': []}
                        category_factors[factor_name]['count'] += 1
                        category_factors[factor_name]['strengths'].append(factor.causal_strength)
            
            # Calculate averages and sort
            for factor_name, data in category_factors.items():
                data['avg_strength'] = sum(data['strengths']) / len(data['strengths'])
            
            top_factors = sorted(category_factors.items(), 
                               key=lambda x: (x[1]['count'], x[1]['avg_strength']), reverse=True)[:3]
            
            if top_factors:
                f.write("- **Common Root Causes:**\n")
                for factor_name, data in top_factors:
                    f.write(f"  - {factor_name}: {data['count']} instances (avg strength: {data['avg_strength']:.2f})\n")
            
            # Common recommendations for this category
            category_recs = {}
            for report in category_reports:
                if hasattr(report.recommendation_suite, 'recommendations'):
                    for rec in report.recommendation_suite.recommendations:
                        if rec.priority_score > 0.5:  # Only high-priority recommendations
                            rec_type = rec.recommendation_type.value if hasattr(rec.recommendation_type, 'value') else str(rec.recommendation_type)
                            if rec_type not in category_recs:
                                category_recs[rec_type] = {'count': 0, 'avg_priority': 0, 'priorities': []}
                            category_recs[rec_type]['count'] += 1
                            category_recs[rec_type]['priorities'].append(rec.priority_score)
            
            # Calculate averages
            for rec_type, data in category_recs.items():
                data['avg_priority'] = sum(data['priorities']) / len(data['priorities'])
            
            top_recs = sorted(category_recs.items(), 
                            key=lambda x: (x[1]['count'], x[1]['avg_priority']), reverse=True)[:3]
            
            if top_recs:
                f.write("- **Recommended Actions:**\n")
                for rec_type, data in top_recs:
                    f.write(f"  - {rec_type}: {data['count']} instances (avg priority: {data['avg_priority']:.2f})\n")
            
            # Show a sample of instances for this category with content for easier identification
            if len(category_reports) <= 3:
                f.write("- **Instance Details:**\n")
                for report in category_reports:
                    if hasattr(report, 'task_type') and report.task_type == "NL2CODE":
                        f.write(f"  - `{report.instance_id}`: Code generation task\n")
                    elif hasattr(report, 'task_type') and report.task_type == "NL2NL":
                        # Try to get question from context or input_text
                        question = "Question text not available"
                        if hasattr(report, 'input_text') and len(report.input_text) < 100:
                            question = report.input_text.strip()
                        f.write(f"  - `{report.instance_id}`: {question}\n")
                    else:
                        f.write(f"  - `{report.instance_id}`: Unknown task type\n")
            
            f.write("\n")
        
        # Confidence-based analysis
        f.write("### High-Confidence Failure Patterns\n\n")
        
        if confidence_stats['high']:
            f.write(f"**{len(confidence_stats['high'])} instances with confidence ≥ 0.7:**\n\n")
            
            # Most common failure categories in high-confidence cases
            high_conf_categories = {}
            for report in confidence_stats['high']:
                cat = report.failure_classification.failure_category
                high_conf_categories[cat] = high_conf_categories.get(cat, 0) + 1
            
            f.write("**Most Reliable Failure Classifications:**\n")
            for cat, count in sorted(high_conf_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(confidence_stats['high'])) * 100
                f.write(f"- {cat}: {count} instances ({percentage:.1f}%)\n")
            f.write("\n")
        
        # Low-confidence cases that need attention
        if confidence_stats['low']:
            f.write(f"### Low-Confidence Cases Requiring Review\n\n")
            f.write(f"**{len(confidence_stats['low'])} instances with confidence < 0.4:**\n\n")
            
            low_conf_instances = [(r.instance_id, r.confidence_score, r.failure_classification.failure_category) 
                                for r in confidence_stats['low']]
            low_conf_instances.sort(key=lambda x: x[1])  # Sort by confidence (lowest first)
            
            f.write("**Instances Needing Manual Review:**\n")
            for instance_id, conf, category in low_conf_instances[:10]:  # Show worst 10
                f.write(f"- {instance_id}: {category} (confidence: {conf:.2f})\n")
            
            if len(low_conf_instances) > 10:
                f.write(f"- ... and {len(low_conf_instances) - 10} more low-confidence cases\n")
            f.write("\n")
        
        # Performance insights
        f.write("### Performance Insights\n\n")
        
        # Processing time by task type
        task_times = {}
        for report in reports:
            task = report.task_type
            if task not in task_times:
                task_times[task] = []
            task_times[task].append(report.processing_time)
        
        f.write("**Processing Time by Task Type:**\n")
        for task, times in task_times.items():
            avg_time = sum(times) / len(times)
            f.write(f"- {task}: {avg_time:.2f}s average ({len(times)} instances)\n")
        
        f.write("\n---\n\n")

    def _write_additional_reports_section(self, f):
        """Write additional reports section"""
        f.write("## 📈 Supplementary Visual Analysis\n\n")
        f.write("In addition to this comprehensive markdown report, interactive visualizations are available:\n\n")
        f.write("### Interactive Dashboard\n\n")
        f.write("- **Primary Visual Report**: `comprehensive_benchmark_analysis.html`\n")
        f.write("  - Interactive dashboard combining all analysis dimensions\n")
        f.write("  - Clickable charts with drill-down capabilities\n")
        f.write("  - Cross-referenced failure patterns and recommendations\n\n")
        f.write("### Detailed Visualization Components\n\n")
        f.write("The interactive dashboard includes specialized views:\n\n")
        f.write("1. **Failure Distribution Analysis** - Interactive pie charts with category filtering\n")
        f.write("2. **Confidence Analysis Dashboard** - Multi-dimensional confidence correlation plots\n")
        f.write("3. **Task Performance Metrics** - Processing time and quality distributions\n")
        f.write("4. **Root Cause Network Visualization** - Interactive causal factor networks\n")
        f.write("5. **Recommendation Impact Analysis** - Priority vs effort bubble plots\n")
        f.write("6. **Performance Trend Analysis** - Time-series and correlation matrices\n")
        f.write("7. **Causal Network Topology** - Network graphs of failure interdependencies\n\n")
        f.write("### Usage Recommendation\n\n")
        f.write("- **Start with this markdown report** for comprehensive textual analysis\n")
        f.write("- **Use the interactive dashboard** for visual exploration and pattern discovery\n")
        f.write("- **Reference both together** for complete understanding of failure patterns\n\n")

    def _get_impact_description(self, strength):
        """Get qualitative description of impact based on strength"""
        if strength >= 0.8:
            return "Critical - Major contributor to failure"
        elif strength >= 0.6:
            return "High - Significant contributor to failure"
        elif strength >= 0.4:
            return "Medium - Moderate contributor to failure"
        elif strength >= 0.2:
            return "Low - Minor contributor to failure"
        else:
            return "Minimal - Negligible contribution to failure"

def main():
    """Main function to run benchmark tests"""
    parser = argparse.ArgumentParser(description="Run benchmark tests for LLM Explainability Framework")
    parser.add_argument("--datasets", nargs="+", choices=["humaneval", "truthfulqa", "all"], 
                       default=["all"], help="Datasets to test")
    parser.add_argument("--max-instances", type=int, help="Maximum instances per dataset")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkTestRunner(output_dir=args.output_dir)
    
    if "all" in args.datasets or "humaneval" in args.datasets:
        humaneval_results = runner.run_humaneval_benchmark(args.max_instances)
        runner.results["datasets"]["humaneval"] = humaneval_results
    
    if "all" in args.datasets or "truthfulqa" in args.datasets:
        truthfulqa_results = runner.run_truthfulqa_benchmark(args.max_instances)
        runner.results["datasets"]["truthfulqa"] = truthfulqa_results
    
    if "all" in args.datasets:
        comprehensive_results = runner.run_comprehensive_benchmark(args.max_instances)
        runner.generate_benchmark_report(comprehensive_results)
    
    print("\nBenchmark testing completed!")

if __name__ == "__main__":
    main() 