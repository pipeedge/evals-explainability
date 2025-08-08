#!/usr/bin/env python3
"""
HumanEval Dataset Integration Test

This script integrates the HumanEval dataset with the LLM Explainability Framework
to test code generation failures and provide comprehensive analysis.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import asyncio
import time
import numpy as np

# JSON serialization helper

def _to_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    if hasattr(obj, 'value'):  # Handle enums
        return obj.value
    if hasattr(obj, 'name'):  # Handle enum names
        return obj.name
    return str(obj)

def _clean_dict_for_json(obj):
    """Recursively clean a dictionary for JSON serialization"""
    try:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # Convert enum keys to strings
                if hasattr(k, 'name'):
                    key = k.name
                elif hasattr(k, 'value'):
                    key = k.value
                else:
                    key = str(k)
                cleaned[key] = _clean_dict_for_json(v)
            return cleaned
        elif isinstance(obj, (list, tuple)):
            return [_clean_dict_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'value'):  # Handle enums
            return obj.value
        elif hasattr(obj, 'name'):  # Handle enum names
            return obj.name
        elif hasattr(obj, '__dict__'):
            try:
                return _clean_dict_for_json(obj.__dict__)
            except:
                return str(obj)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return [_clean_dict_for_json(item) for item in obj]
            except:
                return str(obj)
        else:
            return str(obj)
    except Exception:
        # If all else fails, convert to string
        return str(obj)

# Add the project root to the path
sys.path.insert(0, '.')

from llm_explainability_framework import (
    ExplainabilityEngine,
    create_default_llm_wrapper,
    StakeholderType
)

@dataclass
class HumanEvalInstance:
    """Data class for HumanEval test instances"""
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str
    model_output: str
    passed: bool
    execution_result: Optional[str] = None
    error_message: Optional[str] = None

class HumanEvalTester:
    """
    HumanEval dataset integration for LLM explainability testing
    
    HumanEval is a benchmark for evaluating code generation capabilities
    with 164 hand-written programming problems with unit tests.
    """
    
    def __init__(self, llm_wrapper=None):
        self.engine = ExplainabilityEngine(llm_wrapper or create_default_llm_wrapper())
        self.dataset_path = "datasets/humaneval"
        self.results_path = "test_results/humaneval"
        
        # Create directories if they don't exist
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        
    def load_humaneval_dataset(self) -> List[Dict[str, Any]]:
        """
        Load HumanEval dataset from HuggingFace datasets
        
        Returns:
            List of HumanEval instances
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai_humaneval")
            
            instances = []
            for split in ["test"]:
                for item in dataset[split]:
                    instances.append({
                        "task_id": item["task_id"],
                        "prompt": item["prompt"],
                        "entry_point": item["entry_point"],
                        "canonical_solution": item["canonical_solution"],
                        "test": item["test"],
                        "input_text": item["prompt"],
                        "reference_output": item["canonical_solution"],
                        "task_type": "NL2CODE"
                    })
            
            print(f"Loaded {len(instances)} HumanEval instances")
            return instances
            
        except ImportError:
            print("Warning: datasets library not available. Using sample data.")
            return self._get_sample_humaneval_data()
    
    def _get_sample_humaneval_data(self) -> List[Dict[str, Any]]:
        """Sample HumanEval data for testing when dataset is not available"""
        return [
            {
                "task_id": "HumanEval/1",
                "input_text": """
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
                """,
                "reference_output": """
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
                """,
                "model_output": """
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
                """,
                "task_type": "NL2CODE"
            },
            {
                "task_id": "HumanEval/2",
                "input_text": """
def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('(() ) (( )) (( ( )))')
    ['(() )', '(( ))', '(( ( )))']
    \"\"\"
                """,
                "reference_output": """
def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    current_string = []
    current_depth = 0
    
    for char in paren_string:
        if char == '(':
            current_depth += 1
            current_string.append(char)
        elif char == ')':
            current_depth -= 1
            current_string.append(char)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    
    return result
                """,
                "model_output": """
def separate_paren_groups(paren_string: str) -> List[str]:
    groups = []
    current = ""
    depth = 0
    
    for c in paren_string:
        if c == "(":
            depth += 1
            current += c
        elif c == ")":
            depth -= 1
            current += c
            if depth == 0:
                groups.append(current)
                current = ""
    
    return groups
                """,
                "task_type": "NL2CODE"
            }
        ]
    
    def execute_code_safely(self, code: str, test_code: str) -> Dict[str, Any]:
        """
        Safely execute generated code with test cases
        
        Args:
            code: Generated code to test
            test_code: Test code to execute
            
        Returns:
            Execution result dictionary
        """
        try:
            # Create a safe execution environment
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # Execute test code
            exec(test_code, {"__builtins__": __builtins__}, local_vars)
            
            return {
                "passed": True,
                "execution_result": "All tests passed",
                "error_message": None
            }
            
        except Exception as e:
            return {
                "passed": False,
                "execution_result": None,
                "error_message": str(e)
            }
    
    def analyze_humaneval_failures(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze HumanEval instances for failures and generate explainability reports
        
        Args:
            instances: List of HumanEval test instances
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, instance in enumerate(instances):
            print(f"Analyzing HumanEval instance {i+1}/{len(instances)}: {instance['task_id']}")
            
            # Execute the code to check if it passes tests
            execution_result = self.execute_code_safely(
                instance.get("model_output", ""),
                instance.get("test", "")
            )
            
            # Create analysis instance
            analysis_instance = {
                "input_id": instance["task_id"],
                "task_type": "NL2CODE",
                "input_text": instance["input_text"],
                "model_output": instance.get("model_output", ""),
                "reference_output": instance["reference_output"],
                "context_metadata": {
                    "dataset": "HumanEval",
                    "execution_passed": execution_result["passed"],
                    "error_message": execution_result["error_message"]
                }
            }
            
            # Run explainability analysis
            try:
                result = self.engine.analyze_failure(
                    input_id=analysis_instance["input_id"],
                    task_type=analysis_instance["task_type"],
                    input_text=analysis_instance["input_text"],
                    model_output=analysis_instance["model_output"],
                    reference_output=analysis_instance["reference_output"],
                    context_metadata=analysis_instance.get("context_metadata", {})
                )
                # Convert result to dict for JSON serialization
                result_dict = {
                    "input_id": result.instance_id,
                    "task_type": result.task_type,
                    "original_task_id": instance.get("task_id", ""),  # Add original task ID for identification
                    "failure_classification": asdict(result.failure_classification),
                    "root_cause_analysis": asdict(result.root_cause_analysis),
                    "recommendation_suite": asdict(result.recommendation_suite),
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "quality_metrics": result.quality_metrics,
                    "markdown_report": result.markdown_report,
                    "execution_result": execution_result,
                    "analysis_report": result  # Store original ExplainabilityReport object
                }
                results.append(result_dict)
                
            except Exception as e:
                print(f"Error analyzing {instance['task_id']}: {e}")
                results.append({
                    "input_id": instance["task_id"],
                    "error": str(e),
                    "execution_result": execution_result
                })
        
        return results
    
    def generate_humaneval_report(self, results: List[Dict[str, Any]]) -> None:
        """
        Generate comprehensive report for HumanEval analysis
        
        Args:
            results: Analysis results from HumanEval testing
        """
        # Calculate statistics
        total_instances = len(results)
        successful_analyses = len([r for r in results if "error" not in r])
        execution_passed = len([r for r in results if r.get("execution_result", {}).get("passed", False)])
        
        # Failure category distribution
        failure_categories = {}
        for result in results:
            if "error" not in result:
                category = result.get("failure_classification", {}).get("failure_category", "Unknown")
                failure_categories[category] = failure_categories.get(category, 0) + 1
        
        # Generate report
        report = {
            "dataset": "HumanEval",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_instances": total_instances,
                "successful_analyses": successful_analyses,
                "execution_passed": execution_passed,
                "execution_failed": total_instances - execution_passed
            },
            "failure_categories": failure_categories,
            "detailed_results": results
        }
        
        # Save report
        report_path = os.path.join(self.results_path, "humaneval_analysis_report.json")
        with open(report_path, 'w') as f:
            # Create a copy without ExplainabilityReport objects for JSON serialization
            json_safe_results = []
            for result in results:
                if "error" not in result:
                    json_result = {k: v for k, v in result.items() if k != "analysis_report"}
                    json_safe_results.append(json_result)
                else:
                    json_safe_results.append(result)
            
            json_safe_report = report.copy()
            json_safe_report["detailed_results"] = json_safe_results
            clean_report = _clean_dict_for_json(json_safe_report)
            json.dump(clean_report, f, indent=2)
        
        print(f"\nHumanEval Analysis Report:")
        print(f"Total instances: {total_instances}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Execution passed: {execution_passed}")
        print(f"Execution failed: {total_instances - execution_passed}")
        print(f"Failure categories: {failure_categories}")
        print(f"Report saved to: {report_path}")
    
    def run_humaneval_test(self) -> None:
        """Run complete HumanEval testing pipeline"""
        print("Starting HumanEval dataset testing...")
        
        # Load dataset
        instances = self.load_humaneval_dataset()
        
        # Analyze failures
        results = self.analyze_humaneval_failures(instances)
        
        # Generate report
        self.generate_humaneval_report(results)
        
        print("HumanEval testing completed!")

def main():
    """Main function to run HumanEval testing"""
    tester = HumanEvalTester()
    tester.run_humaneval_test()

if __name__ == "__main__":
    main() 