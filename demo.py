#!/usr/bin/env python3
"""
LLM Explainability Framework Demo

This demo showcases the capabilities of the LLM explainability framework
with sample failure instances across NL2NL, NL2CODE, and CODE2NL tasks.
"""

import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any

from llm_explainability_framework import (
    ExplainabilityEngine,
    ExplainabilityReporter,
    create_default_llm_wrapper,
    StakeholderType
)

# Sample failure instances for demonstration
DEMO_INSTANCES = [
    # NL2NL Examples (Summarization)
    {
        "input_id": "nl2nl_001",
        "task_type": "NL2NL",
        "input_text": """
        Climate change is one of the most pressing issues of our time. Rising global temperatures 
        have led to melting ice caps, rising sea levels, and extreme weather events. Scientists 
        have established clear evidence that human activities, particularly the burning of fossil 
        fuels, are the primary cause of recent climate change. The Intergovernmental Panel on 
        Climate Change (IPCC) reports that global temperatures have risen by approximately 1.1°C 
        since pre-industrial times. Urgent action is needed to reduce greenhouse gas emissions 
        and transition to renewable energy sources to limit further warming to 1.5°C above 
        pre-industrial levels.
        """,
        "model_output": """
        Climate change is happening and it's getting warmer. Ice is melting and there are more 
        storms. Some scientists think humans might be involved through cars and factories. 
        The temperature has gone up by about 2°C in the last 100 years. We should probably 
        do something about it eventually.
        """,
        "reference_output": """
        Climate change, driven primarily by human activities like fossil fuel burning, has caused 
        global temperatures to rise by 1.1°C since pre-industrial times. This has resulted in 
        melting ice caps, rising sea levels, and extreme weather events. The IPCC emphasizes 
        the urgent need to reduce greenhouse gas emissions and transition to renewable energy 
        to limit warming to 1.5°C above pre-industrial levels.
        """,
        "context_metadata": {"source": "climate_report", "difficulty": "medium"}
    },
    
    # NL2CODE Examples (Text-to-Python)
    {
        "input_id": "nl2code_001", 
        "task_type": "NL2CODE",
        "input_text": "Write a Python function that calculates the factorial of a number using recursion. The function should handle edge cases for negative numbers and zero.",
        "model_output": """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
        """,
        "reference_output": """
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
        """,
        "context_metadata": {"programming_language": "python", "difficulty": "easy"}
    },
    
    {
        "input_id": "nl2code_002",
        "task_type": "NL2CODE", 
        "input_text": "Create a function that safely divides two numbers and handles division by zero with appropriate error handling.",
        "model_output": """
def divide(a, b):
    result = a / b
    return result
        """,
        "reference_output": """
def divide(a, b):
    try:
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        return None
        """,
        "context_metadata": {"programming_language": "python", "difficulty": "easy"}
    },
    
    # CODE2NL Examples (Code Documentation)
    {
        "input_id": "code2nl_001",
        "task_type": "CODE2NL",
        "input_text": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
        """,
        "model_output": """
        This function searches for a number in a list. It looks at the middle element 
        and compares it to what we're looking for. Then it keeps searching until it finds it.
        """,
        "reference_output": """
        This function implements binary search algorithm to find the index of a target value 
        in a sorted array. It maintains two pointers (left and right) and repeatedly divides 
        the search space in half by comparing the middle element with the target. If the middle 
        element equals the target, it returns the index. If the middle element is less than 
        the target, it searches the right half; otherwise, it searches the left half. 
        Returns -1 if the target is not found. Time complexity: O(log n), Space complexity: O(1).
        """,
        "context_metadata": {"algorithm_type": "search", "difficulty": "medium"}
    },
    
    # More complex examples
    {
        "input_id": "nl2nl_002",
        "task_type": "NL2NL",
        "input_text": """
        Artificial Intelligence has evolved rapidly in recent years, with Large Language Models 
        (LLMs) like GPT-3 and GPT-4 achieving remarkable capabilities in natural language 
        understanding and generation. These models are trained on vast amounts of text data 
        and use transformer architectures with attention mechanisms. However, they face 
        challenges including hallucination, bias, and lack of explainability. Current research 
        focuses on improving reliability, reducing computational costs, and developing better 
        evaluation metrics.
        """,
        "model_output": """
        AI is getting really good lately. There are these big language models that can write 
        text almost like humans. They learned from reading lots of books and websites. 
        But they sometimes make things up or say biased things. People are working on 
        making them more honest and explaining how they work. Also trying to make them 
        faster and cheaper to run.
        """,
        "reference_output": """
        Recent advances in AI have led to powerful Large Language Models (LLMs) such as 
        GPT-3 and GPT-4, which demonstrate exceptional natural language capabilities through 
        transformer architectures and attention mechanisms. Trained on extensive text corpora, 
        these models excel at language understanding and generation but face significant 
        challenges including hallucination, bias, and limited explainability. Current research 
        priorities include enhancing reliability, reducing computational costs, and developing 
        more effective evaluation methodologies.
        """,
        "context_metadata": {"source": "ai_research", "difficulty": "hard"}
    },
    
    {
        "input_id": "nl2code_003",
        "task_type": "NL2CODE",
        "input_text": "Implement a function that merges two sorted linked lists into a single sorted linked list.",
        "model_output": """
def merge_lists(list1, list2):
    result = []
    result.extend(list1)
    result.extend(list2)
    result.sort()
    return result
        """,
        "reference_output": """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_sorted_lists(list1, list2):
    dummy = ListNode(0)
    current = dummy
    
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    current.next = list1 or list2
    return dummy.next
        """,
        "context_metadata": {"data_structure": "linked_list", "difficulty": "medium"}
    }
]


class DemoRunner:
    """
    Demo runner that orchestrates the explainability framework demonstration
    """
    
    def __init__(self):
        print("🚀 Initializing LLM Explainability Framework Demo...")
        
        # Initialize LLM wrapper
        try:
            self.llm_wrapper = create_default_llm_wrapper()
            print("✅ LLM wrapper initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize LLM wrapper: {e}")
            print("🔄 Creating mock LLM wrapper for demonstration")
            self.llm_wrapper = self._create_mock_llm_wrapper()
        
        # Initialize explainability engine
        self.engine = ExplainabilityEngine(self.llm_wrapper)
        
        # Initialize reporter
        self.reporter = ExplainabilityReporter(output_dir="demo_reports")
        
        print("🎯 Demo initialization complete!")
    
    def _create_mock_llm_wrapper(self):
        """Create a mock LLM wrapper for demonstration when real LLM is unavailable"""
        class MockLLMWrapper:
            def invoke(self, prompt: str, **kwargs) -> str:
                # Provide mock responses based on prompt content
                if "classify the error" in prompt.lower():
                    if "syntax" in prompt.lower() or "code" in prompt.lower():
                        return '{"failure_category": "syntax_error"}'
                    elif "factual" in prompt.lower() or "information" in prompt.lower():
                        return '{"failure_category": "factual_inconsistency"}'
                    else:
                        return '{"failure_category": "incomplete_explanation"}'
                
                elif "root cause analysis" in prompt.lower():
                    return """
### Root Cause Analysis Report

**1. Analysis of Input Intent:**
   - The user requested a specific task that required careful attention to detail and accuracy.

**2. Key Discrepancies Observed:**
   - The model output lacks important details present in the reference output
   - There are accuracy issues in the generated response
   - The structure and completeness do not match expectations

**3. Explanation of Failure:**
   - The model appears to have oversimplified the task requirements
   - Important constraints and edge cases were not properly addressed

**4. Inferred Root Cause:**
   - Insufficient attention to task-specific requirements and constraints
"""
                
                elif "actionable recommendations" in prompt.lower():
                    return """
### Actionable Recommendations

**1. For Prompt Engineering / Input Refinement:**
   - Add more specific instructions about required output format and completeness
   - Include examples of expected output quality and detail level
   - Specify edge cases and constraints that must be handled

**2. For Data Augmentation / Fine-Tuning:**
   - Include more training examples with similar complexity and requirements
   - Add negative examples showing common failure patterns to avoid
   - Enhance training data with expert-annotated examples

**3. For Model Configuration (If Applicable):**
   - Adjust temperature parameter to reduce randomness in outputs
   - Consider using higher top_p values for more diverse yet accurate responses
   - Implement output validation and constraint checking
"""
                
                else:
                    return "Mock LLM response for demonstration purposes."
            
            def get_model_info(self):
                return {"provider": "mock", "type": "demo"}
        
        return MockLLMWrapper()
    
    def run_single_analysis_demo(self, instance_index: int = 0) -> None:
        """Run a single analysis demonstration"""
        if instance_index >= len(DEMO_INSTANCES):
            print(f"❌ Instance index {instance_index} out of range")
            return
        
        instance = DEMO_INSTANCES[instance_index]
        print(f"\n🔍 Analyzing single instance: {instance['input_id']}")
        print(f"📋 Task type: {instance['task_type']}")
        
        # Run analysis
        report = self.engine.analyze_failure(**instance)
        
        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"   - Failure Category: {report.failure_classification.failure_category}")
        print(f"   - Confidence Score: {report.confidence_score:.3f}")
        print(f"   - Processing Time: {report.processing_time:.2f}s")
        print(f"   - Primary Root Cause: {report.root_cause_analysis.primary_cause}")
        print(f"   - Recommendations: {len(report.recommendation_suite.recommendations)} generated")
        
        # Save individual report
        self.engine.save_report(report, output_dir="demo_reports/individual")
        
        return report
    
    def run_batch_analysis_demo(self) -> List[Any]:
        """Run batch analysis demonstration"""
        print(f"\n📊 Running batch analysis on {len(DEMO_INSTANCES)} instances...")
        
        # Run batch analysis
        reports = self.engine.batch_analyze(DEMO_INSTANCES)
        
        # Save individual reports for each instance
        print(f"🚀 Starting batch analysis of {len(DEMO_INSTANCES)} instances...")
        for i, report in enumerate(reports):
            print(f"📊 Processing instance {i+1}/{len(reports)}")
            print(f"🔍 Classifying failure for {report.instance_id}...")
            print(f"🧬 Analyzing root cause for {report.instance_id}...")
            print(f"💡 Generating recommendations for {report.instance_id}...")
            print(f"📄 Generating report for {report.instance_id}...")
            print(f"✅ Analysis completed for {report.instance_id} in {report.processing_time:.2f}s")
            
            # Save individual report
            self.engine.save_report(report, output_dir="demo_reports/individual")
        
        # Display summary
        print(f"\n📈 Batch Analysis Summary:")
        print(f"   - Total instances analyzed: {len(reports)}")
        
        # Analyze results
        failure_categories = {}
        total_time = 0
        total_confidence = 0
        
        for report in reports:
            category = report.failure_classification.failure_category
            failure_categories[category] = failure_categories.get(category, 0) + 1
            total_time += report.processing_time
            total_confidence += report.confidence_score
        
        print(f"   - Average processing time: {total_time/len(reports):.2f}s")
        print(f"   - Average confidence: {total_confidence/len(reports):.3f}")
        print(f"   - Failure category distribution:")
        for category, count in failure_categories.items():
            print(f"     * {category}: {count} instances")
        
        return reports
    
    def run_stakeholder_analysis_demo(self) -> None:
        """Demonstrate stakeholder-specific analysis"""
        print(f"\n👥 Running stakeholder-specific analysis...")
        
        # Select a sample instance
        instance = DEMO_INSTANCES[0]
        
        stakeholders = [
            StakeholderType.DEVELOPER,
            StakeholderType.MANAGER,
            StakeholderType.RESEARCHER
        ]
        
        stakeholder_reports = {}
        
        for stakeholder in stakeholders:
            print(f"   🎯 Analyzing for {stakeholder.value}...")
            report = self.engine.analyze_failure(
                **instance,
                target_stakeholder=stakeholder
            )
            stakeholder_reports[stakeholder] = report
            
            # Show stakeholder-specific insights
            recs = report.recommendation_suite.recommendations
            stakeholder_recs = [r for r in recs if r.stakeholder_type == stakeholder]
            print(f"     - {len(stakeholder_recs)} stakeholder-specific recommendations")
            if stakeholder_recs:
                top_rec = max(stakeholder_recs, key=lambda x: x.priority_score)
                print(f"     - Top recommendation: {top_rec.title}")
        
        return stakeholder_reports
    
    def run_visualization_demo(self, reports: List[Any]) -> None:
        """Demonstrate visualization capabilities"""
        print(f"\n📊 Generating comprehensive visualizations...")
        
        # Generate comprehensive report
        html_report_path = self.reporter.generate_comprehensive_report(
            reports, output_name="demo_analysis"
        )
        
        print(f"   ✅ Comprehensive report generated: {html_report_path}")
        
        # Generate individual HTML reports
        for i, report in enumerate(reports[:3]):  # First 3 reports
            html_content = self.reporter.generate_individual_report_html(report)
            html_path = Path("demo_reports") / f"{report.instance_id}_individual.html"
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"   📄 Individual report saved: {html_path}")
    
    def run_performance_analysis_demo(self) -> None:
        """Demonstrate performance analysis capabilities"""
        print(f"\n⚡ Performance Analysis Demo...")
        
        # Get engine performance stats
        perf_stats = self.engine.get_performance_stats()
        print(f"   📊 Engine Performance:")
        for metric, value in perf_stats.items():
            if isinstance(value, float):
                print(f"     - {metric}: {value:.3f}")
            else:
                print(f"     - {metric}: {value}")
        
        # Get analysis summary
        summary = self.engine.get_analysis_summary()
        print(f"\n   📈 Analysis Summary:")
        if "total_analyses" in summary:
            print(f"     - Total analyses: {summary['total_analyses']}")
            print(f"     - Average confidence: {summary.get('average_confidence', 0):.3f}")
            print(f"     - Average processing time: {summary.get('average_processing_time', 0):.3f}s")
    
    def demonstrate_innovative_features(self) -> None:
        """Highlight the innovative features of the framework"""
        print(f"\n🚀 Innovative Features Demonstration:")
        
        print(f"\n   🧠 1. Multi-Dimensional Failure Analysis:")
        print(f"      - Semantic attention patterns")
        print(f"      - Causal graph discovery") 
        print(f"      - Counterfactual reasoning")
        
        print(f"\n   🔗 2. Hybrid AI Pipeline:")
        print(f"      - Automated semantic classification")
        print(f"      - LLM-enhanced root cause analysis")
        print(f"      - Adaptive recommendation ranking")
        
        print(f"\n   👥 3. Multi-Stakeholder Optimization:")
        print(f"      - Developer-focused technical recommendations")
        print(f"      - Manager-oriented resource planning")
        print(f"      - Researcher-targeted methodological insights")
        
        print(f"\n   📊 4. Comprehensive Explainability:")
        print(f"      - Interactive visualization dashboards")
        print(f"      - Causal network analysis")
        print(f"      - Performance trend analysis")
        
        print(f"\n   🔄 5. Adaptive Learning:")
        print(f"      - Feedback-driven improvement")
        print(f"      - Context-aware optimization")
        print(f"      - Continuous model refinement")
    
    def run_complete_demo(self) -> None:
        """Run the complete demonstration"""
        print("=" * 80)
        print("🎯 LLM EXPLAINABILITY FRAMEWORK - COMPREHENSIVE DEMO")
        print("=" * 80)
        
        try:
            # 1. Single analysis demo
            print("\n" + "🔍 PHASE 1: SINGLE ANALYSIS DEMONSTRATION".center(80))
            single_report = self.run_single_analysis_demo(0)
            
            # 2. Batch analysis demo
            print("\n" + "📊 PHASE 2: BATCH ANALYSIS DEMONSTRATION".center(80))
            batch_reports = self.run_batch_analysis_demo()
            
            # 3. Stakeholder analysis demo
            print("\n" + "👥 PHASE 3: STAKEHOLDER-SPECIFIC ANALYSIS".center(80))
            stakeholder_reports = self.run_stakeholder_analysis_demo()
            
            # 4. Visualization demo
            print("\n" + "📊 PHASE 4: VISUALIZATION AND REPORTING".center(80))
            self.run_visualization_demo(batch_reports)
            
            # 5. Performance analysis
            print("\n" + "⚡ PHASE 5: PERFORMANCE ANALYSIS".center(80))
            self.run_performance_analysis_demo()
            
            # 6. Innovative features highlight
            print("\n" + "🚀 PHASE 6: INNOVATIVE FEATURES SHOWCASE".center(80))
            self.demonstrate_innovative_features()
            
            print("\n" + "=" * 80)
            print("✅ DEMO COMPLETED SUCCESSFULLY!")
            print("📁 Check 'demo_reports' directory for generated reports")
            print("🌐 Open HTML files in browser for interactive visualizations")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for the demo"""
    try:
        demo = DemoRunner()
        demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed to initialize: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 