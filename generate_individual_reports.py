#!/usr/bin/env python3
"""
Script to generate individual reports for all demo instances
"""

import sys
import os
sys.path.insert(0, '.')

from llm_explainability_framework import ExplainabilityEngine, create_default_llm_wrapper

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
        "model_output": "This function performs a binary search algorithm to find a target value in a sorted array.",
        "reference_output": "This function implements binary search to find a target value in a sorted array. It uses two pointers (left and right) and compares the middle element with the target, adjusting the search range accordingly. Returns the index of the target if found, or -1 if not found.",
        "context_metadata": {"programming_language": "python", "difficulty": "medium"}
    },
    
    # Additional NL2NL example
    {
        "input_id": "nl2nl_002",
        "task_type": "NL2NL",
        "input_text": "Summarize the key benefits of renewable energy sources in 2-3 sentences.",
        "model_output": "Renewable energy is good for the environment. It doesn't pollute and comes from natural sources like sun and wind. It's becoming cheaper and more popular.",
        "reference_output": "Renewable energy sources provide clean, sustainable power that significantly reduces greenhouse gas emissions and air pollution. They offer energy security through domestic production and create jobs in manufacturing and installation sectors. Additionally, renewable energy costs have decreased dramatically, making them increasingly competitive with fossil fuels.",
        "context_metadata": {"source": "energy_report", "difficulty": "easy"}
    },
    
    # Additional NL2CODE example
    {
        "input_id": "nl2code_003",
        "task_type": "NL2CODE",
        "input_text": "Write a function to check if a string is a palindrome (reads the same forwards and backwards).",
        "model_output": """
def is_palindrome(s):
    return s == s[::-1]
        """,
        "reference_output": """
def is_palindrome(s):
    # Remove non-alphanumeric characters and convert to lowercase
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
        """,
        "context_metadata": {"programming_language": "python", "difficulty": "easy"}
    }
]

def main():
    print("🚀 Generating individual reports for all demo instances...")
    
    # Initialize the engine
    llm_wrapper = create_default_llm_wrapper()
    engine = ExplainabilityEngine(llm_wrapper)
    
    # Create output directory
    os.makedirs("demo_reports/individual", exist_ok=True)
    
    # Generate reports for each instance
    for i, instance in enumerate(DEMO_INSTANCES):
        print(f"\n📊 Processing instance {i+1}/{len(DEMO_INSTANCES)}: {instance['input_id']}")
        
        # Run analysis
        report = engine.analyze_failure(**instance)
        
        # Save individual report
        engine.save_report(report, output_dir="demo_reports/individual")
        
        print(f"✅ Generated report for {instance['input_id']}")
        print(f"   - Failure Category: {report.failure_classification.failure_category}")
        print(f"   - Confidence Score: {report.confidence_score:.3f}")
        print(f"   - Processing Time: {report.processing_time:.2f}s")
        print(f"   - Recommendations: {len(report.recommendation_suite.recommendations)} generated")
    
    print(f"\n✅ Successfully generated individual reports for all {len(DEMO_INSTANCES)} instances!")
    print("📁 Check 'demo_reports/individual' directory for the reports")

if __name__ == "__main__":
    main() 