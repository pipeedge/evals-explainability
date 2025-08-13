#!/usr/bin/env python3
"""
Integration of Surrogate Model Selection into MADF Framework

This example shows how to use the surrogate model selector to improve
cross-attention approximation in the MADF framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm_explainability_framework'))

from surrogate_model_selector import (
    SurrogateModelSelector, SelectionCriteria, TaskComplexity, DomainType
)
from llm_explainability_framework.utils.metrics import AttentionAnalyzer
from llm_explainability_framework.core.explainability_engine import ExplainabilityEngine
from llm_explainability_framework.core.failure_classifier import FailureClassifier
from llm_explainability_framework.models.llm_wrapper import LLMWrapper

class EnhancedMADFFramework:
    """Enhanced MADF framework with adaptive surrogate model selection"""
    
    def __init__(self):
        self.surrogate_selector = SurrogateModelSelector()
        self.current_surrogate_model = None
        self.llm_wrapper = LLMWrapper()
        self.failure_classifier = FailureClassifier(self.llm_wrapper)
        self.explainability_engine = ExplainabilityEngine()
        
    def analyze_failure_with_adaptive_surrogate(self, task_type: str, input_text: str, 
                                              model_output: str, reference_output: str) -> dict:
        """
        Analyze failure using adaptively selected surrogate model
        """
        
        print(f"🔍 Analyzing failure with adaptive surrogate model selection")
        print(f"Task type: {task_type}")
        print(f"Input length: {len(input_text)} chars")
        print(f"Output length: {len(model_output)} chars")
        
        # Determine task characteristics
        task_characteristics = self._analyze_task_characteristics(
            task_type, input_text, model_output, reference_output
        )
        
        # Select appropriate surrogate model
        criteria = self._create_selection_criteria(task_characteristics)
        surrogate_model = self.surrogate_selector.select_surrogate_model(criteria)
        
        # Update attention analyzer with selected surrogate
        self._update_attention_analyzer(surrogate_model)
        
        # Perform failure analysis
        failure_instance = self._create_failure_instance(
            task_type, input_text, model_output, reference_output
        )
        
        # Get comprehensive analysis
        analysis_result = self.explainability_engine.analyze_failure(
            task_type, input_text, model_output, reference_output
        )
        
        # Add surrogate model information
        analysis_result["surrogate_model_info"] = {
            "model_name": surrogate_model.name,
            "model_path": surrogate_model.model_path,
            "domain": surrogate_model.domain.value,
            "complexity": surrogate_model.complexity.value,
            "parameters": surrogate_model.parameters,
            "layers": surrogate_model.layers,
            "attention_heads": surrogate_model.attention_heads
        }
        
        return analysis_result
    
    def _analyze_task_characteristics(self, task_type: str, input_text: str, 
                                    model_output: str, reference_output: str) -> dict:
        """Analyze task characteristics for surrogate model selection"""
        
        characteristics = {
            "task_type": task_type,
            "avg_input_length": len(input_text),
            "avg_output_length": len(model_output),
            "input_complexity": self._assess_text_complexity(input_text),
            "output_complexity": self._assess_text_complexity(model_output),
            "domain": self._determine_domain(task_type, input_text)
        }
        
        return characteristics
    
    def _assess_text_complexity(self, text: str) -> float:
        """Assess text complexity based on various factors"""
        
        complexity_score = 0.0
        
        # Length factor
        if len(text) > 1000:
            complexity_score += 0.3
        elif len(text) > 500:
            complexity_score += 0.2
        elif len(text) > 100:
            complexity_score += 0.1
        
        # Code indicators
        code_indicators = ['def ', 'class ', 'import ', 'function', 'return', 'if ', 'for ', 'while ']
        for indicator in code_indicators:
            if indicator in text:
                complexity_score += 0.1
        
        # Technical terms
        technical_terms = ['algorithm', 'complexity', 'optimization', 'architecture', 'framework']
        for term in technical_terms:
            if term in text:
                complexity_score += 0.05
        
        return min(1.0, complexity_score)
    
    def _determine_domain(self, task_type: str, input_text: str) -> str:
        """Determine the domain of the task"""
        
        if task_type in ['nl2code', 'code2nl'] or any(code_indicator in input_text.lower() 
                                                     for code_indicator in ['def ', 'class ', 'import ']):
            return 'code'
        elif task_type in ['translation']:
            return 'translation'
        elif task_type in ['summarization']:
            return 'summarization'
        else:
            return 'general'
    
    def _create_selection_criteria(self, task_characteristics: dict) -> SelectionCriteria:
        """Create selection criteria based on task characteristics"""
        
        # Determine complexity
        complexity = self.surrogate_selector.assess_task_complexity(task_characteristics)
        
        # Determine domain
        domain_mapping = {
            'code': DomainType.CODE_GENERATION,
            'translation': DomainType.TRANSLATION,
            'summarization': DomainType.SUMMARIZATION,
            'general': DomainType.TEXT_GENERATION
        }
        domain = domain_mapping.get(task_characteristics['domain'], DomainType.TEXT_GENERATION)
        
        # Set resource constraints based on complexity
        if complexity == TaskComplexity.LOW:
            max_inference_time = 500.0
            max_memory = 1024.0
            interpretability_priority = 0.9
        elif complexity == TaskComplexity.MEDIUM:
            max_inference_time = 1000.0
            max_memory = 2048.0
            interpretability_priority = 0.7
        else:  # HIGH
            max_inference_time = 2000.0
            max_memory = 4096.0
            interpretability_priority = 0.5
        
        return SelectionCriteria(
            task_complexity=complexity,
            domain=domain,
            max_inference_time_ms=max_inference_time,
            max_memory_mb=max_memory,
            interpretability_priority=interpretability_priority
        )
    
    def _update_attention_analyzer(self, surrogate_model):
        """Update the attention analyzer with the selected surrogate model"""
        
        print(f"🔄 Updating attention analyzer with {surrogate_model.name}")
        
        # In a real implementation, you would update the attention analyzer
        # to use the selected surrogate model for embeddings
        # For now, we'll just store the selection
        self.current_surrogate_model = surrogate_model
        
        print(f"   Model: {surrogate_model.name}")
        print(f"   Domain: {surrogate_model.domain.value}")
        print(f"   Complexity: {surrogate_model.complexity.value}")
        print(f"   Parameters: {surrogate_model.parameters:,}")
    
    def _create_failure_instance(self, task_type: str, input_text: str, 
                               model_output: str, reference_output: str):
        """Create a failure instance for analysis"""
        
        from llm_explainability_framework.core.failure_classifier import FailureInstance
        
        return FailureInstance(
            input_id=f"adaptive_{task_type}_{hash(input_text) % 10000}",
            task_type=task_type,
            input_text=input_text,
            model_output=model_output,
            reference_output=reference_output,
            context_metadata={
                "surrogate_model": self.current_surrogate_model.name if self.current_surrogate_model else "unknown",
                "analysis_timestamp": "2024-01-01T00:00:00Z"
            }
        )
    
    def compare_surrogate_models(self, task_type: str, input_text: str, 
                               model_output: str, reference_output: str) -> dict:
        """Compare different surrogate models for the same task"""
        
        print(f"🔍 Comparing surrogate models for {task_type} task")
        
        # Analyze task characteristics
        task_characteristics = self._analyze_task_characteristics(
            task_type, input_text, model_output, reference_output
        )
        
        # Get model recommendations
        criteria = self._create_selection_criteria(task_characteristics)
        recommendations = self.surrogate_selector.get_model_recommendations(criteria)
        
        comparison_results = {
            "task_characteristics": task_characteristics,
            "selection_criteria": {
                "complexity": criteria.task_complexity.value,
                "domain": criteria.domain.value,
                "max_inference_time_ms": criteria.max_inference_time_ms,
                "max_memory_mb": criteria.max_memory_mb,
                "interpretability_priority": criteria.interpretability_priority
            },
            "model_recommendations": []
        }
        
        # Test each recommended model
        for model, score in recommendations:
            print(f"\n📊 Testing {model.name} (score: {score:.3f})")
            
            # Validate model
            test_cases = [{"input": input_text, "output": model_output}]
            validation_metrics = self.surrogate_selector.validate_surrogate_model(model, test_cases)
            
            model_result = {
                "model_name": model.name,
                "model_path": model.model_path,
                "selection_score": score,
                "domain": model.domain.value,
                "complexity": model.complexity.value,
                "parameters": model.parameters,
                "layers": model.layers,
                "attention_heads": model.attention_heads,
                "validation_metrics": validation_metrics,
                "pros": model.pros,
                "cons": model.cons
            }
            
            comparison_results["model_recommendations"].append(model_result)
        
        return comparison_results

def main():
    """Example usage of enhanced MADF framework with adaptive surrogate selection"""
    
    enhanced_MADF = EnhancedMADFFramework()
    
    # Example 1: Code generation failure
    print("=" * 80)
    print("EXAMPLE 1: Code Generation Failure Analysis")
    print("=" * 80)
    
    code_input = "Write a Python function to calculate the factorial of a number"
    code_output = "def factorial(n): return n * factorial(n-1)"  # Missing base case
    code_reference = "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"
    
    code_analysis = enhanced_MADF.analyze_failure_with_adaptive_surrogate(
        "nl2code", code_input, code_output, code_reference
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"Selected surrogate: {code_analysis['surrogate_model_info']['model_name']}")
    print(f"Failure category: {code_analysis['failure_classification']['failure_category']}")
    print(f"Confidence: {code_analysis['failure_classification']['confidence_score']:.3f}")
    
    # Example 2: Text generation failure
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Text Generation Failure Analysis")
    print("=" * 80)
    
    text_input = "Summarize the key points of machine learning"
    text_output = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed."
    text_reference = "Machine learning is a field of AI that focuses on algorithms and statistical models to enable computers to improve performance on tasks through experience."
    
    text_analysis = enhanced_MADF.analyze_failure_with_adaptive_surrogate(
        "nl2nl", text_input, text_output, text_reference
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"Selected surrogate: {text_analysis['surrogate_model_info']['model_name']}")
    print(f"Failure category: {text_analysis['failure_classification']['failure_category']}")
    print(f"Confidence: {text_analysis['failure_classification']['confidence_score']:.3f}")
    
    # Example 3: Model comparison
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Surrogate Model Comparison")
    print("=" * 80)
    
    comparison = enhanced_MADF.compare_surrogate_models(
        "nl2code", code_input, code_output, code_reference
    )
    
    print(f"\n📊 Model Comparison Results:")
    print(f"Task complexity: {comparison['selection_criteria']['complexity']}")
    print(f"Domain: {comparison['selection_criteria']['domain']}")
    
    for i, model_result in enumerate(comparison['model_recommendations'][:3], 1):
        print(f"\n{i}. {model_result['model_name']}")
        print(f"   Score: {model_result['selection_score']:.3f}")
        print(f"   Parameters: {model_result['parameters']:,}")
        print(f"   Validation: {model_result['validation_metrics']['inference_success']:.3f}")
        print(f"   Pros: {', '.join(model_result['pros'][:2])}")

if __name__ == "__main__":
    main() 