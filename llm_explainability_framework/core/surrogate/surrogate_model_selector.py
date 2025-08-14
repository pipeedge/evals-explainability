#!/usr/bin/env python3
"""
Surrogate Model Selector for Black-Box LLM Explanation

This module provides a systematic approach to selecting appropriate surrogate models
for explaining black-box LLM behavior in the MADF framework.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import psutil
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TaskComplexity(Enum):
    """Task complexity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class DomainType(Enum):
    """Domain types for model selection"""
    CODE_GENERATION = "code_generation"
    TEXT_GENERATION = "text_generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    GENERAL = "general"

@dataclass
class SurrogateModel:
    """Surrogate model configuration"""
    name: str
    model_path: str
    size_mb: float
    parameters: int
    layers: int
    attention_heads: int
    domain: DomainType
    complexity: TaskComplexity
    pros: List[str]
    cons: List[str]
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

@dataclass
class SelectionCriteria:
    """Criteria for surrogate model selection"""
    task_complexity: TaskComplexity
    domain: DomainType
    max_inference_time_ms: float = 1000.0
    max_memory_mb: float = 2048.0
    min_accuracy: float = 0.7
    interpretability_priority: float = 0.5  # 0-1, higher = more interpretable

class SurrogateModelSelector:
    """Systematic surrogate model selector"""
    
    def __init__(self):
        self.available_models = self._initialize_available_models()
        self.performance_cache = {}
        
    def _initialize_available_models(self) -> Dict[str, SurrogateModel]:
        """Initialize available surrogate models"""
        
        models = {
            # Small, fast models
            "distilbert-base-uncased": SurrogateModel(
                name="DistilBERT Base",
                model_path="distilbert-base-uncased",
                size_mb=260.0,
                parameters=66_000_000,
                layers=6,
                attention_heads=12,
                domain=DomainType.GENERAL,
                complexity=TaskComplexity.LOW,
                pros=["Fast inference", "High interpretability", "Good attention visibility"],
                cons=["Limited capacity", "May miss complex patterns"]
            ),
            
            "albert-base-v2": SurrogateModel(
                name="ALBERT Base",
                model_path="albert-base-v2", 
                size_mb=45.0,
                parameters=12_000_000,
                layers=12,
                attention_heads=12,
                domain=DomainType.GENERAL,
                complexity=TaskComplexity.LOW,
                pros=["Parameter efficient", "Good balance", "Fast"],
                cons=["Still limited for complex tasks"]
            ),
            
            # Medium complexity models
            "bert-base-uncased": SurrogateModel(
                name="BERT Base",
                model_path="bert-base-uncased",
                size_mb=440.0,
                parameters=110_000_000,
                layers=12,
                attention_heads=12,
                domain=DomainType.GENERAL,
                complexity=TaskComplexity.MEDIUM,
                pros=["Good capacity", "Well-studied", "Balanced"],
                cons=["Slower than smaller models", "Medium interpretability"]
            ),
            
            # Code-specific models
            "microsoft-codebert-base": SurrogateModel(
                name="CodeBERT Base",
                model_path="microsoft/codebert-base",
                size_mb=500.0,
                parameters=125_000_000,
                layers=12,
                attention_heads=12,
                domain=DomainType.CODE_GENERATION,
                complexity=TaskComplexity.HIGH,
                pros=["Code-specific", "AST-aware", "Good for programming"],
                cons=["Larger model", "Slower inference", "Domain-specific"]
            ),
            
            "microsoft-graphcodebert-base": SurrogateModel(
                name="GraphCodeBERT Base", 
                model_path="microsoft/graphcodebert-base",
                size_mb=550.0,
                parameters=125_000_000,
                layers=12,
                attention_heads=12,
                domain=DomainType.CODE_GENERATION,
                complexity=TaskComplexity.HIGH,
                pros=["AST-aware", "Graph structure", "Advanced code understanding"],
                cons=["Largest model", "Slowest inference", "Complex"]
            ),
            
            # Text generation models
            "facebook-bart-base": SurrogateModel(
                name="BART Base",
                model_path="facebook/bart-base",
                size_mb=500.0,
                parameters=140_000_000,
                layers=12,
                attention_heads=16,
                domain=DomainType.TEXT_GENERATION,
                complexity=TaskComplexity.MEDIUM,
                pros=["Good for text generation", "Encoder-decoder", "Versatile"],
                cons=["Larger than BERT", "Slower inference"]
            ),
            
            # Translation models
            "Helsinki-NLP-opus-mt-en-fr": SurrogateModel(
                name="Helsinki MT EN-FR",
                model_path="Helsinki-NLP/opus-mt-en-fr",
                size_mb=300.0,
                parameters=75_000_000,
                layers=6,
                attention_heads=8,
                domain=DomainType.TRANSLATION,
                complexity=TaskComplexity.MEDIUM,
                pros=["Translation-specific", "Fast", "Good quality"],
                cons=["Language-specific", "Limited to EN-FR"]
            )
        }
        
        return models
    
    def assess_task_complexity(self, task_characteristics: Dict[str, Any]) -> TaskComplexity:
        """Assess task complexity based on characteristics"""
        
        complexity_score = 0
        
        # Input length complexity
        if task_characteristics.get("avg_input_length", 0) > 1000:
            complexity_score += 2
        elif task_characteristics.get("avg_input_length", 0) > 500:
            complexity_score += 1
            
        # Output length complexity
        if task_characteristics.get("avg_output_length", 0) > 500:
            complexity_score += 2
        elif task_characteristics.get("avg_output_length", 0) > 200:
            complexity_score += 1
            
        # Task type complexity
        task_type = task_characteristics.get("task_type", "general")
        if task_type in ["code_generation", "complex_reasoning"]:
            complexity_score += 3
        elif task_type in ["summarization", "translation"]:
            complexity_score += 2
        elif task_type in ["classification", "simple_generation"]:
            complexity_score += 1
            
        # Domain complexity
        domain = task_characteristics.get("domain", "general")
        if domain == "code":
            complexity_score += 2
        elif domain == "technical":
            complexity_score += 1
            
        # Determine complexity level
        if complexity_score >= 6:
            return TaskComplexity.HIGH
        elif complexity_score >= 3:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.LOW
    
    def measure_model_performance(self, model_name: str) -> Dict[str, float]:
        """Measure surrogate model performance metrics"""
        
        if model_name in self.performance_cache:
            return self.performance_cache[model_name]
        
        model_config = self.available_models[model_name]
        
        try:
            # Load model and measure performance
            start_time = time.time()
            model = SentenceTransformer(model_config.model_path)
            load_time = (time.time() - start_time) * 1000  # ms
            
            # Measure inference time
            test_text = "This is a test sentence for performance measurement."
            start_time = time.time()
            embeddings = model.encode([test_text])
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Measure memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            performance = {
                "load_time_ms": load_time,
                "inference_time_ms": inference_time,
                "memory_usage_mb": memory_usage,
                "embedding_dimension": embeddings.shape[1]
            }
            
            self.performance_cache[model_name] = performance
            return performance
            
        except Exception as e:
            print(f"Warning: Could not measure performance for {model_name}: {e}")
            return {
                "load_time_ms": 1000.0,
                "inference_time_ms": 100.0,
                "memory_usage_mb": model_config.size_mb,
                "embedding_dimension": 768
            }
    
    def select_surrogate_model(self, criteria: SelectionCriteria) -> SurrogateModel:
        """Select the best surrogate model based on criteria"""
        
        print(f"🔍 Selecting surrogate model for {criteria.domain.value} task (complexity: {criteria.task_complexity.value})")
        
        # Filter models by criteria
        candidate_models = []
        
        for model_name, model in self.available_models.items():
            # Check domain alignment
            if model.domain != criteria.domain and model.domain != DomainType.GENERAL:
                continue
                
            # Check complexity match
            if model.complexity.value != criteria.task_complexity.value:
                # Allow one level up for better approximation
                if not (criteria.task_complexity == TaskComplexity.LOW and model.complexity == TaskComplexity.MEDIUM):
                    continue
            
            # Measure performance
            performance = self.measure_model_performance(model_name)
            
            # Check resource constraints
            if performance["inference_time_ms"] > criteria.max_inference_time_ms:
                continue
                
            if performance["memory_usage_mb"] > criteria.max_memory_mb:
                continue
            
            # Calculate selection score
            score = self._calculate_selection_score(model, performance, criteria)
            
            candidate_models.append((model, score, performance))
        
        if not candidate_models:
            print("⚠️ No models meet the criteria, using fallback")
            return self._get_fallback_model(criteria)
        
        # Sort by score and return best
        candidate_models.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score, best_performance = candidate_models[0]
        
        print(f"✅ Selected: {best_model.name} (score: {best_score:.3f})")
        print(f"   Inference time: {best_performance['inference_time_ms']:.1f}ms")
        print(f"   Memory usage: {best_performance['memory_usage_mb']:.1f}MB")
        
        return best_model
    
    def _calculate_selection_score(self, model: SurrogateModel, performance: Dict[str, float], 
                                 criteria: SelectionCriteria) -> float:
        """Calculate selection score for a model"""
        
        score = 0.0
        
        # Base score from complexity match
        if model.complexity == criteria.task_complexity:
            score += 10.0
        elif model.complexity.value == "medium" and criteria.task_complexity.value == "low":
            score += 8.0
        else:
            score += 5.0
        
        # Domain alignment bonus
        if model.domain == criteria.domain:
            score += 5.0
        elif model.domain == DomainType.GENERAL:
            score += 2.0
        
        # Performance score (lower is better)
        time_score = max(0, 10 - (performance["inference_time_ms"] / 100))
        memory_score = max(0, 10 - (performance["memory_usage_mb"] / 200))
        
        score += time_score * 0.3
        score += memory_score * 0.2
        
        # Interpretability score
        interpretability_score = self._calculate_interpretability_score(model)
        score += interpretability_score * criteria.interpretability_priority * 10
        
        return score
    
    def _calculate_interpretability_score(self, model: SurrogateModel) -> float:
        """Calculate interpretability score for a model"""
        
        score = 0.0
        
        # Smaller models are more interpretable
        if model.parameters < 50_000_000:
            score += 1.0
        elif model.parameters < 100_000_000:
            score += 0.7
        else:
            score += 0.4
        
        # Fewer layers = more interpretable
        if model.layers <= 6:
            score += 1.0
        elif model.layers <= 12:
            score += 0.7
        else:
            score += 0.4
        
        # Fewer attention heads = more interpretable
        if model.attention_heads <= 8:
            score += 1.0
        elif model.attention_heads <= 12:
            score += 0.8
        else:
            score += 0.6
        
        return score / 3.0  # Normalize to 0-1
    
    def _get_fallback_model(self, criteria: SelectionCriteria) -> SurrogateModel:
        """Get fallback model when no models meet criteria"""
        
        # Return the smallest model that matches domain
        fallback_models = [
            "distilbert-base-uncased",
            "albert-base-v2", 
            "bert-base-uncased"
        ]
        
        for model_name in fallback_models:
            if model_name in self.available_models:
                return self.available_models[model_name]
        
        # Ultimate fallback
        return self.available_models["distilbert-base-uncased"]
    
    def validate_surrogate_model(self, surrogate_model: SurrogateModel, 
                               test_cases: List[Dict[str, str]]) -> Dict[str, float]:
        """Validate surrogate model performance on test cases"""
        
        print(f"🔍 Validating surrogate model: {surrogate_model.name}")
        
        try:
            model = SentenceTransformer(surrogate_model.model_path)
            
            validation_metrics = {
                "load_success": 1.0,
                "inference_success": 0.0,
                "embedding_quality": 0.0,
                "attention_visibility": 0.0
            }
            
            # Test inference on sample cases
            successful_inferences = 0
            total_embeddings = []
            
            for i, test_case in enumerate(test_cases[:10]):  # Test first 10 cases
                try:
                    input_text = test_case.get("input", "")
                    output_text = test_case.get("output", "")
                    
                    if input_text and output_text:
                        # Test embedding generation
                        input_emb = model.encode([input_text])[0]
                        output_emb = model.encode([output_text])[0]
                        
                        total_embeddings.extend([input_emb, output_emb])
                        successful_inferences += 1
                        
                except Exception as e:
                    print(f"Warning: Inference failed on test case {i}: {e}")
            
            validation_metrics["inference_success"] = successful_inferences / len(test_cases[:10])
            
            # Test embedding quality (diversity)
            if total_embeddings:
                embeddings_array = np.array(total_embeddings)
                embedding_variance = np.var(embeddings_array)
                validation_metrics["embedding_quality"] = min(1.0, embedding_variance / 10.0)
            
            # Attention visibility (simplified)
            if surrogate_model.layers <= 6 and surrogate_model.attention_heads <= 12:
                validation_metrics["attention_visibility"] = 0.8
            else:
                validation_metrics["attention_visibility"] = 0.5
            
            print(f"✅ Validation complete:")
            for metric, value in validation_metrics.items():
                print(f"   {metric}: {value:.3f}")
            
            return validation_metrics
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {
                "load_success": 0.0,
                "inference_success": 0.0,
                "embedding_quality": 0.0,
                "attention_visibility": 0.0
            }
    
    def get_model_recommendations(self, criteria: SelectionCriteria) -> List[Tuple[SurrogateModel, float]]:
        """Get ranked list of model recommendations"""
        
        recommendations = []
        
        for model_name, model in self.available_models.items():
            # Check basic compatibility
            if model.domain != criteria.domain and model.domain != DomainType.GENERAL:
                continue
            
            # Measure performance
            performance = self.measure_model_performance(model_name)
            
            # Calculate score
            score = self._calculate_selection_score(model, performance, criteria)
            
            recommendations.append((model, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations

def main():
    """Example usage of surrogate model selector"""
    
    selector = SurrogateModelSelector()
    
    # Example 1: Code generation task
    print("=" * 60)
    print("EXAMPLE 1: Code Generation Task")
    print("=" * 60)
    
    code_criteria = SelectionCriteria(
        task_complexity=TaskComplexity.HIGH,
        domain=DomainType.CODE_GENERATION,
        max_inference_time_ms=2000.0,
        max_memory_mb=4096.0,
        interpretability_priority=0.7
    )
    
    code_model = selector.select_surrogate_model(code_criteria)
    
    # Example 2: Simple text generation task
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Simple Text Generation Task")
    print("=" * 60)
    
    text_criteria = SelectionCriteria(
        task_complexity=TaskComplexity.LOW,
        domain=DomainType.TEXT_GENERATION,
        max_inference_time_ms=500.0,
        max_memory_mb=1024.0,
        interpretability_priority=0.9
    )
    
    text_model = selector.select_surrogate_model(text_criteria)
    
    # Example 3: Get recommendations
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Model Recommendations for Translation")
    print("=" * 60)
    
    translation_criteria = SelectionCriteria(
        task_complexity=TaskComplexity.MEDIUM,
        domain=DomainType.TRANSLATION,
        max_inference_time_ms=1000.0,
        max_memory_mb=2048.0,
        interpretability_priority=0.6
    )
    
    recommendations = selector.get_model_recommendations(translation_criteria)
    
    print("Top recommendations:")
    for i, (model, score) in enumerate(recommendations, 1):
        print(f"{i}. {model.name} (score: {score:.3f})")
        print(f"   Domain: {model.domain.value}, Complexity: {model.complexity.value}")
        print(f"   Parameters: {model.parameters:,}, Layers: {model.layers}")

if __name__ == "__main__":
    main() 