#!/usr/bin/env python3
"""
Integration of Surrogate Model Selection into MADF Framework

This module shows how to use the surrogate model selector to improve
cross-attention approximation in the MADF framework.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .surrogate_model_selector import (
    SurrogateModelSelector, SelectionCriteria, TaskComplexity, DomainType
)
from ...utils.metrics import AttentionAnalyzer
from ..explainability_engine import ExplainabilityEngine
from ..failure_classifier import FailureClassifier
from ...models.llm_wrapper import LLMWrapper

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not available. Using fallback similarity computation.")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers not available. Using fallback attention computation.")

@dataclass
class SurrogateAttentionResult:
    """Result from surrogate model attention computation"""
    attention_weights: np.ndarray
    model_name: str
    confidence: float
    computation_time: float
    fallback_used: bool = False

class SurrogateModelIntegration:
    """Integration layer for surrogate models in MADF framework"""
    
    def __init__(self):
        self.selector = SurrogateModelSelector()
        self.current_surrogate_model = None
        self.current_tokenizer = None
        self.current_model = None
        self.attention_cache = {}
        
    def compute_enhanced_cross_attention(self, input_text: str, output_text: str,
                                       task_type: str = "general") -> SurrogateAttentionResult:
        """Compute cross-attention using surrogate model when available"""
        
        import time
        start_time = time.time()
        
        try:
            # Select and load surrogate model
            surrogate_model = self._select_and_load_surrogate_model(
                task_type, input_text, output_text
            )
            
            # Try to use surrogate model for attention computation
            if self.current_model is not None and TRANSFORMERS_AVAILABLE:
                attention_weights = self._compute_surrogate_attention(
                    input_text, output_text, surrogate_model
                )
                confidence = 0.8  # High confidence when surrogate model is used
                fallback_used = False
            else:
                # Fallback to basic similarity-based attention
                attention_weights = self._compute_fallback_attention(input_text, output_text)
                confidence = 0.5  # Lower confidence for fallback
                fallback_used = True
                
        except Exception as e:
            print(f"⚠️  Surrogate attention computation failed: {e}")
            # Fallback to basic attention
            attention_weights = self._compute_fallback_attention(input_text, output_text)
            confidence = 0.3  # Low confidence for error fallback
            fallback_used = True
        
        computation_time = time.time() - start_time
        
        return SurrogateAttentionResult(
            attention_weights=attention_weights,
            model_name=surrogate_model.name if 'surrogate_model' in locals() else "fallback",
            confidence=confidence,
            computation_time=computation_time,
            fallback_used=fallback_used
        )
    
    def _select_and_load_surrogate_model(self, task_type: str, input_text: str, 
                                      output_text: str):
        """Select and load appropriate surrogate model for the task"""
        
        # Analyze task characteristics
        task_characteristics = self._analyze_task_characteristics(
            task_type, input_text, output_text
        )
        
        # Create selection criteria
        criteria = self._create_selection_criteria(task_characteristics)
        
        # Select surrogate model
        surrogate_model = self.selector.select_surrogate_model(criteria)
        
        # Load the model if different from current
        if (self.current_surrogate_model is None or 
            self.current_surrogate_model.model_path != surrogate_model.model_path):
            
            self._load_surrogate_model(surrogate_model)
        
        return surrogate_model
    
    def _analyze_task_characteristics(self, task_type: str, input_text: str, 
                                    output_text: str) -> Dict[str, Any]:
        """Analyze task characteristics for surrogate model selection"""
        
        # Determine domain
        domain = self._determine_domain(task_type, input_text)
        
        # Assess complexity
        complexity = self._assess_complexity(input_text, output_text)
        
        # Estimate resource requirements
        input_length = len(input_text.split())
        output_length = len(output_text.split())
        
        return {
            "domain": domain,
            "complexity": complexity,
            "input_length": input_length,
            "output_length": output_length,
            "total_length": input_length + output_length
        }
    
    def _determine_domain(self, task_type: str, input_text: str):
        """Determine the domain of the task"""
        
        # Check for code-related patterns
        code_indicators = ['def ', 'class ', 'import ', 'function', 'return', 'if ', 'for ', 'while ']
        if any(indicator in input_text for indicator in code_indicators):
            return DomainType.CODE_GENERATION
        
        # Check for translation patterns
        if task_type in ['translation', 'translate']:
            return DomainType.TRANSLATION
        
        # Check for summarization patterns
        if task_type in ['summarization', 'summarize']:
            return DomainType.SUMMARIZATION
        
        # Check for question answering patterns
        if '?' in input_text or task_type in ['qa', 'question_answering']:
            return DomainType.QUESTION_ANSWERING
        
        # Default to general
        return DomainType.GENERAL
    
    def _assess_complexity(self, input_text: str, output_text: str):
        """Assess task complexity based on text characteristics"""
        
        total_length = len(input_text.split()) + len(output_text.split())
        
        if total_length < 50:
            return TaskComplexity.LOW
        elif total_length < 200:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.HIGH
    
    def _create_selection_criteria(self, characteristics: Dict[str, Any]):
        """Create selection criteria based on task characteristics"""
        
        # Adjust resource limits based on complexity
        if characteristics["complexity"] == TaskComplexity.LOW:
            max_inference_time = 200.0
            max_memory = 512.0
        elif characteristics["complexity"] == TaskComplexity.MEDIUM:
            max_inference_time = 500.0
            max_memory = 1024.0
        else:
            max_inference_time = 1000.0
            max_memory = 2048.0
        
        return SelectionCriteria(
            task_complexity=characteristics["complexity"],
            domain=characteristics["domain"],
            max_inference_time_ms=max_inference_time,
            max_memory_mb=max_memory,
            interpretability_priority=0.7  # High priority for explainability
        )
    
    def _load_surrogate_model(self, surrogate_model):
        """Load surrogate model and tokenizer"""
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  Transformers not available, skipping model loading")
            self.current_surrogate_model = surrogate_model
            self.current_tokenizer = None
            self.current_model = None
            return
        
        try:
            print(f"🔄 Loading surrogate model: {surrogate_model.name}")
            
            # Load tokenizer and model
            self.current_tokenizer = AutoTokenizer.from_pretrained(surrogate_model.model_path)
            self.current_model = AutoModel.from_pretrained(surrogate_model.model_path)
            
            # Set model to evaluation mode
            self.current_model.eval()
            
            self.current_surrogate_model = surrogate_model
            
            print(f"✅ Successfully loaded: {surrogate_model.name}")
            
        except Exception as e:
            print(f"❌ Failed to load surrogate model: {e}")
            print("🔄 Using fallback attention computation")
            self.current_surrogate_model = surrogate_model
            self.current_tokenizer = None
            self.current_model = None
    
    def _compute_surrogate_attention(self, input_text: str, output_text: str, surrogate_model):
        """Compute attention using surrogate model"""
        
        if self.current_tokenizer is None or self.current_model is None:
            raise ValueError("Surrogate model not loaded")
        
        # Tokenize input and output
        input_tokens = self.current_tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        output_tokens = self.current_tokenizer(
            output_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Get model outputs
        with torch.no_grad():
            input_outputs = self.current_model(**input_tokens, output_attentions=True)
            output_outputs = self.current_model(**output_tokens, output_attentions=True)
        
        # Extract attention weights from the last layer
        input_attention = input_outputs.attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
        output_attention = output_outputs.attentions[-1][0]
        
        # Average across attention heads
        input_attention_avg = input_attention.mean(dim=0)  # Shape: [seq_len, seq_len]
        output_attention_avg = output_attention.mean(dim=0)
        
        # Convert to numpy and handle dimension mismatch
        input_attn_np = input_attention_avg.numpy()
        output_attn_np = output_attention_avg.numpy()
        
        # Create cross-attention matrix (simplified)
        cross_attention = np.outer(
            input_attn_np.mean(axis=1),  # Average attention for each input token
            output_attn_np.mean(axis=1)  # Average attention for each output token
        )
        
        # Normalize
        cross_attention = cross_attention / (cross_attention.sum() + 1e-8)
        
        return cross_attention
    
    def _compute_fallback_attention(self, input_text: str, output_text: str) -> np.ndarray:
        """Compute fallback attention using basic similarity"""
        
        # Simple word-level tokenization
        input_tokens = input_text.split() if input_text.strip() else ["empty"]
        output_tokens = output_text.split() if output_text.strip() else ["empty"]
        
        # Limit token count
        input_tokens = input_tokens[:50]
        output_tokens = output_tokens[:50]
        
        # Initialize attention matrix
        attention_matrix = np.zeros((len(input_tokens), len(output_tokens)))
        
        # Compute simple similarity-based attention
        for i, input_token in enumerate(input_tokens):
            for j, output_token in enumerate(output_tokens):
                # Simple character-level similarity
                similarity = self._simple_similarity(input_token, output_token)
                attention_matrix[i, j] = similarity
        
        # Handle all-zero matrices
        if np.all(attention_matrix == 0):
            np.fill_diagonal(attention_matrix, 1e-8)
        
        # Clean and normalize
        attention_matrix = np.nan_to_num(attention_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        attention_matrix = attention_matrix + 1e-10
        
        # Apply softmax
        attention_tensor = torch.tensor(attention_matrix, dtype=torch.float32)
        attention_weights = F.softmax(attention_tensor, dim=1).numpy()
        
        return attention_weights
    
    def _simple_similarity(self, token1: str, token2: str) -> float:
        """Compute simple similarity between two tokens"""
        
        # Character-level Jaccard similarity
        set1 = set(token1.lower())
        set2 = set(token2.lower())
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

class EnhancedMADFFramework:
    """Enhanced MADF framework with adaptive surrogate model selection"""
    
    def __init__(self, pattern_library=None):
        self.surrogate_integration = SurrogateModelIntegration()
        self.llm_wrapper = LLMWrapper()
        self.failure_classifier = FailureClassifier(self.llm_wrapper, pattern_library=pattern_library)
        self.explainability_engine = ExplainabilityEngine(pattern_library=pattern_library)
        
    def analyze_failure_with_adaptive_surrogate(self, task_type: str, input_text: str, 
                                              model_output: str, reference_output: str) -> dict:
        """
        Analyze failure using adaptively selected surrogate model
        """
        
        print(f"🔍 Analyzing failure with adaptive surrogate model selection")
        print(f"Task type: {task_type}")
        print(f"Input length: {len(input_text)} chars")
        print(f"Output length: {len(model_output)} chars")
        
        # Get enhanced cross-attention using surrogate models
        surrogate_result = self.surrogate_integration.compute_enhanced_cross_attention(
            input_text, model_output, task_type
        )
        
        print(f"🔧 Surrogate model used: {surrogate_result.model_name}")
        print(f"   Confidence: {surrogate_result.confidence:.3f}")
        print(f"   Computation time: {surrogate_result.computation_time:.3f}s")
        print(f"   Fallback used: {surrogate_result.fallback_used}")
        
        # Perform failure analysis with enhanced attention
        analysis_result = self.explainability_engine.analyze_failure(
            task_type, input_text, model_output, reference_output
        )
        
        # Add surrogate model information
        analysis_result.surrogate_model_info = {
            "model_name": surrogate_result.model_name,
            "confidence": surrogate_result.confidence,
            "computation_time": surrogate_result.computation_time,
            "fallback_used": surrogate_result.fallback_used,
            "attention_shape": surrogate_result.attention_weights.shape
        }
        
        return analysis_result

def main():
    """Example usage of surrogate model integration"""
    
    integration = SurrogateModelIntegration()
    
    # Example inputs
    input_text = "Write a function to calculate factorial"
    output_text = "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"
    
    # Compute enhanced cross-attention
    result = integration.compute_enhanced_cross_attention(
        input_text, output_text, task_type="code_generation"
    )
    
    print(f"✅ Cross-attention computed:")
    print(f"   Model: {result.model_name}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Computation time: {result.computation_time:.3f}s")
    print(f"   Fallback used: {result.fallback_used}")
    print(f"   Attention shape: {result.attention_weights.shape}")

if __name__ == "__main__":
    main() 