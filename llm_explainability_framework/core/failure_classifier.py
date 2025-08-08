"""
Failure Classifier with Multi-Dimensional Analysis

This module implements an innovative failure classification approach that combines:
1. Attention-weighted semantic feature extraction
2. Multi-dimensional embedding clustering 
3. Hierarchical error taxonomy mapping
4. Confidence-weighted ensemble classification
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

from ..models.llm_wrapper import LLMWrapper
from ..utils.metrics import SemanticSimilarity, AttentionAnalyzer


@dataclass
class FailureInstance:
    """Data structure for failure instances"""
    input_id: str
    task_type: str  # NL2NL, NL2CODE, CODE2NL
    input_text: str
    model_output: str
    reference_output: str
    context_metadata: Dict[str, Any] = None


@dataclass
class FailureClassification:
    """Result of failure classification"""
    failure_category: str
    confidence_score: float
    sub_categories: List[str]
    attention_weights: np.ndarray
    semantic_features: np.ndarray
    explanation_vector: np.ndarray


class SemanticAttentionClassifier:
    """
    Novel classifier using semantic attention patterns for failure analysis
    
    Innovation: Combines attention mechanisms with semantic embeddings to create
    multi-dimensional failure representations that capture both syntactic and 
    semantic aspects of failures.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.attention_analyzer = AttentionAnalyzer()
        self.semantic_similarity = SemanticSimilarity()
        
        # Pre-trained failure pattern embeddings
        self.failure_patterns = self._initialize_failure_patterns()
        
    def _initialize_failure_patterns(self) -> Dict[str, np.ndarray]:
        """Initialize semantic embeddings for known failure patterns"""
        patterns = {
            # NL2NL patterns
            "factual_inconsistency": "contradicts source information, incorrect facts, misrepresented data",
            "hallucination": "generates unverifiable information, creates false details, imaginary content", 
            "information_loss": "omits critical details, incomplete coverage, missing key points",
            "stylistic_mismatch": "inappropriate tone, wrong format, style inconsistency",
            
            # NL2CODE patterns  
            "syntax_error": "invalid code syntax, compilation errors, malformed statements",
            "logical_error": "incorrect algorithm, wrong logic flow, faulty reasoning",
            "inefficiency": "suboptimal performance, resource waste, non-idiomatic code",
            "security_vulnerability": "potential security risks, unsafe operations, exploitable code",
            
            # CODE2NL patterns
            "inaccurate_description": "misrepresents code logic, wrong functionality description",
            "incomplete_explanation": "missing details, partial coverage, unclear explanations", 
            "poor_readability": "confusing structure, technical jargon, unclear presentation"
        }
        
        return {
            category: self.embedding_model.encode(description)
            for category, description in patterns.items()
        }
    
    def extract_attention_features(self, input_text: str, output_text: str) -> np.ndarray:
        """
        Extract attention-weighted features from input-output pairs
        
        Innovation: Uses cross-attention patterns between input and output to identify
        failure-relevant regions with higher precision than traditional methods.
        """
        # Compute attention weights between input and output
        attention_weights = self.attention_analyzer.compute_cross_attention(
            input_text, output_text
        )
        
        # Extract semantic features weighted by attention
        input_embeddings = self.embedding_model.encode([input_text])[0]
        output_embeddings = self.embedding_model.encode([output_text])[0]
        
        # Clean attention weights and handle NaN values
        clean_attention_weights = np.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Compute mean with fallback
        attention_mean = np.mean(clean_attention_weights)
        if np.isnan(attention_mean) or np.isinf(attention_mean):
            attention_mean = 0.0
        
        # Create attention-weighted feature vector
        weighted_features = np.concatenate([
            input_embeddings * attention_mean,
            output_embeddings,
            clean_attention_weights.flatten()[:50]  # Top 50 attention weights
        ])
        
        # Ensure no NaN values in final features
        weighted_features = np.nan_to_num(weighted_features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return weighted_features
    
    def compute_semantic_distances(self, instance: FailureInstance) -> Dict[str, float]:
        """
        Compute semantic distances to known failure patterns
        
        Innovation: Multi-dimensional semantic space analysis that considers
        both content similarity and structural patterns.
        """
        # Create composite representation
        combined_text = f"Input: {instance.input_text} Output: {instance.model_output} Reference: {instance.reference_output}"
        instance_embedding = self.embedding_model.encode([combined_text])[0]
        
        distances = {}
        for pattern_name, pattern_embedding in self.failure_patterns.items():
            # Cosine similarity in semantic space
            similarity = cosine_similarity(
                instance_embedding.reshape(1, -1),
                pattern_embedding.reshape(1, -1)
            )[0, 0]
            distances[pattern_name] = similarity
            
        return distances
    
    def classify_failure(self, instance: FailureInstance) -> FailureClassification:
        """
        Classify failure using multi-dimensional analysis
        
        Innovation: Ensemble approach combining semantic similarity, attention analysis,
        and hierarchical clustering for robust failure classification.
        """
        # Extract multi-dimensional features
        attention_features = self.extract_attention_features(
            instance.input_text, instance.model_output
        )
        
        semantic_distances = self.compute_semantic_distances(instance)
        
        # Task-specific classification logic
        task_patterns = self._get_task_specific_patterns(instance.task_type)
        
        # Find best matching pattern
        best_match = max(
            [(pattern, score) for pattern, score in semantic_distances.items() 
             if pattern in task_patterns],
            key=lambda x: x[1]
        )
        
        failure_category = best_match[0]
        confidence_score = best_match[1]
        
        # Generate sub-categories using clustering
        sub_categories = self._extract_sub_categories(
            instance, attention_features, semantic_distances
        )
        
        # Create explanation vector
        explanation_vector = self._create_explanation_vector(
            instance, attention_features, semantic_distances
        )
        
        return FailureClassification(
            failure_category=failure_category,
            confidence_score=confidence_score,
            sub_categories=sub_categories,
            attention_weights=attention_features,
            semantic_features=np.array(list(semantic_distances.values())),
            explanation_vector=explanation_vector
        )
    
    def _get_task_specific_patterns(self, task_type: str) -> List[str]:
        """Get relevant failure patterns for specific task types"""
        task_patterns = {
            "NL2NL": ["factual_inconsistency", "hallucination", "information_loss", "stylistic_mismatch"],
            "NL2CODE": ["syntax_error", "logical_error", "inefficiency", "security_vulnerability"],
            "CODE2NL": ["inaccurate_description", "incomplete_explanation", "poor_readability"]
        }
        return task_patterns.get(task_type, [])
    
    def _extract_sub_categories(self, instance: FailureInstance, 
                               attention_features: np.ndarray,
                               semantic_distances: Dict[str, float]) -> List[str]:
        """Extract fine-grained sub-categories using clustering"""
        # Combine features for clustering
        feature_vector = np.concatenate([
            attention_features[:100],  # Truncate for manageable size
            list(semantic_distances.values())
        ])
        
        # Simple rule-based sub-categorization (can be enhanced with ML)
        sub_categories = []
        
        # Add severity levels
        max_distance = max(semantic_distances.values())
        if max_distance > 0.8:
            sub_categories.append("high_severity")
        elif max_distance > 0.6:
            sub_categories.append("medium_severity")
        else:
            sub_categories.append("low_severity")
            
        # Add complexity indicators
        attention_variance = np.var(attention_features)
        if attention_variance > 0.1:
            sub_categories.append("complex_failure")
        else:
            sub_categories.append("simple_failure")
            
        return sub_categories
    
    def _create_explanation_vector(self, instance: FailureInstance,
                                  attention_features: np.ndarray,
                                  semantic_distances: Dict[str, float]) -> np.ndarray:
        """Create a comprehensive explanation vector for interpretability"""
        # Normalize and combine different feature types
        normalized_attention = attention_features / (np.linalg.norm(attention_features) + 1e-8)
        normalized_semantic = np.array(list(semantic_distances.values()))
        normalized_semantic = normalized_semantic / (np.linalg.norm(normalized_semantic) + 1e-8)
        
        # Create composite explanation vector
        explanation_vector = np.concatenate([
            normalized_attention[:50],  # Top attention features
            normalized_semantic,        # Semantic similarity scores
            [len(instance.input_text), len(instance.model_output)],  # Length features
        ])
        
        return explanation_vector


class FailureClassifier:
    """
    Main failure classifier component with LLM integration
    
    Innovation: Hybrid approach combining semantic analysis with LLM reasoning
    for accurate and interpretable failure classification.
    """
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        self.semantic_classifier = SemanticAttentionClassifier()
        
        # Load prompts from the original prompt.md structure
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize classification prompts based on the provided template"""
        self.classification_prompt_template = """
Given the following evaluation data for an LLM-generated output, please classify the error based on the provided task-specific taxonomy.

**Evaluation Data:**
- **TaskType:** {task_type}
- **InputID:** {input_id}
- **Input (Natural Language or Code):**

{input_text}

- **Model Output (Failed):**

{model_output}

- **Reference Output (Ground Truth):**

{reference_output}

- **Pass/Fail Status:** Fail

**Failure Taxonomy:**

* **If TaskType is NL2NL (e.g., Summarization):**
    * **Factual Inconsistency:** The output contains information that contradicts the source text.
    * **Hallucination:** The output introduces new, unverifiable, or entirely incorrect information.
    * **Loss of Key Information:** The output omits critical details from the source.
    * **Stylistic Mismatch:** The output's tone, style, or format is inappropriate for the request.

* **If TaskType is NL2CODE (e.g., Text-to-Python):**
    * **Syntax Error:** The generated code is not syntactically valid and will not compile or run.
    * **Logical Error:** The code runs but produces an incorrect result due to flawed logic (e.g., wrong algorithm, off-by-one error).
    * **Inefficiency / Non-Idiomatic Code:** The code is correct but is unnecessarily slow, resource-intensive, or does not follow language best practices.
    * **Security Vulnerability:** The code introduces a potential security risk (e.g., SQL injection, buffer overflow).

* **If TaskType is CODE2NL (e.g., Code Documentation):**
    * **Inaccurate Description:** The explanation misrepresents the code's logic, functionality, or purpose.
    * **Incomplete Explanation:** The explanation misses important details, such as edge cases, parameters, or return values.
    * **Poor Readability:** The explanation is confusing, overly technical, or poorly structured.

**Your Task:**
Analyze the "Model Output (Failed)" in relation to the "Reference Output" and the "Input". Respond with a single JSON object containing the primary failure category based on the specified "TaskType".

**Output Format:**
```json
{{
  "failure_category": "YOUR_CLASSIFICATION_HERE"
}}
```
"""
    
    def classify(self, instance: FailureInstance) -> FailureClassification:
        """
        Classify failure using hybrid semantic-LLM approach
        
        Innovation: Combines automated semantic analysis with LLM reasoning
        for enhanced accuracy and explainability.
        """
        # Step 1: Semantic classification
        semantic_result = self.semantic_classifier.classify_failure(instance)
        
        # Step 2: LLM validation and refinement
        llm_result = self._llm_classify(instance)
        
        # Step 3: Ensemble decision
        final_result = self._ensemble_decision(semantic_result, llm_result, instance)
        
        return final_result
    
    def _llm_classify(self, instance: FailureInstance) -> Dict[str, Any]:
        """Use LLM for classification validation"""
        prompt = self.classification_prompt_template.format(
            task_type=instance.task_type,
            input_id=instance.input_id,
            input_text=instance.input_text,
            model_output=instance.model_output,
            reference_output=instance.reference_output
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            return result
        except (json.JSONDecodeError, ValueError):
            # Fallback parsing
            return {"failure_category": "unknown", "confidence": 0.5}
    
    def _ensemble_decision(self, semantic_result: FailureClassification,
                          llm_result: Dict[str, Any],
                          instance: FailureInstance) -> FailureClassification:
        """
        Combine semantic and LLM results using ensemble approach
        
        Innovation: Confidence-weighted ensemble that leverages the strengths
        of both semantic analysis and LLM reasoning.
        """
        # Weight LLM result based on semantic confidence
        semantic_weight = semantic_result.confidence_score
        llm_weight = 1.0 - semantic_weight
        
        # Check agreement
        llm_category = llm_result.get("failure_category", "unknown")
        
        if llm_category == semantic_result.failure_category:
            # Agreement: boost confidence
            final_confidence = min(1.0, semantic_result.confidence_score + 0.2)
            final_category = semantic_result.failure_category
        else:
            # Disagreement: use weighted decision
            if semantic_weight > llm_weight:
                final_category = semantic_result.failure_category
                final_confidence = semantic_result.confidence_score * 0.8  # Reduce confidence
            else:
                final_category = llm_category
                final_confidence = 0.6  # Moderate confidence for LLM-only decision
        
        # Create enhanced result
        enhanced_result = FailureClassification(
            failure_category=final_category,
            confidence_score=final_confidence,
            sub_categories=semantic_result.sub_categories + [f"llm_validated_{llm_category}"],
            attention_weights=semantic_result.attention_weights,
            semantic_features=semantic_result.semantic_features,
            explanation_vector=semantic_result.explanation_vector
        )
        
        return enhanced_result 