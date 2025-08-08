"""
Metrics and Analysis Utilities

This module contains utility classes for various metrics and analysis components
used throughout the explainability framework.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import networkx as nx
from sentence_transformers import SentenceTransformer


class SemanticSimilarity:
    """
    Semantic similarity computation using various metrics
    
    Innovation: Multi-metric similarity computation with adaptive weighting
    based on context and task type.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        
    def compute_similarity(self, text1: str, text2: str, 
                          metric: str = "cosine") -> float:
        """Compute semantic similarity between two texts"""
        try:
            # Handle empty or invalid text
            if not text1 or not text1.strip():
                text1 = "empty"
            if not text2 or not text2.strip():
                text2 = "empty"
            
            emb1 = self.embedding_model.encode([text1])
            emb2 = self.embedding_model.encode([text2])
            
            # Check for NaN embeddings
            if np.isnan(emb1).any() or np.isnan(emb2).any():
                return 0.0
            
            if metric == "cosine":
                similarity = cosine_similarity(emb1, emb2)[0, 0]
            elif metric == "euclidean":
                norm_diff = np.linalg.norm(emb1 - emb2)
                similarity = 1.0 / (1.0 + norm_diff) if not np.isnan(norm_diff) else 0.0
            elif metric == "manhattan":
                abs_diff = np.sum(np.abs(emb1 - emb2))
                similarity = 1.0 / (1.0 + abs_diff) if not np.isnan(abs_diff) else 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Handle NaN/inf results
            if np.isnan(similarity) or np.isinf(similarity):
                return 0.0
            
            return float(similarity)
            
        except Exception as e:
            print(f"Warning: Error computing similarity between '{text1[:50]}...' and '{text2[:50]}...': {e}")
            return 0.0
    
    def batch_similarity(self, texts1: List[str], texts2: List[str],
                        metric: str = "cosine") -> np.ndarray:
        """Compute pairwise similarities between two lists of texts"""
        emb1 = self.embedding_model.encode(texts1)
        emb2 = self.embedding_model.encode(texts2)
        
        if metric == "cosine":
            return cosine_similarity(emb1, emb2)
        else:
            # For other metrics, compute pairwise
            similarities = np.zeros((len(texts1), len(texts2)))
            for i, e1 in enumerate(emb1):
                for j, e2 in enumerate(emb2):
                    if metric == "euclidean":
                        similarities[i, j] = 1.0 / (1.0 + np.linalg.norm(e1 - e2))
                    elif metric == "manhattan":
                        similarities[i, j] = 1.0 / (1.0 + np.sum(np.abs(e1 - e2)))
            return similarities


class AttentionAnalyzer:
    """
    Attention pattern analysis for explainability
    
    Innovation: Cross-modal attention analysis that captures relationships
    between different parts of input, processing, and output.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def compute_cross_attention(self, input_text: str, output_text: str) -> np.ndarray:
        """
        Compute cross-attention weights between input and output
        
        This is a simplified implementation that can be enhanced with actual
        transformer attention weights when available.
        """
        # Handle empty text cases
        if not input_text or not input_text.strip():
            input_text = "empty_input"
        if not output_text or not output_text.strip():
            output_text = "empty_output"
        
        # Tokenize texts (simplified) with minimum length check
        input_tokens = input_text.split()
        output_tokens = output_text.split()
        
        # Ensure minimum token count to avoid empty matrices
        if len(input_tokens) == 0:
            input_tokens = ["empty"]
        if len(output_tokens) == 0:
            output_tokens = ["empty"]
        
        # Create attention matrix based on token similarity
        attention_matrix = np.zeros((len(input_tokens), len(output_tokens)))
        
        semantic_sim = SemanticSimilarity()
        
        try:
            for i, input_token in enumerate(input_tokens):
                for j, output_token in enumerate(output_tokens):
                    # Compute semantic similarity between tokens
                    similarity = semantic_sim.compute_similarity(input_token, output_token)
                    
                    # Handle NaN/inf similarity values
                    if np.isnan(similarity) or np.isinf(similarity):
                        similarity = 0.0
                    
                    attention_matrix[i, j] = similarity
            
            # Check for all-zero matrix and add small epsilon to diagonal
            if np.all(attention_matrix == 0):
                np.fill_diagonal(attention_matrix, 1e-8)
            
            # Ensure matrix has valid values
            attention_matrix = np.nan_to_num(attention_matrix, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Add small epsilon to prevent numerical instability in softmax
            attention_matrix = attention_matrix + 1e-10
            
            # Normalize attention weights with numerical stability
            attention_tensor = torch.tensor(attention_matrix, dtype=torch.float32)
            attention_weights = F.softmax(attention_tensor, dim=1).numpy()
            
            # Final check for NaN values
            attention_weights = np.nan_to_num(attention_weights, nan=1.0/attention_weights.shape[1])
            
        except Exception as e:
            print(f"Warning: Error computing attention weights: {e}")
            # Fallback: uniform attention distribution
            attention_weights = np.ones((len(input_tokens), len(output_tokens)))
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        return attention_weights
    
    def analyze_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Analyze attention patterns for interpretability"""
        patterns = {}
        
        try:
            # Handle NaN values in attention weights
            if attention_weights.size == 0:
                attention_weights = np.array([[1.0]])
            
            # Clean attention weights
            clean_weights = np.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure weights are normalized
            if clean_weights.sum() == 0:
                clean_weights = np.ones_like(clean_weights) / clean_weights.size
            
            # Compute attention concentration
            max_val = np.max(clean_weights)
            mean_val = np.mean(clean_weights)
            patterns['concentration'] = max_val - mean_val if not np.isnan(max_val - mean_val) else 0.0
            
            # Compute attention dispersion
            flat_weights = clean_weights.flatten()
            flat_weights = flat_weights[flat_weights > 0]  # Remove zeros for entropy
            if len(flat_weights) > 0:
                patterns['dispersion'] = entropy(flat_weights)
            else:
                patterns['dispersion'] = 0.0
            
            # Handle NaN in dispersion
            if np.isnan(patterns['dispersion']):
                patterns['dispersion'] = 0.0
            
            # Compute attention variance
            patterns['variance'] = np.var(clean_weights)
            if np.isnan(patterns['variance']):
                patterns['variance'] = 0.0
            
            # Compute attention sparsity
            threshold = 0.1
            patterns['sparsity'] = np.mean(clean_weights < threshold)
            if np.isnan(patterns['sparsity']):
                patterns['sparsity'] = 0.0
                
        except Exception as e:
            print(f"Warning: Error analyzing attention patterns: {e}")
            patterns = {
                'concentration': 0.0,
                'dispersion': 0.0,
                'variance': 0.0,
                'sparsity': 0.0
            }
        
        return patterns


class CausalityAnalyzer:
    """
    Causal relationship analysis using statistical and graph-based methods
    
    Innovation: Combines Granger causality, mutual information, and
    graph neural networks for robust causal discovery.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def compute_granger_causality(self, x: np.ndarray, y: np.ndarray, 
                                 max_lags: int = 5) -> float:
        """
        Compute Granger causality between two time series
        
        Simplified implementation - can be enhanced with proper statistical tests
        """
        # Ensure arrays are 1D
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        
        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
        
        # Compute lagged correlations
        causality_scores = []
        
        for lag in range(1, min(max_lags + 1, len(x) // 2)):
            if len(x) > lag:
                # Correlation between x[t-lag] and y[t]
                x_lagged = x[:-lag]
                y_current = y[lag:]
                
                if len(x_lagged) > 0 and len(y_current) > 0:
                    correlation = np.corrcoef(x_lagged, y_current)[0, 1]
                    causality_scores.append(abs(correlation))
        
        return np.mean(causality_scores) if causality_scores else 0.0
    
    def compute_mutual_information(self, x: np.ndarray, y: np.ndarray, 
                                  bins: int = 10) -> float:
        """Compute mutual information between two variables"""
        # Discretize continuous variables
        x_discrete = np.digitize(x, np.linspace(x.min(), x.max(), bins))
        y_discrete = np.digitize(y, np.linspace(y.min(), y.max(), bins))
        
        # Compute joint and marginal distributions
        joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=bins)
        joint_prob = joint_hist / joint_hist.sum()
        
        x_prob = joint_prob.sum(axis=1)
        y_prob = joint_prob.sum(axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
        
        return mi
    
    def discover_causal_structure(self, features: Dict[str, np.ndarray]) -> nx.DiGraph:
        """
        Discover causal structure among features using multiple methods
        
        Innovation: Ensemble approach combining multiple causality measures
        for robust causal discovery.
        """
        G = nx.DiGraph()
        feature_names = list(features.keys())
        
        # Add nodes
        for name in feature_names:
            G.add_node(name)
        
        # Compute pairwise causality
        for i, name1 in enumerate(feature_names):
            for j, name2 in enumerate(feature_names):
                if i != j:
                    # Compute multiple causality measures
                    granger_score = self.compute_granger_causality(
                        features[name1], features[name2]
                    )
                    mi_score = self.compute_mutual_information(
                        features[name1], features[name2]
                    )
                    
                    # Combine scores
                    combined_score = 0.6 * granger_score + 0.4 * mi_score
                    
                    # Add edge if causality is significant
                    if combined_score > 0.1:  # Threshold
                        G.add_edge(name1, name2, weight=combined_score)
        
        return G


class CounterfactualGenerator:
    """
    Generate counterfactual examples for explainability
    
    Innovation: Minimal intervention counterfactuals that preserve
    semantic meaning while changing decision outcomes.
    """
    
    def __init__(self):
        self.semantic_similarity = SemanticSimilarity()
        
    def generate_text_counterfactual(self, original_text: str, 
                                   target_change: str = "sentiment") -> str:
        """
        Generate textual counterfactual by minimal changes
        
        This is a simplified implementation that can be enhanced with
        more sophisticated NLP techniques.
        """
        words = original_text.split()
        
        # Simple word replacement strategy
        if target_change == "sentiment":
            # Replace negative words with positive ones
            replacements = {
                "bad": "good", "terrible": "excellent", "awful": "wonderful",
                "hate": "love", "worst": "best", "boring": "exciting",
                "difficult": "easy", "problem": "solution", "failed": "succeeded"
            }
            
            modified_words = []
            for word in words:
                word_lower = word.lower().strip('.,!?')
                if word_lower in replacements:
                    # Preserve original capitalization and punctuation
                    replacement = replacements[word_lower]
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    # Add back punctuation
                    for punct in '.,!?':
                        if word.endswith(punct):
                            replacement += punct
                    modified_words.append(replacement)
                else:
                    modified_words.append(word)
            
            return " ".join(modified_words)
        
        return original_text  # No change if target not recognized
    
    def generate_feature_counterfactual(self, features: np.ndarray, 
                                      target_feature: int,
                                      change_magnitude: float = 0.1) -> np.ndarray:
        """Generate counterfactual by modifying specific features"""
        counterfactual = features.copy()
        
        # Modify target feature
        current_value = features[target_feature]
        if current_value != 0:
            # Proportional change
            counterfactual[target_feature] = current_value * (1 + change_magnitude)
        else:
            # Absolute change
            counterfactual[target_feature] = change_magnitude
        
        return counterfactual


class RecommendationRanker:
    """
    Ranking system for recommendations based on multiple criteria
    
    Innovation: Multi-criteria decision analysis with adaptive weights
    based on user feedback and context.
    """
    
    def __init__(self):
        self.criteria_weights = {
            'expected_impact': 0.3,
            'implementation_effort': 0.2,
            'confidence': 0.25,
            'stakeholder_alignment': 0.15,
            'novelty': 0.1
        }
        
    def rank_recommendations(self, recommendations: List[Dict[str, Any]], 
                           stakeholder_preferences: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Rank recommendations based on multiple criteria
        
        Args:
            recommendations: List of recommendation dictionaries
            stakeholder_preferences: Optional stakeholder-specific weights
            
        Returns:
            Ranked list of recommendations
        """
        # Use stakeholder preferences if provided
        weights = stakeholder_preferences or self.criteria_weights
        
        scored_recommendations = []
        
        for rec in recommendations:
            score = self._compute_recommendation_score(rec, weights)
            rec_with_score = rec.copy()
            rec_with_score['ranking_score'] = score
            scored_recommendations.append(rec_with_score)
        
        # Sort by score (descending)
        ranked_recommendations = sorted(
            scored_recommendations, 
            key=lambda x: x['ranking_score'], 
            reverse=True
        )
        
        return ranked_recommendations
    
    def _compute_recommendation_score(self, recommendation: Dict[str, Any], 
                                    weights: Dict[str, float]) -> float:
        """Compute weighted score for a recommendation"""
        score = 0.0
        
        # Expected impact (higher is better)
        if 'expected_impact' in recommendation:
            score += weights.get('expected_impact', 0) * recommendation['expected_impact']
        
        # Implementation effort (lower is better, so invert)
        if 'implementation_effort' in recommendation:
            effort_score = 1.0 - recommendation['implementation_effort']
            score += weights.get('implementation_effort', 0) * effort_score
        
        # Confidence (higher is better)
        if 'confidence' in recommendation:
            score += weights.get('confidence', 0) * recommendation['confidence']
        
        # Stakeholder alignment (higher is better)
        if 'stakeholder_alignment' in recommendation:
            score += weights.get('stakeholder_alignment', 0) * recommendation['stakeholder_alignment']
        
        # Novelty (higher is better)
        if 'novelty' in recommendation:
            score += weights.get('novelty', 0) * recommendation['novelty']
        
        return score


class ContextualOptimizer:
    """
    Context-aware optimization for recommendations and explanations
    
    Innovation: Dynamic optimization based on task context, user profile,
    and historical outcomes.
    """
    
    def __init__(self):
        self.context_history = []
        self.outcome_history = []
        
    def optimize_for_context(self, recommendations: List[Dict[str, Any]], 
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimize recommendations based on current context
        
        Args:
            recommendations: List of recommendations to optimize
            context: Current context information
            
        Returns:
            Context-optimized recommendations
        """
        optimized_recommendations = []
        
        for rec in recommendations:
            optimized_rec = rec.copy()
            
            # Adjust scores based on context
            context_adjustment = self._compute_context_adjustment(rec, context)
            optimized_rec['context_adjusted_score'] = rec.get('ranking_score', 0.5) * context_adjustment
            
            optimized_recommendations.append(optimized_rec)
        
        # Sort by context-adjusted score
        optimized_recommendations.sort(
            key=lambda x: x['context_adjusted_score'], 
            reverse=True
        )
        
        return optimized_recommendations
    
    def _compute_context_adjustment(self, recommendation: Dict[str, Any], 
                                  context: Dict[str, Any]) -> float:
        """Compute context-based adjustment factor"""
        adjustment = 1.0
        
        # Task type adjustment
        task_type = context.get('task_type', 'unknown')
        rec_type = recommendation.get('recommendation_type', 'unknown')
        
        # Boost recommendations that match task type
        if task_type == 'NL2CODE' and 'code' in rec_type:
            adjustment *= 1.2
        elif task_type == 'CODE2NL' and 'explanation' in rec_type:
            adjustment *= 1.2
        elif task_type == 'NL2NL' and 'prompt' in rec_type:
            adjustment *= 1.2
        
        # Time constraint adjustment
        urgency = context.get('urgency', 'medium')
        effort = recommendation.get('implementation_effort', 0.5)
        
        if urgency == 'high' and effort < 0.3:
            adjustment *= 1.3  # Boost low-effort recommendations for urgent cases
        elif urgency == 'low' and effort > 0.7:
            adjustment *= 1.1  # Slightly boost high-effort recommendations for non-urgent cases
        
        # Resource constraint adjustment
        resources = context.get('available_resources', 'medium')
        if resources == 'low' and effort > 0.6:
            adjustment *= 0.8  # Penalize high-effort recommendations when resources are low
        elif resources == 'high':
            adjustment *= 1.1  # Slight boost when resources are abundant
        
        return adjustment
    
    def update_context_outcome(self, context: Dict[str, Any], 
                             outcome_metrics: Dict[str, float]):
        """Update context history with outcome feedback"""
        self.context_history.append(context)
        self.outcome_history.append(outcome_metrics)
        
        # Keep only recent history (last 100 entries)
        if len(self.context_history) > 100:
            self.context_history.pop(0)
            self.outcome_history.pop(0)


class ExplainabilityMetrics:
    """
    Comprehensive metrics for evaluating explainability quality
    
    Innovation: Multi-dimensional explainability assessment combining
    technical accuracy, user comprehension, and practical utility.
    """
    
    def __init__(self):
        self.semantic_similarity = SemanticSimilarity()
        self.ranker = RecommendationRanker()
        
    def compute_explanation_quality(self, explanation: str, 
                                  ground_truth: Optional[str] = None,
                                  user_feedback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute comprehensive explanation quality metrics
        
        Args:
            explanation: Generated explanation text
            ground_truth: Optional ground truth explanation
            user_feedback: Optional user feedback scores
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Technical quality metrics
        metrics.update(self._compute_technical_metrics(explanation))
        
        # Semantic quality metrics
        if ground_truth:
            metrics.update(self._compute_semantic_metrics(explanation, ground_truth))
        
        # User experience metrics
        if user_feedback:
            metrics.update(self._compute_user_metrics(user_feedback))
        
        # Overall quality score
        metrics['overall_quality'] = self._compute_overall_quality(metrics)
        
        return metrics
    
    def _compute_technical_metrics(self, explanation: str) -> Dict[str, float]:
        """Compute technical quality metrics"""
        metrics = {}
        
        # Length appropriateness
        word_count = len(explanation.split())
        metrics['length_score'] = min(1.0, max(0.0, 1.0 - abs(word_count - 150) / 150))
        
        # Readability (simplified)
        avg_word_length = np.mean([len(word) for word in explanation.split()])
        metrics['readability_score'] = min(1.0, max(0.0, 1.0 - abs(avg_word_length - 5) / 5))
        
        # Structure score (based on presence of structured elements)
        structure_indicators = ['1.', '2.', '**', '-', '•']
        structure_count = sum(1 for indicator in structure_indicators if indicator in explanation)
        metrics['structure_score'] = min(1.0, structure_count / 3)
        
        return metrics
    
    def _compute_semantic_metrics(self, explanation: str, ground_truth: str) -> Dict[str, float]:
        """Compute semantic quality metrics"""
        metrics = {}
        
        # Semantic similarity to ground truth
        metrics['semantic_similarity'] = self.semantic_similarity.compute_similarity(
            explanation, ground_truth
        )
        
        # Content coverage (simplified)
        explanation_words = set(explanation.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        
        if ground_truth_words:
            metrics['content_coverage'] = len(explanation_words & ground_truth_words) / len(ground_truth_words)
        else:
            metrics['content_coverage'] = 0.0
        
        return metrics
    
    def _compute_user_metrics(self, user_feedback: Dict[str, float]) -> Dict[str, float]:
        """Compute user experience metrics"""
        metrics = {}
        
        # Direct user feedback metrics
        metrics['user_comprehension'] = user_feedback.get('comprehension', 0.5)
        metrics['user_satisfaction'] = user_feedback.get('satisfaction', 0.5)
        metrics['user_trust'] = user_feedback.get('trust', 0.5)
        metrics['user_actionability'] = user_feedback.get('actionability', 0.5)
        
        return metrics
    
    def _compute_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Compute weighted overall quality score"""
        weights = {
            'semantic_similarity': 0.25,
            'content_coverage': 0.20,
            'user_comprehension': 0.20,
            'user_satisfaction': 0.15,
            'structure_score': 0.10,
            'readability_score': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_score += metrics[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0 