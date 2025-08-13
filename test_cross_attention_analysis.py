#!/usr/bin/env python3
"""
Cross-Attention Behavior Analysis

This script demonstrates the behaviors of cross-attention models and compares
our approximation with real transformer attention patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import sys
import os

# Add the framework to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm_explainability_framework'))

from utils.metrics import AttentionAnalyzer, SemanticSimilarity

class CrossAttentionAnalyzer:
    """Analyze cross-attention behaviors and patterns"""
    
    def __init__(self):
        self.attention_analyzer = AttentionAnalyzer()
        self.semantic_sim = SemanticSimilarity()
        
    def analyze_attention_behaviors(self, input_text: str, output_text: str) -> Dict:
        """Analyze cross-attention behaviors between input and output"""
        
        print(f"🔍 Analyzing Cross-Attention Behaviors")
        print(f"Input: {input_text}")
        print(f"Output: {output_text}")
        print("-" * 60)
        
        # Get our approximation
        our_attention = self.attention_analyzer.compute_cross_attention(input_text, output_text)
        
        # Analyze behaviors
        behaviors = {
            "attention_matrix": our_attention,
            "patterns": self.attention_analyzer.analyze_attention_patterns(our_attention),
            "token_analysis": self._analyze_token_attention(input_text, output_text, our_attention),
            "positional_analysis": self._analyze_positional_effects(input_text, output_text),
            "semantic_analysis": self._analyze_semantic_relationships(input_text, output_text)
        }
        
        return behaviors
    
    def _analyze_token_attention(self, input_text: str, output_text: str, 
                               attention_matrix: np.ndarray) -> Dict:
        """Analyze attention patterns at token level"""
        
        input_tokens = input_text.split()
        output_tokens = output_text.split()
        
        # Find high-attention token pairs
        high_attention_pairs = []
        for i in range(min(len(input_tokens), attention_matrix.shape[0])):
            for j in range(min(len(output_tokens), attention_matrix.shape[1])):
                attention_score = attention_matrix[i, j]
                if attention_score > 0.1:  # Threshold for high attention
                    high_attention_pairs.append({
                        "input_token": input_tokens[i],
                        "output_token": output_tokens[j],
                        "attention_score": attention_score,
                        "position": (i, j)
                    })
        
        # Sort by attention score
        high_attention_pairs.sort(key=lambda x: x["attention_score"], reverse=True)
        
        return {
            "high_attention_pairs": high_attention_pairs[:10],  # Top 10
            "attention_distribution": {
                "mean": float(np.mean(attention_matrix)),
                "std": float(np.std(attention_matrix)),
                "max": float(np.max(attention_matrix)),
                "min": float(np.min(attention_matrix))
            }
        }
    
    def _analyze_positional_effects(self, input_text: str, output_text: str) -> Dict:
        """Analyze how position affects attention patterns"""
        
        input_tokens = input_text.split()
        output_tokens = output_text.split()
        
        # Test different positions
        positional_effects = {}
        
        # Test early vs late positions
        if len(input_tokens) > 2 and len(output_tokens) > 2:
            early_input = " ".join(input_tokens[:2])
            late_input = " ".join(input_tokens[-2:])
            
            early_attention = self.attention_analyzer.compute_cross_attention(early_input, output_text)
            late_attention = self.attention_analyzer.compute_cross_attention(late_input, output_text)
            
            positional_effects["early_vs_late"] = {
                "early_mean_attention": float(np.mean(early_attention)),
                "late_mean_attention": float(np.mean(late_attention)),
                "position_bias": float(np.mean(early_attention) - np.mean(late_attention))
            }
        
        return positional_effects
    
    def _analyze_semantic_relationships(self, input_text: str, output_text: str) -> Dict:
        """Analyze semantic relationships in cross-attention"""
        
        input_tokens = input_text.split()
        output_tokens = output_text.split()
        
        semantic_relationships = []
        
        # Analyze semantic similarity between corresponding tokens
        for i, input_token in enumerate(input_tokens):
            for j, output_token in enumerate(output_tokens):
                similarity = self.semantic_sim.compute_similarity(input_token, output_token)
                
                if similarity > 0.5:  # High semantic similarity
                    semantic_relationships.append({
                        "input_token": input_token,
                        "output_token": output_token,
                        "semantic_similarity": similarity,
                        "relationship_type": self._classify_relationship(input_token, output_token)
                    })
        
        return {
            "high_similarity_pairs": semantic_relationships,
            "semantic_coherence": float(np.mean([r["semantic_similarity"] for r in semantic_relationships])) if semantic_relationships else 0.0
        }
    
    def _classify_relationship(self, token1: str, token2: str) -> str:
        """Classify the type of relationship between tokens"""
        
        # Exact match
        if token1.lower() == token2.lower():
            return "exact_match"
        
        # Stemming match
        if token1.lower().startswith(token2.lower()) or token2.lower().startswith(token1.lower()):
            return "stemming_match"
        
        # Code-specific relationships
        code_patterns = {
            ("def", "function"): "function_declaration",
            ("class", "object"): "class_definition", 
            ("import", "from"): "import_statement",
            ("return", "output"): "return_statement",
            ("if", "condition"): "conditional_statement",
            ("for", "loop"): "loop_statement"
        }
        
        for (pattern1, pattern2), rel_type in code_patterns.items():
            if (token1.lower() in pattern1 or token2.lower() in pattern1) and \
               (token1.lower() in pattern2 or token2.lower() in pattern2):
                return rel_type
        
        return "semantic_similarity"
    
    def compare_with_real_attention(self, input_text: str, output_text: str) -> Dict:
        """Compare our approximation with real transformer attention (if available)"""
        
        print("🔄 Comparing with Real Transformer Attention")
        
        comparison = {
            "our_approximation": self.attention_analyzer.compute_cross_attention(input_text, output_text),
            "limitations": self._identify_limitations(),
            "improvements": self._suggest_improvements()
        }
        
        return comparison
    
    def _identify_limitations(self) -> List[str]:
        """Identify limitations of our approximation"""
        
        limitations = [
            "No learned attention weights (using random projections)",
            "Simplified tokenization (word-level vs subword)",
            "Limited positional encoding (basic sinusoidal)",
            "No layer-specific patterns (single layer approximation)",
            "No attention masking (assumes full attention)",
            "No residual connections or layer normalization",
            "Fixed number of attention heads (8 vs variable)",
            "No pre-training on domain-specific data"
        ]
        
        return limitations
    
    def _suggest_improvements(self) -> List[str]:
        """Suggest improvements to our approximation"""
        
        improvements = [
            "Use pre-trained transformer models for attention extraction",
            "Implement proper subword tokenization (BPE/WordPiece)",
            "Add learned positional encodings",
            "Incorporate attention masking for causal relationships",
            "Use domain-specific pre-training (code, text)",
            "Implement multi-layer attention aggregation",
            "Add attention head specialization",
            "Include residual connections and normalization"
        ]
        
        return improvements
    
    def visualize_attention_patterns(self, attention_matrix: np.ndarray, 
                                   input_text: str, output_text: str):
        """Visualize attention patterns"""
        
        input_tokens = input_text.split()
        output_tokens = output_text.split()
        
        # Limit for visualization
        max_tokens = 20
        if len(input_tokens) > max_tokens:
            input_tokens = input_tokens[:max_tokens]
        if len(output_tokens) > max_tokens:
            output_tokens = output_tokens[:max_tokens]
        
        # Crop attention matrix
        attention_vis = attention_matrix[:len(input_tokens), :len(output_tokens)]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_vis, 
                   xticklabels=output_tokens,
                   yticklabels=input_tokens,
                   cmap='Blues',
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title('Cross-Attention Pattern Visualization')
        plt.xlabel('Output Tokens')
        plt.ylabel('Input Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('cross_attention_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Attention visualization saved as 'cross_attention_visualization.png'")

def main():
    """Main analysis function"""
    
    analyzer = CrossAttentionAnalyzer()
    
    # Test cases
    test_cases = [
        {
            "name": "Code Generation",
            "input": "Write a Python function to calculate factorial",
            "output": "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"
        },
        {
            "name": "Text Summarization", 
            "input": "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet.",
            "output": "A fox jumps over a dog in a pangram."
        },
        {
            "name": "Translation",
            "input": "Hello, how are you today?",
            "output": "Bonjour, comment allez-vous aujourd'hui?"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        # Analyze behaviors
        behaviors = analyzer.analyze_attention_behaviors(
            test_case["input"], test_case["output"]
        )
        
        # Print key insights
        print(f"\n📈 Attention Patterns:")
        for pattern, value in behaviors["patterns"].items():
            print(f"  {pattern}: {value:.4f}")
        
        print(f"\n🔗 High Attention Token Pairs:")
        for pair in behaviors["token_analysis"]["high_attention_pairs"][:5]:
            print(f"  '{pair['input_token']}' → '{pair['output_token']}': {pair['attention_score']:.3f}")
        
        print(f"\n📊 Attention Distribution:")
        dist = behaviors["token_analysis"]["attention_distribution"]
        print(f"  Mean: {dist['mean']:.4f}, Std: {dist['std']:.4f}")
        print(f"  Max: {dist['max']:.4f}, Min: {dist['min']:.4f}")
        
        # Visualize for the first test case
        if i == 0:
            analyzer.visualize_attention_patterns(
                behaviors["attention_matrix"],
                test_case["input"], 
                test_case["output"]
            )
    
    # Compare with real attention
    print(f"\n{'='*60}")
    print("COMPARISON WITH REAL TRANSFORMER ATTENTION")
    print(f"{'='*60}")
    
    comparison = analyzer.compare_with_real_attention(
        test_cases[0]["input"], test_cases[0]["output"]
    )
    
    print(f"\n❌ Limitations of Our Approximation:")
    for limitation in comparison["limitations"]:
        print(f"  • {limitation}")
    
    print(f"\n✅ Suggested Improvements:")
    for improvement in comparison["improvements"]:
        print(f"  • {improvement}")

if __name__ == "__main__":
    main() 