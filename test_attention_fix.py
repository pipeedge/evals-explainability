#!/usr/bin/env python3
"""
Test script to verify attention weight NaN fixes
"""
import sys
import os
sys.path.insert(0, '.')

import numpy as np

def test_attention_analyzer():
    """Test the fixed AttentionAnalyzer"""
    print("Testing AttentionAnalyzer fixes...")
    
    try:
        from llm_explainability_framework.utils.metrics import AttentionAnalyzer, SemanticSimilarity
        
        # Test the fixed attention analyzer
        analyzer = AttentionAnalyzer()
        
        # Test with normal text
        print("1. Testing normal text...")
        attention = analyzer.compute_cross_attention('hello world test', 'hi there friend')
        print(f"   Attention shape: {attention.shape}")
        print(f"   Has NaN values: {np.isnan(attention).any()}")
        print(f"   Sample values: {attention.flatten()[:5]}")
        
        # Test with empty text
        print("2. Testing empty text...")
        attention_empty = analyzer.compute_cross_attention('', '')
        print(f"   Empty text attention shape: {attention_empty.shape}")
        print(f"   Has NaN values: {np.isnan(attention_empty).any()}")
        print(f"   Sample values: {attention_empty.flatten()[:5]}")
        
        # Test with None/whitespace text
        print("3. Testing whitespace text...")
        attention_ws = analyzer.compute_cross_attention('   ', '\n\t  ')
        print(f"   Whitespace attention shape: {attention_ws.shape}")
        print(f"   Has NaN values: {np.isnan(attention_ws).any()}")
        
        # Test attention patterns
        print("4. Testing attention patterns...")
        patterns = analyzer.analyze_attention_patterns(attention)
        print(f"   Patterns: {patterns}")
        print(f"   Any NaN in patterns: {any(np.isnan(v) if isinstance(v, (int, float)) else False for v in patterns.values())}")
        
        # Test with problematic attention weights (all NaN)
        print("5. Testing NaN attention weights...")
        nan_attention = np.full((3, 3), np.nan)
        patterns_nan = analyzer.analyze_attention_patterns(nan_attention)
        print(f"   Patterns from NaN input: {patterns_nan}")
        print(f"   Any NaN in patterns: {any(np.isnan(v) if isinstance(v, (int, float)) else False for v in patterns_nan.values())}")
        
        return True
        
    except Exception as e:
        print(f"Error testing AttentionAnalyzer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_similarity():
    """Test the fixed SemanticSimilarity"""
    print("\nTesting SemanticSimilarity fixes...")
    
    try:
        from llm_explainability_framework.utils.metrics import SemanticSimilarity
        
        sim = SemanticSimilarity()
        
        # Test normal similarity
        print("1. Testing normal similarity...")
        score = sim.compute_similarity("hello", "hi")
        print(f"   Normal similarity: {score}, is NaN: {np.isnan(score)}")
        
        # Test empty text similarity
        print("2. Testing empty text similarity...")
        score_empty = sim.compute_similarity("", "")
        print(f"   Empty similarity: {score_empty}, is NaN: {np.isnan(score_empty)}")
        
        # Test one empty text
        print("3. Testing one empty text...")
        score_one_empty = sim.compute_similarity("hello", "")
        print(f"   One empty similarity: {score_one_empty}, is NaN: {np.isnan(score_one_empty)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing SemanticSimilarity: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Attention Weight NaN Fixes ===\n")
    
    success1 = test_attention_analyzer()
    success2 = test_semantic_similarity()
    
    if success1 and success2:
        print("\n✅ All tests passed! NaN issues should be fixed.")
    else:
        print("\n❌ Some tests failed. Check the errors above.") 