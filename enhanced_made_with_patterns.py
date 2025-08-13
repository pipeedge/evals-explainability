#!/usr/bin/env python3
"""
Enhanced MADF Framework with Systematic Pattern Library

This example demonstrates how the systematic pattern library system
improves the MADF framework's failure analysis capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm_explainability_framework'))

from pattern_library_system import (
    PatternLibrarySystem, FailurePattern, PatternExample, CounterfactualFix,
    PatternSeverity, PatternSource
)
from llm_explainability_framework.core.explainability_engine import ExplainabilityEngine
from llm_explainability_framework.core.failure_classifier import FailureClassifier
from llm_explainability_framework.models.llm_wrapper import LLMWrapper

class EnhancedMADFWithPatterns:
    """Enhanced MADF framework with systematic pattern library"""
    
    def __init__(self):
        self.pattern_library = PatternLibrarySystem()
        self.llm_wrapper = LLMWrapper()
        self.failure_classifier = FailureClassifier(self.llm_wrapper)
        self.explainability_engine = ExplainabilityEngine()
        
        # Initialize pattern library with comprehensive patterns
        self._initialize_comprehensive_patterns()
    
    def _initialize_comprehensive_patterns(self):
        """Initialize the pattern library with comprehensive patterns"""
        
        print("🔧 Initializing Comprehensive Pattern Library")
        print("=" * 60)
        
        # Collect patterns from multiple sources
        literature_patterns = self.pattern_library.collect_patterns_from_literature()
        dataset_patterns = self.pattern_library.collect_patterns_from_datasets()
        
        # Add custom high-quality patterns
        custom_patterns = self._create_custom_patterns()
        
        # Add all patterns to library
        all_patterns = literature_patterns + dataset_patterns + custom_patterns
        
        for pattern in all_patterns:
            self.pattern_library.add_pattern(pattern)
        
        print(f"✅ Initialized pattern library with {len(self.pattern_library.patterns)} patterns")
    
    def _create_custom_patterns(self) -> list:
        """Create custom high-quality patterns"""
        
        custom_patterns = [
            # Code Generation - Security Vulnerabilities
            FailurePattern(
                pattern_id="custom_sql_injection",
                category="code_generation",
                subcategory="security_vulnerability",
                description="SQL injection vulnerability in generated code",
                severity=PatternSeverity.CRITICAL,
                examples=[
                    PatternExample(
                        input_text="Write a function to search users by name",
                        model_output="def search_users(name):\n    query = f\"SELECT * FROM users WHERE name = '{name}'\"\n    return execute_query(query)",
                        reference_output="def search_users(name):\n    query = \"SELECT * FROM users WHERE name = %s\"\n    return execute_query(query, (name,))",
                        failure_type="security_vulnerability",
                        failure_description="SQL injection vulnerability due to string formatting",
                        severity_score=0.9,
                        detection_method="security_analysis",
                        source_dataset="custom"
                    )
                ],
                detection_criteria=[
                    "string formatting in SQL queries",
                    "direct variable interpolation",
                    "lack of parameterized queries"
                ],
                failure_indicators=[
                    "SQL injection risk",
                    "security vulnerability",
                    "unsafe query construction"
                ],
                counterfactuals=[
                    CounterfactualFix(
                        fix_description="Use parameterized queries to prevent SQL injection",
                        fix_code="query = \"SELECT * FROM users WHERE name = %s\"\nexecute_query(query, (name,))",
                        fix_strategy="security_hardening",
                        success_rate=0.95,
                        implementation_difficulty="medium"
                    )
                ],
                prevention_strategies=[
                    "Always use parameterized queries",
                    "Validate and sanitize inputs",
                    "Use ORM frameworks"
                ],
                validation_metrics={"security_expert_verified": 0.95},
                source=PatternSource.EXPERT,
                confidence=0.9
            ),
            
            # Text Generation - Factual Inconsistency
            FailurePattern(
                pattern_id="custom_factual_inconsistency",
                category="text_generation",
                subcategory="factual_error",
                description="Factual inconsistency in generated text",
                severity=PatternSeverity.HIGH,
                examples=[
                    PatternExample(
                        input_text="What is the capital of France?",
                        model_output="The capital of France is Paris, which has a population of 2.2 million people.",
                        reference_output="The capital of France is Paris.",
                        failure_type="factual_inconsistency",
                        failure_description="Incorrect population figure for Paris",
                        severity_score=0.7,
                        detection_method="fact_checking",
                        source_dataset="custom"
                    )
                ],
                detection_criteria=[
                    "contradicts known facts",
                    "incorrect numerical data",
                    "misleading information"
                ],
                failure_indicators=[
                    "factual error",
                    "incorrect data",
                    "misinformation"
                ],
                counterfactuals=[
                    CounterfactualFix(
                        fix_description="Verify facts before including them in text",
                        fix_text="The capital of France is Paris.",
                        fix_strategy="fact_verification",
                        success_rate=0.8,
                        implementation_difficulty="medium"
                    )
                ],
                prevention_strategies=[
                    "Fact-check all information",
                    "Use reliable sources",
                    "Verify numerical data"
                ],
                validation_metrics={"fact_checker_verified": 0.85},
                source=PatternSource.EXPERT,
                confidence=0.8
            )
        ]
        
        return custom_patterns
    
    def analyze_failure_with_enhanced_patterns(self, task_type: str, input_text: str, 
                                            model_output: str, reference_output: str) -> dict:
        """
        Analyze failure using enhanced pattern library
        """
        
        print(f"🔍 Enhanced Failure Analysis with Comprehensive Patterns")
        print(f"Task type: {task_type}")
        print(f"Input length: {len(input_text)} chars")
        print(f"Output length: {len(model_output)} chars")
        
        # Get relevant patterns for this task
        task_characteristics = self._analyze_task_characteristics(task_type, input_text, model_output)
        relevant_patterns = self.pattern_library.get_patterns_for_task(task_type, task_characteristics)
        
        print(f"Found {len(relevant_patterns)} relevant patterns")
        
        # Perform pattern-based analysis
        pattern_analysis = self._analyze_with_patterns(
            input_text, model_output, reference_output, relevant_patterns
        )
        
        # Perform standard MADF analysis
        standard_analysis = self.explainability_engine.analyze_failure(
            task_type, input_text, model_output, reference_output
        )
        
        # Combine analyses
        enhanced_analysis = self._combine_analyses(standard_analysis, pattern_analysis)
        
        # Add pattern library information
        enhanced_analysis["pattern_library_info"] = {
            "total_patterns": len(self.pattern_library.patterns),
            "relevant_patterns": len(relevant_patterns),
            "pattern_coverage": len(relevant_patterns) / len(self.pattern_library.patterns),
            "high_confidence_patterns": len([p for p in relevant_patterns if p.confidence > 0.8])
        }
        
        return enhanced_analysis
    
    def _analyze_task_characteristics(self, task_type: str, input_text: str, 
                                   model_output: str) -> dict:
        """Analyze task characteristics for pattern matching"""
        
        characteristics = {
            "task_type": task_type,
            "input_length": len(input_text),
            "output_length": len(model_output),
            "has_code": any(keyword in input_text.lower() for keyword in ['def ', 'class ', 'import ']),
            "has_security_keywords": any(keyword in input_text.lower() for keyword in ['sql', 'query', 'database', 'user']),
            "has_factual_keywords": any(keyword in input_text.lower() for keyword in ['what', 'when', 'where', 'how many', 'capital', 'population'])
        }
        
        # Determine severity
        if characteristics["has_security_keywords"]:
            characteristics["severity"] = "critical"
        elif characteristics["has_factual_keywords"]:
            characteristics["severity"] = "high"
        else:
            characteristics["severity"] = "medium"
        
        return characteristics
    
    def _analyze_with_patterns(self, input_text: str, model_output: str, 
                             reference_output: str, patterns: list) -> dict:
        """Analyze failure using specific patterns"""
        
        pattern_matches = []
        detected_failures = []
        
        for pattern in patterns:
            # Check if pattern matches this failure
            match_score = self._calculate_pattern_match(pattern, input_text, model_output, reference_output)
            
            if match_score > 0.6:  # Threshold for pattern match
                pattern_matches.append({
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.description,
                    "match_score": match_score,
                    "severity": pattern.severity.value,
                    "confidence": pattern.confidence,
                    "detection_criteria": pattern.detection_criteria,
                    "counterfactuals": [
                        {
                            "description": cf.fix_description,
                            "strategy": cf.fix_strategy,
                            "success_rate": cf.success_rate,
                            "difficulty": cf.implementation_difficulty
                        }
                        for cf in pattern.counterfactuals
                    ],
                    "prevention_strategies": pattern.prevention_strategies
                })
                
                detected_failures.append({
                    "failure_type": pattern.subcategory,
                    "description": pattern.description,
                    "severity": pattern.severity.value,
                    "confidence": pattern.confidence * match_score
                })
        
        return {
            "pattern_matches": pattern_matches,
            "detected_failures": detected_failures,
            "total_patterns_checked": len(patterns),
            "matching_patterns": len(pattern_matches)
        }
    
    def _calculate_pattern_match(self, pattern: FailurePattern, input_text: str, 
                               model_output: str, reference_output: str) -> float:
        """Calculate how well a pattern matches the current failure"""
        
        match_score = 0.0
        
        # Check detection criteria
        for criterion in pattern.detection_criteria:
            if criterion.lower() in input_text.lower() or criterion.lower() in model_output.lower():
                match_score += 0.2
        
        # Check failure indicators
        for indicator in pattern.failure_indicators:
            if indicator.lower() in model_output.lower():
                match_score += 0.3
        
        # Check example similarity
        for example in pattern.examples:
            if self._similar_failure(example, input_text, model_output, reference_output):
                match_score += 0.5
                break
        
        return min(1.0, match_score)
    
    def _similar_failure(self, example: PatternExample, input_text: str, 
                        model_output: str, reference_output: str) -> bool:
        """Check if current failure is similar to pattern example"""
        
        # Simple similarity check based on failure type and keywords
        if example.failure_type in input_text.lower() or example.failure_type in model_output.lower():
            return True
        
        # Check for common keywords
        common_keywords = ['error', 'fail', 'wrong', 'incorrect', 'invalid', 'missing']
        for keyword in common_keywords:
            if keyword in example.failure_description.lower() and keyword in model_output.lower():
                return True
        
        return False
    
    def _combine_analyses(self, standard_analysis: dict, pattern_analysis: dict) -> dict:
        """Combine standard MADF analysis with pattern-based analysis"""
        
        enhanced_analysis = standard_analysis.copy()
        
        # Enhance failure classification with pattern information
        if pattern_analysis["detected_failures"]:
            # Use pattern-based failure detection if available
            best_pattern_match = max(pattern_analysis["pattern_matches"], 
                                   key=lambda x: x["match_score"])
            
            enhanced_analysis["failure_classification"]["failure_category"] = best_pattern_match["pattern_name"]
            enhanced_analysis["failure_classification"]["confidence_score"] = best_pattern_match["match_score"]
            
            # Add pattern-specific information
            enhanced_analysis["failure_classification"]["pattern_based"] = True
            enhanced_analysis["failure_classification"]["pattern_id"] = best_pattern_match["pattern_id"]
        
        # Enhance root cause analysis
        if pattern_analysis["pattern_matches"]:
            enhanced_analysis["root_cause_analysis"]["pattern_insights"] = [
                {
                    "pattern": match["pattern_name"],
                    "root_cause": match["detection_criteria"],
                    "confidence": match["confidence"]
                }
                for match in pattern_analysis["pattern_matches"]
            ]
        
        # Enhance recommendations with pattern-based counterfactuals
        if pattern_analysis["pattern_matches"]:
            pattern_recommendations = []
            for match in pattern_analysis["pattern_matches"]:
                for cf in match["counterfactuals"]:
                    pattern_recommendations.append({
                        "type": "pattern_based",
                        "description": cf["description"],
                        "strategy": cf["strategy"],
                        "success_rate": cf["success_rate"],
                        "difficulty": cf["difficulty"],
                        "source_pattern": match["pattern_id"]
                    })
            
            enhanced_analysis["recommendation_suite"]["pattern_recommendations"] = pattern_recommendations
        
        # Add pattern analysis summary
        enhanced_analysis["pattern_analysis"] = pattern_analysis
        
        return enhanced_analysis
    
    def get_pattern_statistics(self) -> dict:
        """Get statistics about the pattern library"""
        
        stats = {
            "total_patterns": len(self.pattern_library.patterns),
            "patterns_by_category": {},
            "patterns_by_severity": {},
            "patterns_by_source": {},
            "average_confidence": 0.0,
            "high_confidence_patterns": 0
        }
        
        if self.pattern_library.patterns:
            # Category distribution
            for pattern in self.pattern_library.patterns.values():
                category = pattern.category
                stats["patterns_by_category"][category] = stats["patterns_by_category"].get(category, 0) + 1
                
                severity = pattern.severity.value
                stats["patterns_by_severity"][severity] = stats["patterns_by_severity"].get(severity, 0) + 1
                
                source = pattern.source.value
                stats["patterns_by_source"][source] = stats["patterns_by_source"].get(source, 0) + 1
                
                if pattern.confidence > 0.8:
                    stats["high_confidence_patterns"] += 1
            
            # Calculate average confidence
            confidences = [p.confidence for p in self.pattern_library.patterns.values()]
            stats["average_confidence"] = sum(confidences) / len(confidences)
        
        return stats

def main():
    """Example usage of enhanced MADF framework with pattern library"""
    
    enhanced_MADF = EnhancedMADFWithPatterns()
    
    # Get pattern library statistics
    stats = enhanced_MADF.get_pattern_statistics()
    print("\n📊 Pattern Library Statistics")
    print("=" * 40)
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    print(f"High confidence patterns: {stats['high_confidence_patterns']}")
    
    print("\nPatterns by category:")
    for category, count in stats["patterns_by_category"].items():
        print(f"  - {category}: {count}")
    
    print("\nPatterns by severity:")
    for severity, count in stats["patterns_by_severity"].items():
        print(f"  - {severity}: {count}")
    
    # Example 1: Code generation with security vulnerability
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Code Generation with Security Vulnerability")
    print("=" * 80)
    
    code_input = "Write a function to search users by name from database"
    code_output = "def search_users(name):\n    query = f\"SELECT * FROM users WHERE name = '{name}'\"\n    return execute_query(query)"
    code_reference = "def search_users(name):\n    query = \"SELECT * FROM users WHERE name = %s\"\n    return execute_query(query, (name,))"
    
    code_analysis = enhanced_MADF.analyze_failure_with_enhanced_patterns(
        "nl2code", code_input, code_output, code_reference
    )
    
    print(f"\n✅ Enhanced Analysis Complete!")
    print(f"Pattern matches: {code_analysis['pattern_analysis']['matching_patterns']}")
    print(f"Failure category: {code_analysis['failure_classification']['failure_category']}")
    print(f"Confidence: {code_analysis['failure_classification']['confidence_score']:.3f}")
    
    if code_analysis['pattern_analysis']['pattern_matches']:
        print(f"\n🔍 Pattern Matches:")
        for match in code_analysis['pattern_analysis']['pattern_matches']:
            print(f"  - {match['pattern_name']} (score: {match['match_score']:.3f})")
            print(f"    Severity: {match['severity']}")
            print(f"    Counterfactuals: {len(match['counterfactuals'])}")
    
    # Example 2: Text generation with factual error
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Text Generation with Factual Error")
    print("=" * 80)
    
    text_input = "What is the capital of France and its population?"
    text_output = "The capital of France is Paris, which has a population of 2.2 million people."
    text_reference = "The capital of France is Paris, which has a population of approximately 2.1 million people."
    
    text_analysis = enhanced_MADF.analyze_failure_with_enhanced_patterns(
        "nl2nl", text_input, text_output, text_reference
    )
    
    print(f"\n✅ Enhanced Analysis Complete!")
    print(f"Pattern matches: {text_analysis['pattern_analysis']['matching_patterns']}")
    print(f"Failure category: {text_analysis['failure_classification']['failure_category']}")
    print(f"Confidence: {text_analysis['failure_classification']['confidence_score']:.3f}")
    
    # Example 3: Pattern-based recommendations
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Pattern-Based Recommendations")
    print("=" * 80)
    
    if 'pattern_recommendations' in text_analysis['recommendation_suite']:
        print(f"\n🎯 Pattern-Based Recommendations:")
        for rec in text_analysis['recommendation_suite']['pattern_recommendations']:
            print(f"  - {rec['description']}")
            print(f"    Strategy: {rec['strategy']}")
            print(f"    Success rate: {rec['success_rate']:.2f}")
            print(f"    Difficulty: {rec['difficulty']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Enhanced MADF Framework Benefits")
    print("=" * 80)
    
    print("✅ Benefits of Systematic Pattern Library:")
    print("  1. Comprehensive failure coverage")
    print("  2. High-quality pattern validation")
    print("  3. Actionable counterfactual fixes")
    print("  4. Domain-specific pattern matching")
    print("  5. Continuous learning and improvement")
    
    print("\n✅ Pattern Library Quality Metrics:")
    print(f"  - Pattern coverage: {code_analysis['pattern_library_info']['pattern_coverage']:.2f}")
    print(f"  - High-confidence patterns: {code_analysis['pattern_library_info']['high_confidence_patterns']}")
    print(f"  - Relevant pattern matching: {code_analysis['pattern_analysis']['matching_patterns']}")

if __name__ == "__main__":
    main() 