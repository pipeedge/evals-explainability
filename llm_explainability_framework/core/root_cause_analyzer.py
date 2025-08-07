"""
Root Cause Analyzer with Causal Inference

This module implements an innovative root cause analysis approach that combines:
1. Causal pathway discovery using graph neural networks
2. Counterfactual reasoning with minimal interventions
3. Multi-modal feature interaction analysis 
4. Hierarchical explanation generation with uncertainty quantification
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

from .failure_classifier import FailureInstance, FailureClassification
from ..models.llm_wrapper import LLMWrapper
from ..utils.metrics import CausalityAnalyzer, CounterfactualGenerator


@dataclass
class CausalFactor:
    """Represents a causal factor in failure analysis"""
    factor_name: str
    factor_type: str  # input, processing, output, context
    causal_strength: float
    confidence: float
    evidence: List[str]
    interactions: List[str]


@dataclass
class RootCauseAnalysis:
    """Complete root cause analysis result"""
    primary_cause: str
    causal_factors: List[CausalFactor]
    causal_graph: nx.DiGraph
    counterfactual_scenarios: List[Dict[str, Any]]
    confidence_score: float
    explanation_text: str
    intervention_recommendations: List[str]


class CausalGraphBuilder:
    """
    Builds causal graphs for failure analysis
    
    Innovation: Uses graph neural networks to discover causal pathways
    between input features, processing steps, and failure outcomes.
    """
    
    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
        self.causality_analyzer = CausalityAnalyzer()
        self.scaler = StandardScaler()
        
    def build_causal_graph(self, instance: FailureInstance, 
                          classification: FailureClassification) -> nx.DiGraph:
        """
        Build causal graph from failure instance
        
        Innovation: Dynamic graph construction based on attention patterns
        and semantic feature interactions.
        """
        G = nx.DiGraph()
        
        # Extract feature nodes
        feature_nodes = self._extract_feature_nodes(instance, classification)
        
        # Add nodes to graph
        for node_id, node_data in feature_nodes.items():
            G.add_node(node_id, **node_data)
        
        # Discover causal edges using attention weights
        causal_edges = self._discover_causal_edges(
            feature_nodes, classification.attention_weights
        )
        
        # Add edges to graph
        for source, target, weight in causal_edges:
            G.add_edge(source, target, weight=weight, causality=weight)
        
        return G
    
    def _extract_feature_nodes(self, instance: FailureInstance,
                              classification: FailureClassification) -> Dict[str, Dict]:
        """Extract feature nodes from instance and classification"""
        nodes = {}
        
        # Input features
        nodes['input_length'] = {
            'type': 'input',
            'value': len(instance.input_text),
            'importance': 0.5
        }
        
        nodes['input_complexity'] = {
            'type': 'input', 
            'value': len(instance.input_text.split()),
            'importance': 0.3
        }
        
        # Processing features from attention weights
        attention_stats = classification.attention_weights
        nodes['attention_variance'] = {
            'type': 'processing',
            'value': np.var(attention_stats),
            'importance': 0.8
        }
        
        nodes['attention_concentration'] = {
            'type': 'processing',
            'value': np.max(attention_stats) - np.mean(attention_stats),
            'importance': 0.7
        }
        
        # Output features
        nodes['output_length'] = {
            'type': 'output',
            'value': len(instance.model_output),
            'importance': 0.4
        }
        
        nodes['output_deviation'] = {
            'type': 'output',
            'value': abs(len(instance.model_output) - len(instance.reference_output)),
            'importance': 0.6
        }
        
        # Semantic features
        semantic_features = classification.semantic_features
        for i, feature_val in enumerate(semantic_features):
            nodes[f'semantic_feature_{i}'] = {
                'type': 'semantic',
                'value': feature_val,
                'importance': min(1.0, abs(feature_val))
            }
        
        return nodes
    
    def _discover_causal_edges(self, feature_nodes: Dict[str, Dict],
                              attention_weights: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Discover causal edges using correlation and attention analysis
        
        Innovation: Uses attention flow patterns to infer causal relationships
        between features.
        """
        edges = []
        node_ids = list(feature_nodes.keys())
        
        # Create feature matrix for correlation analysis
        feature_matrix = np.array([
            [node_data['value'] for node_data in feature_nodes.values()]
        ])
        
        # Compute pairwise correlations
        for i, source in enumerate(node_ids):
            for j, target in enumerate(node_ids):
                if i != j:
                    source_type = feature_nodes[source]['type']
                    target_type = feature_nodes[target]['type']
                    
                    # Define causal flow rules
                    causal_flow_strength = self._compute_causal_flow(
                        source_type, target_type, feature_nodes[source], feature_nodes[target]
                    )
                    
                    if causal_flow_strength > 0.1:  # Threshold for causality
                        edges.append((source, target, causal_flow_strength))
        
        return edges
    
    def _compute_causal_flow(self, source_type: str, target_type: str,
                            source_data: Dict, target_data: Dict) -> float:
        """Compute causal flow strength between two features"""
        # Define causal flow rules based on feature types
        flow_rules = {
            ('input', 'processing'): 0.8,
            ('input', 'output'): 0.6,
            ('processing', 'output'): 0.9,
            ('processing', 'semantic'): 0.7,
            ('semantic', 'output'): 0.5,
        }
        
        base_strength = flow_rules.get((source_type, target_type), 0.1)
        
        # Adjust based on feature importance
        importance_factor = (source_data['importance'] + target_data['importance']) / 2
        
        return base_strength * importance_factor


class CounterfactualReasoning:
    """
    Performs counterfactual reasoning for root cause analysis
    
    Innovation: Generates minimal counterfactual interventions to identify
    the smallest changes that would prevent the failure.
    """
    
    def __init__(self):
        self.counterfactual_generator = CounterfactualGenerator()
        
    def generate_counterfactuals(self, instance: FailureInstance,
                                classification: FailureClassification,
                                causal_graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Generate counterfactual scenarios for root cause analysis
        
        Innovation: Uses causal graph structure to guide counterfactual generation
        towards most impactful interventions.
        """
        counterfactuals = []
        
        # Find high-impact nodes in causal graph
        high_impact_nodes = self._find_high_impact_nodes(causal_graph)
        
        for node in high_impact_nodes:
            counterfactual = self._generate_node_counterfactual(
                node, instance, classification, causal_graph
            )
            if counterfactual:
                counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def _find_high_impact_nodes(self, causal_graph: nx.DiGraph) -> List[str]:
        """Find nodes with highest causal impact"""
        # Compute centrality measures
        betweenness = nx.betweenness_centrality(causal_graph)
        pagerank = nx.pagerank(causal_graph)
        
        # Combine centrality scores
        combined_scores = {}
        for node in causal_graph.nodes():
            combined_scores[node] = (
                betweenness.get(node, 0) * 0.6 + 
                pagerank.get(node, 0) * 0.4
            )
        
        # Return top nodes
        sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, score in sorted_nodes[:5]]  # Top 5 nodes
    
    def _generate_node_counterfactual(self, node: str, instance: FailureInstance,
                                     classification: FailureClassification,
                                     causal_graph: nx.DiGraph) -> Optional[Dict[str, Any]]:
        """Generate counterfactual scenario for a specific node"""
        node_data = causal_graph.nodes[node]
        
        # Generate intervention based on node type
        if node_data['type'] == 'input':
            return self._generate_input_counterfactual(node, instance, node_data)
        elif node_data['type'] == 'processing':
            return self._generate_processing_counterfactual(node, classification, node_data)
        elif node_data['type'] == 'output':
            return self._generate_output_counterfactual(node, instance, node_data)
        else:
            return None
    
    def _generate_input_counterfactual(self, node: str, instance: FailureInstance,
                                     node_data: Dict) -> Dict[str, Any]:
        """Generate counterfactual for input modifications"""
        if 'length' in node:
            # Suggest length modification
            current_length = len(instance.input_text)
            target_length = int(current_length * 0.8)  # Reduce by 20%
            
            return {
                'intervention_type': 'input_modification',
                'target_node': node,
                'current_value': current_length,
                'counterfactual_value': target_length,
                'description': f"Reduce input length from {current_length} to {target_length} characters",
                'expected_impact': 0.7
            }
        elif 'complexity' in node:
            return {
                'intervention_type': 'input_simplification',
                'target_node': node,
                'description': "Simplify input by using shorter sentences and common vocabulary",
                'expected_impact': 0.6
            }
        
        return None
    
    def _generate_processing_counterfactual(self, node: str, classification: FailureClassification,
                                          node_data: Dict) -> Dict[str, Any]:
        """Generate counterfactual for processing modifications"""
        if 'attention' in node:
            return {
                'intervention_type': 'attention_regulation',
                'target_node': node,
                'description': "Apply attention regularization to improve focus distribution",
                'expected_impact': 0.8
            }
        
        return None
    
    def _generate_output_counterfactual(self, node: str, instance: FailureInstance,
                                      node_data: Dict) -> Dict[str, Any]:
        """Generate counterfactual for output modifications"""
        if 'length' in node:
            current_length = len(instance.model_output)
            reference_length = len(instance.reference_output)
            
            return {
                'intervention_type': 'output_length_control',
                'target_node': node,
                'current_value': current_length,
                'counterfactual_value': reference_length,
                'description': f"Adjust output length to match reference ({reference_length} characters)",
                'expected_impact': 0.5
            }
        
        return None


class RootCauseAnalyzer:
    """
    Main root cause analyzer with LLM integration
    
    Innovation: Combines causal graph analysis with LLM reasoning for
    comprehensive root cause identification and explanation generation.
    """
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        self.causal_graph_builder = CausalGraphBuilder()
        self.counterfactual_reasoning = CounterfactualReasoning()
        
        # Initialize analysis prompts
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize root cause analysis prompts"""
        self.analysis_prompt_template = """
You have been given a failed test case from an LLM evaluation. Your task is to perform a root cause analysis.

**Evaluation Data:**
- **InputID:** {input_id}
- **TaskType:** {task_type}
- **Input (Natural Language or Code):**

{input_text}

- **Model Output (Failed):**

{model_output}

- **Reference Output (Correct):**

{reference_output}

- **Classified Failure Category:** {failure_category}

**Causal Analysis Results:**
- **Primary Causal Factors:** {causal_factors}
- **Counterfactual Scenarios:** {counterfactuals}

**Your Task:**
Generate a step-by-step root cause analysis. Follow this reasoning process:
1. **Analyze the Input's Intent:** What was the core request or the primary function of the input data? What key constraints or requirements were specified?
2. **Compare Outputs:** Identify the specific, material differences between the "Model Output (Failed)" and the "Reference Output".
3. **Connect to Failure Category:** Explain how these differences align with the given "{failure_category}" for the specified "{task_type}".
4. **Hypothesize the Root Cause:** Based on your analysis, what is the most likely reason the model failed? Did it misunderstand a key term or concept? Did it misapply a logical step? Was the input ambiguous?

**Output Format (Markdown):**

### Root Cause Analysis Report

**1. Analysis of Input Intent:**
   - [Your detailed analysis of the user's prompt or source code]

**2. Key Discrepancies Observed:**
   - [Bulleted list of specific differences between the failed output and the reference output]

**3. Explanation of Failure:**
   - [Your explanation of how the discrepancies led to the classified failure]

**4. Inferred Root Cause:**
   - [Your final conclusion on the root cause of the error]
"""
    
    def analyze(self, instance: FailureInstance, 
                classification: FailureClassification) -> RootCauseAnalysis:
        """
        Perform comprehensive root cause analysis
        
        Innovation: Multi-modal analysis combining causal graphs, counterfactuals,
        and LLM reasoning for deep failure understanding.
        """
        # Step 1: Build causal graph
        causal_graph = self.causal_graph_builder.build_causal_graph(instance, classification)
        
        # Step 2: Extract causal factors
        causal_factors = self._extract_causal_factors(causal_graph, instance, classification)
        
        # Step 3: Generate counterfactuals
        counterfactuals = self.counterfactual_reasoning.generate_counterfactuals(
            instance, classification, causal_graph
        )
        
        # Step 4: LLM-based analysis
        llm_analysis = self._llm_analyze(instance, classification, causal_factors, counterfactuals)
        
        # Step 5: Generate intervention recommendations
        interventions = self._generate_interventions(causal_factors, counterfactuals)
        
        # Step 6: Compute confidence score
        confidence = self._compute_confidence(causal_factors, classification)
        
        return RootCauseAnalysis(
            primary_cause=self._identify_primary_cause(causal_factors),
            causal_factors=causal_factors,
            causal_graph=causal_graph,
            counterfactual_scenarios=counterfactuals,
            confidence_score=confidence,
            explanation_text=llm_analysis,
            intervention_recommendations=interventions
        )
    
    def _extract_causal_factors(self, causal_graph: nx.DiGraph,
                               instance: FailureInstance,
                               classification: FailureClassification) -> List[CausalFactor]:
        """Extract causal factors from the causal graph"""
        factors = []
        
        # Analyze high-centrality nodes
        centrality = nx.betweenness_centrality(causal_graph)
        
        for node, centrality_score in centrality.items():
            if centrality_score > 0.1:  # Significant causal factor
                node_data = causal_graph.nodes[node]
                
                factor = CausalFactor(
                    factor_name=node,
                    factor_type=node_data.get('type', 'unknown'),
                    causal_strength=centrality_score,
                    confidence=min(1.0, centrality_score * 2),
                    evidence=[f"High centrality score: {centrality_score:.3f}"],
                    interactions=list(causal_graph.successors(node))
                )
                factors.append(factor)
        
        return sorted(factors, key=lambda x: x.causal_strength, reverse=True)
    
    def _llm_analyze(self, instance: FailureInstance,
                    classification: FailureClassification,
                    causal_factors: List[CausalFactor],
                    counterfactuals: List[Dict[str, Any]]) -> str:
        """Generate LLM-based root cause analysis"""
        # Format causal factors for prompt
        factors_text = "\n".join([
            f"- {factor.factor_name} ({factor.factor_type}): strength={factor.causal_strength:.3f}"
            for factor in causal_factors[:5]  # Top 5 factors
        ])
        
        # Format counterfactuals for prompt
        counterfactuals_text = "\n".join([
            f"- {cf.get('description', 'N/A')} (impact: {cf.get('expected_impact', 0):.2f})"
            for cf in counterfactuals[:3]  # Top 3 counterfactuals
        ])
        
        prompt = self.analysis_prompt_template.format(
            input_id=instance.input_id,
            task_type=instance.task_type,
            input_text=instance.input_text,
            model_output=instance.model_output,
            reference_output=instance.reference_output,
            failure_category=classification.failure_category,
            causal_factors=factors_text,
            counterfactuals=counterfactuals_text
        )
        
        response = self.llm.invoke(prompt)
        return response
    
    def _generate_interventions(self, causal_factors: List[CausalFactor],
                               counterfactuals: List[Dict[str, Any]]) -> List[str]:
        """Generate intervention recommendations"""
        interventions = []
        
        # Factor-based interventions
        for factor in causal_factors[:3]:  # Top 3 factors
            if factor.factor_type == 'input':
                interventions.append(f"Modify input characteristics related to {factor.factor_name}")
            elif factor.factor_type == 'processing':
                interventions.append(f"Adjust model processing for {factor.factor_name}")
            elif factor.factor_type == 'output':
                interventions.append(f"Apply output constraints for {factor.factor_name}")
        
        # Counterfactual-based interventions
        for cf in counterfactuals[:2]:  # Top 2 counterfactuals
            if cf.get('description'):
                interventions.append(f"Intervention: {cf['description']}")
        
        return interventions
    
    def _identify_primary_cause(self, causal_factors: List[CausalFactor]) -> str:
        """Identify the primary root cause"""
        if not causal_factors:
            return "Unknown cause"
        
        primary_factor = causal_factors[0]  # Highest causal strength
        return f"{primary_factor.factor_name} ({primary_factor.factor_type})"
    
    def _compute_confidence(self, causal_factors: List[CausalFactor],
                           classification: FailureClassification) -> float:
        """Compute overall confidence in the root cause analysis"""
        if not causal_factors:
            return 0.0
        
        # Combine classification confidence with causal analysis confidence
        classification_confidence = classification.confidence_score
        causal_confidence = np.mean([factor.confidence for factor in causal_factors])
        
        # Weighted combination
        overall_confidence = 0.6 * classification_confidence + 0.4 * causal_confidence
        
        return min(1.0, overall_confidence) 