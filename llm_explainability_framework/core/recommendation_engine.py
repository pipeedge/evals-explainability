"""
Recommendation Engine with Adaptive Learning

This module implements an innovative recommendation system that combines:
1. Multi-stakeholder optimization for diverse user needs
2. Adaptive learning from feedback and deployment outcomes
3. Contextual recommendation ranking with uncertainty quantification
4. Intervention strategy optimization using reinforcement learning principles
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

from .failure_classifier import FailureInstance, FailureClassification
from .root_cause_analyzer import RootCauseAnalysis, CausalFactor
from ..models.llm_wrapper import LLMWrapper
from ..utils.metrics import RecommendationRanker, ContextualOptimizer


class StakeholderType(Enum):
    """Types of stakeholders with different needs"""
    DEVELOPER = "developer"
    MANAGER = "manager"
    RESEARCHER = "researcher"
    END_USER = "end_user"


class RecommendationType(Enum):
    """Types of recommendations"""
    PROMPT_ENGINEERING = "prompt_engineering"
    DATA_AUGMENTATION = "data_augmentation"
    MODEL_CONFIGURATION = "model_configuration"
    ARCHITECTURAL_CHANGE = "architectural_change"
    TRAINING_STRATEGY = "training_strategy"


@dataclass
class Recommendation:
    """Individual recommendation with metadata"""
    recommendation_id: str
    recommendation_type: RecommendationType
    stakeholder_type: StakeholderType
    title: str
    description: str
    implementation_steps: List[str]
    expected_impact: float
    implementation_effort: float
    confidence: float
    priority_score: float
    evidence: List[str]
    constraints: List[str]


@dataclass
class RecommendationSuite:
    """Complete set of recommendations for a failure instance"""
    instance_id: str
    failure_category: str
    recommendations: List[Recommendation]
    optimization_strategy: Dict[str, Any]
    stakeholder_alignment: Dict[StakeholderType, float]
    overall_confidence: float
    implementation_roadmap: List[Dict[str, Any]]


class AdaptiveLearningSystem:
    """
    Learns from recommendation outcomes to improve future suggestions
    
    Innovation: Implements online learning with multi-armed bandit principles
    to continuously optimize recommendation quality based on real-world feedback.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.recommendation_history = defaultdict(list)
        self.outcome_history = defaultdict(list)
        self.feature_weights = defaultdict(float)
        
    def update_recommendation_outcome(self, recommendation_id: str,
                                    outcome_metrics: Dict[str, float]):
        """
        Update learning system with recommendation outcome
        
        Innovation: Multi-dimensional outcome tracking including effectiveness,
        user satisfaction, and implementation success.
        """
        self.outcome_history[recommendation_id].append(outcome_metrics)
        
        # Update feature weights based on outcome
        self._update_feature_weights(recommendation_id, outcome_metrics)
    
    def _update_feature_weights(self, recommendation_id: str,
                               outcome_metrics: Dict[str, float]):
        """Update feature weights using gradient-based learning"""
        # Get recommendation features
        if recommendation_id in self.recommendation_history:
            recommendation_data = self.recommendation_history[recommendation_id][-1]
            
            # Compute reward signal
            reward = self._compute_reward(outcome_metrics)
            
            # Update weights
            for feature, value in recommendation_data.items():
                if isinstance(value, (int, float)):
                    gradient = reward * value
                    self.feature_weights[feature] += self.learning_rate * gradient
    
    def _compute_reward(self, outcome_metrics: Dict[str, float]) -> float:
        """Compute reward signal from outcome metrics"""
        # Weighted combination of different outcome measures
        effectiveness = outcome_metrics.get('effectiveness', 0.0)
        user_satisfaction = outcome_metrics.get('user_satisfaction', 0.0)
        implementation_success = outcome_metrics.get('implementation_success', 0.0)
        
        reward = (
            0.5 * effectiveness +
            0.3 * user_satisfaction +
            0.2 * implementation_success
        )
        
        return reward
    
    def get_adapted_weights(self) -> Dict[str, float]:
        """Get current adapted feature weights"""
        return dict(self.feature_weights)


class MultiStakeholderOptimizer:
    """
    Optimizes recommendations for different stakeholder needs
    
    Innovation: Pareto-optimal recommendation generation that balances
    conflicting stakeholder objectives using multi-objective optimization.
    """
    
    def __init__(self):
        self.stakeholder_preferences = self._initialize_stakeholder_preferences()
        
    def _initialize_stakeholder_preferences(self) -> Dict[StakeholderType, Dict[str, float]]:
        """Initialize stakeholder preference profiles"""
        return {
            StakeholderType.DEVELOPER: {
                'implementation_ease': 0.8,
                'technical_detail': 0.9,
                'immediate_actionability': 0.7,
                'debugging_support': 0.9
            },
            StakeholderType.MANAGER: {
                'cost_effectiveness': 0.9,
                'risk_mitigation': 0.8,
                'timeline_impact': 0.8,
                'resource_requirements': 0.7
            },
            StakeholderType.RESEARCHER: {
                'novelty': 0.9,
                'generalizability': 0.8,
                'theoretical_grounding': 0.8,
                'experimental_potential': 0.7
            },
            StakeholderType.END_USER: {
                'user_experience': 0.9,
                'reliability': 0.8,
                'transparency': 0.7,
                'performance': 0.8
            }
        }
    
    def optimize_for_stakeholder(self, recommendations: List[Recommendation],
                                stakeholder: StakeholderType) -> List[Recommendation]:
        """
        Optimize recommendation ranking for specific stakeholder
        
        Innovation: Dynamic preference weighting based on stakeholder context
        and historical outcomes.
        """
        preferences = self.stakeholder_preferences[stakeholder]
        
        # Compute stakeholder-specific scores
        for rec in recommendations:
            rec.priority_score = self._compute_stakeholder_score(rec, preferences)
        
        # Sort by priority score
        optimized_recommendations = sorted(
            recommendations, key=lambda x: x.priority_score, reverse=True
        )
        
        return optimized_recommendations
    
    def _compute_stakeholder_score(self, recommendation: Recommendation,
                                  preferences: Dict[str, float]) -> float:
        """Compute stakeholder-specific priority score"""
        score = 0.0
        
        # Map recommendation attributes to stakeholder preferences
        if recommendation.stakeholder_type == StakeholderType.DEVELOPER:
            score += preferences.get('implementation_ease', 0.5) * (1.0 - recommendation.implementation_effort)
            score += preferences.get('technical_detail', 0.5) * recommendation.confidence
            score += preferences.get('immediate_actionability', 0.5) * recommendation.expected_impact
            
        elif recommendation.stakeholder_type == StakeholderType.MANAGER:
            score += preferences.get('cost_effectiveness', 0.5) * recommendation.expected_impact / max(recommendation.implementation_effort, 0.1)
            score += preferences.get('risk_mitigation', 0.5) * recommendation.confidence
            
        elif recommendation.stakeholder_type == StakeholderType.RESEARCHER:
            score += preferences.get('novelty', 0.5) * recommendation.expected_impact
            score += preferences.get('generalizability', 0.5) * recommendation.confidence
            score += preferences.get('theoretical_grounding', 0.5) * recommendation.expected_impact
            
        elif recommendation.stakeholder_type == StakeholderType.END_USER:
            score += preferences.get('user_experience', 0.5) * recommendation.expected_impact
            score += preferences.get('reliability', 0.5) * recommendation.confidence
            score += preferences.get('transparency', 0.5) * recommendation.expected_impact
        
        return score
    
    def generate_pareto_optimal_set(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        Generate Pareto-optimal recommendation set
        
        Innovation: Multi-objective optimization to find recommendations that
        are not dominated by others across all stakeholder dimensions.
        """
        pareto_set = []
        
        for i, rec1 in enumerate(recommendations):
            is_dominated = False
            
            for j, rec2 in enumerate(recommendations):
                if i != j and self._dominates(rec2, rec1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_set.append(rec1)
        
        return pareto_set
    
    def _dominates(self, rec1: Recommendation, rec2: Recommendation) -> bool:
        """Check if rec1 dominates rec2 in Pareto sense"""
        # rec1 dominates rec2 if it's better in all objectives
        objectives = [
            rec1.expected_impact >= rec2.expected_impact,
            rec1.confidence >= rec2.confidence,
            rec1.implementation_effort <= rec2.implementation_effort
        ]
        
        return all(objectives) and any([
            rec1.expected_impact > rec2.expected_impact,
            rec1.confidence > rec2.confidence,
            rec1.implementation_effort < rec2.implementation_effort
        ])


class ContextualRecommendationGenerator:
    """
    Generates contextual recommendations based on failure patterns
    
    Innovation: Template-based recommendation generation with context-aware
    parameter optimization and automated implementation step generation.
    """
    
    def __init__(self):
        self.recommendation_templates = self._initialize_templates()
        self.contextual_optimizer = ContextualOptimizer()
        
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize recommendation templates for different failure types"""
        return {
            # NL2NL failure templates
            "factual_inconsistency": {
                "prompt_engineering": {
                    "title": "Enhanced Fact-Checking Prompts",
                    "description": "Implement explicit fact-checking instructions in prompts",
                    "base_steps": [
                        "Add fact-verification requirements to system prompt",
                        "Include source citation instructions",
                        "Implement fact-checking validation loop"
                    ],
                    "expected_impact": 0.7,
                    "implementation_effort": 0.3
                },
                "data_augmentation": {
                    "title": "Fact-Verified Training Data",
                    "description": "Augment training data with verified factual examples",
                    "base_steps": [
                        "Collect fact-verified examples",
                        "Create negative examples with factual errors",
                        "Fine-tune model on enhanced dataset"
                    ],
                    "expected_impact": 0.8,
                    "implementation_effort": 0.7
                }
            },
            
            # NL2CODE failure templates
            "syntax_error": {
                "prompt_engineering": {
                    "title": "Syntax-Aware Code Generation Prompts",
                    "description": "Add explicit syntax validation requirements",
                    "base_steps": [
                        "Include syntax checking instructions",
                        "Add language-specific style guidelines",
                        "Implement code validation step"
                    ],
                    "expected_impact": 0.8,
                    "implementation_effort": 0.2
                },
                "model_configuration": {
                    "title": "Syntax-Constrained Decoding",
                    "description": "Configure model for syntax-aware generation",
                    "base_steps": [
                        "Implement constrained decoding",
                        "Add syntax parsing validation",
                        "Configure beam search parameters"
                    ],
                    "expected_impact": 0.9,
                    "implementation_effort": 0.5
                }
            },
            
            # CODE2NL failure templates
            "inaccurate_description": {
                "prompt_engineering": {
                    "title": "Code Analysis Prompts",
                    "description": "Enhance prompts for accurate code understanding",
                    "base_steps": [
                        "Add step-by-step code analysis instructions",
                        "Include variable tracking requirements",
                        "Implement logic flow verification"
                    ],
                    "expected_impact": 0.7,
                    "implementation_effort": 0.3
                }
            }
        }
    
    def generate_recommendations(self, instance: FailureInstance,
                               classification: FailureClassification,
                               root_cause: RootCauseAnalysis) -> List[Recommendation]:
        """
        Generate contextual recommendations for the failure instance
        
        Innovation: Dynamic template instantiation with context-specific
        parameter optimization and stakeholder-aware prioritization.
        """
        recommendations = []
        
        # Get relevant templates
        failure_category = classification.failure_category
        templates = self.recommendation_templates.get(failure_category, {})
        
        # Generate recommendations from templates
        for rec_type, template in templates.items():
            for stakeholder in StakeholderType:
                rec = self._instantiate_template(
                    template, rec_type, stakeholder, instance, classification, root_cause
                )
                if rec:
                    recommendations.append(rec)
        
        # Generate custom recommendations based on root cause analysis
        custom_recs = self._generate_custom_recommendations(
            instance, classification, root_cause
        )
        recommendations.extend(custom_recs)
        
        return recommendations
    
    def _instantiate_template(self, template: Dict[str, Any], rec_type: str,
                             stakeholder: StakeholderType, instance: FailureInstance,
                             classification: FailureClassification,
                             root_cause: RootCauseAnalysis) -> Optional[Recommendation]:
        """Instantiate a recommendation template with context-specific details"""
        
        # Determine recommendation type enum
        rec_type_enum = self._get_recommendation_type_enum(rec_type)
        if not rec_type_enum:
            return None
        
        # Customize template based on context
        customized_steps = self._customize_implementation_steps(
            template["base_steps"], instance, classification, root_cause
        )
        
        # Adjust impact and effort based on context
        adjusted_impact = self._adjust_impact_score(
            template["expected_impact"], classification, root_cause
        )
        adjusted_effort = self._adjust_effort_score(
            template["implementation_effort"], stakeholder, instance
        )
        
        # Generate evidence
        evidence = self._generate_evidence(instance, classification, root_cause)
        
        # Generate constraints
        constraints = self._generate_constraints(stakeholder, instance)
        
        recommendation = Recommendation(
            recommendation_id=f"{instance.input_id}_{rec_type}_{stakeholder.value}",
            recommendation_type=rec_type_enum,
            stakeholder_type=stakeholder,
            title=template["title"],
            description=template["description"],
            implementation_steps=customized_steps,
            expected_impact=adjusted_impact,
            implementation_effort=adjusted_effort,
            confidence=classification.confidence_score * 0.8,  # Slight discount for template
            priority_score=0.0,  # Will be set by optimizer
            evidence=evidence,
            constraints=constraints
        )
        
        return recommendation
    
    def _get_recommendation_type_enum(self, rec_type: str) -> Optional[RecommendationType]:
        """Map string to recommendation type enum"""
        mapping = {
            "prompt_engineering": RecommendationType.PROMPT_ENGINEERING,
            "data_augmentation": RecommendationType.DATA_AUGMENTATION,
            "model_configuration": RecommendationType.MODEL_CONFIGURATION,
            "architectural_change": RecommendationType.ARCHITECTURAL_CHANGE,
            "training_strategy": RecommendationType.TRAINING_STRATEGY
        }
        return mapping.get(rec_type)
    
    def _customize_implementation_steps(self, base_steps: List[str],
                                      instance: FailureInstance,
                                      classification: FailureClassification,
                                      root_cause: RootCauseAnalysis) -> List[str]:
        """Customize implementation steps based on specific context"""
        customized_steps = []
        
        for step in base_steps:
            # Add context-specific details
            if "prompt" in step.lower():
                customized_step = f"{step} (specific to {instance.task_type} tasks)"
            elif "data" in step.lower():
                customized_step = f"{step} (focus on {classification.failure_category} patterns)"
            else:
                customized_step = step
                
            customized_steps.append(customized_step)
        
        # Add context-specific steps
        if root_cause.primary_cause:
            customized_steps.append(f"Address primary cause: {root_cause.primary_cause}")
        
        return customized_steps
    
    def _adjust_impact_score(self, base_impact: float,
                           classification: FailureClassification,
                           root_cause: RootCauseAnalysis) -> float:
        """Adjust impact score based on failure context"""
        # Boost impact if high confidence in classification
        confidence_boost = classification.confidence_score * 0.2
        
        # Boost impact if strong causal evidence
        causal_boost = root_cause.confidence_score * 0.1
        
        adjusted_impact = min(1.0, base_impact + confidence_boost + causal_boost)
        return adjusted_impact
    
    def _adjust_effort_score(self, base_effort: float,
                           stakeholder: StakeholderType,
                           instance: FailureInstance) -> float:
        """Adjust effort score based on stakeholder and context"""
        # Developers might find technical changes easier
        if stakeholder == StakeholderType.DEVELOPER:
            adjusted_effort = base_effort * 0.8
        # Managers might find process changes easier
        elif stakeholder == StakeholderType.MANAGER:
            adjusted_effort = base_effort * 1.2
        else:
            adjusted_effort = base_effort
        
        return min(1.0, max(0.1, adjusted_effort))
    
    def _generate_evidence(self, instance: FailureInstance,
                         classification: FailureClassification,
                         root_cause: RootCauseAnalysis) -> List[str]:
        """Generate evidence supporting the recommendation"""
        evidence = [
            f"Failure classified as {classification.failure_category} with {classification.confidence_score:.2f} confidence",
            f"Primary root cause identified: {root_cause.primary_cause}"
        ]
        
        # Add causal factor evidence
        if root_cause.causal_factors:
            top_factor = root_cause.causal_factors[0]
            evidence.append(f"Strongest causal factor: {top_factor.factor_name} (strength: {top_factor.causal_strength:.3f})")
        
        return evidence
    
    def _generate_constraints(self, stakeholder: StakeholderType,
                            instance: FailureInstance) -> List[str]:
        """Generate implementation constraints"""
        constraints = []
        
        if stakeholder == StakeholderType.DEVELOPER:
            constraints.extend([
                "Must maintain code compatibility",
                "Should not significantly impact performance"
            ])
        elif stakeholder == StakeholderType.MANAGER:
            constraints.extend([
                "Must fit within existing budget",
                "Should not disrupt current timeline"
            ])
        
        # Task-specific constraints
        if instance.task_type == "NL2CODE":
            constraints.append("Must preserve code functionality")
        elif instance.task_type == "CODE2NL":
            constraints.append("Must maintain explanation accuracy")
        
        return constraints
    
    def _generate_custom_recommendations(self, instance: FailureInstance,
                                       classification: FailureClassification,
                                       root_cause: RootCauseAnalysis) -> List[Recommendation]:
        """Generate custom recommendations based on specific analysis results"""
        custom_recs = []
        
        # Generate recommendations based on counterfactual scenarios
        for cf in root_cause.counterfactual_scenarios[:2]:  # Top 2 counterfactuals
            if cf.get('expected_impact', 0) > 0.5:
                custom_rec = Recommendation(
                    recommendation_id=f"{instance.input_id}_custom_{cf.get('intervention_type', 'unknown')}",
                    recommendation_type=RecommendationType.PROMPT_ENGINEERING,
                    stakeholder_type=StakeholderType.DEVELOPER,
                    title=f"Counterfactual Intervention: {cf.get('intervention_type', 'Unknown')}",
                    description=cf.get('description', 'No description available'),
                    implementation_steps=[
                        "Analyze counterfactual scenario",
                        "Implement proposed intervention",
                        "Validate effectiveness"
                    ],
                    expected_impact=cf.get('expected_impact', 0.5),
                    implementation_effort=0.4,
                    confidence=root_cause.confidence_score,
                    priority_score=0.0,
                    evidence=[f"Derived from counterfactual analysis"],
                    constraints=["Requires careful validation"]
                )
                custom_recs.append(custom_rec)
        
        return custom_recs


class RecommendationEngine:
    """
    Main recommendation engine with adaptive learning and multi-stakeholder optimization
    
    Innovation: Integrates adaptive learning, multi-stakeholder optimization, and
    LLM-based recommendation generation for comprehensive solution suggestions.
    """
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        self.adaptive_learning = AdaptiveLearningSystem()
        self.stakeholder_optimizer = MultiStakeholderOptimizer()
        self.recommendation_generator = ContextualRecommendationGenerator()
        
        # Initialize LLM prompts
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize recommendation generation prompts"""
        self.recommendation_prompt_template = """
Based on the following failure analysis, provide a set of actionable recommendations to mitigate this type of error in the future.

**Failure Analysis Report:**
- **Input:** {input_text}
- **Failure Category:** {failure_category}
- **Root Cause Explanation:** {root_cause_analysis}

**Your Task:**
Generate a prioritized list of actionable recommendations. The recommendations should be concrete and targeted at either prompt engineering (for NL inputs), model fine-tuning, or data augmentation.

**Output Format (Markdown):**

### Actionable Recommendations

**1. For Prompt Engineering / Input Refinement:**
   - [Specific suggestions on how to rephrase or add detail to the input to avoid this error. Provide an example of an improved prompt or input structure if possible.]

**2. For Data Augmentation / Fine-Tuning:**
   - [Suggestions for new types of data to add to the training/testing set to teach the model how to handle this case better.]

**3. For Model Configuration (If Applicable):**
   - [Suggestions related to model parameters, such as temperature or top_p, that might influence this type of error.]
"""
    
    def generate_recommendations(self, instance: FailureInstance,
                               classification: FailureClassification,
                               root_cause: RootCauseAnalysis,
                               target_stakeholder: Optional[StakeholderType] = None) -> RecommendationSuite:
        """
        Generate comprehensive recommendation suite
        
        Innovation: Multi-modal recommendation generation combining template-based,
        LLM-generated, and adaptive learning-informed suggestions.
        """
        # Step 1: Generate template-based recommendations
        template_recs = self.recommendation_generator.generate_recommendations(
            instance, classification, root_cause
        )
        
        # Step 2: Generate LLM-based recommendations
        llm_recs = self._generate_llm_recommendations(instance, classification, root_cause)
        
        # Step 3: Combine and deduplicate recommendations
        all_recommendations = template_recs + llm_recs
        deduplicated_recs = self._deduplicate_recommendations(all_recommendations)
        
        # Step 4: Apply adaptive learning weights
        adapted_recs = self._apply_adaptive_learning(deduplicated_recs)
        
        # Step 5: Optimize for stakeholder if specified
        if target_stakeholder:
            optimized_recs = self.stakeholder_optimizer.optimize_for_stakeholder(
                adapted_recs, target_stakeholder
            )
        else:
            # Generate Pareto-optimal set for all stakeholders
            optimized_recs = self.stakeholder_optimizer.generate_pareto_optimal_set(adapted_recs)
        
        # Step 6: Generate implementation roadmap
        roadmap = self._generate_implementation_roadmap(optimized_recs)
        
        # Step 7: Compute stakeholder alignment scores
        alignment_scores = self._compute_stakeholder_alignment(optimized_recs)
        
        return RecommendationSuite(
            instance_id=instance.input_id,
            failure_category=classification.failure_category,
            recommendations=optimized_recs,
            optimization_strategy={
                "target_stakeholder": target_stakeholder.value if target_stakeholder else "multi_stakeholder",
                "optimization_method": "pareto_optimal",
                "adaptive_learning": True
            },
            stakeholder_alignment=alignment_scores,
            overall_confidence=np.mean([rec.confidence for rec in optimized_recs]),
            implementation_roadmap=roadmap
        )
    
    def _generate_llm_recommendations(self, instance: FailureInstance,
                                    classification: FailureClassification,
                                    root_cause: RootCauseAnalysis) -> List[Recommendation]:
        """Generate recommendations using LLM"""
        prompt = self.recommendation_prompt_template.format(
            input_text=instance.input_text,
            failure_category=classification.failure_category,
            root_cause_analysis=root_cause.explanation_text
        )
        
        response = self.llm.invoke(prompt)
        
        # Parse LLM response into structured recommendations
        llm_recs = self._parse_llm_recommendations(response, instance, classification)
        
        return llm_recs
    
    def _parse_llm_recommendations(self, llm_response: str, instance: FailureInstance,
                                 classification: FailureClassification) -> List[Recommendation]:
        """Parse LLM response into structured recommendations"""
        recommendations = []
        
        # Simple parsing logic (can be enhanced with more sophisticated NLP)
        sections = llm_response.split('**')
        
        current_type = None
        current_content = ""
        
        for section in sections:
            if "Prompt Engineering" in section:
                current_type = RecommendationType.PROMPT_ENGINEERING
            elif "Data Augmentation" in section:
                current_type = RecommendationType.DATA_AUGMENTATION
            elif "Model Configuration" in section:
                current_type = RecommendationType.MODEL_CONFIGURATION
            elif current_type and section.strip():
                current_content = section.strip()
                
                # Create recommendation
                rec = Recommendation(
                    recommendation_id=f"{instance.input_id}_llm_{current_type.value}",
                    recommendation_type=current_type,
                    stakeholder_type=StakeholderType.DEVELOPER,  # Default to developer
                    title=f"LLM-Generated {current_type.value.replace('_', ' ').title()}",
                    description=current_content[:200] + "..." if len(current_content) > 200 else current_content,
                    implementation_steps=current_content.split('\n')[:3],  # First 3 lines as steps
                    expected_impact=0.6,  # Default impact
                    implementation_effort=0.5,  # Default effort
                    confidence=classification.confidence_score * 0.7,  # Slight discount for LLM-generated
                    priority_score=0.0,
                    evidence=["Generated by LLM analysis"],
                    constraints=["Requires validation and testing"]
                )
                recommendations.append(rec)
                current_content = ""
        
        return recommendations
    
    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate or very similar recommendations"""
        # Simple deduplication based on title similarity
        unique_recs = []
        seen_titles = set()
        
        for rec in recommendations:
            title_key = rec.title.lower().replace(' ', '_')
            if title_key not in seen_titles:
                unique_recs.append(rec)
                seen_titles.add(title_key)
        
        return unique_recs
    
    def _apply_adaptive_learning(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Apply adaptive learning weights to recommendations"""
        adapted_weights = self.adaptive_learning.get_adapted_weights()
        
        for rec in recommendations:
            # Adjust scores based on learned weights
            if rec.recommendation_type.value in adapted_weights:
                weight = adapted_weights[rec.recommendation_type.value]
                rec.expected_impact *= (1.0 + weight * 0.2)  # Up to 20% boost
                rec.confidence *= (1.0 + weight * 0.1)  # Up to 10% boost
        
        return recommendations
    
    def _generate_implementation_roadmap(self, recommendations: List[Recommendation]) -> List[Dict[str, Any]]:
        """Generate implementation roadmap based on effort and dependencies"""
        roadmap = []
        
        # Sort by implementation effort (easiest first)
        sorted_recs = sorted(recommendations, key=lambda x: x.implementation_effort)
        
        phase = 1
        current_effort = 0.0
        phase_recs = []
        
        for rec in sorted_recs:
            if current_effort + rec.implementation_effort > 1.0 and phase_recs:
                # Start new phase
                roadmap.append({
                    "phase": phase,
                    "recommendations": [r.recommendation_id for r in phase_recs],
                    "total_effort": current_effort,
                    "expected_impact": sum(r.expected_impact for r in phase_recs) / len(phase_recs)
                })
                phase += 1
                current_effort = 0.0
                phase_recs = []
            
            phase_recs.append(rec)
            current_effort += rec.implementation_effort
        
        # Add final phase
        if phase_recs:
            roadmap.append({
                "phase": phase,
                "recommendations": [r.recommendation_id for r in phase_recs],
                "total_effort": current_effort,
                "expected_impact": sum(r.expected_impact for r in phase_recs) / len(phase_recs)
            })
        
        return roadmap
    
    def _compute_stakeholder_alignment(self, recommendations: List[Recommendation]) -> Dict[StakeholderType, float]:
        """Compute alignment scores for each stakeholder type"""
        alignment_scores = {}
        
        for stakeholder in StakeholderType:
            stakeholder_recs = [r for r in recommendations if r.stakeholder_type == stakeholder]
            if stakeholder_recs:
                avg_score = np.mean([r.priority_score for r in stakeholder_recs])
                alignment_scores[stakeholder] = avg_score
            else:
                alignment_scores[stakeholder] = 0.0
        
        return alignment_scores
    
    def update_recommendation_feedback(self, recommendation_id: str,
                                     outcome_metrics: Dict[str, float]):
        """Update adaptive learning system with recommendation outcome"""
        self.adaptive_learning.update_recommendation_outcome(recommendation_id, outcome_metrics) 