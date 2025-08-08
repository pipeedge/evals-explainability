# LLM Explainability Framework: Implementation Details

## 1. Attention Analysis Implementation

### 1.1 Cross-Attention Computation Algorithm

```python
def compute_cross_attention(self, input_text: str, output_text: str) -> np.ndarray:
    """
    Compute cross-attention weights between input and output tokens
    
    Algorithm:
    1. Tokenize input and output texts
    2. Compute semantic similarity between all token pairs
    3. Apply softmax normalization for attention weights
    4. Handle numerical stability issues
    """
    # Tokenization with fallback for empty text
    input_tokens = input_text.split() if input_text.strip() else ["empty"]
    output_tokens = output_text.split() if output_text.strip() else ["empty"]
    
    # Initialize attention matrix
    attention_matrix = np.zeros((len(input_tokens), len(output_tokens)))
    
    # Compute semantic similarities
    semantic_sim = SemanticSimilarity()
    for i, input_token in enumerate(input_tokens):
        for j, output_token in enumerate(output_tokens):
            similarity = semantic_sim.compute_similarity(input_token, output_token)
            attention_matrix[i, j] = similarity
    
    # Numerical stability: handle all-zero matrices
    if np.all(attention_matrix == 0):
        np.fill_diagonal(attention_matrix, 1e-8)
    
    # Clean NaN/inf values
    attention_matrix = np.nan_to_num(attention_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Add epsilon for softmax stability
    attention_matrix = attention_matrix + 1e-10
    
    # Apply softmax normalization
    attention_tensor = torch.tensor(attention_matrix, dtype=torch.float32)
    attention_weights = F.softmax(attention_tensor, dim=1).numpy()
    
    return attention_weights
```

### 1.2 Attention Pattern Analysis

```python
def analyze_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, float]:
    """
    Analyze attention patterns for interpretability
    
    Metrics computed:
    - Concentration: How focused attention is
    - Dispersion: Entropy of attention distribution
    - Variance: Variability in attention weights
    - Sparsity: Proportion of low-attention tokens
    """
    # Clean attention weights
    clean_weights = np.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Ensure normalization
    if clean_weights.sum() == 0:
        clean_weights = np.ones_like(clean_weights) / clean_weights.size
    
    patterns = {}
    
    # Concentration: max - mean
    patterns['concentration'] = np.max(clean_weights) - np.mean(clean_weights)
    
    # Dispersion: entropy of non-zero weights
    flat_weights = clean_weights.flatten()
    flat_weights = flat_weights[flat_weights > 0]
    patterns['dispersion'] = entropy(flat_weights) if len(flat_weights) > 0 else 0.0
    
    # Variance
    patterns['variance'] = np.var(clean_weights)
    
    # Sparsity: proportion below threshold
    threshold = 0.1
    patterns['sparsity'] = np.mean(clean_weights < threshold)
    
    return patterns
```

## 2. Semantic Similarity Implementation

### 2.1 Multi-Metric Similarity Computation

```python
def compute_similarity(self, text1: str, text2: str, metric: str = "cosine") -> float:
    """
    Compute semantic similarity between two texts
    
    Supported metrics:
    - cosine: Cosine similarity between embeddings
    - euclidean: Inverse Euclidean distance
    - manhattan: Inverse Manhattan distance
    """
    # Handle empty text
    if not text1 or not text1.strip():
        text1 = "empty"
    if not text2 or not text2.strip():
        text2 = "empty"
    
    # Generate embeddings
    emb1 = self.embedding_model.encode([text1])
    emb2 = self.embedding_model.encode([text2])
    
    # Check for NaN embeddings
    if np.isnan(emb1).any() or np.isnan(emb2).any():
        return 0.0
    
    # Compute similarity based on metric
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
```

## 3. Causality Analysis Implementation

### 3.1 Granger Causality Algorithm

```python
def compute_granger_causality(self, x: np.ndarray, y: np.ndarray, max_lags: int = 5) -> float:
    """
    Compute Granger causality between two time series
    
    Algorithm:
    1. Ensure arrays are 1D and same length
    2. Compute lagged correlations for multiple lags
    3. Average correlation scores
    """
    # Ensure arrays are 1D
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    # Align lengths
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
```

### 3.2 Mutual Information Computation

```python
def compute_mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """
    Compute mutual information between two variables
    
    Algorithm:
    1. Discretize continuous variables into bins
    2. Compute joint and marginal probability distributions
    3. Calculate mutual information using KL divergence
    """
    # Discretize continuous variables
    x_discrete = np.digitize(x, np.linspace(x.min(), x.max(), bins))
    y_discrete = np.digitize(y, np.linspace(y.min(), y.max(), bins))
    
    # Compute joint distribution
    joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()
    
    # Compute marginal distributions
    x_prob = joint_prob.sum(axis=1)
    y_prob = joint_prob.sum(axis=0)
    
    # Compute mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
    
    return mi
```

## 4. Quality Assessment Implementation

### 4.1 Technical Quality Metrics

```python
def _compute_technical_metrics(self, explanation: str) -> Dict[str, float]:
    """
    Compute technical quality metrics for explanation text
    
    Metrics:
    - Length appropriateness
    - Readability score
    - Structure score
    """
    metrics = {}
    
    # Length appropriateness
    word_count = len(explanation.split())
    metrics['length_score'] = min(1.0, max(0.0, 1.0 - abs(word_count - 150) / 150))
    
    # Readability (based on average word length)
    words = explanation.split()
    if words:
        avg_word_length = np.mean([len(word) for word in words])
        metrics['readability_score'] = min(1.0, max(0.0, 1.0 - abs(avg_word_length - 5) / 5))
    else:
        metrics['readability_score'] = 0.0
    
    # Structure score (presence of structured elements)
    structure_indicators = ['1.', '2.', '**', '-', '•']
    structure_count = sum(1 for indicator in structure_indicators if indicator in explanation)
    metrics['structure_score'] = min(1.0, structure_count / 3)
    
    return metrics
```

### 4.2 Semantic Quality Metrics

```python
def _compute_semantic_metrics(self, explanation: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute semantic quality metrics
    
    Metrics:
    - Semantic similarity to ground truth
    - Content coverage
    """
    metrics = {}
    
    # Semantic similarity
    metrics['semantic_similarity'] = self.semantic_similarity.compute_similarity(
        explanation, ground_truth
    )
    
    # Content coverage
    explanation_words = set(explanation.lower().split())
    ground_truth_words = set(ground_truth.lower().split())
    
    if ground_truth_words:
        metrics['content_coverage'] = len(explanation_words & ground_truth_words) / len(ground_truth_words)
    else:
        metrics['content_coverage'] = 0.0
    
    return metrics
```

### 4.3 Overall Quality Computation

```python
def _compute_overall_quality(self, metrics: Dict[str, float]) -> float:
    """
    Compute weighted overall quality score
    
    Formula: Σ(metric_i * weight_i) / Σ(weight_i)
    """
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
```

## 5. Recommendation Quality Assessment

### 5.1 Multi-Criteria Decision Analysis

```python
def _compute_recommendation_score(self, recommendation: Dict[str, Any], 
                                weights: Dict[str, float]) -> float:
    """
    Compute weighted score for a recommendation
    
    Criteria:
    - Expected impact (higher is better)
    - Implementation effort (lower is better)
    - Confidence (higher is better)
    - Stakeholder alignment (higher is better)
    - Novelty (higher is better)
    """
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
```

### 5.2 Context-Aware Optimization

```python
def _compute_context_adjustment(self, recommendation: Dict[str, Any], 
                              context: Dict[str, Any]) -> float:
    """
    Compute context-based adjustment factor
    
    Adjustments based on:
    - Task type alignment
    - Time constraints
    - Resource availability
    """
    adjustment = 1.0
    
    # Task type adjustment
    task_type = context.get('task_type', 'unknown')
    rec_type = recommendation.get('recommendation_type', 'unknown')
    
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
```

## 6. Performance Monitoring Implementation

### 6.1 Real-Time Performance Tracking

```python
def _update_performance_stats(self, processing_time: float, success: bool):
    """
    Update performance statistics in real-time
    
    Metrics tracked:
    - Total analyses
    - Average processing time
    - Success rate
    """
    self.performance_stats['total_analyses'] += 1
    self.performance_stats['total_time'] += processing_time
    
    # Update average processing time
    if self.performance_stats['total_analyses'] > 0:
        self.performance_stats['average_time'] = (
            self.performance_stats['total_time'] / self.performance_stats['total_analyses']
        )
    
    # Update success rate
    if success:
        current_successes = self.performance_stats['success_rate'] * (self.performance_stats['total_analyses'] - 1)
        self.performance_stats['success_rate'] = (current_successes + 1) / self.performance_stats['total_analyses']
    else:
        current_successes = self.performance_stats['success_rate'] * (self.performance_stats['total_analyses'] - 1)
        self.performance_stats['success_rate'] = current_successes / self.performance_stats['total_analyses']
```

### 6.2 Confidence Assessment

```python
def _compute_overall_confidence(self, 
                               classification: FailureClassification,
                               root_cause: RootCauseAnalysis,
                               recommendations: RecommendationSuite) -> float:
    """
    Compute overall confidence in the analysis
    
    Formula: weighted average of component confidences
    """
    confidences = [
        classification.confidence_score,
        root_cause.confidence_score,
        recommendations.overall_confidence
    ]
    
    # Weighted average with more weight on classification and root cause
    weights = [0.4, 0.4, 0.2]
    overall_confidence = sum(c * w for c, w in zip(confidences, weights))
    
    return overall_confidence
```

## 7. Mathematical Formulations

### 7.1 Attention Concentration Formula

```
Concentration = max(attention_weights) - mean(attention_weights)
```

**Interpretation**: Higher values indicate more focused attention on specific tokens.

### 7.2 Attention Dispersion Formula

```
Dispersion = entropy(attention_weights)
```

Where entropy is calculated as:
```
entropy(p) = -Σ p_i * log(p_i)
```

**Interpretation**: Higher values indicate more distributed attention across tokens.

### 7.3 Semantic Similarity Formula

```
Similarity = cosine_similarity(emb1, emb2) = (emb1 · emb2) / (||emb1|| * ||emb2||)
```

**Range**: [-1, 1] where 1 indicates perfect semantic alignment.

### 7.4 Granger Causality Formula

```
Causality = mean(|correlation(x[t-lag], y[t])|)
```

For multiple lags l ∈ {1, 2, ..., max_lags}:
```
Causality = (1/max_lags) * Σ |corr(x[t-l], y[t])|
```

### 7.5 Mutual Information Formula

```
MI(X,Y) = Σ p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

**Properties**: 
- MI(X,Y) ≥ 0
- MI(X,Y) = 0 if and only if X and Y are independent
- Captures both linear and non-linear dependencies

### 7.6 Overall Quality Formula

```
Overall_Quality = Σ(metric_i * weight_i) / Σ(weight_i)
```

Where weights sum to 1.0 and represent the relative importance of each metric.

## 8. Algorithm Complexity Analysis

### 8.1 Attention Computation
- **Time Complexity**: O(n × m) where n = input tokens, m = output tokens
- **Space Complexity**: O(n × m) for attention matrix storage
- **Optimization**: Parallel similarity computation for large token sets

### 8.2 Causality Analysis
- **Time Complexity**: O(n² × max_lags) for Granger causality
- **Space Complexity**: O(n) for correlation storage
- **Optimization**: Early termination for low correlation values

### 8.3 Quality Assessment
- **Time Complexity**: O(n) where n = number of metrics
- **Space Complexity**: O(1) for metric storage
- **Optimization**: Cached embedding computation

## 9. Error Handling and Robustness

### 9.1 Numerical Stability
```python
# Handle NaN/inf values in attention weights
attention_weights = np.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)

# Add epsilon for softmax stability
attention_matrix = attention_matrix + 1e-10
```

### 9.2 Edge Case Handling
```python
# Handle empty text inputs
if not input_text or not input_text.strip():
    input_text = "empty_input"

# Handle all-zero attention matrices
if np.all(attention_matrix == 0):
    np.fill_diagonal(attention_matrix, 1e-8)
```

### 9.3 Fallback Mechanisms
```python
# Fallback for similarity computation errors
try:
    similarity = semantic_sim.compute_similarity(text1, text2)
except Exception as e:
    similarity = 0.0  # Default fallback value
```

## 10. Validation and Testing

### 10.1 Unit Tests
```python
def test_attention_computation():
    """Test attention weight computation"""
    analyzer = AttentionAnalyzer()
    
    # Test normal case
    attention = analyzer.compute_cross_attention("hello world", "hi there")
    assert attention.shape == (2, 2)
    assert not np.isnan(attention).any()
    
    # Test empty input
    attention_empty = analyzer.compute_cross_attention("", "")
    assert attention_empty.shape == (1, 1)
    assert not np.isnan(attention_empty).any()
```

### 10.2 Integration Tests
```python
def test_end_to_end_quality_assessment():
    """Test complete quality assessment pipeline"""
    metrics_evaluator = ExplainabilityMetrics()
    
    explanation = "The model failed due to syntax error in line 5."
    ground_truth = "The code contains a syntax error."
    
    quality_metrics = metrics_evaluator.compute_explanation_quality(
        explanation, ground_truth
    )
    
    assert 'overall_quality' in quality_metrics
    assert 0.0 <= quality_metrics['overall_quality'] <= 1.0
```

This implementation provides a robust, scalable foundation for evaluating LLM explainability with comprehensive metrics covering technical quality, semantic accuracy, user experience, and practical utility. 