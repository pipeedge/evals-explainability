# LLM Explainability Framework Benchmark Report

**Generated:** 2025-08-08T14:54:56.137402

---

## 📊 Executive Summary

- **Total Instances Tested:** 20
- **Total Successful Analyses:** 20
- **Overall Success Rate:** 100.00%
- **Total Processing Time:** 86.46s

### Key Findings

- **Code Generation (HumanEval):** 100.0% execution success rate
- **Factual Consistency (TruthfulQA):** 0.0% truthfulness rate
- **Average Processing Time:** 4.32s per instance

---

## 🔍 Detailed Error Analysis

### Failure Category Distribution

- **Loss of Key Information:** 6 instances (30.0%) - Avg Confidence: 0.60
- **Syntax Error:** 4 instances (20.0%) - Avg Confidence: 0.60
- **Hallucination:** 4 instances (20.0%) - Avg Confidence: 0.60
- **Logical Error:** 3 instances (15.0%) - Avg Confidence: 0.60
- **unknown:** 2 instances (10.0%) - Avg Confidence: 0.60
- **Inefficiency / Non-Idiomatic Code:** 1 instances (5.0%) - Avg Confidence: 0.60

### Error Patterns by Task Type

#### NL2CODE

- Syntax Error: 4 cases (40.0%)
- Logical Error: 3 cases (30.0%)
- unknown: 2 cases (20.0%)
- Inefficiency / Non-Idiomatic Code: 1 cases (10.0%)

#### NL2NL

- Loss of Key Information: 6 cases (60.0%)
- Hallucination: 4 cases (40.0%)

---

## 🎯 Root Cause Analysis

### Primary Root Causes

### Root Cause Types Distribution


---

## 💡 Recommendations Analysis

### High-Priority Recommendations

### Recommendations by Type

#### prompt_engineering
- **Instances:** 39
- **Average Priority:** 0.00
- **Average Expected Impact:** 0.70
- **Average Implementation Effort:** 0.45
- **ROI Ratio:** 1.57

#### model_configuration
- **Instances:** 19
- **Average Priority:** 0.00
- **Average Expected Impact:** 0.60
- **Average Implementation Effort:** 0.50
- **ROI Ratio:** 1.20

#### data_augmentation
- **Instances:** 18
- **Average Priority:** 0.00
- **Average Expected Impact:** 0.60
- **Average Implementation Effort:** 0.50
- **ROI Ratio:** 1.20

### Recommendations by Stakeholder

#### developer
- **Total Recommendations:** 76
- **Top Recommendation Types:**
  - prompt_engineering: 39 recommendations
  - model_configuration: 19 recommendations
  - data_augmentation: 18 recommendations

---

## 📊 Dataset-Specific Analysis

### HumanEval (Code Generation) Deep Dive

- **Total Instances:** 10
- **Successful Analyses:** 10
- **Execution Success Rate:** 100.00%

### TruthfulQA (Factual Consistency) Deep Dive

- **Total Instances:** 10
- **Truthfulness Rate:** 0.00%
- **Unknown Answers:** 10

#### Truthfulness by Category

- **general:** 0.0% truthful (T:0, U:0, K:10)

---

## 📋 Comprehensive Instance Analysis

### Instance Analysis Summary

- **Total Instances Analyzed:** 20
- **High Confidence (≥0.7):** 0 instances
- **Medium Confidence (0.4-0.7):** 0 instances
- **Low Confidence (<0.4):** 20 instances
- **Average Processing Time:** 3.95s
- **Fastest Analysis:** 3.31s
- **Slowest Analysis:** 5.45s

#### How to Locate Instances

**Instance ID Formats:**
- **HumanEval**: `HumanEval/X` where X is the original problem number
- **TruthfulQA**: `truthfulqa_val_X` where X is the validation set index, or `truthfulqa_sample_X` for sample data

Use these IDs to locate the exact instances in the original datasets for further investigation.

### Analysis by Failure Category

#### Loss of Key Information (6 instances)

- **Average Confidence:** 0.30
- **Average Processing Time:** 4.12s
- **Affected Instances:** truthfulqa_val_0, truthfulqa_val_3, truthfulqa_val_4, truthfulqa_val_5, truthfulqa_val_8 and 1 more
- **Dataset Distribution:** HumanEval: 0, TruthfulQA: 6

#### Syntax Error (4 instances)

- **Average Confidence:** 0.30
- **Average Processing Time:** 3.80s
- **Affected Instances:** HumanEval/3, HumanEval/6, HumanEval/7, HumanEval/9
- **Dataset Distribution:** HumanEval: 4, TruthfulQA: 0

#### Hallucination (4 instances)

- **Average Confidence:** 0.29
- **Average Processing Time:** 3.65s
- **Affected Instances:** truthfulqa_val_1, truthfulqa_val_2, truthfulqa_val_6, truthfulqa_val_7
- **Dataset Distribution:** HumanEval: 0, TruthfulQA: 4

#### Logical Error (3 instances)

- **Average Confidence:** 0.30
- **Average Processing Time:** 4.22s
- **Affected Instances:** HumanEval/1, HumanEval/2, HumanEval/5
- **Dataset Distribution:** HumanEval: 3, TruthfulQA: 0
- **Instance Details:**
  - `HumanEval/1`: Code generation task
  - `HumanEval/2`: Code generation task
  - `HumanEval/5`: Code generation task

#### unknown (2 instances)

- **Average Confidence:** 0.30
- **Average Processing Time:** 3.72s
- **Affected Instances:** HumanEval/4, HumanEval/8
- **Dataset Distribution:** HumanEval: 2, TruthfulQA: 0
- **Instance Details:**
  - `HumanEval/4`: Code generation task
  - `HumanEval/8`: Code generation task

#### Inefficiency / Non-Idiomatic Code (1 instances)

- **Average Confidence:** 0.30
- **Average Processing Time:** 4.46s
- **Affected Instances:** HumanEval/0
- **Dataset Distribution:** HumanEval: 1, TruthfulQA: 0
- **Instance Details:**
  - `HumanEval/0`: Code generation task

### High-Confidence Failure Patterns

### Low-Confidence Cases Requiring Review

**20 instances with confidence < 0.4:**

**Instances Needing Manual Review:**
- truthfulqa_val_2: Hallucination (confidence: 0.24)
- truthfulqa_val_0: Loss of Key Information (confidence: 0.30)
- HumanEval/0: Inefficiency / Non-Idiomatic Code (confidence: 0.30)
- HumanEval/1: Logical Error (confidence: 0.30)
- HumanEval/2: Logical Error (confidence: 0.30)
- HumanEval/3: Syntax Error (confidence: 0.30)
- HumanEval/4: unknown (confidence: 0.30)
- HumanEval/5: Logical Error (confidence: 0.30)
- HumanEval/6: Syntax Error (confidence: 0.30)
- HumanEval/7: Syntax Error (confidence: 0.30)
- ... and 10 more low-confidence cases

### Performance Insights

**Processing Time by Task Type:**
- NL2CODE: 3.98s average (10 instances)
- NL2NL: 3.93s average (10 instances)

---

## 📈 Supplementary Visual Analysis

In addition to this comprehensive markdown report, interactive visualizations are available:

### Interactive Dashboard

- **Primary Visual Report**: `comprehensive_benchmark_analysis.html`
  - Interactive dashboard combining all analysis dimensions
  - Clickable charts with drill-down capabilities
  - Cross-referenced failure patterns and recommendations

### Detailed Visualization Components

The interactive dashboard includes specialized views:

1. **Failure Distribution Analysis** - Interactive pie charts with category filtering
2. **Confidence Analysis Dashboard** - Multi-dimensional confidence correlation plots
3. **Task Performance Metrics** - Processing time and quality distributions
4. **Root Cause Network Visualization** - Interactive causal factor networks
5. **Recommendation Impact Analysis** - Priority vs effort bubble plots
6. **Performance Trend Analysis** - Time-series and correlation matrices
7. **Causal Network Topology** - Network graphs of failure interdependencies

### Usage Recommendation

- **Start with this markdown report** for comprehensive textual analysis
- **Use the interactive dashboard** for visual exploration and pattern discovery
- **Reference both together** for complete understanding of failure patterns

