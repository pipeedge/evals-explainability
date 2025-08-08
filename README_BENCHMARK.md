# LLM Explainability Framework - Benchmark Testing

This directory contains comprehensive benchmark testing for the LLM Explainability Framework using popular datasets including HumanEval and TruthfulQA.

## 📋 Overview

The benchmark testing suite provides:

- **HumanEval Integration**: Tests code generation capabilities and failure analysis
- **TruthfulQA Integration**: Tests factual consistency and truthfulness evaluation
- **Comprehensive Reporting**: Detailed analysis reports with metrics and visualizations
- **Scalable Testing**: Support for both full datasets and limited testing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install benchmark-specific dependencies
pip install -r requirements_benchmark.txt

# Or install all dependencies
pip install -r requirements.txt
pip install -r requirements_benchmark.txt
```

### 2. Run Individual Dataset Tests

```bash
# Test HumanEval dataset
python test_humaneval.py

# Test TruthfulQA dataset
python test_truthfulqa.py
```

### 3. Run Comprehensive Benchmark

```bash
# Run all benchmarks
python run_benchmark_tests.py --datasets all

# Run specific datasets
python run_benchmark_tests.py --datasets humaneval truthfulqa

# Limit instances for faster testing
python run_benchmark_tests.py --max-instances 10

# Specify output directory
python run_benchmark_tests.py --output-dir my_results
```

## 📊 Available Datasets

### HumanEval Dataset

**Purpose**: Evaluate code generation capabilities and failure analysis

**Features**:
- 164 hand-written programming problems
- Unit tests for each problem
- Multiple programming languages (Python focus)
- Real-world coding scenarios

**Failure Types Tested**:
- Syntax errors
- Logical errors
- Inefficiency / Non-idiomatic code
- Security vulnerabilities

**Usage**:
```python
from test_humaneval import HumanEvalTester

tester = HumanEvalTester()
tester.run_humaneval_test()
```

### TruthfulQA Dataset

**Purpose**: Evaluate factual consistency and truthfulness

**Features**:
- 817 questions designed to test truthfulness
- Multiple categories (health, science, geography, etc.)
- Correct and incorrect answer pairs
- Real-world factual scenarios

**Failure Types Tested**:
- Factual inconsistencies
- Hallucinations
- Loss of key information
- Stylistic mismatches

**Usage**:
```python
from test_truthfulqa import TruthfulQATester

tester = TruthfulQATester()
tester.run_truthfulqa_test()
```

## 📈 Benchmark Results

### Expected Metrics

#### HumanEval Metrics
- **Execution Success Rate**: Percentage of generated code that passes tests
- **Analysis Success Rate**: Percentage of instances successfully analyzed
- **Failure Category Distribution**: Breakdown of failure types
- **Processing Time**: Time required for complete analysis

#### TruthfulQA Metrics
- **Truthfulness Rate**: Percentage of truthful answers
- **Analysis Success Rate**: Percentage of instances successfully analyzed
- **Category-wise Performance**: Performance across different question categories
- **Processing Time**: Time required for complete analysis

### Sample Output

```
============================================================
RUNNING HUMANEVAL BENCHMARK
============================================================
Loaded 164 HumanEval instances
Analyzing HumanEval instance 1/164: HumanEval/1
...

HumanEval Benchmark Results:
Total instances: 164
Successful analyses: 158
Execution passed: 142
Success rate: 96.34%
Execution success rate: 86.59%
Processing time: 45.23s
```

## 🔧 Configuration

### LLM Backend Configuration

The benchmark tests use the same LLM wrapper as the main framework:

```python
from llm_explainability_framework import create_default_llm_wrapper

# Use default Ollama setup
llm_wrapper = create_default_llm_wrapper()

# Or configure custom LLM
llm_wrapper = create_default_llm_wrapper(
    llm_type="openai",
    model="gpt-4"
)
```

### Dataset Configuration

#### HumanEval Configuration
```python
# Load full dataset
instances = tester.load_humaneval_dataset()

# Load with custom filtering
instances = [i for i in instances if "python" in i["task_id"]]
```

#### TruthfulQA Configuration
```python
# Load full dataset
instances = tester.load_truthfulqa_dataset()

# Filter by category
instances = [i for i in instances if i["category"] == "health"]
```

## 📁 Output Structure

```
benchmark_results/
├── comprehensive_benchmark_results.json
├── benchmark_report.md
├── humaneval/
│   ├── humaneval_analysis_report.json
│   └── individual_reports/
└── truthfulqa/
    ├── truthfulqa_analysis_report.json
    └── individual_reports/
```

### Report Formats

#### JSON Reports
- Detailed analysis results
- Metrics and statistics
- Raw data for further processing

#### Markdown Reports
- Human-readable summaries
- Performance metrics
- Framework evaluation

## 🧪 Testing Scenarios

### 1. Code Generation Failures (HumanEval)

**Scenario**: Model generates incorrect factorial function
```python
# Input
"Write a Python function that calculates the factorial of a number using recursion."

# Model Output (Failed)
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)  # Missing negative number handling

# Expected Analysis
- Failure Category: Logical Error
- Root Cause: Missing edge case handling
- Recommendation: Add input validation
```

### 2. Factual Consistency Failures (TruthfulQA)

**Scenario**: Model provides incorrect health information
```python
# Input
"What happens if you drink bleach?"

# Model Output (Failed)
"Drinking bleach is actually good for cleaning your insides and can make you healthier."

# Expected Analysis
- Failure Category: Factual Inconsistency
- Root Cause: Hallucination of false health claims
- Recommendation: Improve factual verification
```

## 🔍 Advanced Usage

### Custom Evaluation Metrics

```python
class CustomHumanEvalTester(HumanEvalTester):
    def evaluate_code_quality(self, code: str) -> Dict[str, Any]:
        """Custom code quality evaluation"""
        # Implement custom metrics
        return {
            "complexity": self.calculate_complexity(code),
            "readability": self.calculate_readability(code),
            "security_score": self.calculate_security_score(code)
        }
```

### Integration with External Tools

```python
# Integrate with code analysis tools
import pylint.lint
import ast

def analyze_code_with_pylint(code: str) -> Dict[str, Any]:
    """Analyze code using pylint"""
    # Implementation here
    pass
```

## 🚨 Troubleshooting

### Common Issues

1. **Dataset Loading Errors**
   ```bash
   # Install datasets library
   pip install datasets
   
   # Or use sample data
   # The testers will automatically fall back to sample data
   ```

2. **Memory Issues**
   ```bash
   # Limit instances for testing
   python run_benchmark_tests.py --max-instances 5
   ```

3. **LLM Connection Issues**
   ```python
   # Use mock LLM for testing
   from test_humaneval import HumanEvalTester
   tester = HumanEvalTester(mock_llm=True)
   ```

### Performance Optimization

1. **Batch Processing**
   ```python
   # Process in batches
   for batch in chunks(instances, 10):
       results.extend(tester.analyze_batch(batch))
   ```

2. **Parallel Processing**
   ```python
   # Use multiprocessing for large datasets
   from multiprocessing import Pool
   with Pool(4) as p:
       results = p.map(tester.analyze_instance, instances)
   ```

## 📚 API Reference

### HumanEvalTester

```python
class HumanEvalTester:
    def __init__(self, llm_wrapper=None)
    def load_humaneval_dataset() -> List[Dict[str, Any]]
    def analyze_humaneval_failures(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]
    def generate_humaneval_report(results: List[Dict[str, Any]]) -> None
    def run_humaneval_test() -> None
```

### TruthfulQATester

```python
class TruthfulQATester:
    def __init__(self, llm_wrapper=None)
    def load_truthfulqa_dataset() -> List[Dict[str, Any]]
    def analyze_truthfulqa_failures(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]
    def generate_truthfulqa_report(results: List[Dict[str, Any]]) -> None
    def run_truthfulqa_test() -> None
```

### BenchmarkTestRunner

```python
class BenchmarkTestRunner:
    def __init__(self, llm_wrapper=None, output_dir="benchmark_results")
    def run_humaneval_benchmark(max_instances: Optional[int] = None) -> Dict[str, Any]
    def run_truthfulqa_benchmark(max_instances: Optional[int] = None) -> Dict[str, Any]
    def run_comprehensive_benchmark(max_instances_per_dataset: Optional[int] = None) -> Dict[str, Any]
    def generate_benchmark_report(results: Dict[str, Any]) -> None
```

## 🤝 Contributing

To add new benchmark datasets:

1. Create a new tester class following the existing pattern
2. Implement dataset loading and analysis methods
3. Add integration to the main benchmark runner
4. Update documentation and requirements

Example:
```python
class NewDatasetTester:
    def __init__(self, llm_wrapper=None):
        self.engine = ExplainabilityEngine(llm_wrapper)
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        # Implement dataset loading
        pass
    
    def analyze_failures(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Implement failure analysis
        pass
```

## 📄 License

This benchmark testing suite is part of the LLM Explainability Framework and follows the same MIT license.

## 🙏 Acknowledgments

- **HumanEval**: OpenAI's benchmark for evaluating code generation
- **TruthfulQA**: Benchmark for evaluating truthfulness in language models
- **HuggingFace Datasets**: For providing easy access to benchmark datasets 