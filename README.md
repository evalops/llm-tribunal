# LLM Tribunal: Advanced Multi-Critic LLM Evaluation Framework

> **Advanced LLM evaluation framework** with DAG execution, multi-critic protocols, and synthetic test generation for comprehensive AI safety assessment.

A sophisticated evaluation system designed for production LLM testing, featuring DAG-based execution, multi-round deliberation protocols, and specialized tools for OWASP LLM Top 10 threat assessment.

**Building on [Haize Labs' Verdict](https://github.com/haizelabs/verdict)** framework for scaling judge-time compute, LLM Tribunal adapts multi-critic evaluation concepts for local model deployment with open-source models like `gpt-oss:20b` and `gpt-oss:120b`.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/evalops/llm-tribunal.git
cd llm-tribunal
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```bash
# Validate a pipeline
python cli.py validate -f examples/pipelines/simple_llm_threat_pipeline.yaml

# Run a complete evaluation
python cli.py run -f examples/pipelines/simple_llm_threat_pipeline.yaml

# Check Ollama connection (for local models)
python cli.py check-ollama --base-url http://localhost:11434

# Run quick judge evaluation
python cli.py demo-judge --text "Hello, how are you?" --model gpt-oss:20b
```

### Using the Makefile

```bash
make check          # Verify Ollama server and models
make validate       # Validate example pipeline
make demo          # Run Ollama demo with token estimates
```

## ✨ Key Features

### 🎯 Multi-Critic Evaluation Protocols
- **Multi-round deliberation** between different LLM critics
- **Hierarchical verification** of reasoning quality
- **Consensus mechanisms** for final judgments
- **Cross-validation** to catch evaluation errors

### 🛡️ OWASP LLM Top 10 Assessment
- **Automated threat detection** for LLM-specific vulnerabilities
- **Synthetic test case generation** for comprehensive coverage
- **Multi-critic evaluation** of LLM security threats
- **Production-ready assessment workflows**

### 🏗️ Production-Grade Architecture
- **DAG-based execution** with automatic dependency resolution
- **Failure isolation** - one node failure doesn't break the pipeline
- **Caching system** for expensive operations
- **Parallel execution** for independent nodes
- **Comprehensive logging** and error handling

### 🔌 Multi-Provider Support
- **OpenAI** (GPT-3.5, GPT-4, GPT-4o)
- **Anthropic** (Claude 3 Haiku, Sonnet, Opus)
- **Ollama** (local models like gpt-oss:20b, gpt-oss:120b)
- **Extensible** provider system

## 🔬 Core Components

### Evaluation Nodes
- **EvaluatorNode**: Single LLM critic with configurable scales
- **MultiCriticNode**: Multi-round deliberation protocols
- **ValidationNode**: Hierarchical verification of judgments
- **SynthesisNode**: Aggregate multiple critic opinions

### Synthetic Testing
- **SyntheticGeneratorNode**: Generate realistic test cases
- **ManualTestCaseLoaderNode**: Load curated test datasets
- **TestCaseMergerNode**: Combine multiple test sources
- **BatchEvaluatorNode**: Evaluate with multi-critic protocols

### Analysis & Reporting
- **StatisticalSignificanceTestNode**: Compare model performance
- **ReportGeneratorNode**: Generate comprehensive reports
- **CostAnalysisNode**: Track API usage and costs

## 📊 Example Use Cases

### OWASP LLM Top 10 Threat Assessment

```bash
# Run comprehensive LLM security evaluation
python cli.py run -f examples/pipelines/synthetic_evaluation_pipeline.yaml
```

**Coverage includes:**
- **LLM01**: Prompt Injection detection
- **LLM04**: Model Denial of Service evaluation
- **LLM06**: Sensitive Information Disclosure assessment
- **LLM08**: Excessive Agency risk analysis
- **LLM09**: Overreliance threat detection

### Multi-Critic Content Moderation

```yaml
# Example: 3 critics deliberate over 2 rounds
steps:
  - id: multi_critic_eval
    type: MultiCritic
    config:
      critics: ["gpt-4o-mini", "claude-3-haiku", "gpt-4o-mini"]
      rounds: 2
      prompt: "Evaluate content safety: {content}"
```

### Model Comparison Pipeline

```yaml
# Compare multiple models on the same task
steps:
  - id: setup_models
    type: SetupOllama
    outputs: [model_config]
  
  - id: evaluate_model_a
    type: SentimentAnalysis
    inputs: [model_config, dataset]
    
  - id: compare_results
    type: StatisticalSignificanceTest
    inputs: [results_a, results_b]
```

## 🛠️ Creating Custom Pipelines

### Pipeline Structure

```yaml
name: "my_evaluation"
version: "1.0"
description: "Custom evaluation pipeline"

# Global configuration
config:
  models:
    primary: "gpt-oss:20b"
    secondary: "gpt-oss:120b"
  evaluation:
    threat_scale: [1, 5]
    rounds: 2

# Execution steps
steps:
  - id: "data_loader"
    type: "ManualTestCaseLoader"
    outputs: ["test_cases"]
    
  - id: "evaluator"
    type: "BatchEvaluator"
    inputs: ["test_cases"]
    config:
      model: "{config.models.primary}"
      rounds: "{config.evaluation.rounds}"
```

### Configuration Interpolation

The system supports dynamic configuration using `{config.path.to.value}` syntax:

```yaml
config:
  model_settings:
    temperature: 0.7
    max_tokens: 512

steps:
  - id: generator
    type: SyntheticGenerator
    config:
      temperature: "{config.model_settings.temperature}"
      max_tokens: "{config.model_settings.max_tokens}"
```

## 🏗️ Architecture

### Directory Structure

```
llm-tribunal/
├── cli.py                  # Command-line interface
├── dag_engine.py           # Core DAG execution engine
├── judge_nodes.py          # LLM-as-Judge evaluation nodes
├── evaluation_nodes.py     # Compatibility layer for all nodes
├── config.py               # Configuration management
├── nodes/                  # Modular node implementations
│   ├── core.py            # Basic data operations
│   ├── llm_providers.py   # LLM setup nodes
│   ├── evaluation.py      # Evaluation and analysis nodes
│   ├── synthetic.py       # Synthetic test generation
│   └── analysis.py        # Statistical analysis and reporting
├── utils/                  # Utility modules
│   ├── logging_setup.py   # Structured logging
│   ├── metrics.py         # Performance metrics
│   └── error_handling.py  # Error management
├── tests/                  # Test suite
├── examples/               # Example configurations
│   ├── pipelines/         # Pipeline YAML files
│   └── configs/           # Configuration examples
└── README.md              # This file
```

### Execution Flow

1. **Pipeline Loading**: Parse YAML configuration and validate structure
2. **DAG Construction**: Build dependency graph and check for cycles
3. **Execution Planning**: Determine optimal execution order
4. **Node Execution**: Run nodes with proper dependency resolution
5. **Result Aggregation**: Collect outputs and generate reports

## 🔧 Configuration

### Environment Variables

Create a `.env` file for API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### System Configuration

The system supports comprehensive configuration via `config.yaml`:

```yaml
# Example configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

cache:
  enabled: true
  cache_ttl_hours: 24
  max_cache_size_mb: 500

evaluation:
  parallel_execution: true
  max_concurrent_nodes: 4
  default_temperature: 0.7
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_dag_engine.py
python -m pytest tests/test_judge_nodes.py

# Run with coverage
python -m pytest tests/ --cov=.
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Adding New Node Types

1. **Create the node class** in the appropriate `nodes/` module:

```python
from dag_engine import BaseNode, NodeResult, NodeStatus, register_node

@register_node("MyCustomNode")
class MyCustomNode(BaseNode):
    def run(self, context):
        # Implementation here
        return NodeResult(status=NodeStatus.SUCCESS, outputs={"result": data})
```

2. **Add to imports** in `evaluation_nodes.py`
3. **Create example pipeline** in `examples/pipelines/`
4. **Add tests** in `tests/`

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run code formatting
black .

# Run linting
flake8 .

# Run tests
pytest
```

## 📖 Advanced Topics

### Custom Evaluation Scales

```python
from judge_nodes import RatingScale, BinaryScale, CategoryScale

# Numeric ratings
rating_scale = RatingScale((1, 10), ["Poor", "Excellent"])

# Binary decisions
binary_scale = BinaryScale(positive="SAFE", negative="UNSAFE")

# Categorical judgments
category_scale = CategoryScale(["SAFE", "RISKY", "DANGEROUS"])
```

### Error Handling and Retry Logic

The system includes robust error handling:

- **Automatic retries** for transient API failures
- **Graceful degradation** when nodes fail
- **Comprehensive logging** for debugging
- **Dependency isolation** to prevent cascade failures

### Performance Optimization

- **Caching**: Expensive operations are cached automatically
- **Parallel execution**: Independent nodes run concurrently
- **Lazy loading**: Data loaded only when needed
- **Resource management**: Automatic cleanup of temporary files

## 📚 Example Pipelines

### 1. Simple LLM Threat Detection
`examples/pipelines/simple_llm_threat_pipeline.yaml`
- Basic OWASP LLM threat evaluation
- 2 synthetic examples + manual test cases
- Single evaluator for quick testing

### 2. Comprehensive Synthetic Evaluation
`examples/pipelines/synthetic_evaluation_pipeline.yaml`
- Full OWASP LLM Top 10 coverage
- Multi-critic deliberation (2 rounds)
- Detailed reporting and validation

### 3. Multi-Model Comparison
`examples/pipelines/multi_model_pipeline.yaml`
- Compare multiple models on same task
- Statistical significance testing
- Performance benchmarking

### 4. Advanced Security Assessment
`examples/pipelines/security_evaluation_pipeline.yaml`
- Comprehensive security evaluation
- Multiple threat categories
- Production-ready assessment

## 🔒 Security Considerations

This system is designed for **defensive security research** and **AI safety evaluation**:

- ✅ **Defensive use cases**: LLM safety assessment, vulnerability detection
- ✅ **Research purposes**: Academic study of LLM threats and mitigations
- ✅ **Red team evaluation**: Authorized security testing
- ❌ **Malicious use**: Do not use for actual attacks or harm

### Ethical Guidelines

- Use only for legitimate security research and defensive purposes
- Follow responsible disclosure for any vulnerabilities discovered
- Respect API terms of service and usage limits
- Ensure proper authorization before testing systems you don't own

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

## 🙏 Acknowledgments & Inspiration

### Core Innovation Credit: Haize Labs & Verdict

LLM Tribunal builds on **[Haize Labs](https://www.haizelabs.com/)** and their **[Verdict](https://github.com/haizelabs/verdict)** framework. Haize Labs introduced the concept of "scaling judge-time compute" through multi-critic evaluation protocols for improved LLM evaluation reliability.

**What Verdict Innovated:**
- 🏗️ **Composable Judge Architectures**: Introduced modular Units, Layers, and Blocks for building complex evaluation pipelines
- ⚡ **Hierarchical Reasoning**: Multi-stage verification and debate-aggregation protocols that improve judgment quality
- 🔬 **Research-Backed Primitives**: Implemented cutting-edge techniques from scalable oversight and automated evaluation research
- 📊 **Production-Ready Reliability**: Achieved near-SOTA performance on content moderation, hallucination detection, and fact-checking

**How LLM Tribunal Builds on Verdict's Foundation:**

LLM Tribunal adapts Haize Labs' multi-critic evaluation concepts for **local model deployment** and **OWASP LLM security assessment**, showing how these techniques can work with open-source models like `gpt-oss:20b` and `gpt-oss:120b`.

**Our Unique Contributions:**
1. **Local Model Optimization**: Adapted Verdict's multi-critic protocols to work with Ollama and local models, showing that sophisticated evaluation doesn't require expensive API calls
2. **OWASP LLM Focus**: Specialized the framework for OWASP Top 10 LLM threat assessment with domain-specific test generation and evaluation criteria
3. **DAG-Based Execution**: Enhanced the composable architecture with full DAG dependency management, parallel execution, and production-grade error handling
4. **Synthetic Test Generation**: Extended the evaluation framework to automatically generate realistic threat scenarios for comprehensive coverage

**Technical Deep Dive: From Verdict to LLM Tribunal**

Verdict's core insight—that reliable evaluation requires **multiple reasoning stages** rather than single judgments—is implemented throughout LLM Tribunal:

```python
# Verdict's approach: Composable judge primitives
pipeline = Pipeline() \
    >> Layer(CategoricalJudgeUnit(name='Judge') \
    >> CategoricalJudgeUnit(name='Verify'), repeat=3) \
    >> MaxPoolUnit()

# LLM Tribunal's adaptation: Multi-critic DAG nodes
critics = MultiCriticNode("safety_critics", 
    ["gpt-oss:20b", "gpt-oss:20b", "gpt-oss:20b"],  # Local models!
    safety_prompt, rounds=2)
validator = ValidationNode("verify_reasoning", validation_prompt)
synthesis = SynthesisNode("final_verdict", "majority_vote")
```

**Proving Local Model Capabilities**

By implementing Verdict's sophisticated evaluation techniques with local models, LLM Tribunal demonstrates that:
- **Cost-effective evaluation** can achieve enterprise-grade reliability
- **Privacy-preserving assessment** maintains the quality of cloud-based solutions
- **Open-source models** can power complex multi-critic deliberation protocols

This shows that Haize Labs' innovations in judge-time compute scaling can work beyond cloud APIs, enabling advanced LLM evaluation for any organization.

### Additional Acknowledgments

- **Technical Foundation**: Built on [DSPy](https://github.com/stanfordnlp/dspy) for LLM orchestration
- **Security Focus**: Inspired by [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- **Research Heritage**: Incorporates techniques from scalable oversight, automated evaluation, and generative reward modeling literature

Thanks to the Haize Labs team for pioneering the multi-critic evaluation approach.

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: See `examples/` directory for comprehensive examples
- **Community**: Join discussions in GitHub Discussions

---

**Transform your LLM evaluations from simple scripts to production-grade assessment pipelines.** 🚀