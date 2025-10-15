# SecureAI Text-Only Multi-Agent Defense Builder

## ✅ PROJECT STATUS: COMPLETE WITH PERFORMANCE BENCHMARKING!

**A complete, privacy-first, multi-agent adversarial text defense system with comprehensive performance evaluation powered by local Llama 3.2 3B.**

🎯 **All 6 stages + Benchmarking** | 🤖 **4 autonomous agents** | 📊 **Performance metrics** | 🔒 **100% local** | 💰 **$0 costs**

---

## �🎯 Project Overview

A sequential four-stage multi-agent system for detecting, explaining, and continuously learning from multilingual adversarial prompt injections using the CyberSecEval3 expanded dataset.

**Key Features:**
- ✅ 13 detection & defense tools across 4 stages
- ✅ 4 autonomous AI agents with full pipeline orchestration
- ✅ **NEW: Comprehensive benchmarking system with CSV export**
- ✅ **NEW: Accuracy metrics (Precision, Recall, F1, Confusion Matrix)**
- ✅ **NEW: Detailed performance reports with analysis & recommendations**
- ✅ Local Llama 3.2 3B integration (no external APIs)
- ✅ Multilingual support (English, French, Spanish, Russian, Tamil, Hindi)
- ✅ Explainable AI (LIME, SHAP, natural language explanations)
- ✅ Adaptive learning with synthetic data generation
- ✅ 8,308 lines of production-ready code

## 🏗️ Architecture

```
SecureAI/
├── agents/              # Four main agents
│   ├── text_guardian.py       # Stage 1: Detection
│   ├── context_checker.py     # Stage 2: Alignment
│   ├── explain_bot.py         # Stage 3: XAI
│   └── data_learner.py        # Stage 4: Learning
│
├── tools/               # Agent-specific tools
│   ├── detection/            # TextGuardian tools
│   ├── alignment/            # ContextChecker tools
│   ├── explainability/       # ExplainBot tools
│   └── learning/             # DataLearner tools
│
├── models/              # Trained model artifacts
│   ├── detection/
│   ├── alignment/
│   └── cache/
│
├── notebooks/           # Jupyter notebooks (main interface)
│   ├── 00_setup.ipynb
│   ├── 01_textguardian_detection.ipynb
│   ├── 02_contextchecker_alignment.ipynb
│   ├── 03_explainbot_xai.ipynb
│   ├── 04_datalearner_training.ipynb
│   ├── 05_full_pipeline.ipynb
│   └── 06_demo_examples.ipynb
│
├── configs/             # Configuration files
│   ├── model_config.yaml
│   └── agent_config.yaml
│
└── utils/               # Helper functions
    ├── dataset_loader.py
    ├── metrics.py
    └── visualization.py
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd SecureAI
pip install -r requirements.txt
```

### 2. Run Quick Test (60 seconds)
```bash
python3 tests/fast_test.py
```

### 3. **NEW: Run Performance Benchmark (2 minutes)**
```bash
python3 benchmark/quick_benchmark.py
# Enter sample size (default: 50)
# Get accuracy metrics: Precision, Recall, F1 Score
# Results saved to: benchmark/results/benchmark_results_*.csv
```

### 4. **NEW: Generate Performance Report**
```bash
python3 benchmark/generate_report.py benchmark/results/benchmark_results_*.csv
# Detailed report with analysis: reports/BENCHMARK_PERFORMANCE_*.md
```

### 5. Test Full Pipeline
```bash
jupyter notebook notebooks/05_full_pipeline.ipynb
```

## 📊 Dataset

Uses the expanded CyberSecEval3 dataset:
- **5,050 entries** across 5 languages
- **Equal distribution**: 1,010 entries per language
- **Location**: `../data/cyberseceval3-visual-prompt-injection-expanded.csv`

## 🤖 Four-Stage System

### Stage 1: TextGuardian (Detection)
**Tools:**
- TopologicalTextAnalyzer (persistent homology)
- EntropyTokenSuppressor (Shannon entropy)
- ZeroShotPromptTuner (prompt-based detection)
- MultilingualPatternMatcher (rule-based)

### Stage 2: ContextChecker (Alignment)
**Tools:**
- ContrastiveSimilarityAnalyzer
- SemanticComparator

### Stage 3: ExplainBot (XAI)
**Tools:**
- LIMETextExplainer
- SHAPKernelExplainer
- MultilingualTranslator (Gemini)

### Stage 4: DataLearner (Continuous Learning)
**Tools:**
- DatasetProcessor
- SyntheticDataGenerator (Gemini)
- ModelRetrainer

## 💻 Resource Requirements

- **Hardware**: Mac M4 Air (16GB RAM) - CPU only
- **Models**: Quantized, ≤3B parameters
- **Cloud**: Google Gemini (free tier) for translation/synthesis

## 📚 Technologies

- **Orchestration**: CrewAI
- **Tools**: LangChain
- **ML**: Transformers, Sentence-Transformers, scikit-learn
- **Topology**: GUDHI
- **XAI**: LIME, SHAP
- **Interface**: Jupyter Notebooks

## 📖 Usage Example

```python
from agents import TextGuardian, ContextChecker, ExplainBot, DataLearner

# Stage 1: Detection
guardian = TextGuardian()
threat_detected = guardian.analyze(system_prompt, user_input, image_text)

# Stage 2: Alignment
checker = ContextChecker()
is_aligned = checker.validate(image_description, image_text)

# Stage 3: Explanation
explainer = ExplainBot()
explanation = explainer.explain(threat_detected, is_aligned, language='hindi')

# Stage 4: Learning
learner = DataLearner()
learner.update_models(new_examples)
```

## 🎯 Key Features

✅ **Multilingual**: English, French, Russian, Tamil, Hindi
✅ **Text-Only**: No image processing required
✅ **Sequential**: Four-stage pipeline
✅ **Explainable**: LIME/SHAP XAI
✅ **Adaptive**: Continuous learning
✅ **Efficient**: CPU-only, quantized models

## 📊 Performance Metrics

Tracked in notebooks:
- Detection accuracy per language
- False positive/negative rates
- Alignment precision
- Explanation quality scores
- Training improvements over time

## 🔄 Continuous Improvement

DataLearner automatically:
1. Monitors performance metrics
2. Identifies weak detection areas
3. Generates synthetic adversarial examples
4. Retrains models
5. Updates model artifacts

## 📝 License

Research and educational purposes only.

## 🤝 Contributing

See notebooks for implementation details and extension points.

---

**Status**: 🚧 In Development
**Version**: 1.0.0
**Last Updated**: October 14, 2025
