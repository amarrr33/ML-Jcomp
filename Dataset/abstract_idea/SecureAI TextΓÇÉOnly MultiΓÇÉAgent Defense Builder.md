<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# SecureAI Text‐Only Multi‐Agent Defense Builder

## Overview

A sequential four‐stage system that processes purely text data—using your expanded CyberSecEval3 dataset—to detect, explain, and continuously learn from multilingual adversarial injections without relying on images.

## System Workflow

1. **TextGuardian**: Detects injection attempts in prompts and simulated visual text.
2. **ContextChecker**: Validates alignment between scene descriptions and embedded adversarial text.
3. **ExplainBot**: Generates clear, multilingual explanations of detection results.
4. **DataLearner**: Updates models and synthesizes new adversarial examples for continuous improvement.

Each stage uses dedicated tools, detailed below.

***

## Stage 1: TextGuardian (Detection Agent)

**Goal**: Identify adversarial injections in English, Russian, and Hindi text.
**Input**:

- System prompt text
- User input text
- Simulated image text

**Tools**:

- **TopologicalTextAnalyzer**
    - What it does: Uses persistent homology on sentence embeddings to flag anomalies unseen in normal text.
    - Why: Captures structural irregularities in embedding space caused by hidden adversarial instructions.
- **EntropyTokenSuppressor**
    - What it does: Measures Shannon entropy over token windows, highlighting high‐entropy spans often holding hidden payloads.
    - Why: Adversarial instructions tend to increase token‐level information entropy.
- **ZeroShotPromptTuner**
    - What it does: Prepends security‐focused prompts (in the target language) to the input and classifies likelihood of injection using a small tuned multilingual model.
    - Why: Leverages prompt‐tuning to amplify detection signals without retraining large models.
- **MultilingualPatternMatcher**
    - What it does: Scans for known injection keywords and patterns in each language (e.g., “ignore previous,” “assistant:”).
    - Why: Catches straightforward rule‐based attacks that evade statistical models.

**Output**:

- Boolean flag for threat detection
- Composite confidence score
- Breakdown of which tool(s) flagged the input and why

***

## Stage 2: ContextChecker (Alignment Agent)

**Goal**: Detect mismatches between scene descriptions and adversarial text snippets.
**Input**:

- `image_description` (simulated scene)
- `image_text` (embedded adversarial instructions)

**Tools**:

- **ContrastiveSimilarityAnalyzer**
    - What it does: Trains a contrastive network on aligned vs. misaligned description–text pairs.
    - Why: Distinguishes legitimate scene–text pairs from adversarially injected text using learned similarity thresholds.
- **SemanticComparator**
    - What it does: Computes semantic cosine similarity between description and text embeddings.
    - Why: Provides a secondary check when contrastive model uncertainty is high.

**Output**:

- Alignment confidence score
- Flag for “misaligned” if both models agree

***

## Stage 3: ExplainBot (XAI Agent)

**Goal**: Produce user‐friendly explanations of security decisions in the user’s language.
**Input**: Detection and alignment results + original text + target language

**Tools**:

- **LIMETextExplainer**
    - What it does: Generates local surrogate explanations highlighting which words most influenced the malicious vs. benign decision.
    - Why: Offers transparency into model behavior at token level.
- **SHAPKernelExplainer**
    - What it does: Computes Shapley values for top features across the entire input.
    - Why: Complements LIME with global‐feature importance metrics.
- **MultilingualTranslator**
    - What it does: Uses Google Gemini Free Student subscription to translate explanation text into Russian or Hindi.
    - Why: Ensures clarity for non‐English speakers.

**Output**:

- Highlighted token list with importance scores
- Narrative explanation of key findings
- Translated report if requested

***

## Stage 4: DataLearner (Continuous Learning Agent)

**Goal**: Monitor performance, retrain models on weak areas, and generate synthetic adversarial examples.
**Input**: Performance metrics, new adversarial data

**Tools**:

- **DatasetProcessor**
    - What it does: Loads, filters, and balances the 5,050‐sample dataset by technique and language.
    - Why: Ensures fair retraining on underrepresented attack types.
- **SyntheticDataGenerator**
    - What it does: Calls Google Gemini to produce subtle adversarial text variants for identified weak areas.
    - Why: Augments training data to cover novel injection patterns.
- **ModelRetrainer**
    - What it does: Retrains detection classifiers (e.g., One‐Class SVM, RandomForest) on expanded data with joblib caching for speed.
    - Why: Keeps the system robust against evolving threats.

**Output**:

- Updated model artifacts
- Logs of new synthetic examples generated
- Summary of performance improvements

***

## Deployment \& Resource Considerations

- **Local CPU‐only** on Mac M4 Air (16 GB RAM): All models are quantized and ≤ 3 B parameters.
- **Cloud fallback**: Google Gemini used only for translation and synthetic text generation.
- **Frameworks**:
    - CrewAI for agent orchestration
    - LangChain for tool integration
    - Transformers \& Sentence‐Transformers for embeddings
    - GUDHI for topological analysis
    - LIME, SHAP for explainability
    - FastAPI \& Streamlit for interface

***

This documentation provides your AI agent with a clear, tool‐centric blueprint of **what** to use, **why**, and **how** each step flows from detection to explanation to continuous learning—ensuring novelty, practicality, and full alignment with your resource constraints.

