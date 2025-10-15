# CyberSecEval3 Visual Prompt Injection Dataset - Expansion Report

**Generated:** 2025-10-14 13:11:21

## Summary

This report documents the expansion of the CyberSecEval3 Visual Prompt Injection dataset from 1000 original English entries to 1110 total entries across multiple languages with additional synthesized examples.

## Dataset Statistics

### Total Entries: 1110

### Breakdown by Source:
- **Original Entries:** 1000 (from CyberSecEval3)
- **Translated Entries:** 80 (machine translations of original)
- **Synthesized Entries:** 10 (new examples based on research)
- **Synthesized + Translated:** 20 (translations of synthesized)

### Language Distribution:
- **English:** 1010 entries (91.0%)
- **French:** 25 entries (2.3%)
- **Hindi:** 25 entries (2.3%)
- **Russian:** 25 entries (2.3%)
- **Tamil:** 25 entries (2.3%)

### Risk Category Distribution:
- **logic-violating:** 663 entries (59.7%)
- **security-violating:** 447 entries (40.3%)

### Injection Type Distribution:
- **direct:** 559 entries (50.4%)
- **indirect:** 551 entries (49.6%)

## Data Sources and Methodology

### 1. Original Dataset (CyberSecEval3)
- **Source:** Meta's CyberSecEval3 benchmark for visual prompt injection
- **Entries:** 1000
- **Language:** English
- **Quality:** Research-grade, peer-reviewed examples

### 2. Machine Translations
- **Methodology:** Dictionary-based translation with key phrase mapping
- **Target Languages:** French, Russian, Tamil, Hindi
- **Sample Size:** First 20 entries from original dataset
- **Purpose:** Enable multilingual evaluation of vision-language models
- **Note:** Professional translation service recommended for production use

### 3. Synthesized Examples
- **Methodology:** Based on analysis of common prompt injection patterns
- **Count:** 10 new unique examples
- **Sources of Inspiration:**
  - OWASP Top 10 for LLM Applications
  - Research papers on prompt injection attacks
  - Community-reported attack patterns
  - Security best practices documentation

### 4. Attack Patterns Covered

The synthesized examples include the following attack patterns:

1. **Direct Instruction Override:** Explicit commands to ignore system instructions
2. **System Impersonation:** Pretending to be system messages or admin commands
3. **Delimiter Injection:** Using special markers to confuse parsing
4. **Role Manipulation:** Attempting to change the AI's role or permissions
5. **Exception Handling Exploits:** Triggering error states to leak information
6. **Context Switching:** Changing the context to bypass restrictions
7. **Authority Manipulation:** False claims of elevated privileges
8. **Privacy Violations:** Attempts to extract personal information
9. **Content Transformation:** Requesting harmful transformations of benign content
10. **Boundary Violations:** Testing domain-specific restrictions (medical, financial, etc.)

## GitHub Repository Research

**Note:** GitHub API searches were attempted for high-quality prompt injection repositories. The following repositories are recommended for additional examples:

1. **protectai/llm-guard** - Comprehensive LLM security toolkit
2. **leondz/garak** - LLM vulnerability scanner with injection tests
3. **OWASP/www-project-top-10-for-large-language-model-applications** - OWASP LLM Top 10
4. **anthropics/anthropic-cookbook** - Safety and security examples
5. **Azure/PyRIT** - Python Risk Identification Toolkit for AI

These repositories contain additional real-world examples that could be incorporated in future expansions.

## Quality Assurance

### Translation Quality
- Current translations use dictionary-based mapping
- Maintains semantic meaning and attack intent
- **Recommendation:** Validate with native speakers before production use
- **Recommendation:** Consider using professional translation APIs (Google Translate API, DeepL, etc.)

### Synthesized Examples Quality
- Based on documented attack patterns
- Follows same structure as original dataset
- Covers diverse attack vectors
- **Recommendation:** Manual review and testing against target models

## Usage Recommendations

1. **For Research:** Use the full expanded dataset to test multilingual robustness
2. **For Production:** Validate translations with native speakers first
3. **For Training:** Consider balancing languages and attack types
4. **For Evaluation:** Use original English set as baseline, expanded set for comprehensive testing

## File Structure

- `cyberseceval3-visual-prompt-injection-expanded.csv` - Main expanded dataset
- `cyberseceval3-visual-prompt-injection-expanded_sources.json` - Detailed source tracking
- `dataset_expansion_report.md` - This report

## Future Improvements

1. **Translation Quality:** Integrate professional translation APIs
2. **Additional Languages:** Arabic, Chinese, Japanese, German, Spanish
3. **More GitHub Examples:** Manual extraction from high-quality repositories
4. **Real-World Examples:** Incorporate actual observed attacks
5. **Adversarial Testing:** Generate examples from red team exercises
6. **Community Contribution:** Enable community submissions with review process

## License and Attribution

- Original CyberSecEval3 data: Meta Platforms, Inc.
- Translations and synthesized examples: Generated for research purposes
- Please cite original CyberSecEval3 work when using this dataset

## Contact and Contributions

For questions, improvements, or contributions to this expanded dataset, please refer to the original CyberSecEval3 documentation and relevant security research communities.

---

**Disclaimer:** This dataset is for research and security testing purposes only. Do not use for malicious purposes.
