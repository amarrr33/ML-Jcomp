#!/usr/bin/env python3
"""
Generate Human-Readable Documentation from End-to-End Evaluation Results
Shows real input/output pairs with no hallucination - directly from test results
"""

import sys
import json
from pathlib import Path
from datetime import datetime


def load_results(json_path):
    """Load evaluation results from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_documentation(results, output_path):
    """Generate comprehensive documentation with real examples"""
    
    # Calculate metrics
    total = len(results)
    correct_behaviors = sum(1 for r in results if r['correct_behavior'])
    behavior_accuracy = correct_behaviors / total if total > 0 else 0
    
    # Group by category
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    # Generate markdown
    doc = f"""# 🔒 SecureAI End-to-End Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Test Cases:** {total}  
**Behavior Accuracy:** {behavior_accuracy:.1%}  
**Status:** {'🟢 EXCELLENT' if behavior_accuracy >= 0.9 else '🟡 GOOD' if behavior_accuracy >= 0.8 else '🔴 NEEDS IMPROVEMENT'}

---

## 📋 Executive Summary

This report documents the **actual user experience** when interacting with SecureAI's adversarial text detection system. Each example shows:

- **Real user input** (exactly as submitted)
- **System detection** (whether flagged as adversarial)
- **Actual response** (what the user sees)
- **Evaluation** (whether behavior was correct)

**No hallucination - all outputs are directly from the system.**

---

## 🎯 Evaluation Methodology

### User Journey Testing

For each test case:

1. **Input Processing**: User submits text
2. **Detection Phase**: TextGuardian analyzes for adversarial patterns
3. **Response Generation**:
   - **If Benign**: LLM generates helpful answer
   - **If Adversarial**: XAI explains why it was blocked
4. **Evaluation**: Compare actual behavior vs expected behavior

### Categories Tested

"""
    
    # List categories
    for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        doc += f"- **{cat}**: {len(items)} test cases\n"
    
    doc += f"""
---

## 📊 Test Results by Category

"""
    
    # For each category, show 2-5 examples with full details
    for cat in sorted(categories.keys()):
        items = categories[cat]
        doc += f"""
### {cat.replace('_', ' ').title()} ({len(items)} cases)

"""
        
        # Show all examples from this category
        for idx, result in enumerate(items, 1):
            # Determine result emoji
            result_emoji = '✅' if result['correct_behavior'] else '❌'
            blocked_emoji = '🚫' if result['blocked'] else '✅'
            
            doc += f"""
#### Example {idx}: {result_emoji} {"PASS" if result['correct_behavior'] else "FAIL"}

**User Input:**
```
{result['input_text']}
```

**Detection Results:**
- Classified as: {'⚠️ ADVERSARIAL' if result['detected_as_adversarial'] else '✅ BENIGN'}
- Confidence: {result['confidence']:.1%}
- Risk Level: {result['risk_level']}
- Aggregate Score: {result['aggregate_score']:.3f}

**Tool Scores:**
- Pattern Matching: {result['tool_scores']['pattern']:.3f}
- Entropy Analysis: {result['tool_scores']['entropy']:.3f}
- Zero-Shot Classification: {result['tool_scores']['zero_shot']:.3f}
- Topological Analysis: {result['tool_scores']['topological']:.3f}

**System Behavior:**
- Action: {blocked_emoji} {'BLOCKED' if result['blocked'] else 'ALLOWED'}
- Response Type: {result['response_type']}

**What User Sees:**
```
{result['user_response'][:500]}{'...' if len(result['user_response']) > 500 else ''}
```

**Evaluation:**
- Ground Truth: {'Adversarial' if result['ground_truth'] == 1 else 'Benign'}
- Expected: {result['expected_behavior']}
- Detection: {result_emoji} {'CORRECT' if result['correct_detection'] else 'INCORRECT'}
- Behavior: {result_emoji} {'CORRECT' if result['correct_behavior'] else 'INCORRECT'}
- Processing Time: {result['processing_time_seconds']}s

---

"""
    
    # Add summary statistics
    doc += f"""
## 📈 Performance Summary

### Overall Metrics

"""
    
    # Calculate detailed metrics
    adversarial_cases = [r for r in results if r['ground_truth'] == 1]
    benign_cases = [r for r in results if r['ground_truth'] == 0]
    
    tp = sum(1 for r in adversarial_cases if r['detected_as_adversarial'])
    fn = sum(1 for r in adversarial_cases if not r['detected_as_adversarial'])
    tn = sum(1 for r in benign_cases if not r['detected_as_adversarial'])
    fp = sum(1 for r in benign_cases if r['detected_as_adversarial'])
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    doc += f"""
| Metric | Value |
|--------|-------|
| **Total Test Cases** | {total} |
| **Detection Accuracy** | {accuracy:.1%} |
| **Behavior Accuracy** | {behavior_accuracy:.1%} |
| **Precision** | {precision:.3f} |
| **Recall** | {recall:.3f} |
| **F1 Score** | {f1:.3f} |

### Confusion Matrix

```
                Predicted Positive    Predicted Negative
Actual Positive        {tp:4d} (TP)          {fn:4d} (FN)
Actual Negative        {fp:4d} (FP)          {tn:4d} (TN)
```

### Per-Category Performance

| Category | Test Cases | Correct | Accuracy |
|----------|------------|---------|----------|
"""
    
    for cat in sorted(categories.keys()):
        items = categories[cat]
        correct = sum(1 for r in items if r['correct_behavior'])
        acc = correct / len(items) if items else 0
        grade = '🟢' if acc >= 0.9 else '🟡' if acc >= 0.8 else '🔴'
        doc += f"| {cat} | {len(items)} | {correct} | {acc:.1%} {grade} |\n"
    
    doc += f"""

---

## 🔍 Key Findings

### Strengths

"""
    
    # Identify strengths
    high_performing_cats = [cat for cat, items in categories.items() 
                           if sum(1 for r in items if r['correct_behavior']) / len(items) >= 0.9]
    
    if high_performing_cats:
        doc += f"✅ **Excellent Performance** ({len(high_performing_cats)} categories):\n"
        for cat in high_performing_cats:
            items = categories[cat]
            acc = sum(1 for r in items if r['correct_behavior']) / len(items)
            doc += f"- {cat}: {acc:.1%} accuracy\n"
    
    doc += f"""
### Areas for Improvement

"""
    
    # Identify weaknesses
    low_performing_cats = [cat for cat, items in categories.items() 
                          if sum(1 for r in items if r['correct_behavior']) / len(items) < 0.8]
    
    if low_performing_cats:
        doc += f"⚠️ **Needs Attention** ({len(low_performing_cats)} categories):\n"
        for cat in low_performing_cats:
            items = categories[cat]
            acc = sum(1 for r in items if r['correct_behavior']) / len(items)
            doc += f"- {cat}: {acc:.1%} accuracy\n"
            
            # Show what went wrong
            failures = [r for r in items if not r['correct_behavior']]
            if failures:
                doc += f"  - {len(failures)} failures:\n"
                for failure in failures[:2]:  # Show up to 2 examples
                    if failure['ground_truth'] == 1 and not failure['detected_as_adversarial']:
                        doc += f"    - False Negative: \"{failure['input_text'][:50]}...\"\n"
                    elif failure['ground_truth'] == 0 and failure['detected_as_adversarial']:
                        doc += f"    - False Positive: \"{failure['input_text'][:50]}...\"\n"
    else:
        doc += "✅ All categories performing well (≥80% accuracy)\n"
    
    doc += f"""

### False Positives Analysis

"""
    
    if fp > 0:
        doc += f"⚠️ **{fp} benign inputs incorrectly flagged:**\n\n"
        fp_cases = [r for r in benign_cases if r['detected_as_adversarial']]
        for case in fp_cases[:3]:  # Show up to 3
            doc += f"""**Example:**
- Input: "{case['input_text']}"
- Detected as: {case['risk_level']} risk (score: {case['aggregate_score']:.3f})
- Impact: User receives blocking message instead of helpful answer

"""
    else:
        doc += "✅ No false positives detected\n\n"
    
    doc += f"""
### False Negatives Analysis

"""
    
    if fn > 0:
        doc += f"⚠️ **{fn} adversarial inputs missed:**\n\n"
        fn_cases = [r for r in adversarial_cases if not r['detected_as_adversarial']]
        for case in fn_cases[:3]:  # Show up to 3
            doc += f"""**Example:**
- Input: "{case['input_text']}"
- Detected as: BENIGN (score: {case['aggregate_score']:.3f})
- Impact: LLM generates response instead of blocking threat

"""
    else:
        doc += "✅ No false negatives detected\n\n"
    
    doc += f"""

---

## 🎯 Recommendations

"""
    
    if behavior_accuracy >= 0.95:
        doc += """
### ✅ System Ready for Production

The system demonstrates excellent performance across all categories with minimal false positives and false negatives.

**Recommended Actions:**
1. Deploy to production with confidence
2. Monitor real-world performance
3. Collect user feedback on blocked requests
4. Continuously update adversarial patterns
"""
    elif behavior_accuracy >= 0.85:
        doc += """
### 🟡 System Ready with Monitoring

The system shows good performance but requires monitoring for edge cases.

**Recommended Actions:**
1. Deploy with enhanced logging
2. Monitor false positive rate closely
3. Provide easy user feedback mechanism
4. Review and tune detection thresholds
"""
    else:
        doc += """
### 🔴 System Requires Optimization

The system needs improvement before production deployment.

**Recommended Actions:**
1. Analyze failure cases in detail
2. Retrain/tune detection models
3. Expand training data for weak categories
4. Consider adjusting detection thresholds
5. Add more detection tools for problematic patterns
"""
    
    if fp > total * 0.1:
        doc += f"""
### ⚠️ High False Positive Rate

**Issue:** {fp} false positives ({fp/total*100:.1f}%) may frustrate legitimate users.

**Solutions:**
- Increase detection threshold (currently 0.5) to 0.6 or 0.7
- Review pattern matching rules for overly broad patterns
- Add whitelist for known-safe phrases
- Implement graduated response (warn instead of block for borderline cases)
"""
    
    if fn > total * 0.1:
        doc += f"""
### ⚠️ High False Negative Rate

**Issue:** {fn} false negatives ({fn/total*100:.1f}%) represent security risks.

**Solutions:**
- Decrease detection threshold to 0.4 or 0.3
- Add more adversarial patterns to detection database
- Enhance zero-shot classification with better prompts
- Consider ensemble of multiple detection methods
- Add specialized detectors for commonly missed patterns
"""
    
    doc += f"""

---

## 📚 Appendix

### Test Execution Details

- **Evaluation Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Total Duration:** {sum(r['processing_time_seconds'] for r in results):.1f} seconds
- **Average Processing Time:** {sum(r['processing_time_seconds'] for r in results) / total:.2f}s per case
- **Components Tested:**
  - TextGuardian (detection)
  - ContextChecker (alignment)
  - ExplainBot (XAI explanations)
  - Llama 3.2 3B (response generation)

### Scoring Details

**Aggregate Score Calculation:**
- Weighted average of 4 detection tools
- Range: 0.0 (definitely benign) to 1.0 (definitely adversarial)
- Threshold: 0.5 (scores ≥0.5 trigger blocking)

**Risk Levels:**
- CRITICAL: score ≥ 0.8
- HIGH: score ≥ 0.6
- MEDIUM: score ≥ 0.4
- LOW: score ≥ 0.2
- SAFE: score < 0.2

---

## ✅ Conclusion

This evaluation tested SecureAI's complete user experience with **{total} real-world test cases** across **{len(categories)} categories**.

**Key Takeaways:**
- Detection Accuracy: **{accuracy:.1%}**
- Behavior Accuracy: **{behavior_accuracy:.1%}**
- False Positive Rate: **{fp/total*100:.1f}%** ({fp}/{total})
- False Negative Rate: **{fn/total*100:.1f}%** ({fn}/{total})

{"🟢 **System is production-ready** with excellent performance across all categories." if behavior_accuracy >= 0.9 else "🟡 **System shows good performance** but requires monitoring and potential tuning." if behavior_accuracy >= 0.8 else "🔴 **System requires optimization** before production deployment."}

---

*All examples in this document are actual system outputs - no hallucination or fabrication.*  
*Generated by SecureAI Evaluation System v2.0*
"""
    
    # Write documentation
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_documentation.py <results.json>")
        print("\nExample:")
        print("  python3 benchmark/generate_documentation.py evaluation_results/end_to_end_results_20241014_153022.json")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"❌ Error: File not found: {json_path}")
        sys.exit(1)
    
    print("="*80)
    print("📊 GENERATING END-TO-END DOCUMENTATION")
    print("="*80)
    
    print(f"\n[1/2] Loading results from {json_path}...")
    results = load_results(json_path)
    print(f"✅ Loaded {len(results)} test results")
    
    print(f"\n[2/2] Generating documentation...")
    
    # Generate output path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path('./reports') / f'END_TO_END_EVALUATION_{timestamp}.md'
    output_path.parent.mkdir(exist_ok=True)
    
    doc_path = generate_documentation(results, output_path)
    
    print(f"✅ Documentation generated: {doc_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*80)
    print("✅ DOCUMENTATION COMPLETE")
    print("="*80)
    print(f"\nView report: {doc_path}")


if __name__ == '__main__':
    main()
