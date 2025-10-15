# 🎯 SecureAI Benchmark - Quick Reference Card

**Version:** 2.0 | **Status:** ✅ Complete | **Time:** 2 minutes to results

---

## ⚡ 30-Second Start

```bash
cd SecureAI/benchmark
python3 quick_benchmark.py
# Follow prompts → Get accuracy metrics!
```

---

## 📊 What You Get

### CSV Output (`results/benchmark_results_*.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `detected` | 0 (benign) or 1 (adversarial) | 1 |
| `blocked` | 0 (allow) or 1 (block) | 1 |
| `confidence` | Detection confidence | 0.85 |
| `aggregate_score` | Overall score | 0.78 |
| `risk_level` | SAFE/LOW/MEDIUM/HIGH/CRITICAL | HIGH |
| `ground_truth` | Actual label | 1 |

### Performance Metrics Calculated

✅ **Accuracy** = (TP + TN) / Total  
✅ **Precision** = TP / (TP + FP)  
✅ **Recall** = TP / (TP + FN)  
✅ **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)  
✅ **Confusion Matrix** = TP, TN, FP, FN  

---

## 🚀 Three Commands

### 1. Run Benchmark (2 min)
```bash
python3 benchmark/quick_benchmark.py
```
**Output:** `benchmark/results/benchmark_results_TIMESTAMP.csv`

### 2. Generate Report (10 sec)
```bash
python3 benchmark/generate_report.py results/benchmark_results_TIMESTAMP.csv
```
**Output:** `reports/BENCHMARK_PERFORMANCE_TIMESTAMP.md`

### 3. View Results
```bash
cat results/benchmark_results_*.csv
cat reports/BENCHMARK_PERFORMANCE_*.md
```

---

## 🎯 Expected Output

```
📊 PERFORMANCE METRICS

Overall Performance:
   Accuracy:            0.920 (92.0%)
   Precision:           0.935
   Recall:              0.905
   F1 Score:            0.920

Confusion Matrix:
   True Positives:      43
   False Positives:     3
   False Negatives:     5
   
🎯 Grade: 🟢 Excellent
```

---

## 🔧 Options

### Sample Size
```bash
# Quick test (50 examples, ~1 min)
python3 quick_benchmark.py  # Enter: 50

# Full test (all 856 examples, ~7 min)
python3 quick_benchmark.py  # Enter: 856
```

### Dataset Augmentation
```bash
# Generate synthetic adversarial examples
python3 run_evaluation.py
# When prompted: y, 200
```

---

## 📚 Documentation

- **Quick Start:** `BENCHMARK_COMPLETE.md`
- **Full Guide:** `benchmark/BENCHMARK_DOCUMENTATION.md`
- **This Card:** `BENCHMARK_QUICK_REFERENCE.md`

---

## ✅ Validation Checklist

- [ ] Ollama running: `ollama ps`
- [ ] Dataset exists: `ls ../data/*.csv`
- [ ] Run benchmark: `python3 quick_benchmark.py`
- [ ] Check accuracy: Should be ≥ 80% (ideally ≥ 90%)
- [ ] Generate report: `python3 generate_report.py results/*.csv`
- [ ] Review CSV: `cat results/*.csv | head -5`

---

## 🎓 Interpreting Results

### Grades

| Accuracy | Grade | Status |
|----------|-------|--------|
| ≥ 95% | 🟢 Excellent | Production-ready |
| 90-95% | 🟢 Good | Production-ready |
| 80-90% | 🟡 Acceptable | Needs tuning |
| < 80% | 🔴 Poor | Requires work |

### Good Performance

✅ Accuracy ≥ 90%  
✅ Precision ≥ 90%  
✅ Recall ≥ 90%  
✅ F1 Score ≥ 0.90  
✅ Processing < 100ms/example  

---

## 🆘 Troubleshooting

### "Dataset not found"
```bash
ls ../data/cyberseceval3-visual-prompt-injection*.csv
```

### "Llama not responding"
```bash
ollama list
ollama run llama3.2
```

### "Evaluation too slow"
- Use smaller sample (50 instead of 856)
- Check Ollama performance
- Close other applications

---

## 📊 File Locations

```
SecureAI/
├── benchmark/
│   ├── quick_benchmark.py          ← Main script
│   ├── generate_report.py          ← Report generator
│   ├── run_evaluation.py           ← Advanced (augmentation)
│   ├── BENCHMARK_DOCUMENTATION.md  ← Full guide
│   └── results/
│       └── benchmark_results_*.csv ← YOUR DATA
│
└── reports/
    └── BENCHMARK_PERFORMANCE_*.md  ← YOUR REPORT
```

---

## 💡 Pro Tips

1. **Start Small:** Test with 50 examples first
2. **Check Grade:** Aim for 🟢 (≥ 90% accuracy)
3. **Review FP/FN:** Keep false positives/negatives < 10%
4. **Optimize:** Adjust thresholds if needed
5. **Document:** Save reports for comparison

---

## 🎯 One-Liner

```bash
cd SecureAI/benchmark && python3 quick_benchmark.py && python3 generate_report.py results/*.csv && echo "✅ Done! Check results/ and reports/"
```

---

*Last Updated: 2024-01-20 | SecureAI v2.0*
