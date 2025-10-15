# CyberSecEval3 Visual Prompt Injection Dataset - Organized Structure

## 📁 Project Structure

```
Dataset/
├── data/                          # All dataset files
│   ├── cyberseceval3-visual-prompt-injection.csv
│   ├── cyberseceval3-visual-prompt-injection-expanded.csv
│   └── cyberseceval3-visual-prompt-injection-expanded_sources.json
│
├── scripts/                       # Python utilities
│   ├── expand_dataset.py         # Generate expanded dataset
│   └── view_samples.py           # View and analyze samples
│
├── documentation/                 # All documentation
│   ├── README.md                 # Main usage guide
│   ├── DATASET_SUMMARY.md        # Quick reference
│   ├── DATA_SOURCE_REPORT.md     # Complete source tracking
│   └── PROJECT_SUMMARY.md        # Project overview
│
├── reports/                       # Generated reports
│   └── dataset_expansion_report.md
│
└── STRUCTURE.md                   # This file
```

## 🚀 Quick Start

### 1. Generate Expanded Dataset (with equal language proportions)
```bash
cd Dataset
python3 scripts/expand_dataset.py
```

This will:
- Translate ALL 1,000 original entries to French, Russian, Tamil, Hindi
- Create 10 synthesized examples in English
- Translate all synthesized examples to all 4 languages
- Result: **5,050 total entries** with equal proportions across languages

### 2. View Samples
```bash
python3 scripts/view_samples.py
```

### 3. Access Data
```python
import csv
import json

# Load expanded dataset
with open('data/cyberseceval3-visual-prompt-injection-expanded.csv', 'r', encoding='utf-8') as f:
    data = list(csv.DictReader(f))

# Load source tracking
with open('data/cyberseceval3-visual-prompt-injection-expanded_sources.json', 'r') as f:
    sources = json.load(f)

print(f"Total entries: {len(data)}")
```

## 📊 Expected Results (After Running expand_dataset.py)

### Language Distribution (Equal Proportions)
- 🇬🇧 **English**: 1,010 entries (20%)
- 🇫🇷 **French**: 1,010 entries (20%)
- 🇷🇺 **Russian**: 1,010 entries (20%)
- 🇮🇳 **Tamil**: 1,010 entries (20%)
- 🇮🇳 **Hindi**: 1,010 entries (20%)

**Total: 5,050 entries**

### Breakdown
- Original (English): 1,000
- Synthesized (English): 10
- French translations: 1,010 (1,000 original + 10 synthesized)
- Russian translations: 1,010 (1,000 original + 10 synthesized)
- Tamil translations: 1,010 (1,000 original + 10 synthesized)
- Hindi translations: 1,010 (1,000 original + 10 synthesized)

## 📚 Documentation

- **README.md**: Comprehensive usage guide with code examples
- **DATASET_SUMMARY.md**: Quick reference with statistics
- **DATA_SOURCE_REPORT.md**: Complete audit trail of all sources
- **PROJECT_SUMMARY.md**: Project achievements and overview
- **dataset_expansion_report.md**: Technical methodology (auto-generated)

## 🔧 Modify & Extend

### Add More Languages
Edit `scripts/expand_dataset.py`:
```python
languages = ['french', 'russian', 'tamil', 'hindi', 'spanish', 'arabic']
```

### Add More Synthesized Examples
Edit `SYNTHESIZED_EXAMPLES` list in `scripts/expand_dataset.py`

### Change Sample Size
Already set to translate ALL entries for equal proportions!

## 📦 File Sizes (Estimated After Full Expansion)

- Original CSV: 356 KB
- Expanded CSV: ~2.0 MB (5,050 entries)
- Sources JSON: ~1.0 MB (complete metadata)

## ✅ Current Status

- [x] Original dataset preserved in `data/`
- [x] Scripts organized in `scripts/`
- [x] Documentation in `documentation/`
- [x] Reports in `reports/`
- [x] Equal language proportions configured
- [ ] Run `python3 scripts/expand_dataset.py` to generate full dataset

## 🎯 Next Steps

1. **Generate Full Dataset**:
   ```bash
   python3 scripts/expand_dataset.py
   ```
   
2. **Verify Results**:
   ```bash
   python3 scripts/view_samples.py
   ```

3. **Start Working**: All files organized and ready for continuation!

---

**Note**: The expansion script now translates ALL entries to ALL languages for equal proportions across the dataset.
