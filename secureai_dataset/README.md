# SecureAI Dataset generator

This folder contains a minimal, safe, and reproducible dataset generator for the SecureAI dataset described in `prompt.md`.

Key points:
- The generator creates a directory tree, synthetic benign images, and synthetic "scaling attack" samples for testing and evaluation. For safety the generator uses placeholder/redacted attack payloads (no real malicious commands are embedded).
- Use the scripts to produce a dataset skeleton and to extend them for integration with Anamorpher/CyberSecEval3 when you are ready.

Quick start (macOS / zsh):
```bash
python3 -m pip install -r secureai_dataset/requirements.txt
python3 -m secureai_dataset.setup_complete_dataset --out secureai_dataset/output --quick --count 5
```

Files:
- `setup_complete_dataset.py` - main CLI to build the dataset skeleton and run generators
- `anamorpher_integration.py` - functions to simulate safe scaling-attack-style samples
- `cyberseceval3_processor.py` - downloader/processor stub for CyberSecEval3 (placeholder)
- `benign_collector.py` - helper to collect/generate benign images (placeholder + synthetic generator)
- `evaluation_framework.py` - small evaluation stubs and metadata helpers
- `tests/test_quick_generate.py` - quick smoke test runner

Notes:
- This is a conservative, initial implementation intended for local testing and extension. It does not contain real exploit payloads; instead it stores labels and redacted placeholders in metadata to support safe research and detector training.
