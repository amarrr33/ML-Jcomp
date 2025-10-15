# Ollama & Llama 3.2 3B Setup Guide

## Installation Instructions

### For macOS (Your Current System)

1. **Download Ollama for macOS:**
   - Visit: https://ollama.com/download
   - Download the macOS application
   - Or use Homebrew: `brew install ollama`

2. **Install and Start Ollama:**
   ```bash
   # If downloaded from website, open the .dmg file and drag to Applications
   # Then start Ollama from Applications or menu bar
   
   # If using Homebrew:
   brew install ollama
   ```

3. **Verify Ollama is Running:**
   ```bash
   # Ollama should start automatically and run in the background
   # Check if it's running:
   curl http://localhost:11434/api/tags
   ```

4. **Pull Llama 3.2 3B Model (Quantized):**
   ```bash
   ollama pull llama3.2:3b-instruct-q4_0
   ```
   
   This will download ~2GB of the quantized model optimized for your Mac M4.

5. **Test the Model:**
   ```bash
   # Test with Ollama CLI
   ollama run llama3.2:3b-instruct-q4_0 "Hello, introduce yourself!"
   
   # Test with Python
   cd SecureAI
   python -c "from utils.llama_client import LlamaClient; c = LlamaClient(); print(c.generate('Test'))"
   ```

---

## Model Information

**Model:** Llama 3.2 3B Instruct (q4_0 quantized)
- **Size:** ~2GB (quantized from 3B parameters)
- **Quantization:** q4_0 (4-bit quantization)
- **Hardware:** Optimized for Mac M4 Air CPU
- **Memory:** Uses ~3-4GB RAM during inference
- **Speed:** ~10-20 tokens/second on M4 Air

**Why This Model:**
- ✅ Small enough to run efficiently on M4 Air (16GB RAM)
- ✅ Fast inference on CPU
- ✅ Good quality for security analysis tasks
- ✅ Supports long contexts (up to 128k tokens)
- ✅ Instruction-tuned for task completion

---

## Usage in SecureAI

Once Ollama is running, the LlamaClient will automatically use it for:

1. **Adversarial Explanations** (replaces Gemini)
2. **Translation** (replaces Gemini)
3. **Synthetic Data Generation** (replaces Gemini)
4. **Defense Recommendations** (new capability)
5. **Error Pattern Analysis** (new capability)

All agents (TextGuardian, ContextChecker, ExplainBot, DataLearner) will use Llama 3.2 via the unified LlamaClient.

---

## Configuration

The model is configured in `configs/model_config.yaml`:

```yaml
llm:
  provider: "ollama"
  model: "llama3.2:3b-instruct-q4_0"
  temperature: 0.3
  max_tokens: 512
  top_p: 0.8
  base_url: "http://localhost:11434"
```

---

## Troubleshooting

### Ollama Not Running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
# On Mac: Open Ollama from Applications or run:
ollama serve
```

### Model Not Found
```bash
# List installed models
ollama list

# If llama3.2:3b-instruct-q4_0 is not listed, pull it:
ollama pull llama3.2:3b-instruct-q4_0
```

### Connection Errors
- Ensure Ollama is running in the background
- Check firewall settings (Ollama uses port 11434)
- Restart Ollama if needed

### Slow Inference
- Normal: 10-20 tokens/second on M4 Air
- Close other applications to free up resources
- Consider using smaller context windows

---

## Alternative Models

If you need different performance/quality tradeoffs:

```bash
# Smaller, faster (1B parameters, ~700MB)
ollama pull llama3.2:1b-instruct-q4_0

# Larger, better quality (7B parameters, ~4GB)
# Note: May be slower on M4 Air
ollama pull llama3.2:7b-instruct-q4_0
```

Update `model` in `configs/model_config.yaml` to use alternative models.

---

## Next Steps

After Ollama is installed and running:

1. Test the LlamaClient: `python utils/llama_client.py`
2. Run Stage 6 tests
3. Benchmark performance
4. Generate documentation

---

**For immediate setup, please:**
1. Visit https://ollama.com/download
2. Download and install Ollama for macOS
3. Run the setup script: `./setup_llama.sh`
