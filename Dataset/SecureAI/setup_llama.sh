#!/bin/bash
# Setup Ollama and Llama 3.2 3B q4_0 for SecureAI

echo "=================================================="
echo "SecureAI - Ollama & Llama 3.2 Setup"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "❌ Ollama is not installed"
    echo ""
    echo "To install Ollama:"
    echo "  Visit: https://ollama.ai"
    echo "  Or run: curl https://ollama.ai/install.sh | sh"
    echo ""
    exit 1
fi

echo "✓ Ollama is installed"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "❌ Ollama is not running"
    echo ""
    echo "Start Ollama with:"
    echo "  ollama serve"
    echo ""
    echo "Then run this script again."
    echo ""
    exit 1
fi

echo "✓ Ollama is running"

# Pull Llama 3.2 3B q4_0 model
echo ""
echo "Pulling Llama 3.2 3B (q4_0 quantized)..."
echo "This may take a few minutes..."
echo ""

ollama pull llama3.2:3b-instruct-q4_0

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Model installed: llama3.2:3b-instruct-q4_0"
    echo "Size: ~2GB (quantized for efficient inference)"
    echo ""
    echo "Test the model:"
    echo "  ollama run llama3.2:3b-instruct-q4_0 \"Hello!\""
    echo ""
    echo "Or test in Python:"
    echo "  python -c 'from utils.llama_client import LlamaClient; c = LlamaClient(); print(c.generate(\"Test\"))'"
    echo ""
else
    echo ""
    echo "❌ Failed to pull model"
    echo ""
    exit 1
fi
