#!/bin/bash

# LLM Tribunal Safety Gateway Startup Script

set -e

echo "🚀 Starting LLM Tribunal Safety Gateway..."

# Check if Python dependencies are installed
if ! python -c "import fastapi, uvicorn, httpx" 2>/dev/null; then
    echo "📦 Installing gateway dependencies..."
    pip install fastapi "uvicorn[standard]" httpx pydantic
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "⚠️  Warning: Ollama not detected at http://localhost:11434"
    echo "   The gateway will work but may fail to forward requests to local models."
    echo "   To fix this:"
    echo "   1. Install Ollama: https://ollama.ai"
    echo "   2. Start Ollama: ollama serve"
    echo "   3. Pull a model: ollama pull gpt-oss:20b"
    echo ""
fi

# Default configuration
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
POLICY=${POLICY:-gateway/policies/default_safety_policy.yaml}

echo "🔧 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT" 
echo "   Policy: $POLICY"
echo ""

# Start the gateway
echo "🌟 Starting gateway server..."
python gateway_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy "$POLICY" \
    "$@"
