#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo " Starting Daily Tech Intelligence Briefing System setup..."
echo "=========================================================="

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed. Please install Docker first."
    exit 1
fi

# Load .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Copying .env.example..."
    cp .env.example .env
    export $(grep -v '^#' .env | xargs)
fi

echo "Spinning up Docker containers..."
# Use docker compose or docker-compose
if docker compose version &> /dev/null; then
    docker compose up -d
else
    docker-compose up -d
fi

echo "Checking Ollama model requirements..."
# Check if the user is using native Ollama on macOS or containerized
if [[ "$OLLAMA_BASE_URL" == *"host.docker.internal"* ]] || [[ "$OLLAMA_BASE_URL" == *"localhost"* ]] || [[ "$OLLAMA_BASE_URL" == *"127.0.0.1"* ]]; then
    echo ""
    echo "=========================================================="
    echo "IMPORTANT NOTE FOR macOS GPU ACCELERATION:"
    echo "----------------------------------------------------------"
    echo "You are pointing to a host-level Ollama service."
    echo "To get full Metal GPU acceleration on your Mac, please run"
    echo "the following commands in a native terminal on your machine:"
    echo ""
    echo "  ollama pull $OLLAMA_EMBED_MODEL"
    echo "  ollama pull $OLLAMA_LLM_MODEL"
    echo "=========================================================="
    echo ""
    
    # Try to check if native ollama is running and pull if command exists
    if command -v ollama &> /dev/null; then
        echo "Found native 'ollama' installation. Checking models..."
        if ! ollama list | grep -q "$OLLAMA_EMBED_MODEL"; then
            echo "Pulling $OLLAMA_EMBED_MODEL..."
            ollama pull $OLLAMA_EMBED_MODEL
        else
            echo "Embedding model '$OLLAMA_EMBED_MODEL' is already installed."
        fi
        
        # Strip tag if needed or match exact tag
        # We can extract the base model name to be safe
        OLLAMA_LLM_CLEAN=$(echo "$OLLAMA_LLM_MODEL" | cut -d':' -f1)
        if ! ollama list | grep -i -q "$OLLAMA_LLM_CLEAN"; then
            echo "Pulling $OLLAMA_LLM_MODEL..."
            ollama pull $OLLAMA_LLM_MODEL
        else
            echo "LLM model '$OLLAMA_LLM_MODEL' is already installed."
        fi
    else
        echo "Please make sure your native Ollama app is running and pull the models."
    fi
else
    # Containerized Ollama
    echo "Containerized Ollama detected. Checking models inside the container..."
    
    echo "Waiting for Ollama service to start..."
    until docker exec briefing_ollama ollama list &> /dev/null; do
        sleep 2
    done
    
    if ! docker exec briefing_ollama ollama list | grep -q "$OLLAMA_EMBED_MODEL"; then
        echo "Pulling embedding model ($OLLAMA_EMBED_MODEL)..."
        docker exec briefing_ollama ollama pull $OLLAMA_EMBED_MODEL
    else
        echo "Embedding model '$OLLAMA_EMBED_MODEL' is already installed in container."
    fi
    
    OLLAMA_LLM_CLEAN=$(echo "$OLLAMA_LLM_MODEL" | cut -d':' -f1)
    if ! docker exec briefing_ollama ollama list | grep -i -q "$OLLAMA_LLM_CLEAN"; then
        echo "Pulling LLM model ($OLLAMA_LLM_MODEL)..."
        docker exec briefing_ollama ollama pull $OLLAMA_LLM_MODEL
    else
        echo "LLM model '$OLLAMA_LLM_MODEL' is already installed in container."
    fi
    echo "Models checked successfully."
fi

echo "=========================================================="
echo " System is running and configured!"
echo "----------------------------------------------------------"
echo " - FastAPI Helper API: http://localhost:${API_PORT:-8000}"
echo " - n8n Admin Panel:    http://localhost:${N8N_PORT:-5678}"
echo "=========================================================="
echo "To check the system health, visit: http://localhost:${API_PORT:-8000}/"
echo "To import the workflow, open n8n, click 'Workflows' -> 'Import from file',"
echo "and select: n8n/workflows/daily_briefing_workflow.json"
echo "=========================================================="
