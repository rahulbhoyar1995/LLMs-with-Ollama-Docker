#!/bin/sh
set -e

# List of models to pull (space-separated)
MODELS="jina/jina-embeddings-v2-base-de"   # Put whichever model you want to put here

# Function to check if a model is already pulled
model_exists() {
  docker compose -f docker-compose.yml exec ollama ollama list | grep -q "$1"
}

# Loop through each model and pull it only if not already present
for MODEL_NAME in $MODELS; do
  if model_exists "$MODEL_NAME"; then
    echo "'$MODEL_NAME' is already pulled, skipping..."
  else
    echo "Pulling '$MODEL_NAME' into the ollama_server container..."
    docker compose -f docker-compose.yml exec ollama ollama pull "$MODEL_NAME"
    echo "Done pulling '$MODEL_NAME'!"
  fi
done

echo "All necessary models are available!"
