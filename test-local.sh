#!/bin/bash

# Quick test script to build and run locally on port 80
set -e

IMAGE_NAME="mineru-local-test"

echo "🔨 Building Docker image for local testing..."
docker build -t $IMAGE_NAME .

echo "🚀 Running container on port 8080 (mapping to container port 80)..."
echo "   API docs will be available at: http://localhost:8080/docs"
echo "   Health check: http://localhost:8080/health"
echo ""
echo "Press Ctrl+C to stop the container"

# Stop and remove existing container if it exists
docker stop $IMAGE_NAME 2>/dev/null || true
docker rm $IMAGE_NAME 2>/dev/null || true

# Run the container
docker run --rm --name $IMAGE_NAME -p 8080:80 $IMAGE_NAME
