#!/bin/bash

# Build and Push Docker Image Script for RunPod Serverless
# This script builds the MinerU Docker image optimized for port 80 and pushes it to Docker Hub

set -e  # Exit on any error

# Configuration - Update these with your Docker Hub details
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-your-dockerhub-username}"
IMAGE_NAME="${IMAGE_NAME:-mineru-api}"
TAG="${TAG:-latest}"
FULL_IMAGE_NAME="$DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MinerU Docker Build and Push Script (Port 80 for RunPod) ===${NC}"
echo -e "${BLUE}Building image: $FULL_IMAGE_NAME${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Validate environment variables
if [ "$DOCKER_HUB_USERNAME" = "your-dockerhub-username" ]; then
    echo -e "${YELLOW}Warning: Please set your Docker Hub username${NC}"
    echo -e "${YELLOW}You can either:${NC}"
    echo -e "${YELLOW}1. Set environment variable: export DOCKER_HUB_USERNAME=yourusername${NC}"
    echo -e "${YELLOW}2. Edit this script and replace 'your-dockerhub-username'${NC}"
    echo ""
    read -p "Enter your Docker Hub username: " DOCKER_HUB_USERNAME
    if [ -z "$DOCKER_HUB_USERNAME" ]; then
        echo -e "${RED}Error: Docker Hub username is required${NC}"
        exit 1
    fi
    FULL_IMAGE_NAME="$DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG"
fi

echo -e "${BLUE}Step 1: Building Docker image...${NC}"
docker build -t $FULL_IMAGE_NAME . || {
    echo -e "${RED}Error: Failed to build Docker image${NC}"
    exit 1
}

echo -e "${GREEN}✓ Docker image built successfully${NC}"
echo ""

# Show image details
echo -e "${BLUE}Image Details:${NC}"
docker images $FULL_IMAGE_NAME

echo ""
echo -e "${BLUE}Step 2: Logging into Docker Hub...${NC}"

# Login to Docker Hub (will prompt for password if not logged in)
if ! docker info | grep -q "Username: $DOCKER_HUB_USERNAME"; then
    echo "Please login to Docker Hub:"
    docker login || {
        echo -e "${RED}Error: Failed to login to Docker Hub${NC}"
        exit 1
    }
fi

echo -e "${GREEN}✓ Successfully logged into Docker Hub${NC}"
echo ""

echo -e "${BLUE}Step 3: Pushing image to Docker Hub...${NC}"
docker push $FULL_IMAGE_NAME || {
    echo -e "${RED}Error: Failed to push image to Docker Hub${NC}"
    exit 1
}

echo -e "${GREEN}✓ Image pushed successfully to Docker Hub${NC}"
echo ""

echo -e "${GREEN}=== Build and Push Complete ===${NC}"
echo -e "${GREEN}Image: $FULL_IMAGE_NAME${NC}"
echo ""
echo -e "${BLUE}For RunPod Serverless deployment:${NC}"
echo -e "${YELLOW}1. Container Image: $FULL_IMAGE_NAME${NC}"
echo -e "${YELLOW}2. Container Port: 80${NC}"
echo -e "${YELLOW}3. HTTP Path: /health (for health checks)${NC}"
echo -e "${YELLOW}4. Timeout: Recommend 300+ seconds for large documents${NC}"
echo -e "${YELLOW}5. Environment Variables (if needed):${NC}"
echo -e "${YELLOW}   - Add any required API keys or configurations${NC}"
echo ""
echo -e "${BLUE}Test locally with:${NC}"
echo -e "${YELLOW}docker run -p 8080:80 $FULL_IMAGE_NAME${NC}"
echo -e "${YELLOW}Then visit: http://localhost:8080/docs${NC}"
echo -e "${YELLOW}Health check: http://localhost:8080/health${NC}"
