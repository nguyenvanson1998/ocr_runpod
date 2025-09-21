#!/bin/bash

# Build and Deploy Script for MinerU API

set -e

echo "🚀 Building MinerU API Docker Container..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
print_status "Building Docker image..."
docker build -t mineru-api:latest .

if [ $? -eq 0 ]; then
    print_status "Docker image built successfully!"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Ask user what to do next
echo ""
echo "What would you like to do next?"
echo "1) Run with docker-compose (recommended)"
echo "2) Run single container"
echo "3) Just build (already done)"
echo "4) Push to registry"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        print_status "Starting services with docker-compose..."
        docker-compose down 2>/dev/null || true
        docker-compose up -d
        echo ""
        print_status "Services started! Check status with: docker-compose ps"
        print_status "View logs with: docker-compose logs -f"
        print_status "API should be available at: http://localhost:8002"
        if [ -f "nginx.conf" ]; then
            print_status "Nginx proxy available at: http://localhost:80"
        fi
        ;;
    2)
        print_status "Running single container..."
        docker stop mineru-api 2>/dev/null || true
        docker rm mineru-api 2>/dev/null || true
        docker run -d \
            --name mineru-api \
            -p 8002:8002 \
            -v $(pwd)/output_minerU:/app/output_minerU \
            -v $(pwd)/files_test:/app/files_test \
            -v $(pwd)/logs:/app/logs \
            mineru-api:latest
        print_status "Container started! API available at: http://localhost:8002"
        print_status "View logs with: docker logs -f mineru-api"
        ;;
    3)
        print_status "Build complete!"
        ;;
    4)
        read -p "Enter registry URL (e.g., your-registry.com/mineru-api): " registry
        if [ ! -z "$registry" ]; then
            print_status "Tagging and pushing to registry..."
            docker tag mineru-api:latest $registry:latest
            docker push $registry:latest
            print_status "Pushed to registry successfully!"
        else
            print_warning "No registry URL provided, skipping push."
        fi
        ;;
    *)
        print_warning "Invalid choice. Build complete!"
        ;;
esac

echo ""
print_status "Docker setup complete! 🎉"
echo ""
echo "Useful commands:"
echo "  docker-compose up -d          # Start all services"
echo "  docker-compose down           # Stop all services"
echo "  docker-compose logs -f        # View logs"
echo "  docker-compose ps             # Check status"
echo "  docker exec -it mineru-api bash  # Access container shell"
