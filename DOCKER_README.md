# MinerU Docker Deployment Guide

## Overview
This guide helps you build and deploy the MinerU API service using Docker, optimized for RunPod Serverless deployment on port 80.

## Quick Start

### 1. Build and Push to Docker Hub

```bash
# Set your Docker Hub username (replace with your actual username)
export DOCKER_HUB_USERNAME=your-dockerhub-username

# Make the script executable and run it
chmod +x build-and-push.sh
./build-and-push.sh
```

### 2. Test Locally

```bash
# Test the container locally
chmod +x test-local.sh
./test-local.sh
```

Then visit:
- API Documentation: http://localhost:8080/docs
- Health Check: http://localhost:8080/health

## RunPod Serverless Configuration

After pushing your image to Docker Hub, configure RunPod Serverless with:

### Container Settings
- **Container Image**: `your-dockerhub-username/mineru-api:latest`
- **Container Port**: `80`
- **HTTP Path**: `/health` (for health checks)

### Recommended Settings
- **Container Disk**: 20 GB minimum (for model storage)
- **Timeout**: 300+ seconds (for processing large documents)
- **Memory**: 8 GB minimum (16 GB recommended)
- **GPU**: Optional (will use CPU if GPU not available)

### Environment Variables (Optional)
You can add environment variables in RunPod if your application requires:
- API keys
- Model configurations
- Custom settings

## API Endpoints

Once deployed, your service will have these endpoints:

### Document Processing (OCR Mode)
```
POST /process_document_ocr_mode
Content-Type: multipart/form-data

Parameters:
- file: PDF/Image file to process
- lang: Language code (default: "ch")
- parse_method: "auto", "txt", "ocr" (default: "auto")
- formula_enable: true/false (default: true)
- table_enable: true/false (default: true)
```

### Document Processing (Text Only)
```
POST /process_document_text_only
Content-Type: multipart/form-data

Parameters:
- file: PDF file to process
- lang: Language code (default: "ch")
```

### Health Check
```
GET /health
```

## Local Development

### Build Only
```bash
docker build -t mineru-api .
```

### Run with Custom Port
```bash
docker run -p 8080:80 mineru-api
```

### Run with Environment Variables
```bash
docker run -p 8080:80 -e YOUR_ENV_VAR=value mineru-api
```

## Troubleshooting

### Common Issues

1. **Build fails with dependency errors**
   - Check `requirements.txt` for incompatible versions
   - Ensure all system dependencies are in Dockerfile

2. **Container starts but health check fails**
   - Check logs: `docker logs container-name`
   - Verify port 80 is accessible inside container

3. **Out of memory during processing**
   - Increase RunPod memory allocation
   - Process smaller documents or reduce worker count

### Logs
```bash
# Check container logs
docker logs container-name

# Follow logs in real-time
docker logs -f container-name
```

## Performance Optimization

### For RunPod Serverless
- Use GPU-enabled templates for faster processing
- Set appropriate timeout based on expected document sizes
- Monitor memory usage and adjust accordingly

### Resource Requirements
- **Minimum**: 4 GB RAM, 10 GB disk
- **Recommended**: 8+ GB RAM, 20+ GB disk
- **GPU**: Optional but recommended for large documents

## Security Considerations
- Container runs as non-root user (appuser)
- No sensitive data in Docker image
- Use environment variables for secrets
- Health check endpoint is public (no authentication)

## Support
If you encounter issues:
1. Check the health endpoint: `/health`
2. Review container logs
3. Verify RunPod configuration matches the requirements above
