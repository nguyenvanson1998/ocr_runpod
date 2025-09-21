# MinerU API Deployment Guide

## Gunicorn Configuration Files

### 1. `gunicorn.conf.py` - CUDA Optimized (Recommended for GPU)
- **Workers**: 1 (CUDA safe)
- **Preload**: Disabled (prevents CUDA conflicts)
- **Memory management**: Aggressive worker recycling
- **Use case**: Production with GPU acceleration

### 2. `gunicorn_cpu.conf.py` - CPU Optimized
- **Workers**: 4 (CPU parallel processing)
- **Preload**: Enabled (better CPU performance)
- **Use case**: CPU-only deployment or fallback mode

## Startup Scripts

### 1. `start_server.sh` - CUDA Mode
```bash
./start_server.sh
```
- Forces CUDA mode with 1 worker
- Recommended for GPU-enabled servers

### 2. `start_server_cpu.sh` - CPU Mode  
```bash
./start_server_cpu.sh
```
- Forces CPU mode with multiple workers
- Recommended when GPU is not available

### 3. `start_server_smart.sh` - Auto Detection (Recommended)
```bash
./start_server_smart.sh
```
- Automatically detects CUDA availability
- Chooses optimal configuration
- Best for flexible deployment

## CUDA Issue Resolution

The original error occurred because:
1. **Multiple workers** + **CUDA** = Process forking conflicts
2. **Solution**: Use single worker OR disable CUDA for multiple workers

## Performance Considerations

### GPU Mode (1 Worker)
- ✅ CUDA acceleration for ML models
- ✅ No process conflicts
- ❌ Lower concurrent request handling

### CPU Mode (Multiple Workers)
- ✅ Higher concurrent request handling
- ✅ No CUDA conflicts
- ❌ Slower ML model processing

## Production Deployment

### For GPU Servers:
```bash
chmod +x start_server_smart.sh
./start_server_smart.sh
```

### For Load Balancing:
Use multiple single-worker instances behind a load balancer:
```bash
# Terminal 1
gunicorn -c gunicorn.conf.py -b 0.0.0.0:8001 minerU_2endpoints:app

# Terminal 2  
gunicorn -c gunicorn.conf.py -b 0.0.0.0:8002 minerU_2endpoints:app

# Then use nginx/haproxy to balance between ports 8001, 8002, etc.
```

## Monitoring

Check worker status:
```bash
ps aux | grep gunicorn
```

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Configuration Files Summary

| File | Workers | CUDA | Use Case |
|------|---------|------|----------|
| `gunicorn.conf.py` | 1 | ✅ | GPU production |
| `gunicorn_cpu.conf.py` | 4 | ❌ | CPU production |
| `start_server_smart.sh` | Auto | Auto | Flexible deployment |

## 📊 New Features

### Timing Information
All API responses now include detailed timing information:
- Parse duration (PDF processing)
- Processing duration (content extraction)
- Total duration with formatted output
- Start/end timestamps
- Configuration details

### Testing Performance
```bash
# Test API performance with timing
python test_timing.py

# Example output shows:
# Parse time: 0m 8s
# Processing time: 0m 2s  
# Total server time: 0m 10s
```

For detailed timing documentation, see [TIMING_GUIDE.md](./TIMING_GUIDE.md)
