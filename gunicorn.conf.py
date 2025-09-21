# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:80"
backlog = 2048

# Worker processes
workers = 8 # Set to 1 for CUDA compatibility
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 10000
worker_tmp_dir = "/dev/shm"
timeout = 72000
keepalive = 10000

# CRITICAL: Disable preload_app for CUDA compatibility
preload_app = False

# Use spawn method for CUDA compatibility (set via environment variable)
# This prevents CUDA re-initialization errors in forked processes

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 100  # Lower for CUDA memory management
max_requests_jitter = 8

# CRITICAL: Disable preload_app for CUDA compatibility
preload_app = False

# Use spawn method for CUDA compatibility (set via environment variable)
# This prevents CUDA re-initialization errors in forked processes

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "mineru_api"

# User and group (uncomment if needed)
# user = "www-data"
# group = "www-data"

# SSL (uncomment if using HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Graceful timeout
graceful_timeout = 120

# Enable reuse port for better performance
reuse_port = True

# Memory and file limits - Increased for large documents
limit_request_line = 100000  # INCREASED: (was 8192)
limit_request_fields = 1000  # INCREASED: (was 100)
limit_request_field_size = 1638000  # INCREASED: (was 8190)

# Temporary directory for uploaded files
tmp_upload_dir = None

# Maximum size of HTTP request header - Increased
max_header_size = 16384  # INCREASED: (was 8192)

# Additional stability settings
worker_abort_on_timeout = False 
