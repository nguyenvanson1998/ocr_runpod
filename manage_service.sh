#!/bin/bash

# MinerU Service Manager
# Usage: ./manage_service.sh [start|stop|restart|status]

SERVICE_NAME="mineru_api"
PID_FILE="/tmp/mineru_api.pid"
LOG_FILE="/tmp/mineru_api.log"

case "$1" in
    start)
        echo "Starting MinerU API service..."
        if [ -f "$PID_FILE" ]; then
            PID=$(cat $PID_FILE)
            if ps -p $PID > /dev/null 2>&1; then
                echo "Service is already running (PID: $PID)"
                exit 1
            else
                echo "Removing stale PID file..."
                rm -f $PID_FILE
            fi
        fi
        
        # Start service in background
        nohup ./start_server_smart.sh > $LOG_FILE 2>&1 & echo $! > $PID_FILE
        echo "Service started with PID: $(cat $PID_FILE)"
        echo "Log file: $LOG_FILE"
        ;;
        
    stop)
        echo "Stopping MinerU API service..."
        if [ -f "$PID_FILE" ]; then
            PID=$(cat $PID_FILE)
            if ps -p $PID > /dev/null 2>&1; then
                # Try graceful shutdown first
                kill -TERM $PID
                sleep 5
                
                # Check if still running
                if ps -p $PID > /dev/null 2>&1; then
                    echo "Graceful shutdown failed, forcing kill..."
                    kill -KILL $PID
                fi
                
                rm -f $PID_FILE
                echo "Service stopped successfully"
            else
                echo "Service is not running"
                rm -f $PID_FILE
            fi
        else
            echo "PID file not found, checking for running processes..."
            # Kill any gunicorn processes
            pkill -f "gunicorn.*minerU_2endpoints"
            echo "Killed any remaining gunicorn processes"
        fi
        ;;
        
    restart)
        echo "Restarting MinerU API service..."
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo "Checking MinerU API service status..."
        if [ -f "$PID_FILE" ]; then
            PID=$(cat $PID_FILE)
            if ps -p $PID > /dev/null 2>&1; then
                echo "✅ Service is running (PID: $PID)"
                
                # Check what ports are being used
                echo "📡 Ports in use:"
                ss -tlnp | grep gunicorn || echo "No gunicorn ports found"
                
                # Show recent logs
                echo "📋 Recent logs (last 10 lines):"
                tail -n 10 $LOG_FILE 2>/dev/null || echo "No logs available"
            else
                echo "❌ Service is not running (stale PID file)"
                rm -f $PID_FILE
            fi
        else
            # Check for any gunicorn processes
            PIDS=$(pgrep -f "gunicorn.*minerU_2endpoints")
            if [ -n "$PIDS" ]; then
                echo "⚠️ Service processes found but no PID file:"
                echo $PIDS
                echo "📡 Ports in use:"
                ss -tlnp | grep gunicorn
            else
                echo "❌ Service is not running"
            fi
        fi
        ;;
        
    logs)
        echo "📋 Showing service logs..."
        if [ -f "$LOG_FILE" ]; then
            tail -f $LOG_FILE
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the MinerU API service"
        echo "  stop    - Stop the MinerU API service"
        echo "  restart - Restart the MinerU API service"
        echo "  status  - Check service status and show ports"
        echo "  logs    - Show service logs (real-time)"
        echo ""
        echo "Examples:"
        echo "  $0 start    # Start service"
        echo "  $0 status   # Check if running"
        echo "  $0 logs     # Monitor logs"
        echo "  $0 stop     # Stop service"
        exit 1
        ;;
esac
