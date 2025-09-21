#!/bin/bash

# Quick stop all MinerU services
echo "🛑 Stopping all MinerU services..."

# Method 1: Kill by process name
echo "Killing gunicorn processes..."
pkill -f "gunicorn.*minerU_2endpoints" && echo "✅ Killed gunicorn processes" || echo "ℹ️ No gunicorn processes found"

# Method 2: Kill by port (if needed)
echo "Checking for processes on common ports..."
for port in 8000 8001 8002 8003 8004; do
    PID=$(ss -tlnp | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d'=' -f2 | head -1)
    if [ -n "$PID" ]; then
        echo "Killing process on port $port (PID: $PID)"
        kill -TERM $PID
    fi
done

# Wait a moment for graceful shutdown
sleep 3

# Force kill if necessary
echo "Force killing any remaining processes..."
pkill -9 -f "gunicorn.*minerU_2endpoints" 2>/dev/null && echo "✅ Force killed remaining processes" || echo "ℹ️ No processes to force kill"

# Clean up PID files
echo "Cleaning up PID files..."
rm -f /tmp/mineru_api.pid
rm -f /tmp/mineru_*.pid

# Final check
echo "Final status check:"
REMAINING=$(pgrep -f "gunicorn.*minerU_2endpoints")
if [ -z "$REMAINING" ]; then
    echo "✅ All MinerU services stopped successfully"
else
    echo "⚠️ Some processes may still be running:"
    ps aux | grep -E "gunicorn.*minerU_2endpoints" | grep -v grep
fi

echo "🔍 Port status:"
ss -tlnp | grep -E ":(8000|8001|8002|8003|8004)" || echo "ℹ️ No services on common ports"
