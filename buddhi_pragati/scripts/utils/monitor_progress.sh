#!/bin/bash

# Simple progress monitoring utility
# Usage: ./monitor_progress.sh [LOG_DIR]

LOG_DIR="${1:-../../logs}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"

echo "📊 Monitoring Experiment Progress"
echo "Log directory: $LOG_DIR"
echo "Press Ctrl+C to exit"
echo

# Find latest log files
find_latest_logs() {
    find "$LOG_DIR" -name "*.log" -type f -mmin -60 2>/dev/null | head -5
}

while true; do
    clear
    echo "📊 Experiment Progress Monitor - $(date)"
    echo "========================================"
    
    # Show latest log files
    echo "Recent log files (last hour):"
    LOGS=($(find_latest_logs))
    
    if [[ ${#LOGS[@]} -eq 0 ]]; then
        echo "No recent log files found"
    else
        for log in "${LOGS[@]}"; do
            echo "  $(basename "$log")"
        done
    fi
    
    echo
    
    # Show tail of most recent log
    if [[ ${#LOGS[@]} -gt 0 ]]; then
        echo "Latest log activity (${LOGS[0]}):"
        echo "----------------------------------------"
        tail -10 "${LOGS[0]}" 2>/dev/null || echo "Cannot read log file"
    fi
    
    echo
    echo "Press Ctrl+C to exit"
    sleep 5
done