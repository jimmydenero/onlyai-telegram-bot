#!/bin/bash
# Startup script for Railway deployment

# Set default port if not provided
PORT=${PORT:-8080}

# Start the application
python3 -m uvicorn main:app --host 0.0.0.0 --port $PORT
