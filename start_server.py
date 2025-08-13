#!/usr/bin/env python3
"""
Simple server startup script
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
