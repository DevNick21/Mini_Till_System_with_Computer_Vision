#!/usr/bin/env python
"""
Launch script for the BetFred Classification API
"""
import os
import sys
import uvicorn
from pathlib import Path

# Set current directory
current_dir = Path(__file__).parent

if __name__ == "__main__":
    print("\n=== BetFred Handwriting Classification System ===")
    print("=== EfficientNet API Server                 ===")
    print("=== Version 2.1 - July 2025                 ===\n")

    # Configuration for the API server
    config = {
        "app": "classification_api:app",
        "host": "0.0.0.0",
        "port": 8001,
        "reload": True,
        "log_level": "info",
        "use_colors": False
    }

    print("Starting BetFred Classification API server...")
    print(f"   URL: http://localhost:{config['port']}")
    print(f"   Health check: http://localhost:{config['port']}/health")
    print(f"   API docs: http://localhost:{config['port']}/docs\n")

    # Run the API server
    uvicorn.run(**config)
