#!/usr/bin/env python
"""
Launch script for the Classification API
"""
import uvicorn
from pathlib import Path

# Set current directory
current_dir = Path(__file__).parent

if __name__ == "__main__":
    print("\n=== Handwriting Classification System | API Server ===")

    # Configuration for the API server
    config = {
        "app": "classification_api:app",
        "host": "0.0.0.0",
        "port": 8001,
        "reload": True,
        "log_level": "info",
        "use_colors": False
    }

    print("Starting Classification API server...")
    print(f"   URL: http://localhost:{config['port']}")
    print(f"   Health check: http://localhost:{config['port']}/health")
    print(f"   API docs: http://localhost:{config['port']}/docs\n")

    # Run the API server
    uvicorn.run(**config)
