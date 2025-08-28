#!/usr/bin/env python3
"""
Server runner script for the Trending Content Detection System
"""

import uvicorn
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Trending Content Detection System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("test_data").mkdir(exist_ok=True)
    
    print(f"Starting Trending Content Detection System on {args.host}:{args.port}")
    
    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )

if __name__ == "__main__":
    main()