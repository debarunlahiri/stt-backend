#!/usr/bin/env python3
"""
Main Entry Point for Offline STT Backend Server

This script serves as the main entry point for running the STT backend server.
It handles environment setup, configuration loading, and starts the Uvicorn
ASGI server with the FastAPI application.

The script includes special handling for virtual environment PATH issues that
can occur when conda and venv Python installations conflict.

Usage:
    python run.py
    
    Or make executable and run directly:
    chmod +x run.py
    ./run.py

Author: Debarun Lahiri
"""

import os
import sys
import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Fix for conda/venv Python interpreter conflicts
    # Ensure subprocesses use the same Python interpreter as the main process
    # This fixes issues when conda and venv Python installations conflict
    venv_bin_dir = os.path.dirname(sys.executable)
    if os.path.exists(venv_bin_dir) and venv_bin_dir not in os.environ.get('PATH', '').split(os.pathsep)[:1]:
        # Prepend venv bin directory to PATH so subprocesses find the correct Python
        # This ensures that any subprocesses spawned by the application use the same
        # Python interpreter, avoiding import and dependency issues
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = os.pathsep.join([venv_bin_dir, current_path])
    
    # Determine reload directories for development mode
    # When reload is enabled, Uvicorn watches these directories for code changes
    reload_dirs = settings.reload_dirs if settings.reload_dirs else ["app"]
    
    # Start the Uvicorn ASGI server with the FastAPI application
    uvicorn.run(
        "app.main:app",  # Application module path
        host=settings.host,  # Server host (0.0.0.0 for all interfaces)
        port=settings.port,  # Server port (default: 8000)
        reload=settings.reload or settings.debug,  # Enable auto-reload in dev mode
        reload_dirs=reload_dirs if (settings.reload or settings.debug) else None,  # Directories to watch
        workers=1 if (settings.reload or settings.debug) else settings.workers,  # Worker processes (1 in dev, configurable in prod)
        log_level=settings.log_level.lower()  # Logging level (INFO, DEBUG, etc.)
    )

