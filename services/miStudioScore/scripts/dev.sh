#!/bin/bash
#
# Development startup script for the miStudioScore service.
#
# This script sets the necessary PYTHONPATH and launches the FastAPI
# application using uvicorn on the designated port.

echo "Starting miStudioScore service..."

# Add the 'src' directory to the PYTHONPATH to allow uvicorn to find the 'main' module
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src

# Launch the Uvicorn server
# --host 0.0.0.0 makes the service accessible from outside the container/machine
# --port 8004 sets the listening port
# --reload enables auto-reloading for development
uvicorn src.main:app --host 0.0.0.0 --port 8004 --reload