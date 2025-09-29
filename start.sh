#!/bin/bash
set -e  # stop if any command fails

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build frontend if folder exists
if [ -d "frontend" ]; then
    cd frontend
    npm install
    npm run build  # produces build/ folder
    cd ..
fi
