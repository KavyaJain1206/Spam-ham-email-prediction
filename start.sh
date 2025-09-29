#!/bin/bash

# Start backend (FastAPI)
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait 5 seconds for backend to start
sleep 5

# Start frontend
cd frontend
npm install
npm start
