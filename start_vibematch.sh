#!/bin/bash

# Start VibeMatch Frontend and Backend

echo "🎵 Starting VibeMatch..."
echo ""

# Start API server in background
echo "📡 Starting API server on http://localhost:8000..."
cd "$(dirname "$0")"
nohup /Users/juigupte/Desktop/Learning/.venv/bin/python -m uvicorn api:app --port 8000 --host 127.0.0.1 > /tmp/vibematch_api.log 2>&1 &
API_PID=$!
echo "   API PID: $API_PID"

# Wait for API to start
sleep 3

# Start frontend server with Vite
echo ""
echo "🎨 Starting Frontend on http://localhost:3000..."
cd frontend
npm run dev > /tmp/vibematch_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "✅ VibeMatch is running!"
echo ""
echo "📱 Open in browser: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "To stop: kill $API_PID $FRONTEND_PID"
echo ""
