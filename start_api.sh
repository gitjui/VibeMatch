#!/bin/bash
# Start FastAPI Server

cd /Users/juigupte/Desktop/Learning/recsys-foundations
source ../.venv/bin/activate

echo "🚀 Starting Music Identification API..."
echo ""
echo "API will be available at:"
echo "  - http://localhost:8000"
echo "  - http://127.0.0.1:8000"
echo ""
echo "Interactive API docs:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn api:app --reload --port 8000 --host 0.0.0.0
