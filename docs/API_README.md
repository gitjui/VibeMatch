# Music Identification API 🎵

A complete REST API for Shazam-like song identification and music recommendations, powered by deep learning.

## Features

- 🎯 **Song Identification**: Upload audio snippets or files for instant identification
- 🎵 **Smart Recommendations**: Get similar songs based on audio embeddings or metadata
- 📊 **Database Management**: Browse your music collection with filtering
- ⚡ **Fast & Scalable**: Built with FastAPI for high performance
- 📱 **Ready for Frontend**: CORS-enabled, JSON responses, comprehensive docs

## Quick Start

### 1. Start the API

```bash
# Method 1: Using the start script
./start_api.sh

# Method 2: Direct command
cd /Users/juigupte/Desktop/Learning/recsys-foundations
source ../.venv/bin/activate
uvicorn api:app --reload --port 8000
```

### 2. Access the API

- **API Base**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Test the API

```bash
python test_api.py
```

## API Endpoints

### Health & Info

#### `GET /`
Root endpoint with API information

```bash
curl http://localhost:8000/
```

#### `GET /health`
Health check with database status

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-19T10:30:00",
  "database": "connected",
  "songs": 24,
  "model": "loaded"
}
```

#### `GET /stats`
Get database statistics

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "id_database": {
    "total_songs": 24,
    "total_artists": 15,
    "languages": 5
  },
  "recommendation_database": {
    "total_songs": 1000,
    "total_artists": 20
  },
  "embeddings": {
    "cnn_64": {
      "count": 2517,
      "dimension": 64
    }
  }
}
```

### Song Management

#### `GET /songs`
List all songs with pagination and filtering

**Parameters:**
- `limit` (int, default: 50): Number of results
- `offset` (int, default: 0): Pagination offset
- `artist` (string, optional): Filter by artist name
- `language` (string, optional): Filter by language

```bash
# List first 10 songs
curl "http://localhost:8000/songs?limit=10"

# Filter by artist
curl "http://localhost:8000/songs?artist=Ed%20Sheeran"

# Filter by language
curl "http://localhost:8000/songs?language=English&limit=20"
```

**Response:**
```json
{
  "songs": [
    {
      "id": 1,
      "title": "Beautiful Things",
      "artist": "Benson Boone",
      "album": "Beautiful Things",
      "duration_seconds": 192,
      "language": "English",
      "tempo_bpm": 143.6
    }
  ],
  "pagination": {
    "total": 24,
    "limit": 10,
    "offset": 0,
    "has_more": true
  }
}
```

#### `GET /songs/{song_id}`
Get detailed information about a specific song

```bash
curl http://localhost:8000/songs/1
```

**Response:**
```json
{
  "id": 1,
  "title": "Beautiful Things",
  "artist": "Benson Boone",
  "album": "Beautiful Things",
  "duration_seconds": 192,
  "bitrate_kbps": 192.0,
  "tempo_bpm": 143.6,
  "energy_level": "High",
  "language": "English",
  "release_date": "2024-01-18",
  "tags": [
    {"name": "pop", "type": "genre"},
    {"name": "5+ wochen", "type": "tag"}
  ]
}
```

#### `GET /recommendations/{song_id}`
Get song recommendations based on a specific song

**Parameters:**
- `limit` (int, default: 10): Number of recommendations

```bash
curl "http://localhost:8000/recommendations/1?limit=5"
```

**Response:**
```json
{
  "song": {
    "id": 1,
    "title": "Beautiful Things",
    "artist": "Benson Boone"
  },
  "recommendations": [
    {
      "track_id": "SAMPLE000123",
      "title": "Sample Song 124",
      "artist": "Benson Boone",
      "genre": "Pop",
      "year": 2023
    }
  ],
  "total": 5,
  "recommendation_type": "same_artist"
}
```

### Music Identification

#### `POST /identify-url`
Identify a song from a file path

**Request Body:**
```json
{
  "file_path": "/path/to/song.mp3",
  "start_time": 30.0,
  "duration": 5.0
}
```

```bash
curl -X POST "http://localhost:8000/identify-url" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/Users/juigupte/Desktop/Learning/music/mp3/Ed Sheeran - Shape of You (Official Music Video).mp3",
    "start_time": 30.0,
    "duration": 5.0
  }'
```

**Response:**
```json
{
  "success": true,
  "query": {
    "file": "Ed Sheeran - Shape of You.mp3",
    "start_time": 30.0,
    "duration": 5.0
  },
  "exact_match": {
    "found": true,
    "song_id": 12,
    "title": "Shape of You",
    "artist": "Phsycloner",
    "confidence": 0.985,
    "distance": 0.005,
    "match_type": "exact"
  },
  "recommendations": [
    {
      "rank": 1,
      "song_id": "SAMPLE000037",
      "title": "Sample Song 38",
      "artist": "Ed Sheeran",
      "genre": "Pop",
      "year": 2017,
      "similarity": 0.90,
      "type": "metadata_based_same_artist"
    }
  ],
  "summary": {
    "total_recommendations": 10,
    "status": "match_found",
    "recommendation_type": "with_match"
  }
}
```

#### `POST /identify`
Identify a song by uploading an audio file

**Form Data:**
- `file`: Audio file (MP3, WAV, M4A, FLAC)
- `start_time` (float, default: 0.0): Start time in seconds
- `duration` (float, default: 5.0): Duration to analyze

```bash
curl -X POST "http://localhost:8000/identify" \
  -F "file=@/path/to/song.mp3" \
  -F "start_time=30.0" \
  -F "duration=5.0"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/identify"
files = {'file': open('song.mp3', 'rb')}
params = {'start_time': 30.0, 'duration': 5.0}

response = requests.post(url, files=files, params=params)
result = response.json()

if result['exact_match']['found']:
    print(f"Found: {result['exact_match']['title']} by {result['exact_match']['artist']}")
    print(f"Confidence: {result['exact_match']['confidence']*100:.1f}%")
```

**Response:** Same format as `/identify-url`

## Architecture

```
┌─────────────────────────────────────────┐
│           FastAPI Application            │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │   DualDatabaseSystem                │ │
│  │                                     │ │
│  │   ┌──────────────┐ ┌─────────────┐ │ │
│  │   │  ID Database │ │ Rec Database│ │ │
│  │   │  (24 songs)  │ │ (1K songs)  │ │ │
│  │   │  CNN 64-dim  │ │ MSD 12-dim  │ │ │
│  │   └──────────────┘ └─────────────┘ │ │
│  │                                     │ │
│  │   ┌──────────────────────────────┐ │ │
│  │   │   CNN Model (TensorFlow)     │ │ │
│  │   │   track_classifier.keras     │ │ │
│  │   └──────────────────────────────┘ │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Response Formats

### Success Response (Exact Match Found)
```json
{
  "success": true,
  "exact_match": {
    "found": true,
    "song_id": 21,
    "title": "Be the One",
    "artist": "Dua Lipa",
    "confidence": 0.953,
    "distance": 0.014,
    "match_type": "exact"
  },
  "recommendations": [...],
  "summary": {
    "status": "match_found",
    "total_recommendations": 10
  }
}
```

### No Match Response
```json
{
  "success": true,
  "exact_match": {
    "found": false,
    "message": "No exact match found in ID database"
  },
  "recommendations": [...],
  "summary": {
    "status": "no_match",
    "total_recommendations": 10,
    "recommendation_type": "discovery"
  }
}
```

## Error Handling

All errors return proper HTTP status codes and structured messages:

```json
{
  "detail": "File not found: /path/to/song.mp3"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid file type, etc.)
- `404`: Not Found (song ID, file path)
- `500`: Internal Server Error

## Testing

### Run Test Suite
```bash
python test_api.py
```

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# List songs
curl http://localhost:8000/songs?limit=5

# Identify song
curl -X POST http://localhost:8000/identify-url \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/song.mp3", "start_time": 30.0, "duration": 5.0}'
```

### Interactive Documentation

Visit http://localhost:8000/docs for Swagger UI where you can:
- View all endpoints
- Test API calls directly in browser
- See request/response schemas
- Download OpenAPI spec

## Performance

- **Cold start**: ~2-3 seconds (model loading)
- **Warm queries**: ~200-500ms per identification
- **File upload**: ~1-2 seconds (includes temp file handling)
- **Concurrent requests**: Supports multiple simultaneous queries

## Database Schema

### ID Database (songs.db)
- **songs**: 24 songs with 36 metadata fields
- **track_embeddings**: 2,517 CNN embeddings (64-dim)
- **tags**: Genre and tag associations

### Recommendation Database (recommendations.db)
- **recommendation_tracks**: 1,000 songs
- **Embeddings**: 12-dim MSD timbre features

## Development

### Project Structure
```
recsys-foundations/
├── api.py                      # FastAPI application
├── start_api.sh                # API startup script
├── test_api.py                 # Test suite
├── dual_database_system.py     # Core query logic
├── songs.db                    # ID database
├── recommendations.db          # Recommendation database
├── track_classifier.keras      # CNN model
└── API_README.md              # This file
```

### Adding New Songs

1. Add MP3 files to `music/mp3/`
2. Run identification:
   ```bash
   python song_identifier.py "../music/mp3"
   ```
3. Import to database:
   ```python
   from songs_database import SongsDatabase
   db = SongsDatabase('songs.db')
   db.import_from_json('identified_songs.json')
   ```
4. Extract embeddings:
   ```bash
   python cnn_extract_embeddings.py
   ```
5. Restart API

### Environment Variables

```bash
# Optional configurations
export API_HOST="0.0.0.0"
export API_PORT="8000"
export ID_DB_PATH="songs.db"
export REC_DB_PATH="recommendations.db"
export MODEL_PATH="track_classifier.keras"
```

## Production Deployment

### Using Uvicorn with Workers
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn + Uvicorn
```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### API won't start
```bash
# Check if port 8000 is in use
lsof -ti:8000

# Kill process on port 8000
kill -9 $(lsof -ti:8000)
```

### Database errors
```bash
# Check database integrity
sqlite3 songs.db "PRAGMA integrity_check"
```

### Model loading issues
```bash
# Verify model file exists
ls -lh track_classifier.keras

# Check TensorFlow installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

## Next Steps

1. ✅ API running locally
2. 📱 Build frontend (React, Vue, or mobile app)
3. 🚀 Deploy to cloud (AWS, GCP, or Heroku)
4. 📊 Add analytics and logging
5. 🔐 Add authentication (JWT, OAuth)
6. 💾 Scale to larger databases

## Support

For issues or questions:
- Check `/docs` endpoint for API documentation
- Run `python test_api.py` to diagnose problems
- Review logs in terminal where API is running

---

Built with ❤️ using FastAPI, TensorFlow, and Python
