# 🎵 Music Identification API

Shazam-like song identification and recommendation system with REST API and React frontend.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Start both API and frontend
./start_vibematch.sh
```

**Access:**
- 🎨 Frontend: http://localhost:3000
- 📡 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

### Deploy to Production

See [DEPLOY.md](DEPLOY.md) for deployment instructions to Render, Railway, or other platforms.

**API Documentation**: http://localhost:8000/docs

## 📊 Features

- ✅ **Song Identification**: Upload audio file for instant recognition
- ✅ **Recommendations**: Get similar songs based on audio features
- ✅ **Dual Database**: 24-song ID database + 1,000-song recommendation database
- ✅ **CNN Embeddings**: Deep learning model for audio matching
- ✅ **REST API**: FastAPI with auto-generated docs

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/identify` | Upload audio file for identification |
| POST | `/identify-url` | Identify from file path |
| GET | `/songs` | List all songs (with filtering) |
| GET | `/songs/{id}` | Get specific song details |
| GET | `/recommendations/{id}` | Get song recommendations |
| GET | `/health` | API health check |
| GET | `/stats` | Database statistics |

## 📁 Project Structure

```
recsys-foundations/
├── api.py                      # FastAPI application
├── dual_database_system.py     # Core query engine
├── song_identifier.py          # Audio fingerprinting
├── songs_database.py           # Database management
├── cnn_extract_embeddings.py   # Embedding extraction
├── track_classifier.keras      # CNN model (1.2 MB)
├── songs.db                    # ID database (5.6 MB, 24 songs)
├── recommendations.db          # Rec database (252 KB, 1K songs)
├── test_api.py                # Test suite
├── start_api.sh               # Startup script
└── docs/                      # Documentation
```

## 🧪 Testing

```bash
# Run full test suite
python test_api.py

# Test specific endpoint
curl http://localhost:8000/health
```

## 🌐 Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for deployment instructions to:
- Render.com (recommended)
- Railway
- Fly.io

## 📚 Documentation

- [API Documentation](docs/API_README.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture Overview](docs/TWO_TIER_ARCHITECTURE.md)
- [Training Guide](docs/TRAINING_QUICKSTART.md)

## 💡 Usage Example

```python
import requests

# Upload audio file
with open('song.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/identify',
        files={'file': f}
    )

result = response.json()
print(f"Identified: {result['exact_match']['title']}")
print(f"Artist: {result['exact_match']['artist']}")
print(f"Confidence: {result['exact_match']['confidence']}")
```

## 📊 Current Database

- **ID Database**: 24 songs, 6,481 CNN embeddings (64-dim)
- **Recommendation DB**: 1,000 songs with metadata
- **Total Size**: 11 MB

## 🔧 Tech Stack

- **Backend**: FastAPI, Python 3.13
- **ML**: TensorFlow/Keras, librosa
- **Database**: SQLite
- **Audio**: chromaprint, mutagen, soundfile
