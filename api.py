"""
FastAPI Music Identification & Recommendation API
==================================================

Complete REST API for Shazam-like song identification and music recommendations.

Endpoints:
- POST /identify - Upload audio file or snippet for identification
- POST /identify-url - Identify song from audio file path
- GET /songs - List all songs in database
- GET /songs/{song_id} - Get specific song details
- GET /recommendations/{song_id} - Get recommendations for a song
- POST /query - Query with audio snippet (JSON response)
- GET /health - Health check
- GET /stats - Database statistics

Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import sqlite3
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from dual_database_system import DualDatabaseSystem

# Initialize FastAPI
app = FastAPI(
    title="Music Identification API",
    description="Shazam-like song identification and recommendations",
    version="1.0.0"
)

# Add CORS middleware (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize dual database system (lazy loading)
system = None

def get_system():
    """Lazy initialize the dual database system"""
    global system
    if system is None:
        system = DualDatabaseSystem(
            id_db_path='songs.db',
            rec_db_path='recommendations.db',
            model_path='track_classifier.keras',
            exact_match_threshold=0.3
        )
    return system


# Pydantic models for request/response
class QueryRequest(BaseModel):
    file_path: str
    start_time: float = 0.0
    duration: float = 5.0


class IdentifyResponse(BaseModel):
    success: bool
    query: Dict[str, Any]
    exact_match: Optional[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SongDetail(BaseModel):
    id: int
    title: str
    artist: str
    album: Optional[str]
    duration: Optional[float]
    language: Optional[str]
    release_date: Optional[str]
    tempo_bpm: Optional[float]
    energy_level: Optional[str]
    confidence: Optional[float]


class DatabaseStats(BaseModel):
    id_database: Dict[str, Any]
    recommendation_database: Dict[str, Any]
    embeddings: Dict[str, Any]
    total_songs: int


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Music Identification API",
        "version": "1.0.0",
        "endpoints": {
            "POST /identify": "Upload audio file for identification",
            "POST /identify-url": "Identify from file path",
            "GET /songs": "List all songs",
            "GET /songs/{id}": "Get song details",
            "GET /recommendations/{id}": "Get recommendations",
            "POST /query": "Query with audio snippet",
            "GET /health": "Health check",
            "GET /stats": "Database statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        conn = sqlite3.connect('songs.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM songs")
        song_count = cursor.fetchone()[0]
        conn.close()
        
        # Check if model exists
        model_exists = Path('track_classifier.keras').exists()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "songs": song_count,
            "model": "loaded" if model_exists else "missing"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/stats", response_model=DatabaseStats)
async def get_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect('songs.db')
        cursor = conn.cursor()
        
        # ID database stats
        cursor.execute("SELECT COUNT(*) FROM songs")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM songs WHERE artist IS NOT NULL AND artist != 'Unknown'")
        total_artists = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT language) FROM songs WHERE language IS NOT NULL")
        total_languages = cursor.fetchone()[0]
        
        # Embedding stats
        cursor.execute("SELECT model_name, COUNT(*), AVG(embedding_dim) FROM track_embeddings GROUP BY model_name")
        embedding_stats = {}
        total_embeddings = 0
        for model_name, count, avg_dim in cursor.fetchall():
            embedding_stats[model_name] = {
                "count": count,
                "dimension": int(avg_dim)
            }
            total_embeddings += count
        
        conn.close()
        
        # Recommendation database stats
        rec_conn = sqlite3.connect('recommendations.db')
        rec_cursor = rec_conn.cursor()
        rec_cursor.execute("SELECT COUNT(*) FROM recommendation_tracks")
        rec_songs = rec_cursor.fetchone()[0]
        
        rec_cursor.execute("SELECT COUNT(DISTINCT artist) FROM recommendation_tracks")
        rec_artists = rec_cursor.fetchone()[0]
        
        rec_conn.close()
        
        return {
            "id_database": {
                "total_songs": total_songs,
                "total_artists": total_artists,
                "languages": total_languages,
                "database_file": "songs.db"
            },
            "recommendation_database": {
                "total_songs": rec_songs,
                "total_artists": rec_artists,
                "database_file": "recommendations.db"
            },
            "embeddings": embedding_stats,
            "total_songs": total_songs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/songs")
async def list_songs(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    artist: Optional[str] = None,
    language: Optional[str] = None
):
    """List all songs with optional filtering"""
    try:
        conn = sqlite3.connect('songs.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query with filters
        query = "SELECT * FROM songs WHERE 1=1"
        params = []
        
        if artist:
            query += " AND (artist LIKE ? OR id3_artist LIKE ?)"
            params.extend([f"%{artist}%", f"%{artist}%"])
        
        if language:
            query += " AND language = ?"
            params.append(language)
        
        query += " ORDER BY title LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        songs = [dict(row) for row in cursor.fetchall()]
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM songs WHERE 1=1"
        count_params = []
        if artist:
            count_query += " AND (artist LIKE ? OR id3_artist LIKE ?)"
            count_params.extend([f"%{artist}%", f"%{artist}%"])
        if language:
            count_query += " AND language = ?"
            count_params.append(language)
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "songs": songs,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/songs/{song_id}")
async def get_song(song_id: int):
    """Get detailed information about a specific song"""
    try:
        conn = sqlite3.connect('songs.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
        song = cursor.fetchone()
        
        if not song:
            raise HTTPException(status_code=404, detail="Song not found")
        
        # Get tags/genres
        cursor.execute("SELECT tag_name, tag_type FROM tags WHERE song_id = ?", (song_id,))
        tags = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        song_dict = dict(song)
        song_dict['tags'] = tags
        
        return song_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/recommendations/{song_id}")
async def get_recommendations(
    song_id: int,
    limit: int = Query(10, ge=1, le=50)
):
    """Get song recommendations based on a specific song"""
    try:
        # Get song details
        conn = sqlite3.connect('songs.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
        song = cursor.fetchone()
        
        if not song:
            raise HTTPException(status_code=404, detail="Song not found")
        
        song_dict = dict(song)
        artist = song_dict.get('artist') or song_dict.get('id3_artist') or 'Unknown'
        
        # Get recommendations from recommendation database
        rec_conn = sqlite3.connect('recommendations.db')
        rec_conn.row_factory = sqlite3.Row
        rec_cursor = rec_conn.cursor()
        
        # Find similar songs by artist
        rec_cursor.execute("""
            SELECT * FROM recommendation_tracks 
            WHERE artist LIKE ? 
            LIMIT ?
        """, (f"%{artist.split()[0]}%", limit))
        
        recommendations = [dict(row) for row in rec_cursor.fetchall()]
        rec_conn.close()
        conn.close()
        
        return {
            "song": song_dict,
            "recommendations": recommendations,
            "total": len(recommendations),
            "recommendation_type": "same_artist"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/identify-url", response_model=IdentifyResponse)
async def identify_from_url(request: QueryRequest):
    """
    Identify a song from an audio file path
    
    Example request:
    {
        "file_path": "/path/to/song.mp3",
        "start_time": 30.0,
        "duration": 5.0
    }
    """
    try:
        # Check if file exists
        if not Path(request.file_path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Query the system
        sys = get_system()
        results = sys.query(request.file_path, request.start_time, request.duration)
        
        # Format response
        output = {
            "success": True,
            "query": {
                "file": Path(request.file_path).name,
                "start_time": request.start_time,
                "duration": request.duration
            },
            "exact_match": None,
            "recommendations": [],
            "summary": {}
        }
        
        # Add exact match if found
        if results['exact_match']:
            match = results['exact_match']
            output['exact_match'] = {
                "found": True,
                "song_id": int(match.song_id),
                "title": match.title,
                "artist": match.artist,
                "confidence": float(round(match.confidence, 3)),
                "distance": float(round(match.distance, 3)),
                "match_type": match.match_type
            }
        else:
            output['exact_match'] = {
                "found": False,
                "message": "No exact match found in ID database"
            }
        
        # Add recommendations (with metadata fallback if needed)
        if len(results['similar_songs']) == 0 and results['exact_match']:
            # Metadata fallback
            query_filename = Path(request.file_path).stem.lower()
            detected_artist = None
            if ' - ' in query_filename:
                detected_artist = query_filename.split(' - ')[0].strip()
            
            if detected_artist:
                rec_conn = sqlite3.connect('recommendations.db')
                rec_cursor = rec_conn.cursor()
                rec_cursor.execute("""
                    SELECT track_id, title, artist, genre, year
                    FROM recommendation_tracks
                    WHERE LOWER(artist) LIKE ?
                    LIMIT 10
                """, (f"%{detected_artist}%",))
                
                for i, (track_id, title, artist, genre, year) in enumerate(rec_cursor.fetchall(), 1):
                    output['recommendations'].append({
                        "rank": i,
                        "song_id": str(track_id),
                        "title": title,
                        "artist": artist,
                        "genre": genre,
                        "year": year,
                        "similarity": 0.90,
                        "type": "metadata_based_same_artist"
                    })
                rec_conn.close()
        else:
            # Embedding-based recommendations
            for i, rec in enumerate(results['similar_songs'], 1):
                output['recommendations'].append({
                    "rank": i,
                    "song_id": str(rec.song_id),
                    "title": rec.title,
                    "artist": rec.artist,
                    "genre": rec.metadata.get('genre'),
                    "year": rec.metadata.get('year'),
                    "similarity": float(round(rec.similarity, 3)),
                    "type": "embedding_based"
                })
        
        # Summary
        output['summary'] = {
            "total_recommendations": len(output['recommendations']),
            "status": "match_found" if output['exact_match']['found'] else "no_match",
            "recommendation_type": "with_match" if output['exact_match']['found'] else "discovery"
        }
        
        return output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/identify", response_model=IdentifyResponse)
async def identify_from_upload(
    file: UploadFile = File(...),
    start_time: float = Query(0.0, ge=0),
    duration: float = Query(5.0, ge=1, le=30)
):
    """
    Identify a song by uploading an audio file
    
    Upload an MP3/WAV file and get identification + recommendations
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Supported: mp3, wav, m4a, flac"
            )
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Query using the temp file
            sys = get_system()
            results = sys.query(tmp_path, start_time, duration)
            
            # Format response (same as identify-url)
            output = {
                "success": True,
                "query": {
                    "file": file.filename,
                    "start_time": start_time,
                    "duration": duration
                },
                "exact_match": None,
                "recommendations": [],
                "summary": {}
            }
            
            # Add exact match
            if results['exact_match']:
                match = results['exact_match']
                output['exact_match'] = {
                    "found": True,
                    "song_id": int(match.song_id),
                    "title": match.title,
                    "artist": match.artist,
                    "confidence": float(round(match.confidence, 3)),
                    "distance": float(round(match.distance, 3)),
                    "match_type": match.match_type
                }
            else:
                output['exact_match'] = {
                    "found": False,
                    "message": "No exact match found"
                }
            
            # Add recommendations with metadata fallback
            if len(results['similar_songs']) == 0 and results['exact_match']:
                query_filename = Path(file.filename).stem.lower()
                detected_artist = None
                if ' - ' in query_filename:
                    detected_artist = query_filename.split(' - ')[0].strip()
                
                if detected_artist:
                    rec_conn = sqlite3.connect('recommendations.db')
                    rec_cursor = rec_conn.cursor()
                    rec_cursor.execute("""
                        SELECT track_id, title, artist, genre, year
                        FROM recommendation_tracks
                        WHERE LOWER(artist) LIKE ?
                        LIMIT 10
                    """, (f"%{detected_artist}%",))
                    
                    for i, (track_id, title, artist, genre, year) in enumerate(rec_cursor.fetchall(), 1):
                        output['recommendations'].append({
                            "rank": i,
                            "song_id": str(track_id),
                            "title": title,
                            "artist": artist,
                            "genre": genre,
                            "year": year,
                            "similarity": 0.90,
                            "type": "metadata_based_same_artist"
                        })
                    rec_conn.close()
            else:
                for i, rec in enumerate(results['similar_songs'], 1):
                    output['recommendations'].append({
                        "rank": i,
                        "song_id": str(rec.song_id),
                        "title": rec.title,
                        "artist": rec.artist,
                        "genre": rec.metadata.get('genre'),
                        "year": rec.metadata.get('year'),
                        "similarity": float(round(rec.similarity, 3)),
                        "type": "embedding_based"
                    })
            
            # Summary
            output['summary'] = {
                "total_recommendations": len(output['recommendations']),
                "status": "match_found" if output['exact_match']['found'] else "no_match",
                "recommendation_type": "with_match" if output['exact_match']['found'] else "discovery"
            }
            
            return output
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# Run with: uvicorn api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
