"""
Quick Start: Using Your Mini Shazam System
==========================================

This is your cheat sheet for working with the audio matching system.
"""

# ============================================================
# 1. EXTRACT EMBEDDINGS (do this first!)
# ============================================================

# Run this whenever you add new songs to your library:
"""
cd /Users/juigupte/Desktop/Learning/recsys-foundations
python audio_embeddings.py
"""

# This will:
# - Scan all songs in database
# - Split each into 5-second chunks (with 2.5s overlap)
# - Extract 20 MFCC coefficients per chunk
# - Store embeddings in track_embeddings table


# ============================================================
# 2. TEST MATCHING ACCURACY
# ============================================================

# Run random evaluation (20 tests):
"""
python test_audio_matching.py
"""

# Query specific song at specific time:
"""
python test_audio_matching.py "path/to/song.mp3" 30.0
# Queries 5 seconds starting at 30 seconds
"""


# ============================================================
# 3. PROGRAMMATIC USAGE
# ============================================================

from audio_embeddings import MFCCExtractor, EmbeddingMatcher
import librosa
import numpy as np

# Example 1: Query with audio snippet
def find_song(audio_snippet_path: str, start_time: float = 0.0):
    """Find which song matches an audio snippet"""
    
    # Load audio snippet
    y, sr = librosa.load(audio_snippet_path, sr=22050, offset=start_time, duration=5.0)
    
    # Extract features
    extractor = MFCCExtractor()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    query_embedding = np.mean(mfccs, axis=1)
    
    # Search database
    results = EmbeddingMatcher.search_similar(
        query_embedding,
        db_path="songs.db",
        model_name="mfcc_20",
        top_k=5
    )
    
    # Show results
    for i, result in enumerate(results, 1):
        print(f"{i}. Song ID: {result['song_id']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Matched chunk: {result['chunk_start']:.1f}s - {result['chunk_end']:.1f}s")
        print()
    
    return results


# Example 2: Get song metadata from match
def get_song_info(song_id: int):
    """Get metadata for a matched song"""
    import sqlite3
    
    conn = sqlite3.connect("songs.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT title, artist, album, duration_seconds, language
        FROM songs
        WHERE id = ?
    """, (song_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'title': row[0],
            'artist': row[1],
            'album': row[2],
            'duration': row[3],
            'language': row[4]
        }
    return None


# Example 3: Add new song to database
def add_song_to_database(mp3_path: str):
    """Add a new song and extract its embeddings"""
    from song_identifier import scan_and_identify_folder
    from songs_database import SongsDatabase
    from audio_embeddings import MFCCExtractor
    import pickle
    from pathlib import Path
    
    # 1. Scan and identify song (creates JSON)
    folder = str(Path(mp3_path).parent)
    scan_and_identify_folder(folder)
    
    # 2. Import to database (if not already there)
    db = SongsDatabase()
    db.import_from_json("identified_songs.json")
    
    # 3. Extract embeddings
    extractor = MFCCExtractor()
    chunks = extractor.extract_chunks(mp3_path)
    
    # 4. Store embeddings
    import sqlite3
    conn = sqlite3.connect("songs.db")
    cursor = conn.cursor()
    
    # Get song_id
    cursor.execute("SELECT id FROM songs WHERE file_path = ?", (mp3_path,))
    song_id = cursor.fetchone()[0]
    
    for start, end, embedding in chunks:
        blob = pickle.dumps(embedding)
        cursor.execute("""
            INSERT INTO track_embeddings 
            (song_id, chunk_start, chunk_end, embedding, model_name, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (song_id, start, end, blob, "mfcc_20", 20))
    
    conn.commit()
    conn.close()
    
    print(f"✓ Added song with {len(chunks)} chunks")


# ============================================================
# 4. DATABASE QUERIES
# ============================================================

import sqlite3

def show_database_stats():
    """Print database statistics"""
    conn = sqlite3.connect("songs.db")
    cursor = conn.cursor()
    
    # Count songs
    cursor.execute("SELECT COUNT(*) FROM songs")
    num_songs = cursor.fetchone()[0]
    
    # Count embeddings
    cursor.execute("SELECT COUNT(*) FROM track_embeddings")
    num_embeddings = cursor.fetchone()[0]
    
    # Embeddings per song
    cursor.execute("""
        SELECT AVG(chunk_count) FROM (
            SELECT COUNT(*) as chunk_count 
            FROM track_embeddings 
            GROUP BY song_id
        )
    """)
    avg_chunks = cursor.fetchone()[0]
    
    print(f"Songs: {num_songs}")
    print(f"Embeddings: {num_embeddings}")
    print(f"Avg chunks per song: {avg_chunks:.1f}")
    
    conn.close()


def find_similar_songs(song_id: int, top_k: int = 5):
    """Find songs with similar audio characteristics
    
    This compares average embeddings across entire songs.
    """
    import pickle
    import numpy as np
    
    conn = sqlite3.connect("songs.db")
    cursor = conn.cursor()
    
    # Get all embeddings for target song
    cursor.execute("""
        SELECT embedding FROM track_embeddings 
        WHERE song_id = ?
    """, (song_id,))
    
    target_embeddings = [pickle.loads(row[0]) for row in cursor.fetchall()]
    target_avg = np.mean(target_embeddings, axis=0)
    
    # Compare against all other songs
    cursor.execute("SELECT DISTINCT song_id FROM track_embeddings WHERE song_id != ?", (song_id,))
    other_songs = [row[0] for row in cursor.fetchall()]
    
    similarities = []
    for other_id in other_songs:
        cursor.execute("""
            SELECT embedding FROM track_embeddings 
            WHERE song_id = ?
        """, (other_id,))
        
        other_embeddings = [pickle.loads(row[0]) for row in cursor.fetchall()]
        other_avg = np.mean(other_embeddings, axis=0)
        
        # Cosine similarity
        similarity = np.dot(target_avg, other_avg) / (
            np.linalg.norm(target_avg) * np.linalg.norm(other_avg)
        )
        
        similarities.append((other_id, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get metadata for top matches
    results = []
    for sid, sim in similarities[:top_k]:
        cursor.execute("SELECT title, artist FROM songs WHERE id = ?", (sid,))
        title, artist = cursor.fetchone()
        results.append({
            'song_id': sid,
            'title': title,
            'artist': artist,
            'similarity': sim
        })
    
    conn.close()
    return results


# ============================================================
# 5. USEFUL ONELINERS
# ============================================================

# Check database size:
# import os; print(f"{os.path.getsize('songs.db') / 1024 / 1024:.2f} MB")

# Count embeddings per model:
# conn = sqlite3.connect('songs.db')
# cursor = conn.cursor()
# cursor.execute("SELECT model_name, COUNT(*) FROM track_embeddings GROUP BY model_name")
# for row in cursor: print(row)

# Find songs without embeddings:
# cursor.execute("""
#     SELECT id, title FROM songs 
#     WHERE id NOT IN (SELECT DISTINCT song_id FROM track_embeddings)
# """)
# for row in cursor: print(row)

# Delete all embeddings (to re-extract):
# conn = sqlite3.connect('songs.db')
# conn.execute("DELETE FROM track_embeddings")
# conn.commit()


# ============================================================
# 6. TROUBLESHOOTING
# ============================================================

"""
Problem: "No such table: track_embeddings"
Solution: Run `python audio_embeddings.py` to create table

Problem: Matching accuracy is low
Solution: 
  - Ensure embeddings are extracted (check count)
  - Try different chunk sizes (edit audio_embeddings.py)
  - Check audio quality of input files

Problem: "song_id not found"
Solution: Re-run full pipeline:
  1. python song_identifier.py  # Scan MP3s
  2. python songs_database.py   # Create DB
  3. python audio_embeddings.py # Extract embeddings

Problem: Database is too large
Solution:
  - Reduce embedding dimension (n_mfcc parameter)
  - Increase chunk size (fewer chunks per song)
  - Use compression (but may reduce accuracy)
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Mini Shazam System - Quick Reference")
    print("=" * 60)
    print()
    
    # Show current stats
    try:
        show_database_stats()
    except:
        print("Database not found. Run audio_embeddings.py first!")
    
    print()
    print("Import this file for helper functions:")
    print("  from quick_start import find_song, get_song_info")
    print()
    print("Or run test_audio_matching.py for evaluation")
