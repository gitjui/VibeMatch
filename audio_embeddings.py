"""
Audio Embeddings Database - Your mini Shazam Lab
================================================

This module creates a new layer on top of your existing songs database:
- Stores "how songs sound" (embeddings/fingerprints)
- Enables audio matching by similarity search
- Phase 1: MFCC baseline (non-learned features)
- Future: Pre-trained models (VGGish, YAMNet, CLAP)

Architecture:
    songs table (existing) → "what is this song?" (metadata)
    track_embeddings table (new) → "how does this song sound?" (features)
"""

import sqlite3
import numpy as np
import librosa
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime


class AudioEmbeddingsDB:
    """Manages audio embeddings storage and retrieval for song matching"""
    
    def __init__(self, db_path: str = "songs.db"):
        """Initialize embeddings database
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = None
        self.create_embeddings_table()
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def create_embeddings_table(self):
        """Create track_embeddings table if it doesn't exist
        
        Schema:
            id: Primary key
            song_id: Foreign key to songs table (using id field)
            chunk_start: Start time of audio chunk (seconds)
            chunk_end: End time of audio chunk (seconds)
            embedding: Serialized numpy array (blob)
            model_name: Name of model/method used (e.g., 'mfcc_baseline')
            embedding_dim: Dimensionality of the embedding vector
            created_at: Timestamp
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL,
                chunk_start REAL NOT NULL,
                chunk_end REAL NOT NULL,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (song_id) REFERENCES songs(id)
            )
        """)
        
        # Create indexes for fast lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_song_id 
            ON track_embeddings(song_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_name 
            ON track_embeddings(model_name)
        """)
        
        conn.commit()
        conn.close()
        
        print("✓ track_embeddings table created/verified")


class MFCCExtractor:
    """Phase 1: MFCC baseline feature extractor
    
    MFCCs (Mel-Frequency Cepstral Coefficients) are classic audio features
    that capture the spectral envelope of sound. This is your non-learned
    baseline before integrating pre-trained models.
    """
    
    def __init__(self, n_mfcc: int = 20, sr: int = 22050):
        """Initialize MFCC extractor
        
        Args:
            n_mfcc: Number of MFCC coefficients (default 20)
            sr: Sample rate for audio loading (default 22050 Hz)
        """
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.model_name = f"mfcc_{n_mfcc}"
    
    def extract_full_song(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract single MFCC vector for entire song (mean across time)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            numpy array of shape (n_mfcc,) or None if extraction fails
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Average across time → single vector per song
            mfcc_mean = np.mean(mfccs, axis=1)
            
            return mfcc_mean
            
        except Exception as e:
            print(f"Error extracting MFCCs from {audio_path}: {e}")
            return None
    
    def extract_chunks(self, audio_path: str, 
                      chunk_duration: float = 5.0,
                      overlap: float = 2.5) -> List[Tuple[float, float, np.ndarray]]:
        """Extract MFCC vectors from overlapping audio chunks
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds (default 5.0)
            overlap: Overlap between chunks in seconds (default 2.5)
            
        Returns:
            List of (start_time, end_time, embedding) tuples
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate hop size
            hop = chunk_duration - overlap
            chunk_samples = int(chunk_duration * sr)
            hop_samples = int(hop * sr)
            
            chunks = []
            start_sample = 0
            
            while start_sample < len(y):
                end_sample = min(start_sample + chunk_samples, len(y))
                
                # Extract chunk
                chunk_audio = y[start_sample:end_sample]
                
                # Skip if chunk too short
                if len(chunk_audio) < sr * 1.0:  # Minimum 1 second
                    break
                
                # Extract MFCCs for this chunk
                mfccs = librosa.feature.mfcc(y=chunk_audio, sr=sr, n_mfcc=self.n_mfcc)
                mfcc_mean = np.mean(mfccs, axis=1)
                
                # Calculate time boundaries
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                chunks.append((start_time, end_time, mfcc_mean))
                
                start_sample += hop_samples
            
            return chunks
            
        except Exception as e:
            print(f"Error extracting chunked MFCCs from {audio_path}: {e}")
            return []


class EmbeddingMatcher:
    """Similarity search engine for audio matching"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors
        
        Args:
            vec1, vec2: numpy arrays of same shape
            
        Returns:
            Cosine similarity score (0 to 1, higher = more similar)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def search_similar(query_embedding: np.ndarray,
                      db_path: str,
                      model_name: str,
                      top_k: int = 5) -> List[Dict]:
        """Find most similar audio chunks to query
        
        Args:
            query_embedding: Query audio embedding vector
            db_path: Path to SQLite database
            model_name: Model name to filter embeddings
            top_k: Number of top matches to return
            
        Returns:
            List of dicts with match info (song_id, similarity, chunk times)
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all embeddings for this model
        cursor.execute("""
            SELECT id, song_id, chunk_start, chunk_end, embedding
            FROM track_embeddings
            WHERE model_name = ?
        """, (model_name,))
        
        results = []
        for row in cursor.fetchall():
            embedding_id, song_id, chunk_start, chunk_end, embedding_blob = row
            
            # Deserialize embedding
            embedding = pickle.loads(embedding_blob)
            
            # Calculate similarity
            similarity = EmbeddingMatcher.cosine_similarity(query_embedding, embedding)
            
            results.append({
                'embedding_id': embedding_id,
                'song_id': song_id,
                'chunk_start': chunk_start,
                'chunk_end': chunk_end,
                'similarity': similarity
            })
        
        conn.close()
        
        # Sort by similarity (descending) and return top K
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]


def process_song_library(mp3_folder: str, db_path: str = "songs.db",
                        use_chunks: bool = True,
                        chunk_duration: float = 5.0,
                        overlap: float = 2.5):
    """Process all songs in library and store embeddings
    
    Args:
        mp3_folder: Path to folder containing MP3 files
        db_path: Path to SQLite database
        use_chunks: If True, extract chunks; if False, single vector per song
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
    """
    # Initialize
    embeddings_db = AudioEmbeddingsDB(db_path)
    extractor = MFCCExtractor()
    
    # Get all songs from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_path FROM songs WHERE file_path IS NOT NULL")
    songs = cursor.fetchall()
    
    print(f"\n🎵 Processing {len(songs)} songs...")
    print(f"   Mode: {'Chunked' if use_chunks else 'Full song'}")
    if use_chunks:
        print(f"   Chunk size: {chunk_duration}s with {overlap}s overlap")
    print(f"   Model: {extractor.model_name}\n")
    
    total_embeddings = 0
    
    for song_id, file_path in songs:
        if not file_path or not Path(file_path).exists():
            print(f"⚠️  Skipping song ID {song_id}: file not found")
            continue
        
        print(f"Processing: {Path(file_path).name}")
        
        if use_chunks:
            # Extract chunks
            chunks = extractor.extract_chunks(file_path, chunk_duration, overlap)
            
            if chunks:
                print(f"  → {len(chunks)} chunks extracted")
                
                # Store each chunk
                for start_time, end_time, embedding in chunks:
                    embedding_blob = pickle.dumps(embedding)
                    
                    cursor.execute("""
                        INSERT INTO track_embeddings 
                        (song_id, chunk_start, chunk_end, embedding, model_name, embedding_dim)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (song_id, start_time, end_time, embedding_blob, 
                          extractor.model_name, len(embedding)))
                    
                    total_embeddings += 1
        else:
            # Extract full song
            embedding = extractor.extract_full_song(file_path)
            
            if embedding is not None:
                print(f"  → Full song embedding extracted (dim={len(embedding)})")
                
                embedding_blob = pickle.dumps(embedding)
                
                cursor.execute("""
                    INSERT INTO track_embeddings 
                    (song_id, chunk_start, chunk_end, embedding, model_name, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (song_id, 0.0, -1.0, embedding_blob,  # -1 = full song
                      extractor.model_name, len(embedding)))
                
                total_embeddings += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Done! Stored {total_embeddings} embeddings in database")


if __name__ == "__main__":
    # Example usage
    mp3_folder = "/Users/juigupte/Desktop/Learning/music/mp3"
    db_path = "/Users/juigupte/Desktop/Learning/recsys-foundations/songs.db"
    
    print("=" * 60)
    print("Audio Embeddings Database - MFCC Baseline")
    print("=" * 60)
    
    # Process library with chunking
    process_song_library(mp3_folder, db_path, use_chunks=True)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Test matching: python test_audio_matching.py")
    print("  2. Evaluate accuracy on your 17-song library")
    print("  3. Then upgrade to pre-trained models!")
    print("=" * 60)
