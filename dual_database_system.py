"""
Dual-Database Music System
===========================

Combines two databases for a complete music service:

1. **ID Database** (Identification - like Shazam)
   - Your 17 songs (or your owned catalog)
   - Purpose: Exact matching with high confidence
   - Threshold-based matching (distance < 0.3 = exact match)
   
2. **Recommendation Database** (Discovery - like Spotify)
   - Million Song Dataset (or 100M songs)
   - Purpose: Similar songs, playlists, "sounds like"
   - No exact matching needed, just similarity

Flow:
-----
Query → ID DB → Match found? → Return song + similar songs
                 ↓ No match
                 → Recommendation DB → Return "sounds like" songs
"""

import numpy as np
import sqlite3
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import librosa
from tensorflow import keras


@dataclass
class MatchResult:
    """Result from ID database exact matching."""
    song_id: int
    title: str
    artist: str
    confidence: float  # 0-1, higher = more confident
    distance: float    # Embedding distance
    match_type: str    # "exact" or "possible"


@dataclass
class RecommendationResult:
    """Result from recommendation database."""
    song_id: str  # Can be MSD track ID
    title: str
    artist: str
    similarity: float  # 0-1, higher = more similar
    metadata: Dict     # Genre, year, tags, etc.


class DualDatabaseSystem:
    """
    Two-tier music system with ID DB and Recommendation DB.
    """
    
    def __init__(self, 
                 id_db_path: str = 'songs.db',
                 rec_db_path: str = 'recommendations.db',
                 model_path: str = 'track_classifier.keras',
                 exact_match_threshold: float = 0.3):
        """
        Args:
            id_db_path: Path to identification database (your songs)
            rec_db_path: Path to recommendation database (Million Song Dataset)
            model_path: Path to trained CNN model for embeddings
            exact_match_threshold: Distance threshold for exact matches
        """
        self.id_db_path = id_db_path
        self.rec_db_path = rec_db_path
        self.model_path = model_path
        self.exact_match_threshold = exact_match_threshold
        
        # Load embedding model
        print("Loading embedding model...")
        model = keras.models.load_model(model_path)
        embedding_layer = model.get_layer('embeddings')
        self.embedding_model = keras.Model(
            inputs=model.input,
            outputs=embedding_layer.output
        )
        self.embedding_dim = self.embedding_model.output_shape[-1]
        print(f"✓ Loaded {self.embedding_dim}-dim embedding model")
    
    def compute_embedding(self, audio_path: str, start_time: float = 0, 
                         duration: float = 5.0) -> np.ndarray:
        """
        Compute embedding for audio query.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Embedding vector (64-dim)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, offset=start_time, 
                            duration=duration)
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Ensure consistent shape (215 time frames)
        target_frames = 215
        if mel_db.shape[1] < target_frames:
            pad_width = target_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_db.shape[1] > target_frames:
            mel_db = mel_db[:, :target_frames]
        
        # Transpose to (time, freq)
        mel_input = mel_db.T
        
        # Add batch dimension
        mel_input = np.expand_dims(mel_input, axis=0)
        
        # Extract embedding
        embedding = self.embedding_model.predict(mel_input, verbose=0)[0]
        
        return embedding
    
    def query_id_database(self, embedding: np.ndarray, 
                         top_k: int = 5) -> List[MatchResult]:
        """
        Query ID database for exact matches.
        
        Args:
            embedding: Query embedding (64-dim)
            top_k: Number of candidates to return
            
        Returns:
            List of match results, sorted by confidence
        """
        conn = sqlite3.connect(self.id_db_path)
        cursor = conn.cursor()
        
        # Get all embeddings from ID database (your owned songs)
        cursor.execute("""
            SELECT e.id, e.song_id, e.embedding, s.title, s.artist
            FROM track_embeddings e
            JOIN songs s ON e.song_id = s.id
            WHERE e.model_name = 'cnn_64'
        """)
        
        results = []
        for emb_id, song_id, emb_blob, title, artist in cursor.fetchall():
            # Deserialize embedding (stored as pickle)
            try:
                stored_emb = pickle.loads(emb_blob)
            except:
                # Fallback to numpy frombuffer
                stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
            
            # Compute cosine similarity
            similarity = np.dot(embedding, stored_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_emb)
            )
            distance = 1 - similarity
            
            # Determine match type
            if distance < self.exact_match_threshold:
                match_type = "exact"
                confidence = 1 - (distance / self.exact_match_threshold)
            else:
                match_type = "possible"
                confidence = max(0, 1 - distance)
            
            results.append(MatchResult(
                song_id=song_id,
                title=title,
                artist=artist,
                confidence=confidence,
                distance=distance,
                match_type=match_type
            ))
        
        conn.close()
        
        # Sort by distance (lower = better)
        results.sort(key=lambda x: x.distance)
        
        return results[:top_k]
    
    def query_recommendation_database(self, embedding: np.ndarray,
                                     top_k: int = 20,
                                     filters: Optional[Dict] = None) -> List[RecommendationResult]:
        """
        Query recommendation database for similar songs.
        
        Args:
            embedding: Query embedding (64-dim)
            top_k: Number of recommendations to return
            filters: Optional filters (genre, year, artist, etc.)
            
        Returns:
            List of recommendation results, sorted by similarity
        """
        # Check if recommendation DB exists
        import os
        if not os.path.exists(self.rec_db_path):
            print(f"⚠️  Recommendation DB not found: {self.rec_db_path}")
            print("   Run: python build_recommendation_db.py")
            return []
        
        conn = sqlite3.connect(self.rec_db_path)
        cursor = conn.cursor()
        
        # Build query with optional filters
        query = """
            SELECT r.track_id, r.embedding, r.title, r.artist, 
                   r.genre, r.year, r.tags
            FROM recommendation_tracks r
            WHERE 1=1
        """
        params = []
        
        if filters:
            if 'genre' in filters:
                query += " AND r.genre = ?"
                params.append(filters['genre'])
            if 'year_min' in filters:
                query += " AND r.year >= ?"
                params.append(filters['year_min'])
            if 'year_max' in filters:
                query += " AND r.year <= ?"
                params.append(filters['year_max'])
        
        cursor.execute(query, params)
        
        results = []
        for track_id, emb_blob, title, artist, genre, year, tags in cursor.fetchall():
            # Deserialize embedding
            try:
                stored_emb = pickle.loads(emb_blob)
            except:
                # Fallback for raw bytes
                stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
            
            # Ensure it's a numpy array
            if not isinstance(stored_emb, np.ndarray):
                continue
            
            # Check dimension mismatch
            if len(stored_emb) != len(embedding):
                # Skip - embeddings have different dimensions
                # This happens when Rec DB uses MSD features (12-dim)
                # but ID model uses CNN embeddings (64-dim)
                continue
            
            # Compute cosine similarity
            similarity = np.dot(embedding, stored_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_emb)
            )
            
            results.append(RecommendationResult(
                song_id=track_id,
                title=title,
                artist=artist,
                similarity=similarity,
                metadata={
                    'genre': genre,
                    'year': year,
                    'tags': tags
                }
            ))
        
        conn.close()
        
        if len(results) == 0:
            print(f"   ⚠️  No compatible embeddings found (dimension mismatch: "
                  f"query is {len(embedding)}-dim)")
        
        # Sort by similarity (higher = better)
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        return results[:top_k]
    
    def query(self, audio_path: str, start_time: float = 0,
             duration: float = 5.0) -> Dict:
        """
        Complete query flow: ID DB → Recommendation DB.
        
        Args:
            audio_path: Path to query audio
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Dictionary with:
            - exact_match: MatchResult or None
            - similar_songs: List[RecommendationResult]
            - recommendation_type: "with_match" or "no_match"
        """
        print(f"\n🎵 Processing query: {audio_path}")
        
        # Step 1: Compute embedding
        print("   Computing embedding...")
        embedding = self.compute_embedding(audio_path, start_time, duration)
        
        # Step 2: Query ID database
        print("   Querying ID database...")
        id_results = self.query_id_database(embedding, top_k=5)
        
        exact_match = None
        if id_results and id_results[0].match_type == "exact":
            exact_match = id_results[0]
            print(f"   ✓ EXACT MATCH: {exact_match.title} by {exact_match.artist}")
            print(f"     Confidence: {exact_match.confidence:.1%}")
            recommendation_type = "with_match"
        else:
            if id_results:
                print(f"   ✗ No exact match (best: {id_results[0].title}, "
                      f"distance: {id_results[0].distance:.3f})")
            else:
                print(f"   ✗ No exact match (ID database is empty or no embeddings)")
            recommendation_type = "no_match"
        
        # Step 3: Query recommendation database
        print("   Querying recommendation database...")
        similar_songs = self.query_recommendation_database(
            embedding, 
            top_k=10 if exact_match else 20
        )
        
        if similar_songs:
            print(f"   ✓ Found {len(similar_songs)} similar songs")
        else:
            print("   ⚠️  Recommendation database empty")
        
        return {
            'exact_match': exact_match,
            'similar_songs': similar_songs,
            'recommendation_type': recommendation_type,
            'all_id_results': id_results
        }
    
    def format_results(self, query_result: Dict) -> str:
        """Format query results for display."""
        output = []
        output.append("=" * 80)
        
        if query_result['exact_match']:
            match = query_result['exact_match']
            output.append("🎯 EXACT MATCH FOUND")
            output.append("=" * 80)
            output.append(f"Title:      {match.title}")
            output.append(f"Artist:     {match.artist}")
            output.append(f"Confidence: {match.confidence:.1%}")
            output.append(f"Distance:   {match.distance:.3f}")
            output.append("")
        else:
            output.append("❌ NO EXACT MATCH")
            output.append("=" * 80)
            output.append("This song is not in your ID database.")
            output.append("")
        
        similar = query_result['similar_songs']
        if similar:
            output.append("🎵 SIMILAR SONGS (Sounds Like)")
            output.append("=" * 80)
            for i, song in enumerate(similar[:10], 1):
                output.append(f"{i:2d}. {song.title} - {song.artist}")
                output.append(f"    Similarity: {song.similarity:.1%} | "
                            f"Genre: {song.metadata.get('genre', 'Unknown')}")
            output.append("")
        
        if query_result['recommendation_type'] == 'with_match':
            output.append("💡 Because you matched a song, here are similar tracks you might like!")
        else:
            output.append("💡 No exact match, but here are songs that sound similar!")
        
        output.append("=" * 80)
        return "\n".join(output)


# Example usage and testing
if __name__ == '__main__':
    import sys
    
    print("=" * 80)
    print("Dual-Database Music System")
    print("=" * 80)
    
    # Initialize system
    system = DualDatabaseSystem(
        id_db_path='songs.db',
        rec_db_path='recommendations.db',
        model_path='track_classifier.keras',
        exact_match_threshold=0.3
    )
    
    # Example query
    if len(sys.argv) > 1:
        query_file = sys.argv[1]
    else:
        # Default test query
        query_file = '/Users/juigupte/Desktop/Learning/music/mp3/Ed Sheeran - Shape of You (Official Music Video).mp3'
    
    results = system.query(query_file, start_time=30.0, duration=5.0)
    
    # Display results
    print("\n" + system.format_results(results))
    
    # Show what's available
    print("\n📊 System Status:")
    print(f"   ID Database: {system.id_db_path}")
    
    import os
    if os.path.exists(system.rec_db_path):
        conn = sqlite3.connect(system.rec_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM recommendation_tracks")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"   Recommendation DB: {count:,} songs")
    else:
        print(f"   Recommendation DB: NOT FOUND")
        print(f"   → Run: python build_recommendation_db.py")
