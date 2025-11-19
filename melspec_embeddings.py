"""
Mel-Spectrogram Embeddings Extractor - Phase 2
===============================================

Enhanced audio embeddings using mel-spectrograms.
More expressive than MFCCs - captures full spectral information.

Comparison to MFCC:
- MFCC: 20 coefficients (compressed spectral envelope)
- Mel-Spectrogram: 128 mel bins (full spectral detail)
- Expected: Better discrimination for similar-sounding songs
"""

import sqlite3
import numpy as np
import pickle
import librosa
from pathlib import Path
from typing import List, Tuple, Optional


class MelSpectrogramExtractor:
    """Phase 2: Enhanced spectral features
    
    Mel-spectrograms capture more spectral detail than MFCCs.
    This is a step between hand-crafted (MFCC) and deep learned features.
    Many neural audio models use mel-spectrograms as input.
    """
    
    def __init__(self, n_mels: int = 128, sr: int = 22050):
        """Initialize mel-spectrogram extractor
        
        Args:
            n_mels: Number of mel frequency bins (default 128)
            sr: Sample rate for audio loading (default 22050 Hz)
        """
        self.n_mels = n_mels
        self.sr = sr
        self.model_name = f"melspec_{n_mels}"
    
    def extract_full_song(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract single mel-spectrogram vector for entire song (mean across time)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            numpy array of shape (n_mels,) or None if extraction fails
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Average across time → single vector per song
            mel_mean = np.mean(mel_spec_db, axis=1)
            
            return mel_mean
            
        except Exception as e:
            print(f"Error extracting mel-spectrogram from {audio_path}: {e}")
            return None
    
    def extract_chunks(self, audio_path: str,
                      chunk_duration: float = 5.0,
                      overlap: float = 2.5) -> List[Tuple[float, float, np.ndarray]]:
        """Extract mel-spectrogram vectors from overlapping audio chunks
        
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
                
                # Extract mel-spectrogram for this chunk
                mel_spec = librosa.feature.melspectrogram(
                    y=chunk_audio, sr=sr, n_mels=self.n_mels
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_mean = np.mean(mel_spec_db, axis=1)
                
                # Calculate time boundaries
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                chunks.append((start_time, end_time, mel_mean))
                
                start_sample += hop_samples
            
            return chunks
            
        except Exception as e:
            print(f"Error extracting chunked mel-spectrogram from {audio_path}: {e}")
            return []


def process_with_melspec(mp3_folder: str,
                        db_path: str = "songs.db",
                        use_chunks: bool = True,
                        chunk_duration: float = 5.0,
                        overlap: float = 2.5,
                        n_mels: int = 128):
    """Process all songs in library with mel-spectrogram embeddings
    
    Args:
        mp3_folder: Path to folder containing MP3 files
        db_path: Path to SQLite database
        use_chunks: If True, extract chunks; if False, single vector per song
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
        n_mels: Number of mel frequency bins
    """
    from audio_embeddings import AudioEmbeddingsDB
    
    # Initialize
    AudioEmbeddingsDB(db_path)  # Ensure table exists
    extractor = MelSpectrogramExtractor(n_mels=n_mels)
    
    # Get all songs from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_path FROM songs WHERE file_path IS NOT NULL")
    songs = cursor.fetchall()
    
    print(f"\n🎵 Processing {len(songs)} songs with Mel-Spectrograms...")
    print(f"   Mode: {'Chunked' if use_chunks else 'Full song'}")
    if use_chunks:
        print(f"   Chunk size: {chunk_duration}s with {overlap}s overlap")
    print(f"   Model: {extractor.model_name}")
    print(f"   Embedding dim: {n_mels}\n")
    
    total_embeddings = 0
    
    for idx, (song_id, file_path) in enumerate(songs, 1):
        if not file_path or not Path(file_path).exists():
            print(f"⚠️  Skipping song ID {song_id}: file not found")
            continue
        
        print(f"[{idx}/{len(songs)}] Processing: {Path(file_path).name}")
        
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
    
    print(f"\n✓ Done! Stored {total_embeddings} mel-spectrogram embeddings in database")


if __name__ == "__main__":
    mp3_folder = "/Users/juigupte/Desktop/Learning/music/mp3"
    db_path = "/Users/juigupte/Desktop/Learning/recsys-foundations/songs.db"
    
    print("=" * 60)
    print("Mel-Spectrogram Embeddings Extraction - Phase 2")
    print("=" * 60)
    
    # Process library with mel-spectrograms
    process_with_melspec(mp3_folder, db_path, use_chunks=True, n_mels=128)
    
    print("\n" * 60)
    print("Next: Run evaluation to compare MFCC vs Mel-Spec!")
    print("  python test_audio_matching.py")
    print("=" * 60)
