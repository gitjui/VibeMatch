"""
CNN Embedding Extractor
========================

Extract embeddings from trained CNN classifier and store in database.
Uses penultimate layer (before softmax) as learned audio features.

Usage:
    python cnn_extract_embeddings.py                           # baseline model
    python cnn_extract_embeddings.py track_classifier_augmented.keras cnn_aug_64
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import sqlite3
import pickle
import librosa
import sys
from pathlib import Path
from cnn_classifier import CNNTrackClassifier


def extract_and_store_cnn_embeddings(db_path: str = "songs.db",
                                    model_path: str = "track_classifier.keras",
                                    model_name: str = None,
                                    chunk_duration: float = 5.0,
                                    overlap: float = 2.5):
    """Extract CNN embeddings for all songs and store in database
    
    Args:
        db_path: Path to SQLite database
        model_path: Path to trained model
        model_name: Name to store in DB (e.g. 'cnn_64' or 'cnn_aug_64')
        chunk_duration: Chunk duration in seconds
        overlap: Overlap between chunks
    """
    from audio_embeddings import AudioEmbeddingsDB
    
    # Load trained model
    print(f"Loading trained CNN model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Create embedding extractor
    embedding_layer = model.get_layer('embeddings')
    embedding_model = keras.Model(
        inputs=model.input,
        outputs=embedding_layer.output
    )
    
    embedding_dim = embedding_model.output_shape[-1]
    
    # Use provided model_name or auto-generate
    if model_name is None:
        model_name = f"cnn_{embedding_dim}"
    
    print(f"✓ Embedding model ready:")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Model name: {model_name}\n")
    
    # Ensure embeddings table exists
    AudioEmbeddingsDB(db_path)
    
    # Get all songs
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_path FROM songs WHERE file_path IS NOT NULL")
    songs = cursor.fetchall()
    
    print(f"🎵 Processing {len(songs)} songs...")
    print(f"   Chunk size: {chunk_duration}s with {overlap}s overlap\n")
    
    total_embeddings = 0
    hop_length = 512
    chunk_frames = int(chunk_duration * 22050 / hop_length)
    
    for idx, (song_id, file_path) in enumerate(songs, 1):
        if not file_path or not Path(file_path).exists():
            print(f"⚠️  Skipping song ID {song_id}: file not found")
            continue
        
        print(f"[{idx}/{len(songs)}] Processing: {Path(file_path).name}")
        
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=22050, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, hop_length=hop_length
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Split into overlapping chunks
            hop = chunk_duration - overlap
            hop_frames = int(hop * sr / hop_length)
            
            chunks_data = []
            chunk_times = []
            start_frame = 0
            
            while start_frame + chunk_frames <= mel_spec_db.shape[1]:
                end_frame = start_frame + chunk_frames
                
                chunk = mel_spec_db[:, start_frame:end_frame].T  # (time, mels)
                chunks_data.append(chunk)
                
                start_time = start_frame * hop_length / sr
                end_time = end_frame * hop_length / sr
                chunk_times.append((start_time, end_time))
                
                start_frame += hop_frames
            
            if not chunks_data:
                print(f"  ⚠️  No chunks extracted")
                continue
            
            # Extract embeddings in batch
            chunks_array = np.array(chunks_data)
            embeddings = embedding_model.predict(chunks_array, verbose=0)
            
            print(f"  → {len(embeddings)} chunks extracted")
            
            # Store each embedding
            for (start_time, end_time), embedding in zip(chunk_times, embeddings):
                embedding_blob = pickle.dumps(embedding)
                
                cursor.execute("""
                    INSERT INTO track_embeddings 
                    (song_id, chunk_start, chunk_end, embedding, model_name, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (song_id, start_time, end_time, embedding_blob, 
                      model_name, embedding_dim))
                
                total_embeddings += 1
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Done! Stored {total_embeddings} CNN embeddings in database")


if __name__ == "__main__":
    print("=" * 60)
    print("CNN Embedding Extraction")
    print("=" * 60)
    
    # Parse command line args
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        model_path = "track_classifier.keras"
        model_name = "cnn_64"
    
    print(f"Model: {model_path}")
    print(f"Name: {model_name}")
    print()
    
    db_path = "/Users/juigupte/Desktop/Learning/recsys-foundations/songs.db"
    
    extract_and_store_cnn_embeddings(db_path, model_path, model_name=model_name)
    
    print("\n" + "=" * 60)
    print("Next: Run evaluation to compare all models!")
    print("  python test_audio_matching.py")
    print("=" * 60)
