"""
Audio Matching Evaluator - Test Your Mini Shazam
================================================

This script tests your audio matching system by:
1. Taking random snippets from your 17 songs
2. Querying the embeddings database
3. Measuring accuracy (does it find the correct song?)

This validates your Phase 1 MFCC baseline before adding complex models.
"""

import sqlite3
import numpy as np
import librosa
import random
from pathlib import Path
from typing import Dict, List, Tuple
from audio_embeddings import MFCCExtractor, EmbeddingMatcher
from melspec_embeddings import MelSpectrogramExtractor
import tensorflow as tf
from tensorflow import keras


def extract_random_snippet(audio_path: str, 
                           snippet_duration: float = 5.0) -> Tuple[np.ndarray, float, float]:
    """Extract a random snippet from an audio file
    
    Args:
        audio_path: Path to audio file
        snippet_duration: Duration of snippet in seconds
        
    Returns:
        (audio_data, start_time, end_time) tuple
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Pick random start time (avoid edges)
    max_start = max(0, duration - snippet_duration - 1.0)
    start_time = random.uniform(1.0, max_start)
    
    # Extract snippet
    start_sample = int(start_time * sr)
    end_sample = int((start_time + snippet_duration) * sr)
    snippet = y[start_sample:end_sample]
    
    return snippet, start_time, start_time + snippet_duration


def test_single_query(audio_path: str, song_id: int, 
                      db_path: str, model_name: str,
                      snippet_duration: float = 5.0,
                      top_k: int = 5) -> Dict:
    """Test a single query against the database
    
    Args:
        audio_path: Path to audio file
        song_id: Expected song ID
        db_path: Path to database
        model_name: Model name to use ('mfcc_20', 'melspec_128', or 'cnn_64')
        snippet_duration: Duration of test snippet
        top_k: Number of results to retrieve
        
    Returns:
        Dict with test results
    """
    # Extract random snippet
    snippet, start_time, end_time = extract_random_snippet(audio_path, snippet_duration)
    
    # Extract features from snippet based on model
    if model_name.startswith('mfcc'):
        extractor = MFCCExtractor()
        mfccs = librosa.feature.mfcc(y=snippet, sr=extractor.sr, n_mfcc=extractor.n_mfcc)
        query_embedding = np.mean(mfccs, axis=1)
    elif model_name.startswith('melspec'):
        extractor = MelSpectrogramExtractor()
        mel_spec = librosa.feature.melspectrogram(y=snippet, sr=extractor.sr, n_mels=extractor.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        query_embedding = np.mean(mel_spec_db, axis=1)
    elif model_name.startswith('cnn'):
        # Load CNN model and extract embedding
        model_path = "track_classifier.keras"
        try:
            model = keras.models.load_model(model_path)
            embedding_layer = model.get_layer('embeddings')
            embedding_model = keras.Model(inputs=model.input, outputs=embedding_layer.output)
            
            # Prepare mel-spectrogram in same format as training
            hop_length = 512
            mel_spec = librosa.feature.melspectrogram(y=snippet, sr=22050, n_mels=128, hop_length=hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_input = mel_spec_db.T  # (time, mels)
            
            # Pad or trim to expected length
            target_frames = 215  # From training
            if mel_input.shape[0] < target_frames:
                pad_width = ((0, target_frames - mel_input.shape[0]), (0, 0))
                mel_input = np.pad(mel_input, pad_width, mode='constant')
            else:
                mel_input = mel_input[:target_frames, :]
            
            # Extract embedding
            mel_input_batch = np.expand_dims(mel_input, axis=0)
            query_embedding = embedding_model.predict(mel_input_batch, verbose=0)[0]
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            return None
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Search database
    results = EmbeddingMatcher.search_similar(
        query_embedding, db_path, model_name, top_k
    )
    
    # Check if correct song is in results
    correct_in_results = any(r['song_id'] == song_id for r in results)
    top_match = results[0] if results else None
    rank = None
    
    if correct_in_results:
        rank = next(i + 1 for i, r in enumerate(results) if r['song_id'] == song_id)
    
    return {
        'audio_path': audio_path,
        'song_id': song_id,
        'snippet_time': (start_time, end_time),
        'correct_in_results': correct_in_results,
        'rank': rank,
        'top_match': top_match,
        'all_results': results
    }


def run_evaluation(db_path: str = "songs.db",
                  model_name: str = "mfcc_20",
                  num_tests: int = 20,
                  snippet_duration: float = 5.0,
                  top_k: int = 5):
    """Run full evaluation on the song library
    
    Args:
        db_path: Path to database
        model_name: Model name to test
        num_tests: Number of random tests to run
        snippet_duration: Duration of test snippets
        top_k: Number of results to consider
    """
    # Get all songs from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, file_path, title, artist 
        FROM songs 
        WHERE file_path IS NOT NULL
    """)
    songs = cursor.fetchall()
    conn.close()
    
    # Filter songs with valid paths
    valid_songs = [(sid, fp, t, a) for sid, fp, t, a in songs 
                   if fp and Path(fp).exists()]
    
    print("=" * 70)
    print(f"🎯 Audio Matching Evaluation")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Songs in library: {len(valid_songs)}")
    print(f"Test snippets: {num_tests}")
    print(f"Snippet duration: {snippet_duration}s")
    print(f"Top-K results: {top_k}")
    print("=" * 70)
    print()
    
    # Run tests
    results = []
    for i in range(num_tests):
        # Pick random song
        song_id, file_path, title, artist = random.choice(valid_songs)
        
        print(f"Test {i+1}/{num_tests}: {title or Path(file_path).stem}")
        
        try:
            result = test_single_query(
                file_path, song_id, db_path, model_name,
                snippet_duration, top_k
            )
            results.append(result)
            
            if result['correct_in_results']:
                print(f"  ✓ MATCH (rank #{result['rank']}, similarity: {result['top_match']['similarity']:.3f})")
            else:
                print(f"  ✗ MISS (top match: {result['top_match']['song_id'] if result['top_match'] else 'none'})")
        
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
        
        print()
    
    # Calculate metrics
    print("=" * 70)
    print("📊 Results")
    print("=" * 70)
    
    total_tests = len(results)
    if total_tests == 0:
        print("No successful tests!")
        return
    
    # Top-1 accuracy (is correct song the #1 match?)
    top1_correct = sum(1 for r in results if r['rank'] == 1)
    top1_accuracy = top1_correct / total_tests * 100
    
    # Top-K accuracy (is correct song in top K?)
    topk_correct = sum(1 for r in results if r['correct_in_results'])
    topk_accuracy = topk_correct / total_tests * 100
    
    # Average rank of correct song
    ranks = [r['rank'] for r in results if r['rank'] is not None]
    avg_rank = np.mean(ranks) if ranks else 0
    
    # Average similarity scores
    top_similarities = [r['top_match']['similarity'] for r in results if r['top_match']]
    avg_similarity = np.mean(top_similarities) if top_similarities else 0
    
    print(f"Total tests: {total_tests}")
    print(f"\nTop-1 Accuracy: {top1_accuracy:.1f}% ({top1_correct}/{total_tests})")
    print(f"Top-{top_k} Accuracy: {topk_accuracy:.1f}% ({topk_correct}/{total_tests})")
    print(f"\nAverage rank: {avg_rank:.2f}")
    print(f"Average similarity: {avg_similarity:.3f}")
    
    print("\n" + "=" * 70)
    print("💡 Interpretation:")
    print("=" * 70)
    
    if top1_accuracy >= 80:
        print("🎉 Excellent! Your MFCC baseline is working well.")
        print("   → Ready to experiment with pre-trained models")
    elif top1_accuracy >= 60:
        print("👍 Good start! The baseline captures some audio characteristics.")
        print("   → Pre-trained models should improve this significantly")
    elif top1_accuracy >= 40:
        print("🤔 Moderate performance. MFCCs capture basic patterns.")
        print("   → Neural embeddings will help with harder cases")
    else:
        print("⚠️  Low accuracy. Possible issues:")
        print("   1. Not enough embeddings in database (run audio_embeddings.py)")
        print("   2. Songs too similar (test with more diverse library)")
        print("   3. MFCC parameters need tuning")
    
    print("\n" + "=" * 70)


def query_specific_song(audio_path: str, 
                       db_path: str = "songs.db",
                       model_name: str = "mfcc_20",
                       snippet_start: float = 10.0,
                       snippet_duration: float = 5.0,
                       top_k: int = 5):
    """Query with a specific audio clip (for manual testing)
    
    Args:
        audio_path: Path to audio file
        db_path: Path to database
        model_name: Model to use
        snippet_start: Start time of snippet (seconds)
        snippet_duration: Duration of snippet
        top_k: Number of results to show
    """
    print(f"\n🔍 Querying: {Path(audio_path).name}")
    print(f"   Snippet: {snippet_start}s to {snippet_start + snippet_duration}s")
    print()
    
    # Load and extract snippet
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    start_sample = int(snippet_start * sr)
    end_sample = int((snippet_start + snippet_duration) * sr)
    snippet = y[start_sample:end_sample]
    
    # Extract features
    extractor = MFCCExtractor()
    mfccs = librosa.feature.mfcc(y=snippet, sr=sr, n_mfcc=extractor.n_mfcc)
    query_embedding = np.mean(mfccs, axis=1)
    
    # Search
    results = EmbeddingMatcher.search_similar(
        query_embedding, db_path, model_name, top_k
    )
    
    # Get song info
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Top {top_k} matches:")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        cursor.execute("""
            SELECT title, artist FROM songs WHERE id = ?
        """, (result['song_id'],))
        
        row = cursor.fetchone()
        title, artist = row if row else ('Unknown', 'Unknown')
        
        print(f"{i}. {title} - {artist}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Chunk: {result['chunk_start']:.1f}s - {result['chunk_end']:.1f}s")
        print()
    
    conn.close()


if __name__ == "__main__":
    import sys
    
    db_path = "/Users/juigupte/Desktop/Learning/recsys-foundations/songs.db"
    
    if len(sys.argv) > 1:
        # Manual query mode
        audio_path = sys.argv[1]
        snippet_start = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        
        query_specific_song(audio_path, db_path, snippet_start=snippet_start)
    else:
        # Evaluation mode
        run_evaluation(db_path, num_tests=20)
