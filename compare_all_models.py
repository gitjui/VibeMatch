"""
Compare All Models - Final Evaluation
=====================================

Test all three embedding models side-by-side:
1. MFCC (20-dim) - Hand-crafted baseline
2. Mel-Spectrogram (128-dim) - Enhanced features
3. CNN (64-dim) - Learned embeddings

Same evaluation protocol for fair comparison.
"""

from test_audio_matching import run_evaluation
import sqlite3


def compare_all_models(db_path: str = "songs.db", num_tests: int = 30):
    """Run evaluation on all available models
    
    Args:
        db_path: Path to database
        num_tests: Number of test snippets
    """
    # Check which models are available
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT model_name, COUNT(*) FROM track_embeddings GROUP BY model_name")
    available_models = cursor.fetchall()
    conn.close()
    
    print("=" * 70)
    print("FINAL COMPARISON: All Embedding Models")
    print("=" * 70)
    print("\nAvailable models:")
    for model_name, count in available_models:
        print(f"  - {model_name}: {count} embeddings")
    print()
    
    results = {}
    
    for model_name, _ in available_models:
        print("\n" + "=" * 70)
        print(f"Testing: {model_name.upper()}")
        print("=" * 70)
        print()
        
        # Run evaluation
        run_evaluation(
            db_path=db_path,
            model_name=model_name,
            num_tests=num_tests
        )
        
        print("\n")
    
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("Check results above to compare:")
    print("  - Top-1 Accuracy")
    print("  - Top-5 Accuracy")
    print("  - Average Rank")
    print("  - Average Similarity")
    print("\nExpected ranking:")
    print("  CNN (learned) > Mel-Spec (enhanced) > MFCC (baseline)")


if __name__ == "__main__":
    db_path = "/Users/juigupte/Desktop/Learning/recsys-foundations/songs.db"
    
    compare_all_models(db_path, num_tests=30)
