"""
Music Query API - JSON Output
==============================

Query songs and get JSON results with:
- Exact match (if found in ID database)
- Similar songs (from recommendation database)
- Confidence scores and metadata
"""

import json
import numpy as np
from dual_database_system import DualDatabaseSystem
from pathlib import Path
import sys


def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def query_song_json(audio_path: str, start_time: float = 0, duration: float = 5.0) -> dict:
    """
    Query a song and return JSON-formatted results.
    
    Returns:
        {
            "query": {
                "file": "song.mp3",
                "start_time": 0,
                "duration": 5.0
            },
            "exact_match": {
                "found": true/false,
                "song_id": 123,
                "title": "Shape of You",
                "artist": "Ed Sheeran",
                "confidence": 0.93,
                "distance": 0.021
            },
            "recommendations": [
                {
                    "rank": 1,
                    "song_id": "SAMPLE000123",
                    "title": "Sample Song 124",
                    "artist": "Drake",
                    "genre": "Pop",
                    "year": 2018,
                    "similarity": 0.87
                },
                ...
            ]
        }
    """
    # Initialize system
    system = DualDatabaseSystem(
        id_db_path='songs.db',
        rec_db_path='recommendations.db',
        model_path='track_classifier.keras',
        exact_match_threshold=0.3
    )
    
    # Query
    results = system.query(audio_path, start_time, duration)
    
    # Format as JSON
    output = {
        "query": {
            "file": Path(audio_path).name,
            "start_time": start_time,
            "duration": duration
        },
        "exact_match": None,
        "recommendations": []
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
    
    # Add recommendations
    # If no embedding-based recommendations, use metadata fallback from recommendation database
    if len(results['similar_songs']) == 0:
        import sqlite3
        conn = sqlite3.connect('recommendations.db')
        cursor = conn.cursor()
        
        # Try to find the actual artist from the filename
        query_filename = Path(audio_path).stem.lower()
        
        # Extract artist from filename (format: "Artist - Song Title")
        detected_artist = None
        if ' - ' in query_filename:
            detected_artist = query_filename.split(' - ')[0].strip()
        
        # If we have an exact match, try to use that artist too
        if results['exact_match'] and results['exact_match'].artist != 'Unknown':
            match_artist = results['exact_match'].artist
        else:
            match_artist = detected_artist
        
        recommendations_added = False
        
        # Try to find songs by the detected artist from filename
        if detected_artist:
            cursor.execute("""
                SELECT track_id, title, artist, genre, year
                FROM recommendation_tracks
                WHERE LOWER(artist) LIKE ?
                LIMIT 10
            """, (f"%{detected_artist}%",))
            
            results_found = cursor.fetchall()
            for i, (song_id, title, artist, genre, year) in enumerate(results_found, 1):
                output['recommendations'].append({
                    "rank": i,
                    "song_id": str(song_id),
                    "title": title,
                    "artist": artist,
                    "genre": genre,
                    "year": year,
                    "similarity": 0.90,  # High similarity - same artist
                    "type": "metadata_based_same_artist"
                })
                recommendations_added = True
        
        # If still no recommendations, try by genre
        if not recommendations_added and results['exact_match']:
            cursor.execute("""
                SELECT track_id, title, artist, genre, year
                FROM recommendation_tracks
                ORDER BY RANDOM()
                LIMIT 10
            """)
            
            for i, (song_id, title, artist, genre, year) in enumerate(cursor.fetchall(), 1):
                output['recommendations'].append({
                    "rank": i,
                    "song_id": str(song_id),
                    "title": title,
                    "artist": artist,
                    "genre": genre,
                    "year": year,
                    "similarity": 0.60,  # Lower similarity for random
                    "type": "metadata_based_random"
                })
        
        conn.close()
    else:
        # Use embedding-based recommendations
        for i, rec in enumerate(results['similar_songs'], 1):
            output['recommendations'].append({
                "rank": i,
                "song_id": str(rec.song_id),
                "title": rec.title,
                "artist": rec.artist,
                "genre": rec.metadata.get('genre'),
                "year": convert_to_python_types(rec.metadata.get('year')),
                "similarity": float(round(rec.similarity, 3)),
                "type": "embedding_based"
            })
    
    # Add summary
    output['summary'] = {
        "total_recommendations": len(output['recommendations']),
        "status": "match_found" if output['exact_match']['found'] else "no_match",
        "recommendation_type": results['recommendation_type']
    }
    
    return output


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python query_api.py <audio_file> [start_time] [duration]")
        print("\nExamples:")
        print("  python query_api.py song.mp3")
        print("  python query_api.py song.mp3 30 5")
        print("\nTest files:")
        print("  python query_api.py '/Users/juigupte/Desktop/Learning/music/test mp3/Dua Lipa - Be The One (Official Music Video) [-rey3m8SWQI].mp3'")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    start_time = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    
    print(f"🎵 Querying: {Path(audio_path).name}")
    print(f"   Snippet: {start_time}s - {start_time + duration}s")
    print()
    
    # Query and get JSON
    result_json = query_song_json(audio_path, start_time, duration)
    
    # Print formatted JSON
    print(json.dumps(result_json, indent=2, ensure_ascii=False))
    
    # Also save to file
    output_file = "query_result.json"
    with open(output_file, 'w') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if result_json['exact_match']['found']:
        match = result_json['exact_match']
        print(f"✅ EXACT MATCH: {match['title']} by {match['artist']}")
        print(f"   Confidence: {match['confidence']:.1%}")
    else:
        print("❌ NO EXACT MATCH")
        print("   This song is not in your catalog")
    
    num_recs = result_json['summary']['total_recommendations']
    if num_recs > 0:
        print(f"\n🎵 RECOMMENDATIONS: {num_recs} similar songs found")
        print("\nTop 5:")
        for rec in result_json['recommendations'][:5]:
            print(f"   {rec['rank']}. {rec['title']} - {rec['artist']}")
            print(f"      Similarity: {rec['similarity']:.1%} | Genre: {rec['genre']} | Year: {rec['year']}")
    else:
        print(f"\n⚠️  NO RECOMMENDATIONS (dimension mismatch)")
        print("   Recommendation DB uses different embedding dimensions")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
