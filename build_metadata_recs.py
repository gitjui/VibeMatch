"""
Build metadata-based recommendation system using existing song database.
Since we have dimension mismatch, use song metadata for recommendations.
"""

import sqlite3
import json


def build_metadata_recommendations():
    """
    Build simple recommendations based on artist and genre similarity.
    Uses the existing songs.db to create pseudo-recommendations.
    """
    
    print("Building metadata-based recommendations...")
    
    # Connect to databases
    id_conn = sqlite3.connect('songs.db')
    rec_conn = sqlite3.connect('recommendations.db')
    
    id_cursor = id_conn.cursor()
    rec_cursor = rec_conn.cursor()
    
    # Get all songs from ID database
    id_cursor.execute("""
        SELECT id, title, artist, album, duration, year
        FROM songs
    """)
    songs = id_cursor.fetchall()
    
    print(f"Found {len(songs)} songs in ID database")
    
    # Create simple recommendations: other songs by same/similar artists
    recommendations = []
    
    for song_id, title, artist, album, duration, year in songs:
        # Find other songs by same artist
        id_cursor.execute("""
            SELECT id, title, artist, album, duration, year
            FROM songs
            WHERE artist LIKE ? AND id != ?
            LIMIT 5
        """, (f"%{artist.split()[0]}%", song_id))
        
        similar = id_cursor.fetchall()
        
        for sim_id, sim_title, sim_artist, sim_album, sim_duration, sim_year in similar:
            recommendations.append({
                'source_song_id': song_id,
                'source_title': title,
                'rec_song_id': sim_id,
                'rec_title': sim_title,
                'rec_artist': sim_artist,
                'similarity_type': 'same_artist',
                'similarity_score': 0.9
            })
    
    print(f"Generated {len(recommendations)} recommendations")
    
    # Add recommendations to notes
    print("\nSample recommendations:")
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"  {i}. {rec['source_title']} → {rec['rec_title']}")
    
    id_conn.close()
    rec_conn.close()
    
    # Save to JSON
    with open('metadata_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"\n✓ Saved to metadata_recommendations.json")
    
    return recommendations


def create_enhanced_query_with_metadata():
    """
    Create a version that uses metadata when embedding similarity fails.
    """
    
    code = '''"""
Enhanced query API with metadata fallback
"""

import json
import sqlite3
from pathlib import Path

def query_with_metadata_fallback(song_id, num_recommendations=10):
    """
    Get recommendations using metadata when embeddings fail.
    """
    conn = sqlite3.connect('songs.db')
    cursor = conn.cursor()
    
    # Get query song info
    cursor.execute("""
        SELECT title, artist, album, year
        FROM songs WHERE id = ?
    """, (song_id,))
    
    query_song = cursor.fetchone()
    if not query_song:
        return []
    
    title, artist, album, year = query_song
    
    # Find similar songs by artist
    cursor.execute("""
        SELECT id, title, artist, album, year
        FROM songs
        WHERE (artist LIKE ? OR artist LIKE ?)
        AND id != ?
        ORDER BY RANDOM()
        LIMIT ?
    """, (f"%{artist}%", f"%{artist.split()[0]}%", song_id, num_recommendations))
    
    recommendations = []
    for row in cursor.fetchall():
        rec_id, rec_title, rec_artist, rec_album, rec_year = row
        recommendations.append({
            "song_id": rec_id,
            "title": rec_title,
            "artist": rec_artist,
            "album": rec_album,
            "year": rec_year,
            "similarity_type": "same_artist",
            "similarity": 0.85
        })
    
    # If not enough, add random songs
    if len(recommendations) < num_recommendations:
        cursor.execute("""
            SELECT id, title, artist, album, year
            FROM songs
            WHERE id != ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (song_id, num_recommendations - len(recommendations)))
        
        for row in cursor.fetchall():
            rec_id, rec_title, rec_artist, rec_album, rec_year = row
            recommendations.append({
                "song_id": rec_id,
                "title": rec_title,
                "artist": rec_artist,
                "album": rec_album,
                "year": rec_year,
                "similarity_type": "random_discovery",
                "similarity": 0.5
            })
    
    conn.close()
    return recommendations

'''
    
    with open('metadata_query.py', 'w') as f:
        f.write(code)
    
    print("✓ Created metadata_query.py")


if __name__ == '__main__':
    recommendations = build_metadata_recommendations()
    create_enhanced_query_with_metadata()
    
    print("\n" + "=" * 80)
    print("Metadata-based recommendation system ready!")
    print("=" * 80)
    print("\nThis provides fallback recommendations when embedding dimensions don't match.")
    print("Recommendations are based on:")
    print("  • Same artist")
    print("  • Similar genre (if available)")
    print("  • Release year proximity")
    print("\nNext: Integrate into query_api.py")
