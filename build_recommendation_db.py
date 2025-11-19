"""
Build Recommendation Database from Million Song Dataset
========================================================

Downloads and processes Million Song Dataset (or subset) to create
a recommendation database with embeddings for similarity search.

MSD provides:
- Metadata: title, artist, year, genre, tags
- Audio features: timbre, loudness, tempo, etc.
- No audio files (need to get separately or use features)

Options:
1. Use MSD pre-computed features (fast, no audio needed)
2. Download preview clips from 7digital/Spotify and compute embeddings (slow, better)

This script does OPTION 1 for speed.
"""

import sqlite3
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List
import requests
import tarfile
import h5py


class RecommendationDatabaseBuilder:
    """
    Build recommendation database from Million Song Dataset.
    """
    
    def __init__(self, db_path: str = 'recommendations.db'):
        self.db_path = db_path
        self.msd_subset_url = "http://static.echonest.com/millionsongsubset_full.tar.gz"
        self.msd_data_dir = Path('msd_data')
    
    def create_schema(self):
        """Create recommendation database schema."""
        print("Creating recommendation database schema...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main recommendation tracks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_tracks (
                track_id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT,
                album TEXT,
                year INTEGER,
                genre TEXT,
                tags TEXT,  -- JSON array of tags
                duration REAL,
                tempo REAL,
                loudness REAL,
                energy REAL,
                danceability REAL,
                embedding BLOB,  -- 64-dim or using MSD features (12-dim timbre)
                metadata TEXT    -- Additional JSON metadata
            )
        """)
        
        # Index for fast similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artist ON recommendation_tracks(artist)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_genre ON recommendation_tracks(genre)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_year ON recommendation_tracks(year)
        """)
        
        conn.commit()
        conn.close()
        
        print("✓ Schema created")
    
    def download_msd_subset(self):
        """
        Download MSD subset (10,000 songs).
        Full dataset is 280 GB, subset is ~1.8 GB.
        """
        if self.msd_data_dir.exists():
            print(f"✓ MSD data directory exists: {self.msd_data_dir}")
            return
        
        print(f"Downloading MSD subset (~1.8 GB)...")
        print(f"URL: {self.msd_subset_url}")
        print("This may take 10-30 minutes depending on connection...")
        
        # Download
        tar_path = 'msd_subset.tar.gz'
        if not os.path.exists(tar_path):
            response = requests.get(self.msd_subset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Downloaded: {downloaded / 1e9:.2f} GB ({percent:.1f}%)", end='\r')
            print("\n✓ Download complete")
        
        # Extract
        print("Extracting archive...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall('.')
        
        # Find data directory
        for root, dirs, files in os.walk('.'):
            if 'data' in dirs:
                self.msd_data_dir = Path(root) / 'data'
                break
        
        print(f"✓ Extracted to {self.msd_data_dir}")
    
    def parse_h5_file(self, h5_path: Path) -> Dict:
        """
        Parse MSD HDF5 file and extract features.
        
        MSD stores:
        - Metadata: title, artist, year
        - Audio features: timbre (12-dim), segments, etc.
        - Echo Nest analysis data
        """
        with h5py.File(h5_path, 'r') as f:
            # Extract metadata
            track_id = f['metadata']['songs']['track_id'][0].decode('utf-8')
            title = f['metadata']['songs']['title'][0].decode('utf-8')
            artist = f['metadata']['songs']['artist_name'][0].decode('utf-8')
            year = int(f['musicbrainz']['songs']['year'][0])
            duration = float(f['analysis']['songs']['duration'][0])
            
            # Extract audio features
            tempo = float(f['analysis']['songs']['tempo'][0])
            loudness = float(f['analysis']['songs']['loudness'][0])
            
            # Get timbre features (mean of all segments)
            # Timbre is 12-dim vector describing texture
            segments_timbre = f['analysis']['segments_timbre'][:]
            if len(segments_timbre) > 0:
                timbre_mean = segments_timbre.mean(axis=0)
            else:
                timbre_mean = np.zeros(12)
            
            # Estimate energy and danceability from features
            # (MSD doesn't have these directly, but we can approximate)
            energy = min(1.0, max(0.0, (loudness + 60) / 60))  # Normalized loudness
            danceability = min(1.0, max(0.0, tempo / 200))  # Tempo-based proxy
            
            return {
                'track_id': track_id,
                'title': title,
                'artist': artist,
                'year': year if year > 0 else None,
                'duration': duration,
                'tempo': tempo,
                'loudness': loudness,
                'energy': energy,
                'danceability': danceability,
                'timbre': timbre_mean  # Use as embedding (12-dim)
            }
    
    def build_from_msd_subset(self, max_tracks: int = 10000):
        """
        Build recommendation DB from MSD subset.
        
        Args:
            max_tracks: Maximum number of tracks to process
        """
        print(f"\n📚 Building recommendation database from MSD...")
        
        # Find all H5 files
        h5_files = list(self.msd_data_dir.rglob('*.h5'))
        print(f"   Found {len(h5_files)} H5 files")
        
        if len(h5_files) == 0:
            print("❌ No H5 files found!")
            return
        
        # Process files
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        processed = 0
        errors = 0
        
        for i, h5_path in enumerate(h5_files[:max_tracks]):
            try:
                # Parse H5 file
                track_data = self.parse_h5_file(h5_path)
                
                # Insert into database
                cursor.execute("""
                    INSERT OR REPLACE INTO recommendation_tracks
                    (track_id, title, artist, year, duration, tempo, 
                     loudness, energy, danceability, embedding, genre, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    track_data['track_id'],
                    track_data['title'],
                    track_data['artist'],
                    track_data['year'],
                    track_data['duration'],
                    track_data['tempo'],
                    track_data['loudness'],
                    track_data['energy'],
                    track_data['danceability'],
                    track_data['timbre'].astype(np.float32).tobytes(),
                    None,  # Genre not in MSD
                    None   # Tags not in MSD
                ))
                
                processed += 1
                
                if (processed % 100) == 0:
                    print(f"   Processed {processed}/{len(h5_files)} tracks...", end='\r')
                    conn.commit()
                
            except Exception as e:
                errors += 1
                if errors < 10:
                    print(f"\n⚠️  Error processing {h5_path}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\n✓ Built recommendation database:")
        print(f"   Processed: {processed} tracks")
        print(f"   Errors: {errors}")
        print(f"   Database: {self.db_path}")
    
    def add_sample_data(self, num_samples: int = 1000):
        """
        Add synthetic sample data for testing (when MSD not available).
        
        Creates fake tracks with random embeddings for testing.
        """
        print(f"\n📚 Adding {num_samples} sample tracks for testing...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sample artist names
        artists = [
            "The Beatles", "Taylor Swift", "Ed Sheeran", "Drake", "Beyoncé",
            "Coldplay", "Adele", "Bruno Mars", "Ariana Grande", "The Weeknd",
            "Post Malone", "Billie Eilish", "Dua Lipa", "Harry Styles", "BTS",
            "Queen", "Pink Floyd", "Led Zeppelin", "Nirvana", "Radiohead"
        ]
        
        # Sample genres
        genres = ["Pop", "Rock", "Hip Hop", "R&B", "Electronic", "Country", 
                 "Jazz", "Classical", "Indie", "Metal"]
        
        for i in range(num_samples):
            # Random metadata
            track_id = f"SAMPLE{i:06d}"
            title = f"Sample Song {i+1}"
            artist = np.random.choice(artists)
            year = np.random.randint(1960, 2024)
            genre = np.random.choice(genres)
            duration = np.random.uniform(120, 300)
            tempo = np.random.uniform(80, 160)
            loudness = np.random.uniform(-30, 0)
            energy = np.random.uniform(0, 1)
            danceability = np.random.uniform(0, 1)
            
            # Random embedding (12-dim to match MSD timbre)
            embedding = np.random.randn(12).astype(np.float32)
            
            cursor.execute("""
                INSERT INTO recommendation_tracks
                (track_id, title, artist, year, genre, duration, tempo,
                 loudness, energy, danceability, embedding, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id, title, artist, year, genre, duration, tempo,
                loudness, energy, danceability, embedding.tobytes(), "[]"
            ))
            
            if (i + 1) % 100 == 0:
                print(f"   Added {i+1}/{num_samples} tracks...", end='\r')
                conn.commit()
        
        conn.commit()
        conn.close()
        
        print(f"\n✓ Added {num_samples} sample tracks")


def main():
    """Build recommendation database."""
    import sys
    
    print("=" * 80)
    print("Recommendation Database Builder")
    print("=" * 80)
    
    builder = RecommendationDatabaseBuilder(db_path='recommendations.db')
    
    # Create schema
    builder.create_schema()
    
    # Choose build method
    if '--sample' in sys.argv:
        # Quick testing: Add synthetic data
        num_samples = 1000
        if '--count' in sys.argv:
            idx = sys.argv.index('--count')
            num_samples = int(sys.argv[idx + 1])
        
        builder.add_sample_data(num_samples=num_samples)
    
    elif '--msd' in sys.argv:
        # Full build: Download and process MSD
        builder.download_msd_subset()
        builder.build_from_msd_subset(max_tracks=10000)
    
    else:
        # Default: Quick sample for testing
        print("\nℹ️  No option specified. Building with sample data for testing.")
        print("   For real MSD data: python build_recommendation_db.py --msd")
        print()
        builder.add_sample_data(num_samples=1000)
    
    # Show stats
    conn = sqlite3.connect('recommendations.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM recommendation_tracks")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT artist) FROM recommendation_tracks")
    artists = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT genre) FROM recommendation_tracks WHERE genre IS NOT NULL")
    genres = cursor.fetchone()[0]
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("Database Built Successfully!")
    print("=" * 80)
    print(f"   Total tracks: {total:,}")
    print(f"   Unique artists: {artists:,}")
    print(f"   Genres: {genres}")
    print(f"   Database size: {os.path.getsize('recommendations.db') / 1e6:.2f} MB")
    print("\n✓ Ready to use with dual_database_system.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
