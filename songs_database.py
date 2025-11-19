"""
SQLite Database Manager for Music Metadata
Stores comprehensive song information for recommendation system
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime


class SongsDatabase:
    """Manages SQLite database for song metadata"""
    
    def __init__(self, db_path='songs.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        """Create database schema"""
        
        # Main songs table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_path TEXT UNIQUE NOT NULL,
                file_size_mb REAL,
                
                -- Song Identity
                title TEXT,
                artist TEXT,
                album TEXT,
                language TEXT,
                release_date TEXT,
                
                -- Audio Quality
                duration_seconds REAL,
                duration_formatted TEXT,
                bitrate INTEGER,
                bitrate_kbps REAL,
                sample_rate INTEGER,
                channels INTEGER,
                channel_mode TEXT,
                
                -- ID3 Tags
                id3_title TEXT,
                id3_artist TEXT,
                id3_album TEXT,
                id3_genre TEXT,
                id3_date TEXT,
                
                -- Musical Features
                tempo_bpm REAL,
                energy_level TEXT,
                brightness TEXT,
                
                -- Spectral Analysis
                spectral_centroid_mean REAL,
                spectral_centroid_std REAL,
                spectral_rolloff_mean REAL,
                zero_crossing_rate_mean REAL,
                
                -- Audio Normalization
                rms_energy_mean REAL,
                rms_energy_std REAL,
                
                -- Identification Info
                identified BOOLEAN,
                confidence REAL,
                recording_id TEXT,
                source TEXT,
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tags table for genres/tags (many-to-many relationship)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER,
                tag_name TEXT,
                tag_type TEXT,  -- 'genre' or 'tag'
                FOREIGN KEY (song_id) REFERENCES songs (id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for faster queries
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_language ON songs(language)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_tempo ON songs(tempo_bpm)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_energy ON songs(energy_level)')
        
        self.conn.commit()
    
    def import_from_json(self, json_path):
        """Import songs from identified_songs.json"""
        
        with open(json_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        
        print(f"Importing {len(songs)} songs into database...")
        imported = 0
        updated = 0
        
        for song in songs:
            # Check if song already exists
            existing = self.cursor.execute(
                'SELECT id FROM songs WHERE file_path = ?',
                (song.get('file_path'),)
            ).fetchone()
            
            if existing:
                # Update existing record
                self._update_song(existing['id'], song)
                updated += 1
            else:
                # Insert new record
                self._insert_song(song)
                imported += 1
        
        self.conn.commit()
        print(f"✓ Imported {imported} new songs")
        print(f"✓ Updated {updated} existing songs")
        return imported, updated
    
    def _insert_song(self, song):
        """Insert a new song into database"""
        
        self.cursor.execute('''
            INSERT INTO songs (
                file_name, file_path, file_size_mb,
                title, artist, album, language, release_date,
                duration_seconds, duration_formatted, bitrate, bitrate_kbps,
                sample_rate, channels, channel_mode,
                id3_title, id3_artist, id3_album, id3_genre, id3_date,
                tempo_bpm, energy_level, brightness,
                spectral_centroid_mean, spectral_centroid_std,
                spectral_rolloff_mean, zero_crossing_rate_mean,
                rms_energy_mean, rms_energy_std,
                identified, confidence, recording_id, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            song.get('file_name'),
            song.get('file_path'),
            song.get('file_size_mb'),
            song.get('title'),
            song.get('artist'),
            song.get('album'),
            song.get('language'),
            song.get('release_date'),
            song.get('duration_seconds'),
            song.get('duration_formatted'),
            song.get('bitrate'),
            song.get('bitrate_kbps'),
            song.get('sample_rate'),
            song.get('channels'),
            song.get('channel_mode'),
            song.get('id3_title'),
            song.get('id3_artist'),
            song.get('id3_album'),
            song.get('id3_genre'),
            song.get('id3_date'),
            song.get('tempo_bpm'),
            song.get('energy_level'),
            song.get('brightness'),
            song.get('spectral_centroid_mean'),
            song.get('spectral_centroid_std'),
            song.get('spectral_rolloff_mean'),
            song.get('zero_crossing_rate_mean'),
            song.get('rms_energy_mean'),
            song.get('rms_energy_std'),
            song.get('identified'),
            song.get('confidence'),
            song.get('recording_id'),
            song.get('source')
        ))
        
        song_id = self.cursor.lastrowid
        
        # Insert genres and tags
        if song.get('genres'):
            for genre in song['genres']:
                self.cursor.execute(
                    'INSERT INTO tags (song_id, tag_name, tag_type) VALUES (?, ?, ?)',
                    (song_id, genre, 'genre')
                )
        
        if song.get('tags'):
            for tag in song['tags']:
                self.cursor.execute(
                    'INSERT INTO tags (song_id, tag_name, tag_type) VALUES (?, ?, ?)',
                    (song_id, tag, 'tag')
                )
    
    def _update_song(self, song_id, song):
        """Update an existing song in database"""
        
        self.cursor.execute('''
            UPDATE songs SET
                file_name = ?, file_size_mb = ?,
                title = ?, artist = ?, album = ?, language = ?, release_date = ?,
                duration_seconds = ?, duration_formatted = ?, bitrate = ?, bitrate_kbps = ?,
                sample_rate = ?, channels = ?, channel_mode = ?,
                id3_title = ?, id3_artist = ?, id3_album = ?, id3_genre = ?, id3_date = ?,
                tempo_bpm = ?, energy_level = ?, brightness = ?,
                spectral_centroid_mean = ?, spectral_centroid_std = ?,
                spectral_rolloff_mean = ?, zero_crossing_rate_mean = ?,
                rms_energy_mean = ?, rms_energy_std = ?,
                identified = ?, confidence = ?, recording_id = ?, source = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            song.get('file_name'),
            song.get('file_size_mb'),
            song.get('title'),
            song.get('artist'),
            song.get('album'),
            song.get('language'),
            song.get('release_date'),
            song.get('duration_seconds'),
            song.get('duration_formatted'),
            song.get('bitrate'),
            song.get('bitrate_kbps'),
            song.get('sample_rate'),
            song.get('channels'),
            song.get('channel_mode'),
            song.get('id3_title'),
            song.get('id3_artist'),
            song.get('id3_album'),
            song.get('id3_genre'),
            song.get('id3_date'),
            song.get('tempo_bpm'),
            song.get('energy_level'),
            song.get('brightness'),
            song.get('spectral_centroid_mean'),
            song.get('spectral_centroid_std'),
            song.get('spectral_rolloff_mean'),
            song.get('zero_crossing_rate_mean'),
            song.get('rms_energy_mean'),
            song.get('rms_energy_std'),
            song.get('identified'),
            song.get('confidence'),
            song.get('recording_id'),
            song.get('source'),
            song_id
        ))
        
        # Update tags
        self.cursor.execute('DELETE FROM tags WHERE song_id = ?', (song_id,))
        
        if song.get('genres'):
            for genre in song['genres']:
                self.cursor.execute(
                    'INSERT INTO tags (song_id, tag_name, tag_type) VALUES (?, ?, ?)',
                    (song_id, genre, 'genre')
                )
        
        if song.get('tags'):
            for tag in song['tags']:
                self.cursor.execute(
                    'INSERT INTO tags (song_id, tag_name, tag_type) VALUES (?, ?, ?)',
                    (song_id, tag, 'tag')
                )
    
    def get_all_songs(self):
        """Get all songs"""
        return self.cursor.execute('SELECT * FROM songs ORDER BY title').fetchall()
    
    def search_by_artist(self, artist):
        """Search songs by artist"""
        return self.cursor.execute(
            'SELECT * FROM songs WHERE artist LIKE ? ORDER BY title',
            (f'%{artist}%',)
        ).fetchall()
    
    def search_by_language(self, language):
        """Search songs by language"""
        return self.cursor.execute(
            'SELECT * FROM songs WHERE language = ? ORDER BY title',
            (language,)
        ).fetchall()
    
    def search_by_tempo_range(self, min_bpm, max_bpm):
        """Find songs within a tempo range"""
        return self.cursor.execute(
            'SELECT * FROM songs WHERE tempo_bpm BETWEEN ? AND ? ORDER BY tempo_bpm',
            (min_bpm, max_bpm)
        ).fetchall()
    
    def search_by_energy(self, energy_level):
        """Find songs by energy level"""
        return self.cursor.execute(
            'SELECT * FROM songs WHERE energy_level = ? ORDER BY title',
            (energy_level,)
        ).fetchall()
    
    def get_statistics(self):
        """Get database statistics"""
        stats = {}
        
        # Total songs
        stats['total_songs'] = self.cursor.execute('SELECT COUNT(*) FROM songs').fetchone()[0]
        
        # Language distribution
        stats['languages'] = self.cursor.execute('''
            SELECT language, COUNT(*) as count 
            FROM songs 
            GROUP BY language 
            ORDER BY count DESC
        ''').fetchall()
        
        # Energy distribution
        stats['energy'] = self.cursor.execute('''
            SELECT energy_level, COUNT(*) as count 
            FROM songs 
            GROUP BY energy_level 
            ORDER BY count DESC
        ''').fetchall()
        
        # Tempo statistics
        tempo_stats = self.cursor.execute('''
            SELECT 
                MIN(tempo_bpm) as min_tempo,
                MAX(tempo_bpm) as max_tempo,
                AVG(tempo_bpm) as avg_tempo
            FROM songs 
            WHERE tempo_bpm IS NOT NULL
        ''').fetchone()
        stats['tempo'] = tempo_stats
        
        # Top artists
        stats['top_artists'] = self.cursor.execute('''
            SELECT artist, COUNT(*) as count 
            FROM songs 
            WHERE artist != 'Unknown' AND artist IS NOT NULL
            GROUP BY artist 
            ORDER BY count DESC 
            LIMIT 5
        ''').fetchall()
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Import JSON data and display statistics"""
    
    print("="*80)
    print("MUSIC DATABASE MANAGER")
    print("="*80)
    
    # Create database
    db = SongsDatabase('songs.db')
    
    # Import from JSON
    json_path = 'identified_songs.json'
    if Path(json_path).exists():
        db.import_from_json(json_path)
        
        # Display statistics
        print("\n" + "="*80)
        print("DATABASE STATISTICS")
        print("="*80)
        
        stats = db.get_statistics()
        
        print(f"\nTotal Songs: {stats['total_songs']}")
        
        print(f"\nLanguages:")
        for row in stats['languages']:
            print(f"  {row['language']}: {row['count']} songs")
        
        print(f"\nEnergy Levels:")
        for row in stats['energy']:
            if row['energy_level']:
                print(f"  {row['energy_level']}: {row['count']} songs")
        
        if stats['tempo']:
            print(f"\nTempo Range:")
            print(f"  Min: {stats['tempo']['min_tempo']:.1f} BPM")
            print(f"  Max: {stats['tempo']['max_tempo']:.1f} BPM")
            print(f"  Avg: {stats['tempo']['avg_tempo']:.1f} BPM")
        
        print(f"\nTop Artists:")
        for row in stats['top_artists']:
            print(f"  {row['artist']}: {row['count']} songs")
        
        # Example queries
        print("\n" + "="*80)
        print("EXAMPLE QUERIES")
        print("="*80)
        
        # High energy songs
        high_energy = db.search_by_energy('High')
        print(f"\nHigh Energy Songs ({len(high_energy)}):")
        for song in high_energy[:3]:
            print(f"  • {song['title']} - {song['artist']} ({song['tempo_bpm']} BPM)")
        
        # Hindi songs
        hindi_songs = db.search_by_language('Hindi')
        print(f"\nHindi Songs ({len(hindi_songs)}):")
        for song in hindi_songs[:3]:
            print(f"  • {song['title']} - {song['artist']}")
        
        # Songs in tempo range 120-140 BPM
        moderate_tempo = db.search_by_tempo_range(120, 140)
        print(f"\nSongs 120-140 BPM ({len(moderate_tempo)}):")
        for song in moderate_tempo[:3]:
            print(f"  • {song['title']} ({song['tempo_bpm']} BPM)")
        
        print("\n" + "="*80)
        print(f"Database saved to: songs.db")
        print("="*80)
    
    else:
        print(f"Error: {json_path} not found!")
    
    db.close()


if __name__ == '__main__':
    main()
