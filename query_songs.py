"""
Interactive Music Database Query Tool
Query and filter songs from SQLite database
"""

from songs_database import SongsDatabase
import sys


def print_song(song, show_details=False):
    """Pretty print song information"""
    print(f"\n{'='*80}")
    print(f"🎵 {song['title']}")
    print(f"   Artist: {song['artist'] or 'Unknown'}")
    print(f"   Album: {song['album'] or 'Unknown'}")
    print(f"   Language: {song['language']}")
    
    if show_details:
        print(f"\n   Duration: {song['duration_formatted']} | Bitrate: {song['bitrate_kbps']} kbps")
        if song['tempo_bpm']:
            print(f"   Tempo: {song['tempo_bpm']} BPM | Energy: {song['energy_level']} | Brightness: {song['brightness']}")
        if song['identified']:
            print(f"   Identified: ✓ ({song['confidence']*100:.1f}% confidence)")
        print(f"   File: {song['file_name']}")


def search_menu(db):
    """Interactive search menu"""
    
    while True:
        print("\n" + "="*80)
        print("MUSIC DATABASE QUERY TOOL")
        print("="*80)
        print("\n1. Search by Artist")
        print("2. Search by Language")
        print("3. Search by Tempo Range")
        print("4. Search by Energy Level")
        print("5. View All Songs")
        print("6. Get Statistics")
        print("7. Find Similar Tempo Songs")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            artist = input("Enter artist name: ").strip()
            songs = db.search_by_artist(artist)
            print(f"\nFound {len(songs)} songs by '{artist}':")
            for song in songs:
                print_song(song)
        
        elif choice == '2':
            print("\nAvailable languages:")
            stats = db.get_statistics()
            for lang in stats['languages']:
                print(f"  - {lang['language']}")
            language = input("\nEnter language: ").strip()
            songs = db.search_by_language(language)
            print(f"\nFound {len(songs)} {language} songs:")
            for song in songs:
                print_song(song)
        
        elif choice == '3':
            min_bpm = float(input("Enter minimum BPM: ").strip())
            max_bpm = float(input("Enter maximum BPM: ").strip())
            songs = db.search_by_tempo_range(min_bpm, max_bpm)
            print(f"\nFound {len(songs)} songs between {min_bpm}-{max_bpm} BPM:")
            for song in songs:
                print_song(song, show_details=True)
        
        elif choice == '4':
            print("\nEnergy levels: High, Medium, Low")
            energy = input("Enter energy level: ").strip()
            songs = db.search_by_energy(energy)
            print(f"\nFound {len(songs)} {energy} energy songs:")
            for song in songs:
                print_song(song, show_details=True)
        
        elif choice == '5':
            songs = db.get_all_songs()
            print(f"\nAll {len(songs)} songs:")
            for song in songs:
                print_song(song)
        
        elif choice == '6':
            stats = db.get_statistics()
            print("\n" + "="*80)
            print("DATABASE STATISTICS")
            print("="*80)
            print(f"\nTotal Songs: {stats['total_songs']}")
            
            print(f"\nLanguages:")
            for row in stats['languages']:
                print(f"  {row['language']}: {row['count']} songs")
            
            print(f"\nEnergy Distribution:")
            for row in stats['energy']:
                if row['energy_level']:
                    print(f"  {row['energy_level']}: {row['count']} songs")
            
            if stats['tempo']:
                print(f"\nTempo Statistics:")
                print(f"  Min: {stats['tempo']['min_tempo']:.1f} BPM")
                print(f"  Max: {stats['tempo']['max_tempo']:.1f} BPM")
                print(f"  Avg: {stats['tempo']['avg_tempo']:.1f} BPM")
            
            print(f"\nTop Artists:")
            for row in stats['top_artists']:
                print(f"  {row['artist']}: {row['count']} songs")
        
        elif choice == '7':
            # Find songs with similar tempo to a given song
            title = input("Enter song title to find similar tempo: ").strip()
            result = db.cursor.execute(
                'SELECT * FROM songs WHERE title LIKE ? LIMIT 1',
                (f'%{title}%',)
            ).fetchone()
            
            if result and result['tempo_bpm']:
                target_tempo = result['tempo_bpm']
                print(f"\n'{result['title']}' has tempo: {target_tempo} BPM")
                
                # Find songs within ±10 BPM
                similar = db.search_by_tempo_range(target_tempo - 10, target_tempo + 10)
                print(f"\nSongs with similar tempo ({target_tempo-10:.0f}-{target_tempo+10:.0f} BPM):")
                for song in similar:
                    if song['id'] != result['id']:  # Exclude the original song
                        print_song(song, show_details=True)
            else:
                print(f"Song not found or no tempo data available.")
        
        elif choice == '8':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice! Please try again.")


def main():
    """Main function"""
    
    # Open database
    db = SongsDatabase('songs.db')
    
    # Check if database has data
    stats = db.get_statistics()
    if stats['total_songs'] == 0:
        print("Database is empty! Please run 'python songs_database.py' first to import songs.")
        db.close()
        return
    
    # Start interactive menu
    try:
        search_menu(db)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Goodbye!")
    finally:
        db.close()


if __name__ == '__main__':
    main()
