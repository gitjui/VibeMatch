"""
Automatic Song Identification using Audio Fingerprinting
Works for any MP3 file - identifies songs by analyzing audio content
"""

import os
import json
from pathlib import Path
import acoustid
import musicbrainzngs
import ssl
import certifi
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
from langdetect import detect, DetectorFactory
import re
import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set seed for consistent language detection results
DetectorFactory.seed = 0

# Configure MusicBrainz API
musicbrainzngs.set_useragent("MP3Scanner", "1.0", "user@example.com")

# Fix SSL certificate issues
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# Get your FREE API key from: https://acoustid.org/
ACOUSTID_API_KEY = "uSJzjPuEIT"  # Replace with your API key


def format_duration(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def extract_audio_features(file_path, duration=30):
    """
    Extract advanced audio features using librosa
    Analyzes only first 30 seconds for efficiency
    
    Returns dict with:
    - tempo (BPM)
    - energy
    - spectral_centroid (brightness)
    - spectral_rolloff
    - zero_crossing_rate
    - rms_energy (loudness)
    """
    try:
        # Load audio file (first 30 seconds only for speed)
        y, sr = librosa.load(file_path, duration=duration, sr=22050)
        
        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate (texture)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # RMS Energy (loudness)
        rms = librosa.feature.rms(y=y)[0]
        
        # Chroma features (for key detection)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        return {
            'tempo_bpm': round(float(tempo), 1),
            'spectral_centroid_mean': round(float(np.mean(spectral_centroids)), 2),
            'spectral_centroid_std': round(float(np.std(spectral_centroids)), 2),
            'spectral_rolloff_mean': round(float(np.mean(spectral_rolloff)), 2),
            'zero_crossing_rate_mean': round(float(np.mean(zero_crossing_rate)), 4),
            'rms_energy_mean': round(float(np.mean(rms)), 4),
            'rms_energy_std': round(float(np.std(rms)), 4),
            'energy_level': 'High' if np.mean(rms) > 0.1 else 'Medium' if np.mean(rms) > 0.05 else 'Low',
            'brightness': 'Bright' if np.mean(spectral_centroids) > 3000 else 'Medium' if np.mean(spectral_centroids) > 1500 else 'Dark'
        }
    except Exception as e:
        print(f"    Warning: Could not extract audio features: {e}")
        return None


def detect_language(text):
    """
    Detect language from text
    Returns language code (e.g., 'en', 'hi', 'es')
    """
    if not text or text == 'Unknown':
        return 'unknown'
    
    # Remove special characters and clean text
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    clean_text = clean_text.strip()
    
    if not clean_text or len(clean_text) < 3:
        return 'unknown'
    
    try:
        lang = detect(clean_text)
        # Map common language codes to readable names
        lang_map = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-cn': 'Chinese',
            'ar': 'Arabic',
            'ru': 'Russian',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'pa': 'Punjabi',
            'mr': 'Marathi',
            'gu': 'Gujarati'
        }
        return lang_map.get(lang, lang)
    except:
        return 'unknown'


def detect_song_language(title, artist, album, filename):
    """
    Detect language from multiple sources and return best match
    """
    languages = []
    
    # Check for common Indian names/words to identify Hindi/Indian songs
    indian_indicators = [
        'singh', 'kumar', 'arijit', 'pritam', 'tanishk', 'bagchi', 
        'bollywood', 'hindi', 'kumar', 'kiara', 'varun', 'shraddha',
        'khan', 'kapoor', 'chopra', 'sharma', 'rao', 'reddy',
        'jubin', 'nautiyal', 'asees', 'kaur', 'shershaah', 'kamil',
        'bhattacharya', 'tseries', 't-series', 'panday', 'padda'
    ]
    
    combined_text = f"{title} {artist} {album} {filename}".lower()
    
    # Check for Indian indicators
    if any(indicator in combined_text for indicator in indian_indicators):
        return 'Hindi'
    
    # Try to detect from title
    if title and title != 'Unknown':
        lang = detect_language(title)
        if lang != 'unknown':
            languages.append(lang)
    
    # Try to detect from artist
    if artist and artist != 'Unknown':
        lang = detect_language(artist)
        if lang != 'unknown':
            languages.append(lang)
    
    # Try to detect from filename
    if filename:
        # Remove file extension and clean
        name = filename.rsplit('.', 1)[0]
        lang = detect_language(name)
        if lang != 'unknown':
            languages.append(lang)
    
    # Return most common language or unknown
    if languages:
        # Count occurrences and return most common
        from collections import Counter
        most_common = Counter(languages).most_common(1)[0][0]
        return most_common
    
    return 'Unknown'


def extract_from_filename(filename):
    """
    Extract song info from filename
    Common patterns: 
    - "Artist - Title.mp3"
    - "Title - Artist.mp3"
    - "Song Name (Official Video).mp3"
    """
    import re
    
    # Remove file extension
    name = filename.rsplit('.', 1)[0]
    
    # Remove common video/audio indicators
    name = re.sub(r'\s*\(.*?(Official|Music|Video|Audio|Lyric|HD|4K).*?\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\[.*?\]', '', name)  # Remove anything in brackets
    name = name.strip()
    
    title = name
    artist = 'Unknown'
    
    # Try to split by common separators
    if ' - ' in name:
        parts = name.split(' - ', 1)
        # Usually format is "Artist - Title"
        artist = parts[0].strip()
        title = parts[1].strip()
    elif ' _ ' in name:
        parts = name.split(' _ ', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    elif '|' in name:
        parts = name.split('|', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    
    return {
        'title': title,
        'artist': artist,
        'album': 'Unknown',
        'source': 'filename'
    }


def identify_song(file_path, api_key):
    """
    Identify a song using audio fingerprinting
    
    Args:
        file_path: Path to MP3 file
        api_key: AcoustID API key
        
    Returns:
        Dictionary with song metadata or None if not identified
    """
    print(f"  Analyzing audio fingerprint...")
    
    try:
        # Generate fingerprint and lookup
        results = acoustid.match(api_key, file_path)
        
        for score, recording_id, title, artist in results:
            if score > 0.5:  # Confidence threshold
                print(f"  ✓ Match found (confidence: {score:.1%})")
                
                # Get additional metadata from MusicBrainz
                try:
                    recording = musicbrainzngs.get_recording_by_id(
                        recording_id,
                        includes=['artists', 'releases', 'tags']
                    )
                    
                    rec = recording['recording']
                    metadata = {
                        'title': rec.get('title', title),
                        'artist': artist,
                        'recording_id': recording_id,
                        'confidence': round(score, 3),
                        'length': rec.get('length'),
                        'genres': [],
                        'tags': [],
                        'release_date': None,
                        'album': None
                    }
                    
                    # Extract genres/tags
                    if 'tag-list' in rec:
                        metadata['tags'] = [tag['name'] for tag in rec['tag-list'][:5]]
                    
                    if 'genre-list' in rec:
                        metadata['genres'] = [genre['name'] for genre in rec['genre-list']]
                    
                    # Get album and release date
                    if 'release-list' in rec:
                        releases = rec['release-list']
                        if releases:
                            first_release = releases[0]
                            metadata['album'] = first_release.get('title')
                            metadata['release_date'] = first_release.get('date')
                    
                    return metadata
                    
                except Exception as e:
                    print(f"  Warning: Could not fetch extended metadata: {e}")
                    return {
                        'title': title,
                        'artist': artist,
                        'recording_id': recording_id,
                        'confidence': round(score, 3)
                    }
        
        print("  ✗ No confident match found")
        return None
        
    except acoustid.NoBackendError:
        print("  ✗ Error: fpcalc not found. Please install chromaprint:")
        print("    macOS: brew install chromaprint")
        print("    Linux: sudo apt-get install libchromaprint-tools")
        return None
        
    except Exception as e:
        print(f"  ✗ Error identifying song: {e}")
        return None


def scan_and_identify_folder(folder_path, api_key):
    """
    Scan folder and identify all MP3 files
    
    Args:
        folder_path: Path to folder with MP3 files
        api_key: AcoustID API key
        
    Returns:
        List of identified songs with metadata
    """
    folder = Path(folder_path)
    mp3_files = list(folder.glob('*.mp3')) + list(folder.glob('**/*.mp3'))
    mp3_files = list(set(mp3_files))  # Remove duplicates
    
    print(f"Found {len(mp3_files)} MP3 files\n")
    
    identified_songs = []
    
    for idx, file_path in enumerate(mp3_files, 1):
        file_info = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size_mb': round(file_path.stat().st_size / (1024 * 1024), 2)
        }
        
        # Extract audio properties from MP3
        try:
            audio_info = MP3(str(file_path))
            file_info['duration_seconds'] = round(audio_info.info.length, 2)
            file_info['duration_formatted'] = format_duration(audio_info.info.length)
            file_info['bitrate'] = audio_info.info.bitrate
            file_info['bitrate_kbps'] = round(audio_info.info.bitrate / 1000, 1)
            file_info['sample_rate'] = audio_info.info.sample_rate
            file_info['channels'] = audio_info.info.channels
            file_info['channel_mode'] = 'Stereo' if audio_info.info.channels == 2 else 'Mono'
        except Exception as e:
            print(f"  Warning: Could not read audio properties: {e}")
        
        # Try to read existing ID3 tags (if any)
        try:
            audio_tags = EasyID3(str(file_path))
            file_info['id3_title'] = audio_tags.get('title', [''])[0]
            file_info['id3_artist'] = audio_tags.get('artist', [''])[0]
            file_info['id3_album'] = audio_tags.get('album', [''])[0]
            file_info['id3_genre'] = audio_tags.get('genre', [''])[0]
            file_info['id3_date'] = audio_tags.get('date', [''])[0]
        except:
            pass
        
        print(f"[{idx}/{len(mp3_files)}] {file_path.name}")
        
        # Identify the song
        metadata = identify_song(str(file_path), api_key)
        
        if metadata:
            file_info.update(metadata)
            file_info['identified'] = True
        else:
            # Fallback: Try to extract from filename
            file_info['identified'] = False
            filename_info = extract_from_filename(file_path.name)
            file_info.update(filename_info)
            print(f"  ℹ️  Using filename-based extraction")
        
        # Detect language from available metadata
        language = detect_song_language(
            file_info.get('title', ''),
            file_info.get('artist', ''),
            file_info.get('album', ''),
            file_path.name
        )
        file_info['language'] = language
        
        # Extract advanced audio features
        print(f"  🎵 Extracting audio features...")
        audio_features = extract_audio_features(str(file_path))
        if audio_features:
            file_info.update(audio_features)
        
        identified_songs.append(file_info)
        print()
    
    return identified_songs


def display_results(songs):
    """Display identified songs in readable format"""
    print("="*80)
    print("IDENTIFIED SONGS")
    print("="*80)
    
    for idx, song in enumerate(songs, 1):
        status = "✓" if song.get('identified') else "✗"
        print(f"\n[{idx}] {status} {song['file_name']}")
        print("-" * 80)
        
        if song.get('identified'):
            print(f"  Title:          {song.get('title', 'N/A')}")
            print(f"  Artist:         {song.get('artist', 'N/A')}")
            print(f"  Album:          {song.get('album', 'N/A')}")
            print(f"  Language:       {song.get('language', 'N/A')}")
            print(f"  Release Date:   {song.get('release_date', 'N/A')}")
            print(f"  Confidence:     {song.get('confidence', 0)*100:.1f}%")
            
            if song.get('genres'):
                print(f"  Genres:         {', '.join(song['genres'])}")
            if song.get('tags'):
                print(f"  Tags:           {', '.join(song['tags'][:5])}")
            
            print(f"  Recording ID:   {song.get('recording_id', 'N/A')}")
            print()
            print(f"  Audio Properties:")
            print(f"    Duration:     {song.get('duration_formatted', 'N/A')}")
            print(f"    Bitrate:      {song.get('bitrate_kbps', 'N/A')} kbps")
            print(f"    Sample Rate:  {song.get('sample_rate', 'N/A')} Hz")
            print(f"    Channels:     {song.get('channel_mode', 'N/A')}")
            print(f"    File Size:    {song.get('file_size_mb', 'N/A')} MB")
            
            # Advanced audio features
            if song.get('tempo_bpm'):
                print()
                print(f"  Advanced Features:")
                print(f"    Tempo:        {song.get('tempo_bpm', 'N/A')} BPM")
                print(f"    Energy:       {song.get('energy_level', 'N/A')}")
                print(f"    Brightness:   {song.get('brightness', 'N/A')}")
                print(f"    Loudness:     {song.get('rms_energy_mean', 'N/A')}")
        else:
            # Not identified by fingerprint, but we have filename data
            print(f"  Title:          {song.get('title', 'N/A')}")
            print(f"  Artist:         {song.get('artist', 'N/A')}")
            print(f"  Album:          {song.get('album', 'N/A')}")
            print(f"  Language:       {song.get('language', 'N/A')}")
            print(f"  Source:         {song.get('source', 'N/A')} (not fingerprinted)")
            print()
            print(f"  Audio Properties:")
            print(f"    Duration:     {song.get('duration_formatted', 'N/A')}")
            print(f"    Bitrate:      {song.get('bitrate_kbps', 'N/A')} kbps")
            print(f"    Sample Rate:  {song.get('sample_rate', 'N/A')} Hz")
            print(f"    Channels:     {song.get('channel_mode', 'N/A')}")
            print(f"    File Size:    {song.get('file_size_mb')} MB")
            
            # Advanced audio features
            if song.get('tempo_bpm'):
                print()
                print(f"  Advanced Features:")
                print(f"    Tempo:        {song.get('tempo_bpm', 'N/A')} BPM")
                print(f"    Energy:       {song.get('energy_level', 'N/A')}")
                print(f"    Brightness:   {song.get('brightness', 'N/A')}")
                print(f"    Loudness:     {song.get('rms_energy_mean', 'N/A')}")


def save_to_json(songs, output_file='identified_songs.json'):
    """Save identified songs to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(songs, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


def main():
    """Main function"""
    print("Automatic Song Identifier")
    print("="*80)
    
    # Check if API key is set
    if ACOUSTID_API_KEY == "YOUR_API_KEY_HERE" or not ACOUSTID_API_KEY or len(ACOUSTID_API_KEY) < 5:
        print("\n⚠️  WARNING: AcoustID API key not set!")
        print("\nTo use this script:")
        print("1. Get a FREE API key from: https://acoustid.org/new-application")
        print("2. Replace 'YOUR_API_KEY_HERE' in this script with your API key")
        print("\nFor now, I'll show you how it works with a demo...")
        return
    
    # Path to MP3 folder
    mp3_folder = '/Users/juigupte/Desktop/Learning/music/mp3'
    
    print(f"Scanning folder: {mp3_folder}\n")
    
    if not os.path.exists(mp3_folder):
        print(f"Error: Folder not found: {mp3_folder}")
        return
    
    # Scan and identify songs
    identified_songs = scan_and_identify_folder(mp3_folder, ACOUSTID_API_KEY)
    
    # Display results
    display_results(identified_songs)
    
    # Save to JSON
    output_path = '/Users/juigupte/Desktop/Learning/recsys-foundations/identified_songs.json'
    save_to_json(identified_songs, output_path)
    
    # Summary
    identified_count = sum(1 for s in identified_songs if s.get('identified'))
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files:        {len(identified_songs)}")
    print(f"Successfully ID'd:  {identified_count}")
    print(f"Not identified:     {len(identified_songs) - identified_count}")
    
    # Language statistics
    from collections import Counter
    languages = [s.get('language', 'Unknown') for s in identified_songs if s.get('language') != 'Unknown']
    if languages:
        lang_counts = Counter(languages)
        print(f"\nLanguages detected:")
        for lang, count in lang_counts.most_common():
            print(f"  {lang}: {count} song(s)")


if __name__ == '__main__':
    main()
