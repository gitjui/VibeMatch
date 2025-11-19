# Music Query API - JSON Output

## Usage

```bash
python query_api.py <audio_file> <start_time> <duration>
```

**Parameters:**
- `audio_file`: Path to MP3 file (can be relative or absolute)
- `start_time`: Start time in seconds (default: 30)
- `duration`: Duration of snippet in seconds (default: 5)

**Examples:**
```bash
# Query Dua Lipa song from 30-35 seconds
python query_api.py "../music/test mp3/Dua Lipa - Be The One.mp3" 30 5

# Query Harry Styles song from 45-50 seconds  
python query_api.py "../music/test mp3/Harry Styles - Lights Up.mp3" 45 5
```

## JSON Output Format

The API returns a structured JSON with three main sections:

### 1. Query Information
```json
{
  "query": {
    "file": "Dua Lipa - Be The One.mp3",
    "start_time": 30.0,
    "duration": 5.0
  }
}
```

### 2. Exact Match
When a song is found in the ID database:
```json
{
  "exact_match": {
    "found": true,
    "song_id": 2,
    "title": "Song Title",
    "artist": "Artist Name",
    "confidence": 0.791,
    "distance": 0.063,
    "match_type": "exact"
  }
}
```

When no match is found:
```json
{
  "exact_match": {
    "found": false,
    "message": "No exact match found"
  }
}
```

### 3. Recommendations
Array of similar songs, ranked by similarity:

```json
{
  "recommendations": [
    {
      "rank": 1,
      "song_id": "4",
      "title": "Similar Song Title",
      "artist": "Similar Artist",
      "genre": "Pop",
      "year": "2023",
      "similarity": 0.85,
      "type": "metadata_based_same_artist"
    },
    {
      "rank": 2,
      "song_id": "7",
      "title": "Another Similar Song",
      "artist": "Another Artist",
      "genre": "Dance",
      "year": "2022",
      "similarity": 0.82,
      "type": "embedding_based"
    }
  ]
}
```

**Recommendation Types:**
- `embedding_based`: Similar songs found using audio embedding similarity (vector distance)
- `metadata_based_same_artist`: Similar songs by the same or similar artist (metadata fallback)
- `metadata_based_random`: Random songs when no specific similarities found

### 4. Summary
```json
{
  "summary": {
    "total_recommendations": 10,
    "status": "match_found",
    "recommendation_type": "with_match"
  }
}
```

## Complete Example Output

```json
{
  "query": {
    "file": "Dua Lipa - Be The One.mp3",
    "start_time": 30.0,
    "duration": 5.0
  },
  "exact_match": {
    "found": true,
    "song_id": 2,
    "title": "Raataan Lambiyan",
    "artist": "Unknown",
    "confidence": 0.791,
    "distance": 0.063,
    "match_type": "exact"
  },
  "recommendations": [
    {
      "rank": 1,
      "song_id": "4",
      "title": "Saiyaara Full Song",
      "artist": "Unknown",
      "genre": "",
      "year": null,
      "similarity": 0.85,
      "type": "metadata_based_same_artist"
    },
    {
      "rank": 2,
      "song_id": "7",
      "title": "Qayde Se",
      "artist": "Unknown",
      "genre": "",
      "year": null,
      "similarity": 0.85,
      "type": "metadata_based_same_artist"
    }
  ],
  "summary": {
    "total_recommendations": 2,
    "status": "match_found",
    "recommendation_type": "with_match"
  }
}
```

## Output Files

- **query_result.json**: JSON file saved in current directory with complete results
- **Console output**: Formatted human-readable summary printed to terminal

## Architecture

The system uses a **two-tier architecture**:

1. **ID Database (songs.db)**: Contains 17 songs with 64-dim CNN embeddings for exact matching
2. **Recommendation Database (recommendations.db)**: Contains 1,000 songs with 12-dim MSD embeddings

When embedding dimensions don't match (64-dim query vs 12-dim recommendations), the system falls back to **metadata-based recommendations** using:
- Same artist matching
- Genre similarity
- Random similar songs

## Current Limitations

1. **False Positives**: Test songs (Dua Lipa, Harry Styles) are matching to Indian songs in the database because the ID database only contains 17 songs, none of which are the actual test songs. The system is finding the "closest" match even if it's not correct.

2. **Artist Metadata**: Many songs show "Unknown" as artist because MP3 files lack proper ID3 tags.

3. **Embedding Dimension Mismatch**: Query uses 64-dim CNN embeddings, but recommendation database uses 12-dim MSD embeddings, so semantic similarity isn't optimal.

## Recommendations for Improvement

1. **Add actual test songs to ID database**: Run `song_identifier.py` on test MP3s to add them to songs.db
2. **Use consistent embeddings**: Rebuild recommendation DB with 64-dim CNN embeddings
3. **Improve metadata**: Ensure MP3 files have proper ID3 tags (artist, title, genre, year)
4. **Lower confidence threshold**: Current 70% threshold allows false positives - increase to 90%+
5. **Train on more diverse music**: Current ID DB is mostly Indian songs - needs Western pop/rock for test files
