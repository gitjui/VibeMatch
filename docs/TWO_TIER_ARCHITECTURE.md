# Two-Tier Music System: ID + Recommendations

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      QUERY AUDIO (5s clip)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Compute Embedding    │
         │  (CNN 64-dim vector)  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   ID DATABASE (17)    │◄────── Your owned songs
         │  Exact Match Search   │        Shazam-like matching
         └───────────┬───────────┘
                     │
            ┌────────┴────────┐
            │                 │
        Distance < 0.3    Distance ≥ 0.3
        (EXACT MATCH)     (NO MATCH)
            │                 │
            ▼                 ▼
    ┌───────────────┐   ┌───────────────┐
    │ Return:       │   │ Return:       │
    │ • Song info   │   │ • "Unknown"   │
    │ • Confidence  │   │ • Similar     │
    │ • +Top 10     │   │   songs       │
    │   similar     │   │ • Top 20      │
    └───────────────┘   └───────────────┘
            │                 │
            └────────┬────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ RECOMMENDATION DB     │◄────── Million Song Dataset
         │ (1K-100M songs)       │        Spotify-like discovery
         │ Similarity Search     │
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Features:             │
         │ • Sounds like         │
         │ • Artist radio        │
         │ • Mood playlists      │
         │ • Genre neighbors     │
         └───────────────────────┘
```

## Two Databases Explained

### 1. **ID Database** (Identification - Your Catalog)

**Purpose**: Exact matching like Shazam

**Contents**:
- Your 17 songs (or your owned catalog)
- High-quality embeddings (64-dim CNN)
- Metadata: title, artist, album, etc.
- Multiple embeddings per song (5s chunks with overlap)

**Query Method**:
- Cosine similarity search
- Threshold-based: distance < 0.3 = "exact match"
- Returns confidence score (0-100%)

**Use Cases**:
- "What song is this?" (Shazam)
- Copyright detection
- Content ID for your catalog
- Royalty tracking

**Size**: 17 songs → ~1,995 embeddings → ~2 MB

---

### 2. **Recommendation Database** (Discovery - World's Music)

**Purpose**: Similar songs like Spotify Discover

**Contents**:
- Million Song Dataset (or 100M songs)
- Embeddings (12-dim MSD timbre or 64-dim CNN if you have audio)
- Metadata: artist, genre, year, tempo, energy, danceability
- Tags: mood, instruments, style

**Query Method**:
- Similarity search (no threshold)
- Returns top-K most similar
- Can filter by genre, year, artist, etc.

**Use Cases**:
- "Songs like this"
- Artist radio
- Mood playlists
- "If you like X, try Y"
- Discovery when exact match fails

**Size**: 
- 10K songs (subset) → ~20 MB
- 1M songs → ~200 MB
- 100M songs → ~20 GB (use FAISS for speed)

## Query Flow Examples

### Case A: Song IS in ID Database

```
User plays: "Shape of You" by Ed Sheeran (your catalog)

1. Compute embedding → [0.23, -0.45, 0.67, ...]
2. Query ID DB → Match found!
   - Title: Shape of You
   - Artist: Ed Sheeran
   - Confidence: 97%
   - Distance: 0.08 (< 0.3 threshold)

3. Query Recommendation DB → Similar songs:
   1. "Castle on the Hill" - Ed Sheeran (0.92 similarity)
   2. "Thinking Out Loud" - Ed Sheeran (0.89)
   3. "Galway Girl" - Ed Sheeran (0.87)
   4. "Perfect" - Ed Sheeran (0.85)
   5. "Sing" - Ed Sheeran (0.84)
   6. "Photograph" - Ed Sheeran (0.83)
   7. "One Dance" - Drake (0.81)
   8. "Closer" - The Chainsmokers (0.80)
   9. "Can't Stop the Feeling!" - Justin Timberlake (0.79)
   10. "Cheap Thrills" - Sia (0.78)

4. Display:
   ✓ You're listening to: Shape of You - Ed Sheeran
   🎵 Similar songs you might like: [10 songs above]
```

### Case B: Song is NOT in ID Database

```
User plays: "Bohemian Rhapsody" by Queen (not in your catalog)

1. Compute embedding → [0.12, -0.78, 0.34, ...]
2. Query ID DB → No match
   - Best candidate: "We Will Rock You" - Queen
   - Distance: 0.45 (≥ 0.3 threshold)
   - Result: UNKNOWN SONG

3. Query Recommendation DB → Similar songs:
   1. "Don't Stop Me Now" - Queen (0.94)
   2. "Somebody to Love" - Queen (0.92)
   3. "We Are the Champions" - Queen (0.90)
   4. "Radio Ga Ga" - Queen (0.88)
   5. "Under Pressure" - Queen & David Bowie (0.87)
   ...
   [Top 20 most similar tracks]

4. Display:
   ❌ Unknown song (not in our catalog)
   🎵 Based on what we heard, you might like: [20 songs above]
```

## Implementation

### Step 1: Build Recommendation Database

```bash
# Option A: Sample data for testing (1,000 songs, instant)
python build_recommendation_db.py --sample

# Option B: Sample data with custom count
python build_recommendation_db.py --sample --count 10000

# Option C: Real Million Song Dataset (10K songs, ~30 min)
python build_recommendation_db.py --msd
```

This creates `recommendations.db` with:
- Track ID, title, artist, year, genre
- Audio features: tempo, loudness, energy, danceability
- Embeddings (12-dim timbre from MSD)

### Step 2: Query the System

```bash
# Query a song
python dual_database_system.py path/to/audio.mp3

# Interactive demo
python dual_database_system.py
```

### Step 3: Integrate into Your App

```python
from dual_database_system import DualDatabaseSystem

# Initialize
system = DualDatabaseSystem(
    id_db_path='songs.db',
    rec_db_path='recommendations.db',
    model_path='track_classifier.keras',
    exact_match_threshold=0.3
)

# Query
results = system.query('user_recording.mp3', start_time=0, duration=5.0)

# Check if exact match
if results['exact_match']:
    song = results['exact_match']
    print(f"Match: {song.title} by {song.artist} ({song.confidence:.0%} confident)")
else:
    print("Unknown song")

# Show recommendations
for i, rec in enumerate(results['similar_songs'][:10], 1):
    print(f"{i}. {rec.title} - {rec.artist} ({rec.similarity:.0%} similar)")
```

## Features You Can Build

### 1. **"Sounds Like" Feature**
```python
# Find songs similar to a given track
similar = system.query_recommendation_database(
    embedding, 
    top_k=20,
    filters={'genre': 'Rock'}
)
```

### 2. **Artist Radio**
```python
# Get all songs by artist, then find similar songs by other artists
similar = system.query_recommendation_database(
    embedding,
    top_k=50,
    filters={'exclude_artist': 'Ed Sheeran'}  # Not same artist
)
```

### 3. **Mood Playlists**
```python
# Filter by audio features
happy_songs = system.query_recommendation_database(
    embedding,
    top_k=100,
    filters={
        'energy_min': 0.7,      # High energy
        'danceability_min': 0.6, # Danceable
        'tempo_min': 120         # Fast tempo
    }
)
```

### 4. **Genre Discovery**
```python
# Explore different genres
rock_similar = system.query_recommendation_database(
    embedding,
    top_k=20,
    filters={'genre': 'Rock'}
)

pop_similar = system.query_recommendation_database(
    embedding,
    top_k=20,
    filters={'genre': 'Pop'}
)
```

### 5. **Time Travel**
```python
# Find similar songs from a specific era
classic_rock = system.query_recommendation_database(
    embedding,
    top_k=30,
    filters={
        'genre': 'Rock',
        'year_min': 1970,
        'year_max': 1989
    }
)
```

## Scaling to 100M Songs

For large-scale deployment, use **FAISS** or **Annoy** for fast similarity search:

```python
import faiss

# Build FAISS index
embeddings = load_all_embeddings()  # (100M, 64)
index = faiss.IndexFlatIP(64)  # Inner product (cosine with normalized vectors)
faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
index.add(embeddings)

# Query
query_emb = compute_embedding(audio)
faiss.normalize_L2(query_emb.reshape(1, -1))
similarities, indices = index.search(query_emb.reshape(1, -1), k=100)

# indices contains top 100 most similar song IDs
```

**Performance**:
- 100M songs, 64-dim embeddings
- Query time: ~10ms (with GPU)
- Memory: ~24 GB RAM

## Comparison: ID DB vs Recommendation DB

| Feature | ID Database | Recommendation DB |
|---------|-------------|-------------------|
| **Purpose** | Exact matching | Discovery |
| **Size** | 17-10K songs | 1M-100M songs |
| **Threshold** | Yes (< 0.3) | No (top-K) |
| **Accuracy** | High (95%+) | N/A (similarity) |
| **Use Case** | "What song is this?" | "Songs like this" |
| **Update Frequency** | When you add songs | Rarely (static) |
| **Embeddings** | High quality (64-dim CNN) | Lower quality OK (12-dim MSD) |
| **Storage** | ~2 MB | 20 MB - 20 GB |

## Real-World Systems

### Shazam
- ID DB: 70M+ songs
- No recommendation DB (just identification)
- Returns exact match only

### Spotify
- ID DB: 100M+ songs (their catalog)
- Recommendation DB: Same 100M songs (for discovery)
- Uses both for "Discover Weekly", "Daily Mix", etc.

### Your System
- ID DB: 17 songs (your owned catalog)
- Recommendation DB: 10K-100M songs (world's music)
- Best of both: exact matching + discovery

## Next Steps

1. **Build recommendation DB** (5 minutes)
   ```bash
   python build_recommendation_db.py --sample
   ```

2. **Test dual system** (2 minutes)
   ```bash
   python dual_database_system.py
   ```

3. **Scale up** (optional)
   - Download full MSD (10K songs)
   - Add audio files and compute CNN embeddings
   - Use FAISS for fast search at scale

4. **Add features**
   - Playlists
   - Mood detection
   - Genre classification
   - User preferences

---

**You now have a complete music service:** exact matching (Shazam) + discovery (Spotify)! 🎵
