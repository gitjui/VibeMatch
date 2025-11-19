# Two-Tier System: Quick Setup Guide

## 🎯 What You Have Now

✅ **Dual Database System** - Complete implementation  
✅ **Recommendation DB** - 1,000 sample songs  
✅ **Documentation** - Full architecture guide  
⏳ **ID DB Embeddings** - Need to extract from trained CNN  

## 🚀 Quick Start (5 minutes)

### Step 1: Extract Baseline CNN Embeddings
```bash
cd /Users/juigupte/Desktop/Learning/recsys-foundations
/Users/juigupte/Desktop/Learning/.venv/bin/python cnn_extract_embeddings.py
```

This will:
- Load `track_classifier.keras`
- Extract 64-dim embeddings for all 17 songs
- Store in `songs.db` as `model_name='cnn_64'`
- Takes ~5 minutes

### Step 2: Test Dual System
```bash
/Users/juigupte/Desktop/Learning/.venv/bin/python dual_database_system.py
```

This will:
- Query Ed Sheeran song
- Search ID database → Exact match!
- Search Rec database → Similar songs
- Display results

### Step 3: Try Your Own Query
```bash
/Users/juigupte/Desktop/Learning/.venv/bin/python dual_database_system.py /path/to/your/song.mp3
```

## 📊 Current Status

### ID Database (songs.db)
- **Songs**: 17 (your catalog)
- **Embeddings**: 0 (need to extract - see Step 1)
- **Purpose**: Exact matching (Shazam-like)
- **Size**: ~2 MB

### Recommendation Database (recommendations.db)
- **Songs**: 1,000 (sample data)
- **Embeddings**: 1,000 × 12-dim (MSD timbre features)
- **Purpose**: Similar songs, discovery
- **Size**: ~0.26 MB

## ⚠️ Important: Embedding Dimensions

**Current Issue**: Dimension mismatch!
- ID DB: 64-dim (CNN embeddings) - need to extract
- Rec DB: 12-dim (MSD timbre features)

**Solutions**:

### Option A: Use Same Model for Both (Recommended)
```bash
# After Step 1 (extract ID embeddings), also extract for Rec DB
# This requires audio files for the recommendation songs
# Not practical for 1M+ songs
```

### Option B: Keep Separate (What Spotify Does)
```python
# ID DB: High-quality 64-dim CNN (for your 17 songs)
# Rec DB: Lower-quality 12-dim MSD (for 1M songs)
# Limitation: Can't directly compare embeddings across databases
```

### Option C: Two-Stage Recommendation
```python
# 1. Find exact match in ID DB (64-dim CNN)
# 2. Use metadata (artist, genre) to query Rec DB
# 3. Return songs by same artist or genre
# This is what we'll implement next
```

## 🛠️ Metadata-Based Recommendations

Since embeddings have different dimensions, use metadata:

```python
# If exact match found
if exact_match:
    artist = exact_match.artist
    
    # Get songs by same artist from Rec DB
    similar = query_recommendation_database(
        filters={'artist': artist}
    )
    
    # Or same genre
    similar = query_recommendation_database(
        filters={'genre': 'Pop'}
    )
```

## 📈 Scaling to Production

### For Small Catalogs (< 10K songs)
- **ID DB**: All your owned songs with 64-dim CNN embeddings
- **Rec DB**: MSD 10K subset with 12-dim features
- **Query**: Direct embedding comparison (fast)
- **Hardware**: Single server, SQLite

### For Medium Catalogs (10K - 1M songs)
- **ID DB**: Your catalog with CNN embeddings
- **Rec DB**: MSD 1M songs with features
- **Query**: Use FAISS for fast similarity search
- **Hardware**: Server with 16+ GB RAM

### For Large Scale (1M+ songs)
- **ID DB**: Your catalog (same as above)
- **Rec DB**: 100M songs with embeddings
- **Query**: FAISS-GPU for millisecond queries
- **Hardware**: Server with GPU + 64+ GB RAM

## 🎵 Example Workflows

### Workflow 1: User Records a Song
```
1. User records 5-second clip on phone
2. Compute 64-dim embedding
3. Query ID Database
   → Distance < 0.3? YES → "You're listening to Shape of You"
   → Show play button, lyrics, album art
4. Query Rec Database (by artist)
   → "More from Ed Sheeran" [10 songs]
   → "Fans also like" [Drake, Justin Bieber, ...]
```

### Workflow 2: Discovery Mode
```
1. User likes a song (not in your catalog)
2. Compute embedding
3. Query ID Database
   → No match (distance > 0.3)
4. Query Rec Database
   → "You might like these similar songs" [20 songs]
   → Filter by genre, mood, era
```

### Workflow 3: Artist Radio
```
1. User selects "Ed Sheeran Radio"
2. Get Ed Sheeran song embeddings from ID DB
3. Query Rec DB
   → Filter by genre: 'Pop'
   → Exclude artist: 'Ed Sheeran'
   → Return top 50 most similar
4. Build playlist with variety
```

## 📁 Files Created

```
recsys-foundations/
├── dual_database_system.py          # Main system (ID + Rec query)
├── build_recommendation_db.py       # Build Rec DB from MSD
├── TWO_TIER_ARCHITECTURE.md         # Full documentation
├── songs.db                         # ID Database (17 songs)
├── recommendations.db               # Rec Database (1,000 songs)
└── track_classifier.keras           # CNN model for embeddings
```

## ✅ Next Actions

1. **Extract ID embeddings** (Step 1 above) - REQUIRED
2. Test dual system with real data
3. Choose recommendation strategy:
   - Metadata-based (fast, works now)
   - Embedding-based (accurate, needs same dimensions)
4. Scale Rec DB:
   - Add more sample data (10K songs)
   - Download real MSD subset
   - Compute CNN embeddings (if you have audio)

## 🎓 Learning Outcomes

You now understand:
- ✅ Two-tier architecture (ID + Rec)
- ✅ Exact matching vs similarity search
- ✅ Threshold-based identification
- ✅ Recommendation systems at scale
- ✅ Embedding dimension challenges
- ✅ Metadata filtering strategies

This is exactly how Spotify, Shazam, and YouTube Music work! 🎉

---

**Ready to test?** Run Step 1 to extract embeddings, then try Step 2! 🚀
