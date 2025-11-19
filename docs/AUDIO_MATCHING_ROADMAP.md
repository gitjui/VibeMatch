# Audio Matching System Roadmap 🎵

## Your Mini Shazam Lab - Implementation Plan

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Your 17-Song Database                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 1: Metadata (songs table)                            │
│  ├─ "What is this song?"                                     │
│  ├─ Title, artist, album, duration, language                │
│  └─ recording_id (unique key)                               │
│                                                               │
│  Layer 2: Audio Features (track_embeddings table) ← NEW!    │
│  ├─ "How does this song sound?"                             │
│  ├─ Embeddings/fingerprints per audio chunk                 │
│  └─ Links to songs via recording_id                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: MFCC Baseline (Non-Learned) ✅ IMPLEMENTED

**Goal**: Establish baseline accuracy before ML models

**Method**: MFCCs (Mel-Frequency Cepstral Coefficients)
- Classic audio features capturing spectral envelope
- 20-dimensional vectors per audio chunk
- Cosine similarity for matching

**What we built**:
1. ✅ `track_embeddings` table schema
2. ✅ `MFCCExtractor` class (full song + chunking)
3. ✅ `EmbeddingMatcher` with cosine similarity search
4. ✅ `AudioEmbeddingsDB` for storage
5. ✅ Evaluation framework

**How to use**:
```bash
# Step 1: Extract embeddings from your 17 songs
python audio_embeddings.py

# Step 2: Test accuracy with random snippets
python test_audio_matching.py

# Step 3: Query specific song
python test_audio_matching.py "path/to/song.mp3" 10.0
```

**Expected Results**:
- Top-1 accuracy: 40-70% (depending on song diversity)
- This is your baseline to beat with ML models!

---

### Phase 2: Pre-trained Models (Learned Features)

**Goal**: Improve accuracy using models trained on millions of songs

#### Option 2A: VGGish (Audio Classification)
- **What**: CNN trained on YouTube-8M (audio tagging)
- **Output**: 128-dimensional embeddings
- **Good for**: General audio similarity
- **How to use**: `tensorflow`, `tensorflow_hub`

```python
import tensorflow_hub as hub

# Load model
vggish = hub.load('https://tfhub.dev/google/vggish/1')

# Extract features
embeddings = vggish(waveform)
```

#### Option 2B: OpenL3 (Look, Listen, Learn)
- **What**: Self-supervised audio embeddings
- **Output**: 512 or 6144 dimensions
- **Good for**: Music similarity
- **How to use**: `openl3` library

```python
import openl3

# Extract features
emb, timestamps = openl3.get_audio_embedding(audio, sr)
```

#### Option 2C: CLAP (Contrastive Language-Audio)
- **What**: Audio-text joint embeddings
- **Output**: Variable dimensions
- **Good for**: Semantic audio understanding
- **How to use**: `laion_clap` library

```python
from laion_clap import CLAP_Module

# Load model
clap = CLAP_Module(enable_fusion=False)

# Extract audio embeddings
embeddings = clap.get_audio_embedding_from_filelist(file_list)
```

**Implementation Steps**:
1. Create `VGGishExtractor` class (similar to `MFCCExtractor`)
2. Add `model_name='vggish'` to track_embeddings
3. Run evaluation to compare vs MFCC baseline
4. Measure improvement!

---

### Phase 3: Custom Model (Learn Your Own)

**Goal**: Train a model specifically for YOUR music taste/collection

#### Path 3A: Siamese Network
- **What**: Learn to measure similarity directly
- **Training**: Pairs of audio chunks (similar/dissimilar)
- **Loss**: Contrastive or triplet loss

```
Input: Audio chunk A, Audio chunk B
       ↓                ↓
   [Encoder]        [Encoder]  ← Shared weights
       ↓                ↓
   Embedding A     Embedding B
       ↓                ↓
    Similarity metric
       ↓
   Loss (push similar together, dissimilar apart)
```

#### Path 3B: Autoencoder
- **What**: Compress audio into meaningful embeddings
- **Training**: Reconstruct input audio from embedding
- **Use**: Embedding space captures audio characteristics

```
Audio → [Encoder] → Embedding → [Decoder] → Reconstructed Audio
                        ↓
                Use this for matching!
```

---

### Evaluation Metrics

| Metric | What it measures | Target |
|--------|------------------|--------|
| **Top-1 Accuracy** | % correct song is #1 match | > 80% |
| **Top-5 Accuracy** | % correct song in top 5 | > 95% |
| **Average Rank** | Position of correct match | < 1.5 |
| **Avg Similarity** | Confidence of matches | > 0.7 |

---

### Learning Roadmap (Conceptual)

```
Phase 1: MFCC Baseline ← YOU ARE HERE
   ↓
├─ Run evaluation
├─ Measure: Top-1 accuracy = X%
└─ This is your baseline!

Phase 2: Pre-trained Model (e.g., VGGish)
   ↓
├─ Same evaluation protocol
├─ Measure: Top-1 accuracy = Y%
└─ Improvement: (Y - X)%

Phase 3: Custom Model (if needed)
   ↓
├─ Collect training pairs from your library
├─ Train Siamese network
└─ Measure: Top-1 accuracy = Z%
```

---

### Database Schema

```sql
-- Existing table (metadata)
CREATE TABLE songs (
    recording_id TEXT PRIMARY KEY,
    title TEXT,
    artist TEXT,
    -- ... 36 fields total
);

-- NEW: Embeddings table
CREATE TABLE track_embeddings (
    id INTEGER PRIMARY KEY,
    track_id TEXT,                  -- FK to songs.recording_id
    chunk_start REAL,               -- Start time (seconds)
    chunk_end REAL,                 -- End time (seconds)
    embedding BLOB,                 -- Pickled numpy array
    model_name TEXT,                -- 'mfcc_20', 'vggish', etc.
    embedding_dim INTEGER,          -- Vector size
    created_at TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES songs(recording_id)
);

-- Indexes for fast lookup
CREATE INDEX idx_track_id ON track_embeddings(track_id);
CREATE INDEX idx_model_name ON track_embeddings(model_name);
```

---

### Next Steps

1. **Run Phase 1 baseline** ✅
   ```bash
   python audio_embeddings.py
   python test_audio_matching.py
   ```

2. **Interpret results**
   - What's your Top-1 accuracy?
   - Which songs are confused?
   - Are Hindi/English songs clustered separately?

3. **Pick a pre-trained model**
   - Start with VGGish (easiest to integrate)
   - Or OpenL3 (better for music)

4. **Compare results**
   - Same evaluation, different model
   - Quantify improvement

5. **Advanced: Custom training**
   - If you want to learn deep learning
   - Requires more songs + GPU

---

### File Structure

```
recsys-foundations/
├── songs.db                      # Main database
├── audio_embeddings.py           # Phase 1 implementation ✅
├── test_audio_matching.py        # Evaluation framework ✅
├── AUDIO_MATCHING_ROADMAP.md     # This file ✅
│
├── (future) vggish_embeddings.py # Phase 2A
├── (future) openl3_embeddings.py # Phase 2B
├── (future) train_siamese.py     # Phase 3
└── (future) model_comparison.py  # Compare all models
```

---

### Resources for Learning

1. **Audio ML Basics**
   - librosa documentation (feature extraction)
   - "Speech and Language Processing" (Jurafsky & Martin)

2. **Pre-trained Models**
   - TensorFlow Hub (VGGish, YAMNet)
   - OpenL3 paper & repo
   - CLAP (LAION)

3. **Metric Learning**
   - Siamese networks tutorial
   - Triplet loss explained
   - Contrastive learning survey

4. **Evaluation**
   - Information Retrieval metrics (Precision@K, MRR)
   - Audio fingerprinting papers (Shazam algorithm)

---

## Summary

You now have a **complete audio matching pipeline**:

1. ✅ Extract audio embeddings (MFCC baseline)
2. ✅ Store in SQLite (`track_embeddings` table)
3. ✅ Query with similarity search (cosine)
4. ✅ Evaluate accuracy on your 17-song library

**This is a real Shazam lab!** 🎉

The difference from production Shazam:
- They have billions of songs → you have 17
- They use specialized fingerprinting → you use MFCC/VGGish
- They have distributed search → you use SQLite

But the **core ideas are identical**:
1. Audio → Embeddings
2. Store embeddings
3. Query by similarity
4. Return matches

Now go measure your baseline accuracy! 📊
