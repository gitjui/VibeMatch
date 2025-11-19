# Quick Start: Training with Augmentation

## ✅ What's Ready

You now have a complete audio augmentation pipeline for robust Shazam-like training!

### Files Created:
1. **`audio_augmentations.py`** - Augmentation module with 8 transformations
2. **`cnn_classifier_augmented.py`** - CNN trainer with augmentation (80% of samples)
3. **`compare_baseline_vs_augmented.py`** - Evaluation comparing both models
4. **`AUGMENTATION_STRATEGY.md`** - Full documentation

### Augmentation Examples Generated:
- ✅ `augmentation_examples.png` - Visual spectrogram comparison
- ✅ `aug_example_*.wav` - 7 audio files showing each augmentation

### Current Status:
- ✅ Baseline CNN trained (79.19% val accuracy)
- ⏳ Baseline embeddings not yet extracted
- ⏳ Augmented CNN not yet trained
- ⏳ Augmented embeddings not yet extracted

---

## 🚀 Next Steps (in order)

### 1. Extract Baseline Embeddings (5 minutes)
```bash
cd /Users/juigupte/Desktop/Learning/recsys-foundations
/Users/juigupte/Desktop/Learning/.venv/bin/python cnn_extract_embeddings.py
```

This will:
- Load `track_classifier.keras` (baseline model)
- Extract 64-dim embeddings from penultimate layer
- Store ~1,995 embeddings as `model_name='cnn_64'`
- Database grows to ~3 MB

### 2. Train Augmented CNN (20-30 minutes)
```bash
/Users/juigupte/Desktop/Learning/.venv/bin/python cnn_classifier_augmented.py
```

This will:
- Load 744 chunks from 17 songs
- Apply augmentation to 80% of training samples
- Train for 50 epochs (with early stopping)
- Save best model to `track_classifier_augmented.keras`
- Expected: 70-85% val accuracy (on clean audio, but more robust)

**Note**: Training takes longer than baseline due to real-time augmentation!

### 3. Extract Augmented Embeddings (5 minutes)
```bash
/Users/juigupte/Desktop/Learning/.venv/bin/python cnn_extract_embeddings.py \
    track_classifier_augmented.keras \
    cnn_aug_64
```

This will:
- Load augmented model
- Extract embeddings and store as `model_name='cnn_aug_64'`
- Database grows to ~3.5 MB with 5,985 total embeddings

### 4. Compare Baseline vs Augmented (2 minutes)
```bash
/Users/juigupte/Desktop/Learning/.venv/bin/python compare_baseline_vs_augmented.py 30
```

This will:
- Test both models on 30 random queries (clean audio)
- Compare Top-1 accuracy, Top-5 accuracy
- Save results to `model_comparison_results.json`
- Print analysis of augmentation impact

### 5. Compare All Models (3 minutes)
```bash
/Users/juigupte/Desktop/Learning/.venv/bin/python compare_all_models.py
```

This will compare:
- MFCC (20-dim, hand-crafted) - 90% baseline
- Mel-Spec (128-dim, enhanced) - 95% 
- CNN Baseline (64-dim, learned) - Expected 95-100%
- CNN Augmented (64-dim, robust) - Expected 90-95%

---

## 📊 Expected Timeline

| Step | Time | Bottleneck |
|------|------|------------|
| Extract baseline embeddings | 5 min | Mel-spec computation |
| Train augmented CNN | 20-30 min | Real-time augmentation + training |
| Extract augmented embeddings | 5 min | Mel-spec computation |
| Compare models | 2 min | Database queries |
| Compare all | 3 min | Multiple evaluations |
| **Total** | **35-45 min** | |

---

## 🔍 What to Look For

### During Augmented Training:
- **Training accuracy**: Should be lower than baseline (60-80%)
  - This is GOOD! Augmented data is harder to fit
- **Validation accuracy**: May be slightly lower than baseline (70-85% vs 79%)
  - Still good! Model is learning robust features
- **Training time**: ~2x slower than baseline
  - Real-time augmentation adds overhead

### During Comparison:
- **Baseline model**: Higher accuracy on clean queries
- **Augmented model**: More robust, better generalization
- **On clean audio**: Baseline may win by 5-10%
- **On noisy audio**: Augmented should win by 10-20%

### Expected Results:
```
FINAL COMPARISON
================
Model                          Top-1 Acc    Top-5 Acc    Avg Rank    Avg Sim
--------------------------------------------------------------------------------
Baseline CNN (no augment)      96.7%        100.0%       1.13        0.891
Augmented CNN (with augment)   90.0%        100.0%       1.30        0.867
```

**Analysis**: Augmented model trades 5-10% clean accuracy for robustness on degraded audio.

---

## 🎵 Testing with Noisy Queries

To truly test augmentation benefits, add noise to queries:

```python
from audio_augmentations import AudioAugmenter
import librosa

# Load query
y, sr = librosa.load('query.mp3', sr=22050, duration=5.0)

# Add noise
augmenter = AudioAugmenter(sr=sr)
y_noisy = augmenter._add_background_noise(y)

# Test both models on noisy query
# Augmented model should perform MUCH better!
```

---

## 🎯 Key Insights

### Augmentation Benefits:
1. **Robustness**: Handles noisy, degraded audio
2. **Generalization**: Better on unseen conditions
3. **Real-world ready**: Phone recordings, café noise, etc.

### Augmentation Trade-offs:
1. **Training time**: 2x slower (real-time augmentation)
2. **Clean accuracy**: Slight drop (5-10%)
3. **Model complexity**: Same architecture, different training

### When to Use Each Model:

| Scenario | Best Model |
|----------|-----------|
| Studio recordings | Baseline CNN |
| Phone recordings | Augmented CNN |
| Noisy environments | Augmented CNN |
| Production deployment | **Augmented CNN** |

---

## 📁 File Organization

No need to create a separate `training/` folder. The current structure is clean:

```
recsys-foundations/
├── audio_augmentations.py          # Augmentation module
├── cnn_classifier.py                # Baseline trainer
├── cnn_classifier_augmented.py      # Augmented trainer
├── cnn_extract_embeddings.py        # Embedding extractor (both)
├── compare_baseline_vs_augmented.py # Comparison script
├── track_classifier.keras           # Baseline model (79% val)
├── track_classifier_augmented.keras # Augmented model (pending)
├── songs.db                         # Database with embeddings
└── augmentation_examples.png        # Visual examples
```

Everything is in one place, organized by purpose.

---

## 🔧 Tuning Augmentation

If augmented model performs poorly, adjust in `audio_augmentations.py`:

```python
# Reduce augmentation strength
augmenter = AudioAugmenter(
    sr=22050,
    p_time_stretch=0.2,      # Lower from 0.3
    p_pitch_shift=0.2,       # Lower from 0.3
    p_background_noise=0.3,  # Lower from 0.4
    p_reverb=0.1,           # Lower from 0.2
    p_eq=0.2,               # Lower from 0.3
    p_volume=0.4,           # Lower from 0.5
    p_phone_sim=0.1,        # Lower from 0.2
    p_compression=0.1,      # Lower from 0.2
)
```

Or reduce augmentation probability in `cnn_classifier_augmented.py`:
```python
self.augmenter = TrainingAugmenter(
    sr=sr, 
    augment_probability=0.5  # Lower from 0.8
)
```

---

## 📚 Documentation

- **`AUGMENTATION_STRATEGY.md`** - Full technical details
- **`AUDIO_MATCHING_ROADMAP.md`** - Overall project roadmap
- **`PHASE2_COMPARISON.md`** - Mel-spec vs MFCC results
- **`README_MINI_SHAZAM.md`** - System overview

---

## ✅ Checklist

- [x] Augmentation module created
- [x] Augmented trainer created
- [x] Comparison scripts created
- [x] Baseline model trained (79% val)
- [x] Examples generated (PNG + WAV files)
- [ ] Baseline embeddings extracted
- [ ] Augmented model trained
- [ ] Augmented embeddings extracted
- [ ] Models compared
- [ ] Results documented

---

**Ready to train!** Start with step 1 above. 🚀
