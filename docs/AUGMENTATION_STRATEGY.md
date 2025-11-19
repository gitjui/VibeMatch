# Audio Augmentation for Robust Song Matching

## Overview

This document describes the data augmentation pipeline used to train a more robust CNN classifier for Shazam-like audio fingerprinting.

## Why Augmentation?

In real-world scenarios, query audio can be degraded by:
- 📱 Different recording devices (phone mics vs studio mics)
- 🔊 Background noise (café, street, crowd)
- 🏠 Room acoustics (reverb, echo)
- 📉 Low-bitrate compression (MP3, streaming)
- 📞 Phone call quality (band-limited audio)
- 🔉 Volume variations
- ⏱️ Slight speed differences (turntables, tape players)

**Without augmentation**, the model only sees clean studio-quality audio and may fail on degraded queries.

**With augmentation**, the model learns features that are invariant to these real-world distortions.

## Augmentation Pipeline

Located in `audio_augmentations.py`, the `AudioAugmenter` class applies random transformations:

### 1. Time-Based Augmentations

**Time Stretch** (30% probability)
- Speed change: 0.95x - 1.05x (±5%)
- Simulates: Playback speed variations
- Label-preserving: ✅ (small changes don't change the song)

**Pitch Shift** (30% probability)
- Pitch change: ±0.5 semitones
- Simulates: Slight tuning differences
- Label-preserving: ✅ (very small shift)

### 2. Noise Augmentations

**Background Noise** (40% probability)
- Types: White noise, pink noise, brown noise
- SNR: 10-30 dB (noise is quieter than signal)
- Simulates: Café, street, room ambient noise
- Label-preserving: ✅ (song still audible)

### 3. Room Acoustics

**Reverb** (20% probability)
- Delay: 20-50 ms
- Decay: 0.2-0.5
- Simulates: Room echo, hall acoustics
- Label-preserving: ✅ (adds space, doesn't change identity)

### 4. Frequency Response

**EQ (Equalization)** (30% probability)
- Bass boost/cut: ±6 dB
- Treble boost/cut: ±6 dB
- Simulates: Different speakers, headphones, equalizer settings
- Label-preserving: ✅ (tonal balance changes, not identity)

**Phone Simulation** (20% probability)
- Band-pass filter: 300 Hz - 3400 Hz
- Simulates: Phone call quality, low-quality mics
- Label-preserving: ✅ (song recognizable despite bandwidth limit)

### 5. Signal Quality

**Volume Change** (50% probability)
- Gain: ±6 dB
- Simulates: Recording level differences
- Label-preserving: ✅ (amplitude doesn't change identity)

**Compression Artifacts** (20% probability)
- Bit depth reduction: 8-14 bits (from 16)
- Quantization noise
- Simulates: Low-bitrate MP3, streaming artifacts
- Label-preserving: ✅ (song still recognizable)

## Training Strategy

### Baseline Model (No Augmentation)
```bash
python cnn_classifier.py
```
- Trains on clean audio only
- Expected: High accuracy on clean queries, lower on degraded queries
- Embeddings stored as: `cnn_64`

### Augmented Model (With Augmentation)
```bash
python cnn_classifier_augmented.py
```
- 80% of training samples are augmented
- Each sample gets random combination of augmentations
- Expected: Slightly lower accuracy on clean audio, MUCH higher on degraded queries
- Embeddings stored as: `cnn_aug_64`

## Evaluation Workflow

### 1. Train Both Models
```bash
# Train baseline
python cnn_classifier.py

# Extract embeddings
python cnn_extract_embeddings.py

# Train augmented
python cnn_classifier_augmented.py

# Extract augmented embeddings
python cnn_extract_embeddings.py track_classifier_augmented.keras cnn_aug_64
```

### 2. Compare Performance
```bash
python compare_baseline_vs_augmented.py 30
```

This will:
- Test both models on 30 random queries
- Compare Top-1 accuracy, Top-5 accuracy, average rank
- Save results to `model_comparison_results.json`

### 3. Compare All Models
```bash
python compare_all_models.py
```

Compares:
- MFCC (20-dim, hand-crafted features)
- Mel-Spec (128-dim, enhanced hand-crafted)
- CNN Baseline (64-dim, learned features)
- CNN Augmented (64-dim, robust learned features)

## Expected Results

| Model | Clean Queries | Noisy Queries | Real-World |
|-------|--------------|---------------|------------|
| MFCC | 90% | 60-70% | Poor |
| Mel-Spec | 95% | 70-80% | Fair |
| CNN Baseline | 95-100% | 70-80% | Good |
| **CNN Augmented** | **90-95%** | **85-95%** | **Excellent** |

The augmented model trades a small drop on clean queries for significant robustness on degraded audio.

## Augmentation Hyperparameters

In `audio_augmentations.py`, you can tune:

```python
augmenter = AudioAugmenter(
    sr=22050,
    p_time_stretch=0.3,      # Probability of time stretch
    p_pitch_shift=0.3,       # Probability of pitch shift
    p_background_noise=0.4,  # Probability of noise
    p_reverb=0.2,           # Probability of reverb
    p_eq=0.3,               # Probability of EQ
    p_volume=0.5,           # Probability of volume change
    p_phone_sim=0.2,        # Probability of phone sim
    p_compression=0.2,      # Probability of compression
)
```

**Guidelines:**
- Start with p=0.2-0.5 for most augmentations
- Higher probabilities = more robustness but harder training
- Too strong augmentations can make songs unrecognizable
- Balance between robustness and accuracy

## Testing Augmentations

Visualize augmentations:
```bash
python audio_augmentations.py
```

This generates:
- `augmentation_examples.png` - Spectrogram comparison
- `aug_example_*.wav` - Audio files for each augmentation

Listen to ensure songs are still recognizable!

## Real-World Deployment

For a production Shazam-like system:

1. **Use augmented model** for embedding extraction
2. **Store augmented embeddings** in database
3. **Query audio**: No augmentation needed (model handles it)
4. **Expected performance**:
   - Studio recordings: 95-100% accuracy
   - Phone recordings: 85-95% accuracy
   - Noisy environments: 80-90% accuracy
   - Very degraded: 60-80% accuracy

## References

- **AudioSet**: Google's large-scale audio dataset (uses augmentation)
- **Shazam paper**: "An Industrial-Strength Audio Search Algorithm"
- **SpecAugment**: Augmentation for speech recognition (similar principles)
- **Data Augmentation for Audio**: Standard practice in audio ML

## Next Steps

To further improve robustness:

1. **Mixup augmentation**: Mix two songs with low weight
2. **SpecAugment**: Mask time/frequency regions in spectrogram
3. **Room impulse responses**: Real room acoustics from datasets
4. **Device-specific augmentation**: Phone mics, laptop mics, etc.
5. **More training data**: Additional songs = better generalization

---

**Summary**: Augmentation is essential for real-world audio systems. The small accuracy drop on clean audio is worth the massive robustness gain on degraded queries.
