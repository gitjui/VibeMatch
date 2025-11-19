"""
Audio augmentation pipeline for robust Shazam-like training.

Simulates real-world conditions:
- Different devices/microphones
- Background noise
- Room acoustics
- Encoding artifacts
- Phone calls, recordings, etc.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import random


class AudioAugmenter:
    """
    Applies random augmentations to audio to simulate real-world conditions.
    
    Each augmentation has a probability of being applied.
    """
    
    def __init__(
        self,
        sr=22050,
        # Probabilities for each augmentation
        p_time_stretch=0.3,
        p_pitch_shift=0.3,
        p_background_noise=0.4,
        p_reverb=0.2,
        p_eq=0.3,
        p_volume=0.5,
        p_phone_sim=0.2,
        p_compression=0.2,
    ):
        self.sr = sr
        self.p_time_stretch = p_time_stretch
        self.p_pitch_shift = p_pitch_shift
        self.p_background_noise = p_background_noise
        self.p_reverb = p_reverb
        self.p_eq = p_eq
        self.p_volume = p_volume
        self.p_phone_sim = p_phone_sim
        self.p_compression = p_compression
    
    def augment(self, y):
        """
        Apply random augmentations to audio signal.
        
        Args:
            y: Audio time series (1D numpy array)
            
        Returns:
            Augmented audio
        """
        # Apply augmentations with their probabilities
        if random.random() < self.p_time_stretch:
            y = self._time_stretch(y)
        
        if random.random() < self.p_pitch_shift:
            y = self._pitch_shift(y)
        
        if random.random() < self.p_background_noise:
            y = self._add_background_noise(y)
        
        if random.random() < self.p_reverb:
            y = self._add_reverb(y)
        
        if random.random() < self.p_eq:
            y = self._apply_eq(y)
        
        if random.random() < self.p_phone_sim:
            y = self._simulate_phone(y)
        
        if random.random() < self.p_volume:
            y = self._change_volume(y)
        
        if random.random() < self.p_compression:
            y = self._simulate_compression(y)
        
        # Ensure output is valid
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to prevent clipping
        if np.abs(y).max() > 1.0:
            y = y / np.abs(y).max() * 0.95
        
        return y
    
    def _time_stretch(self, y):
        """
        Time-stretch audio (change speed without changing pitch).
        Small range to keep song recognizable.
        """
        rate = random.uniform(0.95, 1.05)  # ±5% speed change
        return librosa.effects.time_stretch(y, rate=rate)
    
    def _pitch_shift(self, y):
        """
        Pitch shift audio.
        Small range (±0.5 semitones) to stay recognizable.
        """
        n_steps = random.uniform(-0.5, 0.5)  # Very small pitch shift
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
    
    def _add_background_noise(self, y):
        """
        Add background noise (white, pink, or colored).
        Simulates café, street, room noise.
        """
        noise_types = ['white', 'pink', 'brown']
        noise_type = random.choice(noise_types)
        
        # Generate noise
        if noise_type == 'white':
            noise = np.random.randn(len(y))
        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            noise = self._generate_pink_noise(len(y))
        else:  # brown
            # Brown noise (1/f^2 spectrum)
            noise = np.cumsum(np.random.randn(len(y)))
            noise = noise / np.abs(noise).max()
        
        # Random SNR between 10-30 dB (noise is quieter than signal)
        snr_db = random.uniform(10, 30)
        
        # Calculate noise scaling factor
        signal_power = np.mean(y ** 2)
        noise_power = np.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        
        return y + noise_scale * noise
    
    def _generate_pink_noise(self, length):
        """Generate pink noise (1/f power spectrum)."""
        # Simple approximation using white noise and filtering
        white = np.random.randn(length)
        # Low-pass filter to approximate pink noise
        b, a = signal.butter(1, 0.5)
        pink = signal.filtfilt(b, a, white)
        return pink / np.abs(pink).max()
    
    def _add_reverb(self, y):
        """
        Add simple reverb (simulate room acoustics).
        Uses exponentially decaying delays.
        """
        # Simple reverb using comb filter
        delay_samples = random.randint(int(0.02 * self.sr), int(0.05 * self.sr))
        decay = random.uniform(0.2, 0.5)
        
        reverb = np.zeros(len(y) + delay_samples)
        reverb[:len(y)] = y
        
        # Add delayed and decayed copies
        for i in range(3):
            delay = delay_samples * (i + 1)
            if len(y) + delay < len(reverb):
                reverb[delay:delay+len(y)] += y * (decay ** (i + 1))
        
        return reverb[:len(y)]
    
    def _apply_eq(self, y):
        """
        Apply random EQ (cut/boost frequencies).
        Simulates different speaker/headphone responses.
        """
        # Randomly boost or cut low and high frequencies
        low_gain_db = random.uniform(-6, 6)
        high_gain_db = random.uniform(-6, 6)
        
        # Apply simple shelving filters
        # Low shelf (boost/cut bass)
        if abs(low_gain_db) > 0.1:
            b, a = signal.butter(2, 500 / (self.sr / 2), btype='low')
            low_shelf = signal.filtfilt(b, a, y)
            y = y + low_shelf * (10 ** (low_gain_db / 20) - 1)
        
        # High shelf (boost/cut treble)
        if abs(high_gain_db) > 0.1:
            b, a = signal.butter(2, 4000 / (self.sr / 2), btype='high')
            high_shelf = signal.filtfilt(b, a, y)
            y = y + high_shelf * (10 ** (high_gain_db / 20) - 1)
        
        return y
    
    def _simulate_phone(self, y):
        """
        Simulate phone microphone (band-limited).
        Typical phone: 300 Hz - 3400 Hz.
        """
        # Band-pass filter
        low_freq = 300
        high_freq = 3400
        
        sos = signal.butter(4, [low_freq / (self.sr / 2), high_freq / (self.sr / 2)], 
                           btype='band', output='sos')
        y_filtered = signal.sosfilt(sos, y)
        
        return y_filtered
    
    def _change_volume(self, y):
        """
        Random volume gain/attenuation.
        Simulates different recording levels.
        """
        gain_db = random.uniform(-6, 6)
        gain_linear = 10 ** (gain_db / 20)
        return y * gain_linear
    
    def _simulate_compression(self, y):
        """
        Simulate lossy compression artifacts.
        Add slight distortion and quantization noise.
        """
        # Simple simulation: add quantization noise
        bits = random.randint(8, 14)  # Simulate lower bit depth
        quantization_levels = 2 ** bits
        
        # Quantize and dequantize
        y_normalized = y / np.abs(y).max() if np.abs(y).max() > 0 else y
        y_quantized = np.round(y_normalized * quantization_levels) / quantization_levels
        
        # Add tiny amount of noise to simulate compression artifacts
        noise = np.random.randn(len(y)) * 0.001
        
        return y_quantized * np.abs(y).max() + noise


class TrainingAugmenter:
    """
    Wrapper for training with augmentation.
    Provides both clean and augmented versions of audio.
    """
    
    def __init__(self, sr=22050, augment_probability=0.8):
        """
        Args:
            sr: Sample rate
            augment_probability: Probability of applying augmentation to a sample
        """
        self.sr = sr
        self.augment_probability = augment_probability
        self.augmenter = AudioAugmenter(sr=sr)
    
    def __call__(self, audio_path, chunk_start, chunk_duration):
        """
        Load audio chunk and optionally apply augmentation.
        
        Args:
            audio_path: Path to audio file
            chunk_start: Start time in seconds
            chunk_duration: Duration in seconds
            
        Returns:
            Audio chunk (augmented with probability augment_probability)
        """
        # Load audio chunk
        y, sr = librosa.load(audio_path, sr=self.sr, offset=chunk_start, 
                            duration=chunk_duration)
        
        # Apply augmentation with probability
        if random.random() < self.augment_probability:
            y = self.augmenter.augment(y)
        
        return y


# Example usage and testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Load a test audio file
    test_file = '/Users/juigupte/Desktop/Learning/music/mp3/Ed Sheeran - Shape of You (Official Music Video).mp3'
    
    y_orig, sr = librosa.load(test_file, sr=22050, duration=5.0)
    
    # Create augmenter
    augmenter = AudioAugmenter(sr=sr)
    
    # Apply augmentations
    augmentations = {
        'Original': y_orig,
        'Time Stretch': augmenter._time_stretch(y_orig.copy()),
        'Pitch Shift': augmenter._pitch_shift(y_orig.copy()),
        'Background Noise': augmenter._add_background_noise(y_orig.copy()),
        'Reverb': augmenter._add_reverb(y_orig.copy()),
        'Phone Sim': augmenter._simulate_phone(y_orig.copy()),
        'Full Pipeline': augmenter.augment(y_orig.copy()),
    }
    
    # Plot spectrograms
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, y) in enumerate(augmentations.items()):
        if i >= len(axes):
            break
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Plot
        img = axes[i].imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(name)
        axes[i].set_ylabel('Mel Frequency')
        axes[i].set_xlabel('Time')
        plt.colorbar(img, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved augmentation examples to augmentation_examples.png")
    
    # Save audio examples
    for name, y in augmentations.items():
        filename = f"aug_example_{name.replace(' ', '_').lower()}.wav"
        sf.write(filename, y, sr)
    print("✓ Saved audio examples")
