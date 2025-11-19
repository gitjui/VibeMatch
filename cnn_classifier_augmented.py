"""
CNN Track Classifier with Audio Augmentation
==============================================

Trains a CNN to classify 17 songs with robust augmentation pipeline.
Uses learned embeddings from penultimate layer for Shazam-like matching.

Key improvements over baseline:
- Data augmentation (time, pitch, noise, reverb, EQ, phone sim)
- More training diversity without needing more songs
- Model learns invariant features robust to real-world conditions
"""

import os
import sqlite3
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from audio_augmentations import TrainingAugmenter


class AugmentedTrackDataset:
    """
    Dataset loader with real-time audio augmentation.
    Loads audio chunks and applies augmentations on-the-fly.
    """
    
    def __init__(self, db_path='songs.db', music_dir='/Users/juigupte/Desktop/Learning/music/mp3',
                 chunk_duration=5.0, chunk_overlap=2.5, max_chunks_per_song=50,
                 sr=22050, augment_train=True):
        """
        Args:
            db_path: Path to songs database
            music_dir: Directory with MP3 files
            chunk_duration: Duration of each chunk (seconds)
            chunk_overlap: Overlap between chunks (seconds)
            max_chunks_per_song: Max chunks to extract per song
            sr: Sample rate
            augment_train: Whether to augment training data
        """
        self.db_path = db_path
        self.music_dir = music_dir
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_song = max_chunks_per_song
        self.sr = sr
        self.augment_train = augment_train
        
        # Create augmenter (80% of training samples will be augmented)
        self.augmenter = TrainingAugmenter(sr=sr, augment_probability=0.8)
        
        self.samples = []  # List of (song_id, file_path, start_time)
        self.song_id_to_label = {}  # Map song_id to integer label
        self.label_to_song_id = {}  # Reverse mapping
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load song metadata and create chunk samples."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all songs with file paths
        cursor.execute("""
            SELECT id, title, artist, file_path, duration 
            FROM songs 
            WHERE file_path IS NOT NULL
            ORDER BY id
        """)
        songs = cursor.fetchall()
        conn.close()
        
        if len(songs) == 0:
            raise ValueError("No songs found in database!")
        
        print(f"📚 Loading dataset from {len(songs)} songs...")
        
        # Create label mapping
        for idx, (song_id, title, artist, file_path, duration) in enumerate(songs):
            self.song_id_to_label[song_id] = idx
            self.label_to_song_id[idx] = song_id
        
        # Create chunks for each song
        for song_id, title, artist, file_path, duration in songs:
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
            
            # Generate chunk start times
            hop_size = self.chunk_duration - self.chunk_overlap
            num_chunks = min(
                int((duration - self.chunk_duration) / hop_size) + 1,
                self.max_chunks_per_song
            )
            
            for i in range(num_chunks):
                start_time = i * hop_size
                if start_time + self.chunk_duration <= duration:
                    self.samples.append((song_id, file_path, start_time))
        
        print(f"✓ Created {len(self.samples)} chunks from {len(songs)} songs")
        print(f"   Chunks per song: {len(self.samples) / len(songs):.1f}")
    
    def load_mel_spectrogram(self, file_path, start_time, augment=False):
        """
        Load audio chunk and compute mel spectrogram.
        
        Args:
            file_path: Path to audio file
            start_time: Start time in seconds
            augment: Whether to apply augmentation
            
        Returns:
            Mel spectrogram (215, 128) - ready for CNN
        """
        # Load audio (with optional augmentation)
        if augment and self.augment_train:
            y = self.augmenter(file_path, start_time, self.chunk_duration)
        else:
            y, _ = librosa.load(file_path, sr=self.sr, offset=start_time, 
                               duration=self.chunk_duration)
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=128, 
            n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Ensure consistent shape (215 time frames for 5s at sr=22050)
        target_frames = 215
        if mel_db.shape[1] < target_frames:
            # Pad if too short
            pad_width = target_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_db.shape[1] > target_frames:
            # Truncate if too long
            mel_db = mel_db[:, :target_frames]
        
        return mel_db.T  # Shape: (215, 128) - time x freq
    
    def get_datasets(self, test_size=0.2, random_state=42):
        """
        Split into train/validation sets.
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Create labels
        labels = [self.song_id_to_label[song_id] for song_id, _, _ in self.samples]
        
        # Split with stratification
        train_idx, val_idx = train_test_split(
            range(len(self.samples)),
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        print(f"\n📊 Dataset split:")
        print(f"   Training samples: {len(train_idx)}")
        print(f"   Validation samples: {len(val_idx)}")
        
        # Load training data (WITH augmentation)
        print("\n🔄 Loading training data with augmentation...")
        X_train = []
        y_train = []
        for i, idx in enumerate(train_idx):
            if (i + 1) % 100 == 0:
                print(f"   Loaded {i+1}/{len(train_idx)} training samples...")
            
            song_id, file_path, start_time = self.samples[idx]
            mel = self.load_mel_spectrogram(file_path, start_time, augment=True)
            X_train.append(mel)
            y_train.append(self.song_id_to_label[song_id])
        
        # Load validation data (NO augmentation - test on clean audio)
        print("\n✓ Loading validation data (clean, no augmentation)...")
        X_val = []
        y_val = []
        for i, idx in enumerate(val_idx):
            if (i + 1) % 50 == 0:
                print(f"   Loaded {i+1}/{len(val_idx)} validation samples...")
            
            song_id, file_path, start_time = self.samples[idx]
            mel = self.load_mel_spectrogram(file_path, start_time, augment=False)
            X_val.append(mel)
            y_val.append(self.song_id_to_label[song_id])
        
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        
        print(f"\n✓ Dataset ready!")
        print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        return X_train, X_val, y_train, y_val


class CNNTrackClassifier:
    """
    CNN model for track classification with augmented training.
    Architecture: Conv blocks → Global pooling → Dense(64) → Softmax(17)
    """
    
    def __init__(self, num_classes=17, embedding_dim=64):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model = None
        self.embedding_model = None
    
    def build_model(self, input_shape=(215, 128)):
        """
        Build CNN architecture.
        Slightly deeper than baseline to handle augmented data.
        """
        inputs = keras.Input(shape=input_shape, name='mel_spectrogram')
        
        # Expand to include channel dimension
        x = tf.expand_dims(inputs, axis=-1)  # (215, 128, 1)
        
        # Conv Block 1
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        # Conv Block 2
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        # Conv Block 3
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        # Global pooling
        x = keras.layers.GlobalAveragePooling2D()(x)
        
        # Embedding layer (this is what we'll extract later)
        embeddings = keras.layers.Dense(self.embedding_dim, activation='relu', 
                                       name='embeddings')(x)
        embeddings = keras.layers.Dropout(0.5)(embeddings)
        
        # Classification layer
        outputs = keras.layers.Dense(self.num_classes, activation='softmax', 
                                     name='classification')(embeddings)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='track_classifier')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print summary
        print("\n🏗️  Model Architecture:")
        self.model.summary()
        print(f"\n📊 Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model with callbacks."""
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'track_classifier_augmented.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print("\n🚀 Training with augmented data...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Find best epoch
        best_epoch = np.argmax(history.history['val_accuracy'])
        best_val_acc = history.history['val_accuracy'][best_epoch]
        best_val_loss = history.history['val_loss'][best_epoch]
        
        print(f"\n📊 Final Results:")
        print(f"   Validation accuracy: {best_val_acc:.2%}")
        print(f"   Validation loss: {best_val_loss:.4f}")
        
        return history
    
    def create_embedding_model(self):
        """Create model that outputs embeddings (not classifications)."""
        if self.model is None:
            raise ValueError("Must train model first!")
        
        # Extract embedding layer
        embedding_layer = self.model.get_layer('embeddings')
        self.embedding_model = keras.Model(
            inputs=self.model.input,
            outputs=embedding_layer.output,
            name='embedding_extractor'
        )
        
        print(f"✓ Created embedding model: {self.model.input.shape} -> ({self.embedding_dim},)")
        
        return self.embedding_model


def main():
    """Main training script with augmentation."""
    
    print("=" * 60)
    print("CNN Track Classifier with Audio Augmentation")
    print("=" * 60)
    
    # Load dataset with augmentation
    dataset = AugmentedTrackDataset(
        db_path='songs.db',
        music_dir='/Users/juigupte/Desktop/Learning/music/mp3',
        chunk_duration=5.0,
        chunk_overlap=2.5,
        max_chunks_per_song=50,
        augment_train=True  # Enable augmentation
    )
    
    # Get train/val splits
    X_train, X_val, y_train, y_val = dataset.get_datasets(test_size=0.2)
    
    # Build model
    num_classes = len(dataset.song_id_to_label)
    classifier = CNNTrackClassifier(num_classes=num_classes, embedding_dim=64)
    classifier.build_model(input_shape=(215, 128))
    
    # Train
    history = classifier.train(
        X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Save model
    classifier.model.save('track_classifier_augmented.keras')
    print("\n✓ Model saved to track_classifier_augmented.keras")
    
    # Create embedding extractor
    classifier.create_embedding_model()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Next: Extract embeddings from all songs")
    print("  python cnn_extract_embeddings.py --model track_classifier_augmented.keras")
    print("=" * 60)


if __name__ == '__main__':
    main()
