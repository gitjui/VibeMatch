"""
CNN Track Classifier - Phase 3
===============================

Train a CNN to classify which of 17 songs a chunk belongs to.
Use penultimate layer as learned embeddings for matching.

Architecture:
    Input: Mel-spectrogram (128 mel bins × time frames)
    CNN: Learn features
    Output: 17-way softmax (one per song)
    Embeddings: Penultimate layer activations

This is supervised learning - the model learns what makes each song unique!
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import sqlite3
import pickle
import librosa
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TrackClassifierDataset:
    """Dataset loader for track classification"""
    
    def __init__(self, db_path: str = "songs.db"):
        self.db_path = db_path
        self.label_encoder = LabelEncoder()
    
    def load_chunks_from_files(self, chunk_duration: float = 5.0,
                               max_chunks_per_song: int = 50) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load mel-spectrogram chunks directly from audio files
        
        Args:
            chunk_duration: Duration of each chunk
            max_chunks_per_song: Limit chunks per song (for balance)
            
        Returns:
            X: Mel-spectrograms (N, time_frames, n_mels)
            y: Song IDs (N,)
            metadata: Dict with song info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, title FROM songs WHERE file_path IS NOT NULL")
        songs = cursor.fetchall()
        conn.close()
        
        print(f"\n📂 Loading chunks from {len(songs)} songs...")
        print(f"   Chunk duration: {chunk_duration}s")
        print(f"   Max per song: {max_chunks_per_song}")
        
        X_list = []
        y_list = []
        song_metadata = {}
        
        for song_id, file_path, title in songs:
            if not file_path or not Path(file_path).exists():
                continue
            
            print(f"Loading: {Path(file_path).name[:50]}...")
            
            try:
                # Load audio
                y_audio, sr = librosa.load(file_path, sr=22050, mono=True)
                
                # Extract chunks with fixed length
                hop_length = 512
                chunk_frames = int(chunk_duration * sr / hop_length)
                
                # Extract mel-spectrogram for entire song
                mel_spec = librosa.feature.melspectrogram(
                    y=y_audio, sr=sr, n_mels=128, hop_length=hop_length
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Split into fixed-size chunks
                num_chunks = mel_spec_db.shape[1] // chunk_frames
                
                for i in range(min(num_chunks, max_chunks_per_song)):
                    start_frame = i * chunk_frames
                    end_frame = start_frame + chunk_frames
                    
                    chunk = mel_spec_db[:, start_frame:end_frame]
                    
                    if chunk.shape[1] == chunk_frames:
                        X_list.append(chunk.T)  # (time, mels)
                        y_list.append(song_id)
                
                song_metadata[song_id] = title or Path(file_path).stem
                print(f"  → Extracted {min(num_chunks, max_chunks_per_song)} chunks")
                
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Encode labels to 0-16 range
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n✓ Dataset loaded:")
        print(f"  Total chunks: {len(X)}")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {len(np.unique(y_encoded))}")
        
        return X, y_encoded, song_metadata


class CNNTrackClassifier:
    """CNN model for track classification"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, 
                 embedding_dim: int = 64):
        """Initialize CNN classifier
        
        Args:
            input_shape: (time_frames, n_mels)
            num_classes: Number of songs (17)
            embedding_dim: Size of embedding layer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model = None
        self.embedding_model = None
        
    def build_model(self):
        """Build CNN architecture
        
        Architecture:
            Input → Conv2D → MaxPool → Conv2D → MaxPool 
                 → Flatten → Dense(embedding_dim) → Dense(num_classes)
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Add channel dimension for Conv2D
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        # Conv Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Conv Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Conv Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)  # Reduce to 128-dim
        x = layers.Dropout(0.4)(x)
        
        # Embedding layer (this is what we'll extract!)
        embeddings = layers.Dense(self.embedding_dim, activation='relu', 
                                  name='embeddings')(x)
        x = layers.Dropout(0.3)(embeddings)
        
        # Classification head
        outputs = layers.Dense(self.num_classes, activation='softmax', 
                              name='classifier')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='track_classifier')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n🏗️  Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs: int = 50, batch_size: int = 32):
        """Train the classifier
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            History object
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        print(f"\n🚀 Training...")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Val samples: {len(X_val)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}\n")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def build_embedding_model(self):
        """Create model that outputs embeddings (penultimate layer)"""
        embedding_layer = self.model.get_layer('embeddings')
        self.embedding_model = keras.Model(
            inputs=self.model.input,
            outputs=embedding_layer.output,
            name='embedding_extractor'
        )
        
        print(f"\n✓ Embedding model created:")
        print(f"   Input: {self.embedding_model.input_shape}")
        print(f"   Output: {self.embedding_model.output_shape}")
        
        return self.embedding_model
    
    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from input data
        
        Args:
            X: Input mel-spectrograms
            
        Returns:
            Embeddings array
        """
        if self.embedding_model is None:
            self.build_embedding_model()
        
        embeddings = self.embedding_model.predict(X, verbose=0)
        return embeddings
    
    def save_model(self, path: str = "track_classifier.keras"):
        """Save trained model"""
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str = "track_classifier.keras"):
        """Load trained model"""
        self.model = keras.models.load_model(path)
        self.build_embedding_model()
        print(f"✓ Model loaded from {path}")


def train_classifier(db_path: str = "songs.db",
                    chunk_duration: float = 5.0,
                    max_chunks_per_song: int = 50,
                    embedding_dim: int = 64,
                    epochs: int = 50):
    """Train track classifier and save embeddings
    
    Args:
        db_path: Path to database
        chunk_duration: Chunk duration in seconds
        max_chunks_per_song: Max chunks to use per song
        embedding_dim: Size of embedding layer
        epochs: Training epochs
    """
    # Load dataset
    dataset = TrackClassifierDataset(db_path)
    X, y, metadata = dataset.load_chunks_from_files(
        chunk_duration=chunk_duration,
        max_chunks_per_song=max_chunks_per_song
    )
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train)} chunks")
    print(f"   Val: {len(X_val)} chunks")
    
    # Build and train model
    classifier = CNNTrackClassifier(
        input_shape=X.shape[1:],
        num_classes=len(np.unique(y)),
        embedding_dim=embedding_dim
    )
    
    classifier.build_model()
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    # Evaluate
    val_loss, val_acc = classifier.model.evaluate(X_val, y_val, verbose=0)
    print(f"\n📊 Final Results:")
    print(f"   Validation accuracy: {val_acc*100:.2f}%")
    print(f"   Validation loss: {val_loss:.4f}")
    
    # Save model
    classifier.save_model("track_classifier.keras")
    
    # Save label encoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(dataset.label_encoder, f)
    
    return classifier, history, dataset.label_encoder


if __name__ == "__main__":
    print("=" * 60)
    print("CNN Track Classifier Training - Phase 3")
    print("=" * 60)
    
    db_path = "/Users/juigupte/Desktop/Learning/recsys-foundations/songs.db"
    
    # Train classifier
    classifier, history, label_encoder = train_classifier(
        db_path=db_path,
        chunk_duration=5.0,
        max_chunks_per_song=50,
        embedding_dim=64,
        epochs=50
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Next: Extract embeddings from all songs")
    print("  python cnn_extract_embeddings.py")
    print("=" * 60)
