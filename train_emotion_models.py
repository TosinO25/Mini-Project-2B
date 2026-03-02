"""
Emotion Recognition - Scaled Neural Network Training
Train multiple models of increasing size on full EEG dataset (28 subjects)
Measure accuracy and runtime for each model configuration

Run on FABRIC with the full 2.4 GB dataset
"""

import numpy as np
import pandas as pd
import time
import os
import glob
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Deep learning library (choose one and install via pip)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


class EmotionDataLoader:
    """Load and preprocess full emotion recognition dataset"""
    
    def __init__(self, dataset_root):
        """
        Initialize data loader
        
        Parameters:
        - dataset_root: path to root of emotion dataset
                       expected structure: dataset_root/S01/Preprocessed/.csv format/
        """
        self.dataset_root = dataset_root
        self.data = []
        self.labels = []
        
    def load_dataset(self, max_subjects=None):
        """
        Load data from all subjects
        
        Parameters:
        - max_subjects: load only first N subjects (for testing)
        
        Returns:
        - X: feature matrix (num_samples, num_features)
        - y: label vector (num_samples,)
        """
        
        print("Loading emotion dataset...")
        
        # Find all subject directories
        subject_dirs = sorted(glob.glob(os.path.join(self.dataset_root, 'S*')))
        
        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]
        
        print(f"Found {len(subject_dirs)} subjects")
        
        all_data = []
        all_labels = []
        
        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            print(f"  Loading {subject_id}...", end='')
            
            # Look for CSV files in Preprocessed/.csv format/
            csv_dir = os.path.join(subject_dir, 'Preprocessed', '.csv format')
            
            if not os.path.exists(csv_dir):
                print(f" (not found)")
                continue
            
            csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
            
            subject_samples = 0
            for i, csv_file in enumerate(csv_files):
                try:
                    # Load CSV - adjust column names based on actual dataset
                    df = pd.read_csv(csv_file)
                    
                    # Extract features (all columns except label columns)
                    # Adjust this based on actual CSV structure
                    feature_cols = [col for col in df.columns if col not in ['label', 'emotion', 'subject']]
                    X_file = df[feature_cols].values
                    
                    # Label corresponds to file number
                    # Adjust emotion mapping based on dataset description
                    emotion_mapping = {
                        '01': 0,  # Happy
                        '02': 1,  # Sad
                        '03': 2,  # Neutral
                        '04': 3,  # Angry
                    }
                    
                    # Extract emotion number from filename
                    filename = os.path.basename(csv_file)
                    emotion_num = filename[:2]  # First two digits
                    emotion_label = emotion_mapping.get(emotion_num, -1)
                    
                    if emotion_label == -1:
                        continue
                    
                    all_data.append(X_file)
                    all_labels.extend([emotion_label] * len(X_file))
                    subject_samples += len(X_file)
                
                except Exception as e:
                    print(f"\n    Error loading {csv_file}: {e}")
                    continue
            
            print(f" ({subject_samples} samples)")
        
        # Combine all data
        X = np.vstack(all_data)
        y = np.array(all_labels)
        
        print(f"\nDataset loaded:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y


class NeuralNetworkExperiment:
    """Train and evaluate neural networks of different sizes"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize experiment
        
        Parameters:
        - X_train, X_test: feature matrices
        - y_train, y_test: label vectors
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []
        
        # Normalize data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
    
    def build_model(self, layers_config, model_name):
        """
        Build a neural network with specified architecture
        
        Parameters:
        - layers_config: list of layer sizes, e.g., [128, 64, 32]
        - model_name: name for this model configuration
        
        Returns:
        - model: compiled Keras model
        """
        
        num_features = self.X_train_scaled.shape[1]
        num_classes = len(np.unique(self.y_train))
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(num_features,)))
        
        # Add hidden layers
        for units in layers_config:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.2))  # Prevent overfitting
        
        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, model_name, epochs=50, batch_size=32):
        """
        Train model and record results
        
        Parameters:
        - model: Keras model
        - model_name: name for results
        - epochs: number of training epochs
        - batch_size: batch size for training
        """
        
        print(f"\nTraining {model_name}...")
        print(f"  Model parameters: {model.count_params()}")
        
        # Train and time
        start_time = time.time()
        history = model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Evaluate on test set and time
        start_time = time.time()
        test_loss, test_acc = model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        test_time = time.time() - start_time
        
        # Train accuracy
        _, train_acc = model.evaluate(self.X_train_scaled, self.y_train, verbose=0)
        
        # Get model architecture info
        num_layers = len(model.layers)
        first_layer_units = model.layers[0].units if hasattr(model.layers[0], 'units') else 0
        
        result = {
            'model_name': model_name,
            'num_layers': num_layers,
            'num_units': first_layer_units,
            'num_parameters': model.count_params(),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_time_seconds': train_time,
            'test_time_seconds': test_time,
        }
        
        self.results.append(result)
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Train Time: {train_time:.2f}s")
        print(f"  Test Time: {test_time:.2f}s")
        
        return result
    
    def run_experiments(self):
        """Run all model configurations"""
        
        # Model configurations to test
        # Format: (model_name, layer_config)
        configs = [
            ('Baseline', [64, 32]),
            ('Model 1', [128, 64, 32]),
            ('Model 2', [256, 128, 64]),
            ('Model 3', [256, 256, 128, 64]),
            ('Model 4', [512, 512, 256, 128, 64]),
            ('Model 5', [1024, 1024, 512, 256, 128]),
        ]
        
        for model_name, layers_config in configs:
            # Build model
            model = self.build_model(layers_config, model_name)
            
            # Train model
            self.train_model(model, model_name, epochs=50, batch_size=32)
            
            # Clear memory
            keras.backend.clear_session()
        
        return pd.DataFrame(self.results)
    
    def save_results(self, filename='emotion_model_results.csv'):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        return df


def main():
    """Main execution"""
    
    print("="*70)
    print("Emotion Recognition - Large-Scale Neural Network Training")
    print("="*70)
    
    # TODO: Set these paths
    DATASET_ROOT = "/path/to/emotion_dataset"  # Change this!
    OUTPUT_FILE = "emotion_model_results.csv"
    
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset not found at {DATASET_ROOT}")
        print("Please update DATASET_ROOT path in the script")
        return
    
    # Load dataset
    loader = EmotionDataLoader(DATASET_ROOT)
    X, y = loader.load_dataset(max_subjects=None)  # Change to max_subjects=5 for testing
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Run experiments
    experiment = NeuralNetworkExperiment(X_train, X_test, y_train, y_test)
    results_df = experiment.run_experiments()
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best test accuracy: {results_df['test_accuracy'].max():.4f}")
    print(f"Best model: {results_df.loc[results_df['test_accuracy'].idxmax(), 'model_name']}")
    print(f"Fastest training: {results_df.loc[results_df['train_time_seconds'].idxmin(), 'model_name']}")
    print(f"Slowest training: {results_df.loc[results_df['train_time_seconds'].idxmax(), 'model_name']}")
    print(f"Training time increase: {results_df['train_time_seconds'].max() / results_df['train_time_seconds'].min():.1f}x")


if __name__ == "__main__":
    main()
