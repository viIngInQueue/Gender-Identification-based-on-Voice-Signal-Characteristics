"""
Data loading and preprocessing utilities.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .feature_extraction import FeatureExtractor


class DataLoader:
    """Load and preprocess audio data for gender identification."""
    
    def __init__(self, feature_extractor=None):
        """
        Initialize data loader.
        
        Args:
            feature_extractor: FeatureExtractor instance
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.scaler = StandardScaler()
    
    def load_dataset(self, data_dir, male_label='male', female_label='female'):
        """
        Load dataset from directory structure.
        Expected structure:
            data_dir/
                male/
                    audio1.wav
                    audio2.wav
                female/
                    audio1.wav
                    audio2.wav
        
        Args:
            data_dir: Root directory containing male and female subdirectories
            male_label: Label for male samples
            female_label: Label for female samples
            
        Returns:
            features: numpy array of features
            labels: numpy array of labels (0 for male, 1 for female)
        """
        features = []
        labels = []
        
        # Load male samples
        male_dir = os.path.join(data_dir, male_label)
        if os.path.exists(male_dir):
            for filename in os.listdir(male_dir):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(male_dir, filename)
                    try:
                        feature = self.feature_extractor.extract_features_from_file(file_path)
                        features.append(feature)
                        labels.append(0)  # 0 for male
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        # Load female samples
        female_dir = os.path.join(data_dir, female_label)
        if os.path.exists(female_dir):
            for filename in os.listdir(female_dir):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(female_dir, filename)
                    try:
                        feature = self.feature_extractor.extract_features_from_file(file_path)
                        features.append(feature)
                        labels.append(1)  # 1 for female
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        if len(features) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
        
        return np.array(features), np.array(labels)
    
    def prepare_data(self, features, labels, test_size=0.2, random_state=42):
        """
        Split and normalize data.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test: Split and normalized data
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def load_single_file(self, file_path, normalize=True):
        """
        Load and process a single audio file.
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize features
            
        Returns:
            Feature vector
        """
        feature = self.feature_extractor.extract_features_from_file(file_path)
        if normalize and hasattr(self.scaler, 'mean_'):
            feature = self.scaler.transform(feature.reshape(1, -1))[0]
        return feature
