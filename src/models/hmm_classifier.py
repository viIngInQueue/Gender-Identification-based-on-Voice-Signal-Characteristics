"""
Hidden Markov Model (HMM) classifier for gender identification.
"""

import numpy as np
import pickle
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, classification_report


class HMMClassifier:
    """HMM-based gender classifier."""
    
    def __init__(self, n_components=5, n_iter=100, random_state=42):
        """
        Initialize HMM classifier.
        
        Args:
            n_components: Number of hidden states
            n_iter: Number of training iterations
            random_state: Random seed
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Create HMM for each gender
        self.hmm_male = hmm.GaussianHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
            covariance_type='diag'
        )
        self.hmm_female = hmm.GaussianHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
            covariance_type='diag'
        )
        
        self.is_trained = False
    
    def _prepare_sequences(self, X, sequence_length=10):
        """
        Convert feature matrix to sequences for HMM training.
        
        Note: This implementation treats each feature dimension as a time step
        by reshaping to (-1, 1). This is a simplification suitable for static
        feature vectors. For better temporal modeling, consider extracting
        frame-level features from audio segments.
        
        Args:
            X: Feature matrix
            sequence_length: Number of frames per sequence (unused in current implementation)
            
        Returns:
            sequences: List of feature sequences
            lengths: List of sequence lengths
        """
        sequences = []
        lengths = []
        
        # Each sample is treated as a single sequence
        # For better HMM training, we could split longer audio files
        for features in X:
            # Reshape single feature vector to sequence
            # Here we treat each feature as a time step
            seq = features.reshape(-1, 1)
            sequences.append(seq)
            lengths.append(len(seq))
        
        # Concatenate all sequences
        X_concat = np.vstack(sequences)
        
        return X_concat, lengths
    
    def train(self, X_train, y_train):
        """
        Train HMM models.
        
        Args:
            X_train: Training features
            y_train: Training labels (0 for male, 1 for female)
        """
        # Separate data by gender
        X_male = X_train[y_train == 0]
        X_female = X_train[y_train == 1]
        
        # Prepare sequences
        print(f"Training male HMM with {len(X_male)} samples...")
        X_male_seq, lengths_male = self._prepare_sequences(X_male)
        self.hmm_male.fit(X_male_seq, lengths_male)
        
        print(f"Training female HMM with {len(X_female)} samples...")
        X_female_seq, lengths_female = self._prepare_sequences(X_female)
        self.hmm_female.fit(X_female_seq, lengths_female)
        
        self.is_trained = True
        print("HMM training completed.")
    
    def predict(self, X):
        """
        Predict gender for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels (0 for male, 1 for female)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        
        for features in X:
            # Prepare sequence
            seq = features.reshape(-1, 1)
            
            # Calculate log-likelihood for each gender
            try:
                log_likelihood_male = self.hmm_male.score(seq)
                log_likelihood_female = self.hmm_female.score(seq)
                
                # Predict based on higher likelihood
                prediction = 1 if log_likelihood_female > log_likelihood_male else 0
                predictions.append(prediction)
            except Exception:
                # If scoring fails, use class balance (0.5 probability for each)
                # This avoids gender bias
                predictions.append(np.random.randint(0, 2))
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            accuracy: Classification accuracy
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print("\n=== HMM Classifier Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Male', 'Female']))
        
        return accuracy
    
    def save_model(self, filepath):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'hmm_male': self.hmm_male,
            'hmm_female': self.hmm_female,
            'n_components': self.n_components,
            'n_iter': self.n_iter
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.hmm_male = model_data['hmm_male']
        self.hmm_female = model_data['hmm_female']
        self.n_components = model_data['n_components']
        self.n_iter = model_data['n_iter']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
