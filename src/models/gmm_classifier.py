"""
Gaussian Mixture Model (GMM) classifier for gender identification.
"""

import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report


class GMMClassifier:
    """GMM-based gender classifier."""
    
    def __init__(self, n_components=8, covariance_type='diag', random_state=42):
        """
        Initialize GMM classifier.
        
        Args:
            n_components: Number of Gaussian components
            covariance_type: Type of covariance parameters
            random_state: Random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        # Create GMM for each gender
        self.gmm_male = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.gmm_female = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train GMM models.
        
        Args:
            X_train: Training features
            y_train: Training labels (0 for male, 1 for female)
        """
        # Separate data by gender
        X_male = X_train[y_train == 0]
        X_female = X_train[y_train == 1]
        
        # Train GMMs
        print(f"Training male GMM with {len(X_male)} samples...")
        self.gmm_male.fit(X_male)
        
        print(f"Training female GMM with {len(X_female)} samples...")
        self.gmm_female.fit(X_female)
        
        self.is_trained = True
        print("GMM training completed.")
    
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
        
        # Calculate log-likelihood for each gender
        log_likelihood_male = self.gmm_male.score_samples(X)
        log_likelihood_female = self.gmm_female.score_samples(X)
        
        # Predict based on higher likelihood
        predictions = (log_likelihood_female > log_likelihood_male).astype(int)
        return predictions
    
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
        
        print("\n=== GMM Classifier Performance ===")
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
            'gmm_male': self.gmm_male,
            'gmm_female': self.gmm_female,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type
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
        
        self.gmm_male = model_data['gmm_male']
        self.gmm_female = model_data['gmm_female']
        self.n_components = model_data['n_components']
        self.covariance_type = model_data['covariance_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
