"""
Support Vector Machine (SVM) classifier for gender identification.
"""

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class SVMClassifier:
    """SVM-based gender classifier."""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            random_state: Random seed
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True
        )
        
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train SVM model.
        
        Args:
            X_train: Training features
            y_train: Training labels (0 for male, 1 for female)
        """
        print(f"Training SVM with {len(X_train)} samples...")
        print(f"Kernel: {self.kernel}, C: {self.C}, Gamma: {self.gamma}")
        
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        print("SVM training completed.")
        
        # Print support vector info
        print(f"Number of support vectors: {len(self.model.support_vectors_)}")
        print(f"Support vectors per class: {self.model.n_support_}")
    
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
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
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
        
        print("\n=== SVM Classifier Performance ===")
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
            'model': self.model,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
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
        
        self.model = model_data['model']
        self.kernel = model_data['kernel']
        self.C = model_data['C']
        self.gamma = model_data['gamma']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
