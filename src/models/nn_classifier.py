"""
Neural Network (NN) classifier for gender identification.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import accuracy_score, classification_report


class NNClassifier:
    """Neural Network-based gender classifier."""
    
    def __init__(self, input_dim=None, hidden_layers=[128, 64, 32], dropout_rate=0.3, random_state=42):
        """
        Initialize NN classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        keras.utils.set_random_seed(random_state)
        
        self.model = None
        self.is_trained = False
    
    def _build_model(self, input_dim):
        """
        Build neural network architecture.
        
        Args:
            input_dim: Input feature dimension
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'hidden_{i+1}'))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\n=== Neural Network Architecture ===")
        model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels (0 for male, 1 for female)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            if self.input_dim is None:
                self.input_dim = X_train.shape[1]
            self._build_model(self.input_dim)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        callback_list = [early_stopping, reduce_lr]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        print("\nTraining Neural Network...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        print("Neural Network training completed.")
        
        return history
    
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
        
        # Get probabilities
        probabilities = self.model.predict(X, verbose=0)
        
        # Convert to binary predictions
        predictions = (probabilities > 0.5).astype(int).flatten()
        
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
        
        print("\n=== Neural Network Classifier Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Male', 'Female']))
        
        # Also show model evaluation
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return accuracy
    
    def save_model(self, filepath):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
