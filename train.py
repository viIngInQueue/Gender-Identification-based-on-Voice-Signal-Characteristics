"""
Main training script for gender identification system.
Trains and evaluates all four classifiers: GMM, HMM, NN, and SVM.
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import FeatureExtractor, DataLoader
from models import GMMClassifier, HMMClassifier, NNClassifier, SVMClassifier


def train_all_models(data_dir, output_dir='models', test_size=0.2, random_state=42):
    """
    Train all classifiers on the dataset.
    
    Args:
        data_dir: Directory containing audio files
        output_dir: Directory to save trained models
        test_size: Proportion of test set
        random_state: Random seed
    """
    print("=" * 80)
    print("Gender Identification based on Voice Signal Characteristics")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature extractor and data loader
    print("\n[1/6] Initializing feature extractor...")
    feature_extractor = FeatureExtractor()
    data_loader = DataLoader(feature_extractor)
    
    # Load and prepare data
    print("\n[2/6] Loading and processing dataset...")
    try:
        features, labels = data_loader.load_dataset(data_dir)
        print(f"Loaded {len(features)} samples")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Male samples: {np.sum(labels == 0)}")
        print(f"Female samples: {np.sum(labels == 1)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nExpected directory structure:")
        print("  data_dir/")
        print("    male/")
        print("      audio1.wav")
        print("      audio2.wav")
        print("    female/")
        print("      audio1.wav")
        print("      audio2.wav")
        return
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = data_loader.prepare_data(
        features, labels, test_size=test_size, random_state=random_state
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    results = {}
    
    # Train GMM classifier
    print("\n" + "=" * 80)
    print("[3/6] Training GMM Classifier")
    print("=" * 80)
    try:
        gmm = GMMClassifier(n_components=8, random_state=random_state)
        gmm.train(X_train, y_train)
        gmm_accuracy = gmm.evaluate(X_test, y_test)
        gmm.save_model(os.path.join(output_dir, 'gmm_model.pkl'))
        results['GMM'] = gmm_accuracy
    except Exception as e:
        print(f"Error training GMM: {e}")
        results['GMM'] = None
    
    # Train HMM classifier
    print("\n" + "=" * 80)
    print("[4/6] Training HMM Classifier")
    print("=" * 80)
    try:
        hmm_clf = HMMClassifier(n_components=5, n_iter=100, random_state=random_state)
        hmm_clf.train(X_train, y_train)
        hmm_accuracy = hmm_clf.evaluate(X_test, y_test)
        hmm_clf.save_model(os.path.join(output_dir, 'hmm_model.pkl'))
        results['HMM'] = hmm_accuracy
    except Exception as e:
        print(f"Error training HMM: {e}")
        results['HMM'] = None
    
    # Train Neural Network classifier
    print("\n" + "=" * 80)
    print("[5/6] Training Neural Network Classifier")
    print("=" * 80)
    try:
        nn = NNClassifier(
            input_dim=X_train.shape[1],
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            random_state=random_state
        )
        # Use validation split from training data
        val_split_idx = int(len(X_train) * 0.8)
        X_train_nn, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_nn, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        nn.train(X_train_nn, y_train_nn, X_val, y_val, epochs=50, batch_size=32)
        nn_accuracy = nn.evaluate(X_test, y_test)
        nn.save_model(os.path.join(output_dir, 'nn_model.h5'))
        results['NN'] = nn_accuracy
    except Exception as e:
        print(f"Error training Neural Network: {e}")
        results['NN'] = None
    
    # Train SVM classifier
    print("\n" + "=" * 80)
    print("[6/6] Training SVM Classifier")
    print("=" * 80)
    try:
        svm = SVMClassifier(kernel='rbf', C=10.0, gamma='scale', random_state=random_state)
        svm.train(X_train, y_train)
        svm_accuracy = svm.evaluate(X_test, y_test)
        svm.save_model(os.path.join(output_dir, 'svm_model.pkl'))
        results['SVM'] = svm_accuracy
    except Exception as e:
        print(f"Error training SVM: {e}")
        results['SVM'] = None
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<10}")
    print("-" * 30)
    for model_name, accuracy in results.items():
        if accuracy is not None:
            print(f"{model_name:<20} {accuracy:.4f}")
        else:
            print(f"{model_name:<20} {'Failed':<10}")
    print("=" * 80)
    
    # Save data scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(data_loader.scaler, f)
    print(f"\nScaler saved to {scaler_path}")
    
    print(f"\nAll models saved to '{output_dir}' directory")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train gender identification classifiers'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing audio files (default: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of test set (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    train_all_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()
