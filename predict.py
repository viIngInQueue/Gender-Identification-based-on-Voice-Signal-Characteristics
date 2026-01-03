"""
Prediction script for gender identification.
Uses trained models to predict gender from audio files.
"""

import os
import sys
import argparse
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import FeatureExtractor
from models import GMMClassifier, HMMClassifier, NNClassifier, SVMClassifier


def predict_gender(audio_file, model_type='all', model_dir='models'):
    """
    Predict gender from audio file.
    
    Args:
        audio_file: Path to audio file
        model_type: Type of model to use ('gmm', 'hmm', 'nn', 'svm', or 'all')
        model_dir: Directory containing trained models
    """
    print("=" * 80)
    print("Gender Identification - Prediction")
    print("=" * 80)
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at '{scaler_path}'")
        print("Please train models first using train.py")
        return
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Extract features
    print(f"\nExtracting features from: {audio_file}")
    feature_extractor = FeatureExtractor()
    try:
        features = feature_extractor.extract_features_from_file(audio_file)
        features = scaler.transform(features.reshape(1, -1))
        print(f"Feature dimension: {features.shape[1]}")
    except Exception as e:
        print(f"Error extracting features: {e}")
        return
    
    # Define models to use
    if model_type.lower() == 'all':
        models_to_test = ['gmm', 'hmm', 'nn', 'svm']
    else:
        models_to_test = [model_type.lower()]
    
    print("\n" + "=" * 80)
    print("Predictions:")
    print("=" * 80)
    
    gender_map = {0: 'Male', 1: 'Female'}
    
    # GMM prediction
    if 'gmm' in models_to_test:
        gmm_path = os.path.join(model_dir, 'gmm_model.pkl')
        if os.path.exists(gmm_path):
            try:
                gmm = GMMClassifier()
                gmm.load_model(gmm_path)
                prediction = gmm.predict(features)[0]
                print(f"GMM:  {gender_map[prediction]}")
            except Exception as e:
                print(f"GMM:  Error - {e}")
        else:
            print(f"GMM:  Model not found at {gmm_path}")
    
    # HMM prediction
    if 'hmm' in models_to_test:
        hmm_path = os.path.join(model_dir, 'hmm_model.pkl')
        if os.path.exists(hmm_path):
            try:
                hmm_clf = HMMClassifier()
                hmm_clf.load_model(hmm_path)
                prediction = hmm_clf.predict(features)[0]
                print(f"HMM:  {gender_map[prediction]}")
            except Exception as e:
                print(f"HMM:  Error - {e}")
        else:
            print(f"HMM:  Model not found at {hmm_path}")
    
    # Neural Network prediction
    if 'nn' in models_to_test:
        nn_path = os.path.join(model_dir, 'nn_model.h5')
        if os.path.exists(nn_path):
            try:
                nn = NNClassifier()
                nn.load_model(nn_path)
                prediction = nn.predict(features)[0]
                print(f"NN:   {gender_map[prediction]}")
            except Exception as e:
                print(f"NN:   Error - {e}")
        else:
            print(f"NN:   Model not found at {nn_path}")
    
    # SVM prediction
    if 'svm' in models_to_test:
        svm_path = os.path.join(model_dir, 'svm_model.pkl')
        if os.path.exists(svm_path):
            try:
                svm = SVMClassifier()
                svm.load_model(svm_path)
                prediction = svm.predict(features)[0]
                print(f"SVM:  {gender_map[prediction]}")
            except Exception as e:
                print(f"SVM:  Error - {e}")
        else:
            print(f"SVM:  Model not found at {svm_path}")
    
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Predict gender from audio file'
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to audio file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all', 'gmm', 'hmm', 'nn', 'svm'],
        help='Model to use for prediction (default: all)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    args = parser.parse_args()
    
    predict_gender(
        audio_file=args.audio_file,
        model_type=args.model,
        model_dir=args.model_dir
    )


if __name__ == '__main__':
    main()
