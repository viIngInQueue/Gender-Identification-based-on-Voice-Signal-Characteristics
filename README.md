# Gender Identification based on Voice Signal Characteristics

A comprehensive machine learning system for identifying gender from voice signals using four different approaches: Gaussian Mixture Models (GMM), Hidden Markov Models (HMM), Neural Networks (NN), and Support Vector Machines (SVM).

## Features

- **Multiple Classifiers**: Implements four different machine learning approaches
  - **GMM** (Gaussian Mixture Model): Probabilistic model using Gaussian distributions
  - **HMM** (Hidden Markov Model): Sequential model for temporal patterns
  - **NN** (Neural Network): Deep learning approach with dropout regularization
  - **SVM** (Support Vector Machine): Kernel-based discriminative classifier
  
- **Rich Feature Extraction**: Extracts comprehensive acoustic features
  - MFCCs (Mel-Frequency Cepstral Coefficients)
  - Pitch characteristics (mean, std, min, max)
  - Spectral features (centroid, rolloff, contrast)
  - Zero-crossing rate
  
- **Easy to Use**: Simple command-line interface for training and prediction
- **Modular Design**: Clean, extensible architecture for adding new models

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/viIngInQueue/Gender-Identification-based-on-Voice-Signal-Characteristics.git
cd Gender-Identification-based-on-Voice-Signal-Characteristics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your audio dataset in the following structure:

```
data/
├── male/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── female/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

Supported audio formats: `.wav`, `.mp3`, `.flac`

## Usage

### Training

Train all four classifiers on your dataset:

```bash
python train.py --data-dir data --output-dir models
```

Options:
- `--data-dir`: Directory containing audio files (default: `data`)
- `--output-dir`: Directory to save trained models (default: `models`)
- `--test-size`: Proportion of test set (default: `0.2`)
- `--random-state`: Random seed for reproducibility (default: `42`)

This will:
1. Extract features from all audio files
2. Train GMM, HMM, NN, and SVM classifiers
3. Evaluate each model on the test set
4. Save trained models to the output directory

### Prediction

Predict gender from a new audio file:

```bash
python predict.py path/to/audio.wav
```

This will use all four trained models to predict the gender.

To use a specific model:
```bash
python predict.py path/to/audio.wav --model gmm
python predict.py path/to/audio.wav --model hmm
python predict.py path/to/audio.wav --model nn
python predict.py path/to/audio.wav --model svm
```

Options:
- `--model`: Model to use (`all`, `gmm`, `hmm`, `nn`, `svm`) (default: `all`)
- `--model-dir`: Directory containing trained models (default: `models`)

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gmm_classifier.py    # GMM implementation
│   │   ├── hmm_classifier.py    # HMM implementation
│   │   ├── nn_classifier.py     # Neural Network implementation
│   │   └── svm_classifier.py    # SVM implementation
│   └── utils/
│       ├── __init__.py
│       ├── feature_extraction.py # Audio feature extraction
│       └── data_loader.py        # Data loading utilities
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Model Details

### GMM (Gaussian Mixture Model)
- Uses separate GMMs for male and female voices
- Classifies based on log-likelihood comparison
- Parameters: 8 Gaussian components, diagonal covariance

### HMM (Hidden Markov Model)
- Learns temporal patterns in voice signals
- Separate HMMs for each gender
- Parameters: 5 hidden states, 100 training iterations

### Neural Network
- Multi-layer perceptron with dropout regularization
- Architecture: Input → 128 → 64 → 32 → Output
- Training: Adam optimizer, early stopping, learning rate reduction
- Parameters: 50 epochs, dropout rate 0.3

### SVM (Support Vector Machine)
- RBF kernel for non-linear classification
- Probability estimates enabled
- Parameters: C=10.0, gamma='scale'

## Performance

Each model is evaluated on:
- Classification accuracy
- Precision, recall, and F1-score per class
- Confusion matrix

Results are printed during training and can be used to compare model performance.

## Example Output

```
================================================================================
Gender Identification based on Voice Signal Characteristics
================================================================================

[1/6] Initializing feature extractor...
[2/6] Loading and processing dataset...
Loaded 1000 samples
Feature dimension: 38
Male samples: 500
Female samples: 500

[3/6] Training GMM Classifier
Training male GMM with 400 samples...
Training female GMM with 400 samples...
GMM training completed.

=== GMM Classifier Performance ===
Accuracy: 0.9250

[4/6] Training HMM Classifier
...

TRAINING SUMMARY
================================================================================
Model                Accuracy  
------------------------------
GMM                  0.9250
HMM                  0.8950
NN                   0.9450
SVM                  0.9500
================================================================================
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- librosa for audio processing
- scikit-learn for machine learning utilities
- hmmlearn for HMM implementation
- TensorFlow for neural network implementation