"""
Feature extraction utilities for voice signal processing.
Extracts various acoustic features from audio files for gender identification.
"""

import numpy as np
import librosa
import soundfile as sf


class FeatureExtractor:
    """Extract acoustic features from audio signals."""
    
    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Target sampling rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(self, file_path):
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio time series
        """
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def extract_mfcc(self, audio):
        """
        Extract MFCC features.
        
        Args:
            audio: Audio time series
            
        Returns:
            MFCC coefficients (mean and std)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extract_pitch(self, audio):
        """
        Extract pitch features.
        
        Args:
            audio: Audio time series
            
        Returns:
            Pitch statistics (mean, std, min, max)
        """
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            return np.array([
                np.mean(pitch_values),
                np.std(pitch_values),
                np.min(pitch_values),
                np.max(pitch_values)
            ])
        else:
            return np.zeros(4)
    
    def extract_spectral_features(self, audio):
        """
        Extract spectral features.
        
        Args:
            audio: Audio time series
            
        Returns:
            Spectral features (centroid, rolloff, contrast)
        """
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        features = np.array([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_contrast),
            np.std(spectral_contrast)
        ])
        return features
    
    def extract_zero_crossing_rate(self, audio):
        """
        Extract zero crossing rate.
        
        Args:
            audio: Audio time series
            
        Returns:
            ZCR statistics (mean, std)
        """
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        return np.array([np.mean(zcr), np.std(zcr)])
    
    def extract_all_features(self, audio):
        """
        Extract all features from audio.
        
        Args:
            audio: Audio time series
            
        Returns:
            Combined feature vector
        """
        mfcc_features = self.extract_mfcc(audio)
        pitch_features = self.extract_pitch(audio)
        spectral_features = self.extract_spectral_features(audio)
        zcr_features = self.extract_zero_crossing_rate(audio)
        
        return np.concatenate([
            mfcc_features,
            pitch_features,
            spectral_features,
            zcr_features
        ])
    
    def extract_features_from_file(self, file_path):
        """
        Extract features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Feature vector
        """
        audio = self.load_audio(file_path)
        return self.extract_all_features(audio)
