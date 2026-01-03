"""
Demo script for gender identification system.
Generates synthetic audio data and demonstrates the system's capabilities.
"""

import os
import sys
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import FeatureExtractor


def generate_synthetic_audio(duration=2.0, sample_rate=22050, pitch_freq=150, output_file='demo_audio.wav'):
    """
    Generate synthetic audio with specified pitch.
    
    Args:
        duration: Duration in seconds
        pitch_freq: Fundamental frequency (Hz)
        sample_rate: Sample rate
        output_file: Output file path
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate fundamental frequency and harmonics
    signal = 0.3 * np.sin(2 * np.pi * pitch_freq * t)  # Fundamental
    signal += 0.2 * np.sin(2 * np.pi * 2 * pitch_freq * t)  # 2nd harmonic
    signal += 0.1 * np.sin(2 * np.pi * 3 * pitch_freq * t)  # 3rd harmonic
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    signal = signal + noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Save audio
    sf.write(output_file, signal, sample_rate)
    print(f"Generated audio: {output_file}")
    return output_file


def demo_feature_extraction():
    """Demonstrate feature extraction on synthetic audio."""
    print("=" * 80)
    print("Gender Identification System - Demo")
    print("=" * 80)
    
    # Create demo directory
    demo_dir = 'demo_data'
    os.makedirs(demo_dir, exist_ok=True)
    
    # Generate synthetic audio samples
    print("\n[1/3] Generating synthetic audio samples...")
    print("\nGenerating male voice samples (lower pitch: 100-150 Hz)...")
    male_dir = os.path.join(demo_dir, 'male')
    os.makedirs(male_dir, exist_ok=True)
    for i in range(5):
        pitch = np.random.uniform(100, 150)
        output_file = os.path.join(male_dir, f'male_{i+1}.wav')
        generate_synthetic_audio(duration=2.0, pitch_freq=pitch, output_file=output_file)
    
    print("\nGenerating female voice samples (higher pitch: 180-250 Hz)...")
    female_dir = os.path.join(demo_dir, 'female')
    os.makedirs(female_dir, exist_ok=True)
    for i in range(5):
        pitch = np.random.uniform(180, 250)
        output_file = os.path.join(female_dir, f'female_{i+1}.wav')
        generate_synthetic_audio(duration=2.0, pitch_freq=pitch, output_file=output_file)
    
    # Extract features from one sample
    print("\n[2/3] Demonstrating feature extraction...")
    feature_extractor = FeatureExtractor()
    
    sample_file = os.path.join(male_dir, 'male_1.wav')
    print(f"\nExtracting features from: {sample_file}")
    
    audio = feature_extractor.load_audio(sample_file)
    print(f"Audio duration: {len(audio) / feature_extractor.sample_rate:.2f} seconds")
    print(f"Sample rate: {feature_extractor.sample_rate} Hz")
    
    # Extract individual features
    mfcc_features = feature_extractor.extract_mfcc(audio)
    print(f"\nMFCC features shape: {mfcc_features.shape}")
    
    pitch_features = feature_extractor.extract_pitch(audio)
    print(f"Pitch features: mean={pitch_features[0]:.2f} Hz, std={pitch_features[1]:.2f} Hz")
    
    spectral_features = feature_extractor.extract_spectral_features(audio)
    print(f"Spectral features shape: {spectral_features.shape}")
    
    zcr_features = feature_extractor.extract_zero_crossing_rate(audio)
    print(f"Zero-crossing rate: mean={zcr_features[0]:.4f}, std={zcr_features[1]:.4f}")
    
    # Extract all features
    all_features = feature_extractor.extract_all_features(audio)
    print(f"\nTotal feature dimension: {all_features.shape[0]}")
    
    # Instructions for training
    print("\n[3/3] Next steps...")
    print("\n" + "=" * 80)
    print("Demo data generated successfully!")
    print("=" * 80)
    print(f"\nSynthetic audio samples saved to: {demo_dir}/")
    print("\nTo train models on this demo data, run:")
    print(f"  python train.py --data-dir {demo_dir}")
    print("\nTo predict gender from an audio file, run:")
    print(f"  python predict.py {os.path.join(male_dir, 'male_1.wav')}")
    print("\nNote: For real-world applications, use actual voice recordings.")
    print("      The synthetic data is for demonstration purposes only.")
    print("=" * 80)


def main():
    """Main function."""
    demo_feature_extraction()


if __name__ == '__main__':
    main()
