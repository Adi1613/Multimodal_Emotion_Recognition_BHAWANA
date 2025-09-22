# Multimodal Emotion Recognition using Pretrained Embedding Fusion

This repository contains the codebase for a multimodal emotion recognition system that leverages pretrained embeddings from EEG, audio, and video data sources. Developed as part of my PhD research, the system demonstrates how deep feature fusion from multiple modalities can improve the robustness and accuracy of emotion classification in naturalistic settings.

## 🧠 Overview

Human emotions are inherently multimodal, expressed via voice, facial expressions, and brain activity. This project proposes a unified deep learning pipeline that:

- Extracts modality-specific embeddings using:
  - **Wav2Vec2.0** for audio (speech)
  - **ResNet50** for video (facial features)
  - **Statistical EEG features** from Muse 2 signals
- Concatenates these features
- Trains a lightweight **Transformer-based classifier** on the fused representations
- Predicts one of 8 emotion labels (every 5 seconds)

The model is evaluated using the **BHAWNA** dataset — a synchronized, naturalistic, multimodal dataset created during this research.

## 📁 Folder Structure

├── data/
│ ├── audio_clips/ # Raw audio files (5s chunks)
│ ├── video_embeddings/ # Precomputed ResNet50 features
│ ├── eeg_embeddings/ # EEG statistical features
│ └── preprocessed_data/ # Combined and cleaned data
├── main.py # Complete pipeline: feature extraction → fusion → classification
└── README.md

@misc{shah2024bhawna,
  author = {Aditya Kumar Shah},
  title = {BHAWNA: A Multimodal Dataset for Emotion Recognition using EEG, Audio, and Video Cues},
  year = {2024},
  note = {Preprint under submission}
}

