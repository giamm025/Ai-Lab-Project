Sign Language Recognition (LSTM + MediaPipe)

An end-to-end Machine Learning pipeline for American Sign Language (ASL) / Sign Language recognition. This project leverages MediaPipe Holistic for extracting spatial keypoints (hands and facial Non-Manual Markers) and a PyTorch LSTM Neural Network for temporal sequence classification.

The pipeline includes automated dataset downloading, processing, data augmentation, model training, evaluation, and real-time webcam inference.

✨ Features

Multi-Dataset Aggregation: Combines data from major Kaggle ASL datasets (ASLLVD, ASL Citizen, MS-ASL, WLASL).

Robust Feature Extraction: Uses MediaPipe to extract keypoints. Supports two modes:

SOLO_MANI (126 features): Left and right hand tracking.

MANI_VOLTO (402 features): Hands + 276 filtered facial keypoints critical for sign language (Non-Manual Markers like eyebrows, eyes, and lips).

Data Augmentation: Balances dataset classes using spatial jitter, temporal dropping, spatial scaling, and horizontal mirroring.

PyTorch LSTM: Custom LSTM architecture with dynamic padding via pack_padded_sequence and confidence thresholding for an "unknown/garbage" class.

Real-time Inference: Live webcam inference script with a custom OpenCV HUD, real-time predictions, and confidence probability distributions.

📂 Repository Structure

Ai-Lab-Project/
│
├── data/                       # Data storage (created dynamically)
│   ├── raw/                    # Raw .mp4 clips
│   └── processed/              # Extracted .npy MediaPipe sequences
│
├── datasets/                   # Downloaded Kaggle datasets
├── models/                     # Saved PyTorch models (.pth)
├── results/                    # Training histories, graphs, and reports
│
└── src/
    ├── config.py               # Global configurations, labels, and target words
    ├── extract_features.py     # MediaPipe video-to-keypoint converter
    ├── count_datset.py         # Utility to count original vs augmented data
    │
    ├── enlarge_dataset/        # Dataset Acquisition & Augmentation
    │   ├── download_datasets.py
    │   ├── augment_dataset.py
    │   └── count_processed_files.py
    │
    ├── parsers/                # Dataset-specific metadata parsers & video clippers
    │   ├── ASLCitizen_parser.py
    │   ├── MSASL_parser.py
    │   └── WLASL_parser.py
    │
    ├── neural_network/         # PyTorch Model, Training, and Evaluation
    │   ├── dataset.py          # PyTorch Dataset & Dataloaders
    │   ├── model.py            # LSTM Architecture
    │   ├── train.py            # Training loop with Early Stopping
    │   ├── test.py             # Evaluation (F1-score, Confusion Matrix, etc.)
    │   ├── automatize_train_and_test.py
    │   └── tuning.py
    │
    └── webcam/                 # Inference Scripts
        ├── predict_single.py   # Run model on a single .mp4 file
        └── webcam_inference.py # Live webcam inference with OpenCV HUD


🚀 Pipeline & Usage Guide

1. Setup & Configuration

Adjust the TARGET_WORDS in src/config.py to select the specific signs you want to classify. By default, the model recognizes: happen, finally, late, not-yet, misunderstand, understand, and an unknown class.

2. Download and Parse Datasets

Download the raw datasets from Kaggle and parse their metadata to extract only the videos corresponding to your target words.

python src/enlarge_dataset/download_datasets.py --dataset all
python src/parsers/MSASL_parser.py
python src/parsers/WLASL_parser.py
python src/parsers/ASLCitizen_parser.py

Note: This will output .mp4 clips to data/raw/.

3. Extract Features (MediaPipe)

Convert the raw .mp4 videos into NumPy arrays (.npy) containing the spatial coordinates of the hands and face across all frames.

python src/extract_features.py


4. Data Augmentation

Balance your dataset by generating synthetic variations of your existing signs to prevent class imbalance and improve model generalization.

python src/enlarge_dataset/augment_dataset.py


5. Train the Model

Train the PyTorch LSTM model. You can specify the mode (MANI_VOLTO or SOLO_MANI), hyperparameters, and experiment tracking versions.

python src/neural_network/train.py --modalita MANI_VOLTO --epochs 200 --batch_size 8


Tip: Use automatize_train_and_test.py or tuning.py to run automated hyperparameter tuning and model comparisons.

6. Evaluate the Model

Generate learning curves, classification reports (Precision, Recall, F1), and a Confusion Matrix for the best-trained model.

python src/neural_network/test.py --modalita MANI_VOLTO


Outputs are saved in the results/ directory.

7. Run Inference

Single Video File:

python src/webcam/predict_single.py --video_path path/to/your/video.mp4 --modalita MANI_VOLTO


Real-Time Webcam (Live):

python src/webcam/webcam_inference.py --modalita MANI_VOLTO


Controls inside Webcam HUD: * Press R to start/stop recording your sign. Upon stopping, inference is immediately calculated and displayed.

Press Q to quit.

🧠 Model Architecture details

The core model (SignLanguageLSTM) relies on a standard Long Short-Term Memory (LSTM) architecture.

Input Layer: Takes sequential keypoint data (Shape: Batch, Sequence_Length, Features).

Hidden Layers: Configurable hidden size and layer count. Uses PyTorch's pack_padded_sequence to dynamically ignore padding zeros in variable-length videos.

Output Layer: Fully connected layer with Softmax mapping to the number of target classes.

Out-of-Distribution (OOD): If the highest prediction confidence falls below a configured threshold (default 60%), the prediction is forced to an unknown class.

🛠️ Requirements

Python 3.9+

torch, torchvision, torchaudio, torchmetrics

mediapipe, opencv-python

numpy, pandas, matplotlib, seaborn, scikit-learn

kagglehub, yt-dlp, moviepy (for dataset acquisition)