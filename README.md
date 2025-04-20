# Egocentric Video Classification Project

This project implements a 3D CNN model for egocentric video classification using PyTorch. The model is designed to process video data and classify different actions or activities.

## Charades-Ego Video Analysis Project

This repository contains implementations for video analysis tasks on the Charades-Ego dataset, focusing on first-person (egocentric) videos.

### Project Structure

```
.
├── data/                  # Data directory
│   ├── CharadesEgo/       # Raw dataset files
│   └── processed/         # Processed dataset files
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementations
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── requirements.txt       # Dependencies
└── README.md              # This file
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/DL_Final_Project_Egocentric.git
cd DL_Final_Project_Egocentric
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Charades-Ego dataset and place it in the `data/CharadesEgo` directory.

### Task 1: Action Recognition - Phase 1 Implementation

We have implemented the data preparation phase for the action recognition task as outlined in the [strategy](strategy.md). This includes:

1. **Dataset Filtering**: Extracting only first-person videos from the Charades-Ego dataset
2. **Temporal Annotation Processing**: Converting action annotations with start/end times into frame-level labels
3. **Data Augmentation Pipeline**: Implementing transformations suitable for egocentric videos

#### How to Run Data Preparation

The data preparation pipeline can be run using the following command:

```bash
python src/data/prepare_data.py --process_labels --extract_clips --visualize
```

Command-line options:
- `--base_dir`: Base directory of the project (default: `../..`)
- `--train_ratio`: Ratio of data to use for training (default: 0.7)
- `--val_ratio`: Ratio of data to use for validation (default: 0.15)
- `--test_ratio`: Ratio of data to use for testing (default: 0.15)
- `--process_test`: Process the test set
- `--process_labels`: Process annotations into frame-level labels
- `--frame_sample_rate`: Sample every N-th frame when processing labels (default: 4)
- `--extract_clips`: Extract video clips
- `--clip_sample_size`: Number of videos to sample for clip extraction (default: 10)
- `--clip_length`: Number of frames in extracted clips (default: 32)
- `--clip_overlap`: Overlap between consecutive clips (default: 0.5)
- `--visualize`: Create visualizations

## Model Architecture

The model uses a 3D CNN architecture with the following components:
- Two 3D convolutional layers with max pooling
- Two fully connected layers
- Input shape: (batch_size, 3, frames, height, width)
- Output: 10 classes

## Requirements

- Python 3.9+
- PyTorch 2.6.0+
- CUDA (for GPU support)
- Jupyter Notebook 