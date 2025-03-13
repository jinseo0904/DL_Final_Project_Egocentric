# Egocentric Video Classification Project

This project implements a 3D CNN model for egocentric video classification using PyTorch. The model is designed to process video data and classify different actions or activities.

## Project Structure

- `ego_video_training.ipynb`: Main Jupyter notebook containing the model implementation and training code
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook ego_video_training.ipynb
```

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