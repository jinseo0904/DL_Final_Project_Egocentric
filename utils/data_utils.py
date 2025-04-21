"""
Utilities for loading and preprocessing the Charades-Ego dataset for TimeSformer.
Working directly with video files.
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor
import logging
import av
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration (Adjust paths as necessary) ---
# Path to the directory containing the generated annotation files
DEFAULT_ANNOTATION_DIR = "data/CharadesEgo"  # MODIFY IF YOUR ANNOTATIONS ARE ELSEWHERE
# Path to the directory containing the video files
DEFAULT_VIDEO_BASE_PATH = "data/CharadesEgo_v1_videos"  # Directory containing video files

# Assumes these files exist in DEFAULT_ANNOTATION_DIR
TRAIN_ANNOTATION_FILE = "charadesego_train_revised_v2_grouped.txt"
VAL_ANNOTATION_FILE = "charadesego_val_revised_v2_grouped.txt"

# TimeSformer default config often uses 8 frames
NUM_FRAMES_TO_SAMPLE = 8
SAMPLING_RATE = 8  # Sample every 8 frames to cover longer duration, adjust as needed

def load_frames_from_video(video_path, indices):
    """
    Load specific frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        indices (list): List of frame indices to load
    
    Returns:
        list: List of loaded frames as numpy arrays in RGB format, or None if loading fails
    """
    try:
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found: {video_path}")
            return None
            
        with av.open(video_path) as container:
            # Get video stream
            stream = container.streams.video[0]
            
            # Get total number of frames in video
            total_frames = stream.frames
            if indices[-1] >= total_frames:
                logging.warning(f"Requested frame index {indices[-1]} exceeds total frames {total_frames} in {video_path}")
                # Adjust indices to stay within bounds
                indices = [min(idx, total_frames - 1) for idx in indices]
            
            frames = []
            # Seek to each frame index and decode
            for idx in indices:
                # Seek to the target frame
                container.seek(int(idx * stream.time_base * stream.rate), stream=stream)
                
                # Read and decode frame
                for frame in container.decode(video=0):
                    # Convert AVFrame to numpy array (RGB format)
                    img = frame.to_ndarray(format='rgb24')
                    frames.append(img)
                    break  # Only take the first frame after seeking
            
            if len(frames) != len(indices):
                logging.warning(f"Could only load {len(frames)} out of {len(indices)} requested frames from {video_path}")
                # If at least some frames were loaded, duplicate the last frame to reach the desired count
                if frames:
                    while len(frames) < len(indices):
                        frames.append(frames[-1])
                else:
                    return None
                    
            return frames
            
    except Exception as e:
        logging.error(f"Error loading frames from {video_path}: {e}")
        logging.error(traceback.format_exc())
        return None

def sample_frame_indices(total_frames, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices. Handles edge cases.
    Args:
        total_frames (int): Total number of frames available. Must be > 0.
        frame_sample_rate (int): Sample every N frames. Must be >= 1.
        seg_len (int): Number of frames to sample. Must be > 0.
    Returns:
        list: List of sampled frame indices
    """
    if total_frames <= 0 or frame_sample_rate < 1 or seg_len <= 0:
        logging.error(f"Invalid input to sample_frame_indices: total_frames={total_frames}, frame_sample_rate={frame_sample_rate}, seg_len={seg_len}")
        return []

    # Calculate the effective number of frames considering the sampling rate
    converted_len = max(1, total_frames // frame_sample_rate)

    if converted_len < seg_len:
        # If not enough frames after sampling, sample with repeats
        indices = np.linspace(0, total_frames - 1, seg_len)
        indices = np.clip(indices, 0, total_frames - 1).astype(int)
    else:
        # Sample uniformly from the 'rate-adjusted' indices, then scale back
        end_idx = converted_len - 1
        indices = np.linspace(0, end_idx, seg_len, dtype=int)
        indices = indices * frame_sample_rate
        indices = np.clip(indices, 0, total_frames - 1).astype(int)

    return sorted(indices)

def get_video_frame_count(video_path):
    """
    Get the total number of frames in a video file.
    Args:
        video_path (str): Path to the video file
    Returns:
        int: Total number of frames or 0 if video couldn't be opened
    """
    try:
        if not os.path.exists(video_path):
            return 0
            
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            return stream.frames
    except Exception as e:
        logging.error(f"Error getting frame count for {video_path}: {e}")
        return 0

def load_charades_ego_splits(annotation_dir=DEFAULT_ANNOTATION_DIR, video_base_path=DEFAULT_VIDEO_BASE_PATH):
    """
    Loads the grouped Charades-Ego annotations and creates Hugging Face Datasets.
    Adds the full video file path to each entry.
    """
    train_ann_path = os.path.join(annotation_dir, TRAIN_ANNOTATION_FILE)
    val_ann_path = os.path.join(annotation_dir, VAL_ANNOTATION_FILE)

    if not os.path.exists(train_ann_path):
        raise FileNotFoundError(f"Training annotation file not found: {train_ann_path}")
    if not os.path.exists(val_ann_path):
        raise FileNotFoundError(f"Validation annotation file not found: {val_ann_path}")

    logging.info(f"Loading datasets from: {annotation_dir}")
    dataset = load_dataset("text", data_files={"train": train_ann_path, "validation": val_ann_path})

    def split_line(example):
        parts = example["text"].strip().split()
        if len(parts) == 2:
            return {"video_id": parts[0], "label_str": parts[1]}
        else:
            logging.warning(f"Skipping malformed line: {example['text']}")
            return {"video_id": None, "label_str": None}

    dataset = dataset.map(split_line, remove_columns=["text"], num_proc=os.cpu_count())
    dataset = dataset.filter(lambda example: example["video_id"] is not None, num_proc=os.cpu_count())

    def cast_label(example):
        try:
            return {"label": int(example["label_str"])}
        except (ValueError, TypeError):
            logging.warning(f"Could not cast label to int: {example['label_str']} for video {example['video_id']}")
            return {"label": -1}

    dataset = dataset.map(cast_label, remove_columns=["label_str"], num_proc=os.cpu_count())
    dataset = dataset.filter(lambda example: example["label"] != -1, num_proc=os.cpu_count())

    def add_video_path(example):
        video_id = example["video_id"]
        # Try multiple video extensions (mp4, avi, mov)
        for ext in ["mp4", "avi", "mov"]:
            video_path = os.path.join(video_base_path, f"{video_id}.{ext}")
            if os.path.exists(video_path):
                # Get the frame count for the video
                total_frames = get_video_frame_count(video_path)
                if total_frames > 0:
                    return {
                        "video_path": video_path,
                        "total_frames": total_frames
                    }
        
        # If no valid video file was found
        return {
            "video_path": None,
            "total_frames": 0
        }

    # Add video paths and remove video_id since we don't need it anymore
    dataset = dataset.map(add_video_path, remove_columns=["video_id"], num_proc=os.cpu_count())
    
    # Filter out examples where videos weren't found
    initial_train_count = len(dataset["train"])
    initial_val_count = len(dataset["validation"])
    
    dataset = dataset.filter(
        lambda example: example["video_path"] is not None and example["total_frames"] > 0,
        num_proc=os.cpu_count()
    )
    
    final_train_count = len(dataset["train"])
    final_val_count = len(dataset["validation"])

    if final_train_count < initial_train_count:
        logging.warning(f"Removed {initial_train_count - final_train_count} train examples due to missing videos.")
    if final_val_count < initial_val_count:
        logging.warning(f"Removed {initial_val_count - final_val_count} validation examples due to missing videos.")

    logging.info(f"Dataset loaded: {dataset}")
    return dataset

class VideoPreprocessor:
    """
    A callable class to preprocess video frames for TimeSformer.
    Uses the provided Hugging Face image processor.
    """

    def __init__(self, image_processor, num_frames=NUM_FRAMES_TO_SAMPLE, sample_rate=SAMPLING_RATE):
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        logging.info(f"VideoPreprocessor initialized with num_frames={num_frames}, sample_rate={sample_rate}")

    def __call__(self, examples):
        """
        Processes a batch of examples.
        Loads frames directly from videos, samples them, and applies processor transformations.
        Returns tensors suitable for model input.
        """
        # Get video_path and total_frames, ensure they're lists
        video_paths = examples["video_path"]
        total_frames = examples["total_frames"]
        if not isinstance(video_paths, list):
            video_paths = [video_paths]
            total_frames = [total_frames]

        labels = examples["label"]
        if not isinstance(labels, list):
            labels = [labels]

        processed_pixel_values = []
        processed_labels = []

        for video_path, num_frames, label in zip(video_paths, total_frames, labels):
            try:
                if video_path is None or num_frames == 0:
                    logging.warning("Skipping example with invalid video path or zero frames")
                    continue

                indices = sample_frame_indices(
                    total_frames=num_frames,
                    frame_sample_rate=self.sample_rate,
                    seg_len=self.num_frames
                )
                
                if not indices:
                    logging.warning(f"Frame sampling failed for {video_path}. Skipping.")
                    continue

                frames = load_frames_from_video(video_path, indices)
                
                if frames is None:
                    logging.warning(f"Could not load required frames from {video_path}. Skipping.")
                    continue

                # Apply image processor - the processor can handle numpy arrays
                processor_inputs = self.image_processor(images=frames, return_tensors="pt")
                processed_pixel_values.append(processor_inputs["pixel_values"].squeeze(0))
                processed_labels.append(label)

            except Exception as e:
                logging.error(f"Unexpected error processing video {video_path}: {e}")
                continue

        if not processed_pixel_values:
            return {"pixel_values": torch.empty(0), "label": torch.empty(0, dtype=torch.long)}

        try:
            batch_pixel_values = torch.stack(processed_pixel_values)
            batch_labels = torch.tensor(processed_labels, dtype=torch.long)
        except Exception as e:
            logging.error(f"Error stacking tensors for batch: {e}")
            return {"pixel_values": torch.empty(0), "label": torch.empty(0, dtype=torch.long)}

        return {"pixel_values": batch_pixel_values, "label": batch_labels}

if __name__ == "__main__":
    # --- Example Usage ---
    logging.info("Starting data utils example...")

    MODEL_ID = "facebook/timesformer-base-finetuned-k400"
    try:
        image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    except Exception as e:
        logging.error(f"Failed to load image processor for {MODEL_ID}: {e}")
        exit()

    try:
        charades_dataset = load_charades_ego_splits()
    except FileNotFoundError as e:
        logging.error(f"Dataset loading failed: {e}")
        logging.warning("Please ensure annotation files exist at expected paths or adjust DEFAULT_ANNOTATION_DIR.")
        logging.warning(f"Looked for: {os.path.join(DEFAULT_ANNOTATION_DIR, TRAIN_ANNOTATION_FILE)} and ..._val...")
        exit()
    except Exception as e:
        logging.error(f"An unexpected error occurred during dataset loading: {e}")
        exit()

    preprocessor = VideoPreprocessor(image_processor)

    try:
        charades_dataset.set_transform(preprocessor)
        logging.info("Dataset transform set. Accessing elements will now apply preprocessing.")

        if len(charades_dataset["train"]) > 0:
            example_element = charades_dataset["train"][0]
            logging.info(f"Processed element keys: {example_element.keys()}")
            logging.info(f"Pixel values shape: {example_element['pixel_values'].shape}")
            logging.info(f"Label: {example_element['label']}")

            logging.info(f"Pixel values type: {type(example_element['pixel_values'])}")
            logging.info(f"Pixel values dtype: {example_element['pixel_values'].dtype}")
            logging.info(f"Label type: {type(example_element['label'])}")
            logging.info(f"Label dtype: {example_element['label'].dtype}")

        else:
            logging.warning("Training dataset is empty, cannot show example element.")

    except Exception as e:
        logging.error(f"Error during preprocessing test using set_transform: {e}")

    logging.info("Data utils example finished.")
