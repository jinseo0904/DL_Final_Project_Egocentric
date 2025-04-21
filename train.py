# File: train_timesformer.py (Your main script)
"""
Main script for fine-tuning the TimeSformer model on the Charades-Ego dataset
for action recognition using the Hugging Face Trainer API.
"""

import logging
import os
import torch
import evaluate  # Hugging Face Evaluate library
import numpy as np
from transformers import TrainingArguments, Trainer

# --- Import your helper modules ---
# Ensure these files exist and paths are correct relative to this script
try:
    from models.action_recognition import load_timesformer_model_and_processor
    from utils.data_utils import load_charades_ego_splits, VideoPreprocessor
except ImportError as e:
    logging.error(f"Failed to import helper modules: {e}")
    logging.error("Make sure 'models/action_recognition.py' and 'utils/data_utils.py' exist and are in PYTHONPATH.")
    exit()  # Exit if helpers can't be imported

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
# Define output directory for final model/processor
OUTPUT_DIR = "./results/timesformer-charades-ego-finetuned"
# Define directory for saving intermediate checkpoints
CHECKPOINT_DIR = "./checkpoints/timesformer-charades-ego"

# Training Hyperparameters (optimized for 32GB VRAM)
LEARNING_RATE = 1e-5  # Decreased learning rate
TRAIN_BATCH_SIZE = 16  # Increased for 32GB VRAM
EVAL_BATCH_SIZE = 32  # Increased for 32GB VRAM
NUM_EPOCHS = 10  # Increased number of epochs
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 4  # Added gradient accumulation to increase effective batch size

# --- Metrics Computation ---
# Load metrics using evaluate library
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """Computes accuracy and F1 score for evaluation predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # Use macro average for multiclass F1 score
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return {"accuracy": acc, "f1_macro": f1}


# --- Main Training Function ---
def main():
    logging.info("Starting TimeSformer fine-tuning script...")

    # 1. Load Model and Processor
    logging.info("Loading pre-trained model and processor...")
    try:
        # Assuming load_timesformer_model_and_processor is defined correctly
        # in models/action_recognition.py for 19 classes
        processor, model = load_timesformer_model_and_processor()
        # Note: Trainer automatically handles moving the model to the correct device (GPU/CPU)
    except Exception as e:
        logging.error(f"Failed to load model/processor: {e}")
        return

    # 2. Load Datasets
    logging.info("Loading Charades-Ego dataset splits...")
    try:
        # Ensure paths in data_utils.py point to your video files
        # This function should return a datasets.DatasetDict
        dataset = load_charades_ego_splits()
    except FileNotFoundError as e:
        logging.error(f"Dataset loading failed: {e}")
        logging.error("Check paths defined in utils/data_utils.py (DEFAULT_VIDEO_BASE_PATH)")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during dataset loading: {e}")
        return

    # 3. Set up Preprocessing
    logging.info("Setting up video preprocessor and applying transformations...")

    # Define paths to save preprocessed dataset
    PREPROCESSED_DATASET_DIR = "./data/preprocessed_dataset"
    os.makedirs(PREPROCESSED_DATASET_DIR, exist_ok=True)

    # Check if preprocessed dataset already exists
    train_cache_path = os.path.join(PREPROCESSED_DATASET_DIR, "train_preprocessed")
    val_cache_path = os.path.join(PREPROCESSED_DATASET_DIR, "val_preprocessed")

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        logging.info("Loading preprocessed dataset from cache...")
        try:
            from datasets import load_from_disk

            # Load the preprocessed datasets
            dataset_train = load_from_disk(train_cache_path)
            dataset_val = load_from_disk(val_cache_path)

            # Recreate the dataset dictionary
            dataset = {}
            dataset["train"] = dataset_train
            dataset["validation"] = dataset_val

            logging.info("Successfully loaded preprocessed dataset from cache.")
        except Exception as e:
            logging.error(f"Error loading preprocessed dataset: {e}")
            logging.info("Will preprocess dataset from scratch...")
            preprocess_from_scratch = True
        else:
            preprocess_from_scratch = False
    else:
        logging.info("No cached preprocessed dataset found. Processing from scratch...")
        preprocess_from_scratch = True

    if preprocess_from_scratch:
        try:
            # Instantiate your preprocessor class from data_utils.py
            # Pass the processor loaded with the model
            preprocessor = VideoPreprocessor(image_processor=processor)

            # Apply the preprocessing function to the datasets
            # This will add the 'pixel_values' column needed by the model
            # Use batched=True for efficiency
            # Assumes load_charades_ego_splits provides a 'video_path' column
            dataset = dataset.map(
                preprocessor,
                batched=True,
                remove_columns=["video_path", "total_frames"],  # Remove original data columns after processing
            )

            # Set the format for PyTorch
            dataset.set_format(type="torch", columns=["pixel_values", "label"])

            # Save the preprocessed dataset to disk
            logging.info("Saving preprocessed dataset to disk for future use...")
            try:
                dataset["train"].save_to_disk(train_cache_path)
                dataset["validation"].save_to_disk(val_cache_path)
                logging.info(f"Preprocessed dataset saved to {PREPROCESSED_DATASET_DIR}")
            except Exception as e:
                logging.error(f"Error saving preprocessed dataset: {e}")
                logging.warning("Continuing without saving dataset cache...")

        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            return

    logging.info(f"Train dataset size: {len(dataset['train'])}")
    logging.info(f"Validation dataset size: {len(dataset['validation'])}")
    if len(dataset["train"]) == 0 or len(dataset["validation"]) == 0:
        logging.error("Training or validation dataset is empty after processing. Cannot proceed.")
        logging.error("Check video paths, file existence, and preprocessing logic.")
        return

    # 4. Define Training Arguments
    logging.info("Defining training arguments...")
    # Ensure the checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Added gradient accumulation
        logging_dir="./logs",
        logging_steps=10,  # Log training loss every 10 steps
        load_best_model_at_end=True,  # Load the best model based on metric_for_best_model
        metric_for_best_model="f1_macro",  # Use F1 score to select the best model
        save_total_limit=2,  # Only keep the latest 2 checkpoints
        # push_to_hub=False,         # Set to True to push to Hugging Face Hub
        fp16=torch.cuda.is_available(),  # Enable mixed precision if CUDA is available
        report_to="tensorboard",  # Enable TensorBoard logging
        remove_unused_columns=False,  # Keep 'label' column
    )

    # 5. Initialize Trainer
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        # Processor can be passed as tokenizer for feature extraction models
        # tokenizer=processor, # Usually not needed for classification if data is preprocessed
        # Data collator is usually handled automatically for pixel_values and labels
    )

    # 6. Train (Handle Resume from Checkpoint)
    # The Trainer automatically handles finding the latest checkpoint in output_dir
    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        # Check if the directory is not empty and might contain checkpoints
        # A more robust check involves looking for specific checkpoint files/folders
        if any(fname.startswith("checkpoint-") for fname in os.listdir(training_args.output_dir)):
            resume_from_checkpoint = True  # Let Trainer find the latest
            logging.info(f"Found potential checkpoints in {training_args.output_dir}. Attempting to resume training.")
        else:
            logging.info(f"Checkpoint directory {training_args.output_dir} exists but contains no checkpoints. Starting fresh.")
    else:
        logging.info("No checkpoint directory found. Starting training from scratch...")

    logging.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  # Saves optimizer, scheduler, etc.

    # 7. Evaluate the Best Model
    logging.info("Evaluating the best model on the validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # 8. Save the Final Model and Processor
    logging.info(f"Saving the fine-tuned model and processor to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Saves the best model loaded at the end of training
    trainer.save_model(OUTPUT_DIR)
    # Save the processor config alongside the model
    processor.save_pretrained(OUTPUT_DIR)

    logging.info("Fine-tuning finished successfully!")
    logging.info(f"Model and processor saved to: {OUTPUT_DIR}")
    logging.info(f"Final evaluation metrics: {eval_metrics}")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Before running:
    # 1. Ensure dependencies are installed (transformers, datasets, evaluate, accelerate, torchvision, av).
    #    The 'av' library is essential for video decoding: pip install av
    # 2. Ensure the annotation files (e.g., charadesego_train_revised_v2_grouped.txt) exist at the path specified in data_utils.py (or update ANNOTATION_DIR).
    # 3. Ensure the video files exist and the paths provided by `load_charades_ego_splits` (likely in `utils/data_utils.py`) point to these video files.
    # 4. Ensure models/action_recognition.py and utils/data_utils.py are created with the necessary code (including VideoPreprocessor handling video files).
    # 5. Adjust batch sizes (TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE) based on your GPU memory.
    # 6. Adjust paths (OUTPUT_DIR, CHECKPOINT_DIR, ANNOTATION_DIR, VIDEO_DIR in data_utils.py).
    # --- ----------- ---
    main()
