"""
Action Recognition Module using Hugging Face Transformers TimeSformer.
"""

import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification

# Define the 19 revised V2 class names for Charades-Ego grouped actions
REVISED_V2_CLASS_NAMES = [
    "Manage Clothing/Footwear",  # 0
    "Interact with Doors/Cabinets/Fridge",  # 1
    "Interact with Furniture/Surfaces",  # 2
    "Use Electronic Devices",  # 3
    "Interact with Containers",  # 4
    "Interact with Reading Material",  # 5
    "Interact with Linens",  # 6
    "Handle Food/Groceries",  # 7
    "Eat/Drink/Take Medicine",  # 8
    "Prepare Food/Cook",  # 9
    "Interact with Pictures",  # 10
    "Interact with Fixtures",  # 11
    "Cleaning / Tidying Actions",  # 12
    "Handle Kitchenware/Tableware",  # 13
    "Movement Actions",  # 14
    "Posture Change/Positioning",  # 15
    "Personal Grooming",  # 16
    "Communication/Expression",  # 17
    "General Object Manipulation",  # 18
]

NUM_CLASSES = len(REVISED_V2_CLASS_NAMES)

# Create label mappings
label2id = {label: i for i, label in enumerate(REVISED_V2_CLASS_NAMES)}
id2label = {i: label for i, label in enumerate(REVISED_V2_CLASS_NAMES)}

# Pretrained model identifier
MODEL_ID = "facebook/timesformer-base-finetuned-k400"


def load_timesformer_model_and_processor(model_id=MODEL_ID, num_classes=NUM_CLASSES, label2id=label2id, id2label=id2label):
    """
    Loads the TimeSformer model and its associated image processor,
    modifying the classification head for the specified number of classes.

    Args:
        model_id (str): The identifier of the pre-trained TimeSformer model.
        num_classes (int): The number of target classes for the new classifier head.
        label2id (dict): Mapping from class label string to class index.
        id2label (dict): Mapping from class index to class label string.

    Returns:
        tuple: A tuple containing:
            - processor (AutoImageProcessor): The image processor for the model.
            - model (TimesformerForVideoClassification): The loaded model with the modified head.
    """
    print(f"Loading image processor for {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

    print(f"Loading TimeSformer model ({model_id}) and modifying head for {num_classes} classes...")
    model = TimesformerForVideoClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # Crucial for replacing the classifier head
    )
    print("Model loaded and classification head modified successfully.")
    return processor, model


if __name__ == "__main__":
    # Example usage: Load the model and processor
    processor, model = load_timesformer_model_and_processor()

    # Print model configuration to verify the number of labels
    print("\nModel Config (showing num_labels):")
    print(model.config)

    # Print the classifier layer to verify its output size
    print("\nClassifier Layer:")
    print(model.classifier)

    # --- Example of how to get hidden states --- #
    # This requires a dummy input tensor in the expected format.
    # The processor helps create this format, but actual video loading is needed.
    # Example: 8 frames, 3 channels (RGB), 224x224 height/width
    # (Replace with actual processed video data)
    num_frames = 8
    dummy_video = list(torch.randn(num_frames, 3, 224, 224))

    # Process the dummy video (this would typically happen in the dataset map function)
    inputs = processor(dummy_video, return_tensors="pt")

    print("\nRunning dummy forward pass to demonstrate hidden state output...")
    with torch.no_grad():  # No need to track gradients for this example
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple
    # The exact layer to use as features might depend on experimentation.
    # Often, the last hidden state before the classification layer is used.
    # The length of the tuple depends on the model architecture.
    print(f"Number of hidden state layers output: {len(outputs.hidden_states)}")

    # Example: Get the last hidden state
    last_hidden_state = outputs.hidden_states[-1]
    print(f"Shape of the last hidden state: {last_hidden_state.shape}")

    # Example: Get logits (predictions before softmax)
    logits = outputs.logits
    print(f"Shape of the output logits: {logits.shape}")  # Should be [batch_size, num_classes]

    print("\nSetup complete. Ready for data preparation and fine-tuning.")
