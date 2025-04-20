import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import numpy as np
import csv
import json
from tqdm import tqdm

# --- Configuration ---
TRAIN_CSV_PATH = "data/CharadesEgo/CharadesEgo_v1_train.csv"  # Assuming this exists
OBJECT_CLASSES_PATH = "data/CharadesEgo/Charades_v1_objectclasses.txt"
# Or optionally use detected objects:
# DETECTION_RESULTS_PATH = "results/object_detection_evaluation.json"
OUTPUT_DIR = "results/scene_predictor"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "scene_transformer_model.pth")
SCENE_MAP_PATH = os.path.join(OUTPUT_DIR, "scene_label_map.json")
OBJECT_VOCAB_PATH = os.path.join(OUTPUT_DIR, "object_vocab.json")

# Model Hyperparameters
VOCAB_SIZE = None  # Will be set after loading data
NUM_SCENE_CLASSES = None  # Will be set after loading data
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 20  # Max number of objects per video to consider
DROPOUT = 0.2

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
VALIDATION_SPLIT = 0.1

# --- Utility Functions ---


def load_object_classes(class_file):
    """Loads object classes (ID -> Name) and creates name -> ID mapping."""
    object_classes = {}
    name_to_id = {}
    try:
        with open(class_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if len(row) >= 2:
                    class_id = row[0]
                    class_name = " ".join(row[1:])
                    object_classes[class_id] = class_name
                    name_to_id[class_name.lower()] = class_id
    except FileNotFoundError:
        print(f"Error: Object classes file not found at {class_file}")
    return object_classes, name_to_id


def load_charades_data(csv_path, name_to_id_map):
    """Loads video data, extracts GT objects and scenes."""
    data = []
    scenes = set()
    object_ids_used = set()
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            video_id = row["id"]
            scene = row["scene"]
            raw_objects = row["objects"].split(";") if pd.notna(row["objects"]) and row["objects"] else []

            # Convert object names/IDs to consistent IDs (handle potential mixed formats)
            gt_object_ids = set()
            for obj_str in raw_objects:
                obj_str = obj_str.strip()
                if obj_str.startswith("o") and len(obj_str) == 4:  # Looks like an ID
                    # Basic validation, could add check against loaded classes if needed
                    gt_object_ids.add(obj_str)
                else:  # Assume it's a name
                    obj_id = name_to_id_map.get(obj_str.lower())
                    if obj_id:
                        gt_object_ids.add(obj_id)
                    # else: print(f"Warning: Object name '{obj_str}' not found in mapping for video {video_id}")

            if scene and gt_object_ids:  # Only include videos with scene label and objects
                # Exclude 'o000' (person/None) if present
                gt_object_ids.discard("o000")
                if gt_object_ids:  # Check again if any objects remain
                    data.append({
                        "id": video_id,
                        "scene": scene,
                        "object_ids": sorted(list(gt_object_ids)),  # Use sorted list
                    })
                    scenes.add(scene)
                    object_ids_used.update(gt_object_ids)

    except FileNotFoundError:
        print(f"Error: Training data file not found at {csv_path}")
        return [], set(), set()
    except Exception as e:
        print(f"Error loading data from {csv_path}: {e}")
        return [], set(), set()

    print(f"Loaded {len(data)} videos with scenes and ground truth objects from {csv_path}")
    return data, scenes, object_ids_used


# --- Dataset Class ---
class SceneObjectDataset(Dataset):
    def __init__(self, data, scene_to_idx, obj_to_idx, max_len):
        self.data = data
        self.scene_to_idx = scene_to_idx
        self.obj_to_idx = obj_to_idx
        self.max_len = max_len
        self.pad_idx = obj_to_idx["<PAD>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        scene_label = self.scene_to_idx[item["scene"]]
        object_ids = item["object_ids"]

        # Convert object IDs to indices, use <UNK> for unknown
        unk_idx = self.obj_to_idx["<UNK>"]
        object_indices = [self.obj_to_idx.get(obj_id, unk_idx) for obj_id in object_ids]

        # Pad sequence
        seq_len = len(object_indices)
        if seq_len < self.max_len:
            padded_indices = object_indices + [self.pad_idx] * (self.max_len - seq_len)
        else:
            padded_indices = object_indices[: self.max_len]  # Truncate

        return {"object_seq": torch.tensor(padded_indices, dtype=torch.long), "scene_label": torch.tensor(scene_label, dtype=torch.long)}


# --- Transformer Model ---
class SceneTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_len, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Assuming PAD is index 0
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))  # Learnable Positional Encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        # Use the output of the first token ([CLS] equivalent, although we don't explicitly add one)
        # Or average pooling over non-padding tokens
        self.fc = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Positional encoding might be initialized differently if needed

    def forward(self, src_seq):
        # src_seq shape: (batch_size, seq_len)
        padding_mask = src_seq == 0  # Assuming PAD index is 0

        embedded = self.embedding(src_seq) * np.sqrt(self.embed_dim)  # (batch_size, seq_len, embed_dim)
        # Add positional encoding
        pos_encoded = embedded + self.pos_encoder[:, : src_seq.size(1), :]  # (batch_size, seq_len, embed_dim)
        pos_encoded = self.dropout(pos_encoded)

        # Pass through transformer encoder
        # TransformerEncoderLayer expects src_key_padding_mask where True indicates padding
        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=padding_mask)
        # transformer_output shape: (batch_size, seq_len, embed_dim)

        # --- Classification Head ---
        # Option 1: Use the output of the first element (like BERT's [CLS] token)
        # pooled_output = transformer_output[:, 0]

        # Option 2: Average pooling over non-padding elements
        # Mask out padding tokens before averaging
        transformer_output = transformer_output.masked_fill(padding_mask.unsqueeze(-1), 0)
        # Sum non-padding elements and divide by the number of non-padding elements
        non_padding_elements = (~padding_mask).sum(dim=1, keepdim=True)
        # Avoid division by zero if a sequence is all padding (shouldn't happen with checks)
        non_padding_elements = torch.max(non_padding_elements, torch.ones_like(non_padding_elements))
        pooled_output = transformer_output.sum(dim=1) / non_padding_elements

        # pooled_output shape: (batch_size, embed_dim)
        logits = self.fc(pooled_output)  # (batch_size, num_classes)
        return logits


# --- Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    print(f"Starting training for {num_epochs} epochs on {device}...")
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        # Wrap train_loader with tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            object_seq = batch['object_seq'].to(device)
            scene_labels = batch['scene_label'].to(device)

            optimizer.zero_grad()
            logits = model(object_seq)
            loss = criterion(logits, scene_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_train += scene_labels.size(0)
            correct_train += (predicted == scene_labels).sum().item()

            # Update tqdm description with current loss
            train_pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        # Wrap val_loader with tqdm for progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                object_seq = batch['object_seq'].to(device)
                scene_labels = batch['scene_label'].to(device)

                logits = model(object_seq)
                loss = criterion(logits, scene_labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total_val += scene_labels.size(0)
                correct_val += (predicted == scene_labels).sum().item()

                # Update tqdm description with current loss
                val_pbar.set_postfix({'loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val

        # Step the scheduler after validation
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  Saving new best model with Val Acc: {best_val_acc:.4f}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Scene Prediction Transformer Training ---")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    charades_objects, name_to_id_map = load_object_classes(OBJECT_CLASSES_PATH)
    if not charades_objects:
        exit(1)

    video_data, unique_scenes, unique_object_ids = load_charades_data(TRAIN_CSV_PATH, name_to_id_map)
    if not video_data:
        print("No training data loaded. Exiting.")
        exit(1)

    # 2. Preprocess Data - Create Mappings and Vocab
    print("Preprocessing data...")
    # Scene mapping
    scene_list = sorted(list(unique_scenes))
    scene_to_idx = {scene: i for i, scene in enumerate(scene_list)}
    idx_to_scene = {i: scene for scene, i in scene_to_idx.items()}
    NUM_SCENE_CLASSES = len(scene_list)
    print(f"Found {NUM_SCENE_CLASSES} unique scene classes.")
    with open(SCENE_MAP_PATH, "w") as f:
        json.dump({"scene_to_idx": scene_to_idx, "idx_to_scene": idx_to_scene}, f, indent=2)
    print(f"Saved scene mapping to {SCENE_MAP_PATH}")

    # Object vocabulary (including special tokens)
    # Ensure PAD is 0, UNK is 1
    obj_list = sorted(list(unique_object_ids))
    obj_to_idx = {"<PAD>": 0, "<UNK>": 1}
    # Start indexing actual objects from 2
    for i, obj_id in enumerate(obj_list):
        obj_to_idx[obj_id] = i + 2
    idx_to_obj = {i: obj for obj, i in obj_to_idx.items()}
    VOCAB_SIZE = len(obj_to_idx)
    print(f"Created object vocabulary with {VOCAB_SIZE} entries (including <PAD>, <UNK>).")
    with open(OBJECT_VOCAB_PATH, "w") as f:
        json.dump({"obj_to_idx": obj_to_idx, "idx_to_obj": idx_to_obj}, f, indent=2)
    print(f"Saved object vocabulary to {OBJECT_VOCAB_PATH}")

    # 3. Create Datasets and DataLoaders
    full_dataset = SceneObjectDataset(video_data, scene_to_idx, obj_to_idx, MAX_SEQ_LEN)

    # Split dataset
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Created datasets: Train ({len(train_dataset)} samples), Validation ({len(val_dataset)} samples)")

    # 4. Initialize Model
    model = SceneTransformerClassifier(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=NUM_SCENE_CLASSES, max_len=MAX_SEQ_LEN, dropout=DROPOUT
    ).to(device)

    print("Model Architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # 5. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Initialize the Cosine Annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 6. Train Model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS)

    print("--- Training Complete ---")
