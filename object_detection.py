from ultralytics import YOLO
import torch
import csv
import os
import cv2


def load_charades_ego_data(csv_path):
    """Load Charades-Ego dataset from CSV file."""
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_data = {
                "id": row["id"],
                "scene": row["scene"],
                "actions": row["actions"].split(";") if row["actions"] else [],
                "length": float(row["length"]),
                "egocentric": row["egocentric"] == "Yes",
            }
            data.append(video_data)
    return data


def process_video(video_path, model, device):
    """Process a single video with the YOLO model."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    results = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 24th frame (1 frame per second for 24fps videos)
        if frame_count % 24 == 0:
            # Run object detection
            detections = model(frame, device=device)
            results.append({"frame": frame_count, "detections": detections})

        frame_count += 1

    cap.release()
    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO model
    model = YOLO("yolo12x.yaml", task="detect").load("yolo12x.pt")
    model.to(device)
    model.info()

    # Set paths
    dataset_dir = "CharadesEgo"  # Update this to your dataset path
    train_csv = os.path.join(dataset_dir, "CharadesEgo_v1_train.csv")
    test_csv = os.path.join(dataset_dir, "CharadesEgo_v1_test.csv")
    videos_dir = os.path.join(dataset_dir, "videos")  # Update this to your videos path

    # Load dataset metadata
    try:
        train_data = load_charades_ego_data(train_csv)
        print(f"Loaded {len(train_data)} training videos metadata")
    except Exception as e:
        print(f"Error loading training data: {e}")
        train_data = []

    # Process videos
    for i, video_data in enumerate(train_data):
        if i >= 5:  # Process just a few videos for testing
            break

        video_id = video_data["id"]
        video_path = os.path.join(videos_dir, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        print(f"Processing video {i + 1}/{len(train_data)}: {video_id}")
        results = process_video(video_path, model, device)

        if results:
            print(f"Processed {len(results)} frames for video {video_id}")
