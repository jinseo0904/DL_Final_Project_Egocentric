# File: models/object_detection.py

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import numpy as np  # Import numpy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ObjectDetectorYOLO:
    """
    Handles object detection using a YOLO model provided by the Ultralytics library.
    Reads input frames, runs YOLO inference, and extracts bounding boxes,
    class predictions, confidences, and feature representations (placeholder).
    """

    def __init__(self, model_path: str | Path):
        """
        Initializes the YOLO object detector by loading the model.

        Args:
            model_path: Path to the pre-trained YOLO model file (e.g., 'yolov12.pt' or 'yolov8n.pt').
                        Ensure this model is compatible with the ultralytics library version.
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            logging.error(f"Model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        # Load the YOLO model
        try:
            # Check for CUDA availability and set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {self.device}")
            # Initialize the model using the path provided
            self.model = YOLO(self.model_path)
            # Move model to the appropriate device
            self.model.to(self.device)
            logging.info(f"Successfully loaded YOLO model from: {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            raise

    def detect_objects(self, frame_path: str | Path | None = None, frame_data: np.ndarray | None = None):
        """
        Detects objects in a single video frame using the loaded YOLO model.

        Accepts either a path to the frame image file or the frame data as a numpy array.

        Args:
            frame_path: Path to the input video frame image file.
            frame_data: Input video frame as an OpenCV numpy array (BGR format).

        Returns:
            A dictionary containing:
            - 'boxes': List of detected bounding boxes [[x1, y1, x2, y2], ...].
            - 'classes': List of predicted class IDs [...].
            - 'class_names': List of predicted class names [...].
            - 'confidences': List of detection confidences [...].
            - 'features': Placeholder for feature representations (currently None).
                         Extracting deep features might require model-specific hooks or modifications.
        """
        source_description = ""  # For logging purposes
        if frame_path:
            frame_path = Path(frame_path)
            if not frame_path.is_file():
                logging.error(f"Input frame file not found: {frame_path}")
                raise FileNotFoundError(f"Input frame file not found: {frame_path}")
            # Load frame using OpenCV
            source = cv2.imread(str(frame_path))
            if source is None:
                logging.error(f"Failed to read frame image: {frame_path}")
                raise ValueError(f"Failed to read frame image: {frame_path}")
            source_description = f"path: {frame_path}"
        elif frame_data is not None:
            # Ensure frame_data is a numpy array
            if not isinstance(frame_data, np.ndarray):
                logging.error("frame_data must be a numpy array.")
                raise TypeError("frame_data must be a numpy array.")
            source = frame_data
            source_description = "data array"
        else:
            logging.error("Either frame_path or frame_data must be provided.")
            raise ValueError("Either frame_path or frame_data must be provided.")

        logging.info(f"Processing frame from {source_description}.")

        # Perform inference using the model's predict method
        # Arguments like conf, imgsz can be added here if needed
        # verbose=False reduces console clutter during loops
        try:
            results = self.model.predict(source=source, device=self.device, verbose=False)
        except Exception as e:
            logging.error(f"Error during YOLO prediction: {e}")
            raise

        # Results is a list, we typically process the first element for single image inference
        if not results:
            logging.warning(f"No results returned from YOLO model for source: {source_description}.")
            return {"boxes": [], "classes": [], "class_names": [], "confidences": [], "features": None}

        # Extract information based on YOLO_data_strcture.md
        result = results[0]  # Get the Results object for the first (or only) image
        boxes = result.boxes  # Access the Boxes object
        names = result.names  # Access the class names map {id: name}

        # Extract data from the Boxes object, moving to CPU and converting to list/float
        detected_boxes = boxes.xyxy.cpu().numpy().tolist() if boxes.xyxy is not None else []
        detected_classes = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else []
        detected_confidences = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []

        # Map class IDs (floats from tensor, convert to int) to actual names
        detected_class_names = [names.get(int(cls_id), "Unknown") for cls_id in detected_classes]

        # --- Feature Extraction Placeholder ---
        # As per strategy.md, feature representations are needed.
        # This often requires accessing intermediate model layers.
        # How to do this depends heavily on your specific YOLO model architecture
        # and the capabilities of the Ultralytics library or custom modifications.
        # Returning None as a placeholder.
        # Example conceptual access (might not work directly):
        # features = result.features
        # or features = self.model.get_features(source) # If model has such method
        features = None
        # --------------------------------------

        logging.info(f"Detected {len(detected_boxes)} objects in frame from {source_description}.")

        return {
            "boxes": detected_boxes,
            "classes": [int(c) for c in detected_classes],  # Ensure class IDs are int
            "class_names": detected_class_names,
            "confidences": detected_confidences,
            "features": features,  # Currently None, needs specific implementation
        }


# Example Usage: Put this in a separate test script or run directly
if __name__ == "__main__":
    logging.info("Starting ObjectDetectorYOLO example usage...")

    # --- ----------- ---
    MODEL_WEIGHTS_PATH = "yolo12x.pt"

    # --- Example Frame ---
    EXAMPLE_FRAME_PATH = "data/CharadesEgo_v1_rgb/0BDCZEGO/0BDCZEGO-000001.jpg"
    # --------------------

    try:
        # 1. Initialize the detector
        logging.info(f"Attempting to load model: {MODEL_WEIGHTS_PATH}")
        detector = ObjectDetectorYOLO(model_path=MODEL_WEIGHTS_PATH)
        logging.info("Object detector initialized successfully.")

        # 2. Detect objects using the frame path
        if Path(EXAMPLE_FRAME_PATH).exists():
            logging.info(f"Processing example frame from path: {EXAMPLE_FRAME_PATH}")
            detection_output = detector.detect_objects(frame_path=EXAMPLE_FRAME_PATH)

            # 3. Print the results
            print("\n--- Detection Results (from file path) ---")
            print(f"Detected {len(detection_output['boxes'])} objects.")
            for i in range(len(detection_output["boxes"])):
                box = [round(coord, 2) for coord in detection_output["boxes"][i]]  # Round coordinates
                class_id = detection_output["classes"][i]
                class_name = detection_output["class_names"][i]
                confidence = detection_output["confidences"][i]
                print(f"  Object {i + 1}: Class='{class_name}' ({class_id}), Confidence={confidence:.4f}, Box={box}")
            # print(f"  Features: {detection_output['features']}") # Currently None
        else:
            logging.warning(f"Example frame '{EXAMPLE_FRAME_PATH}' not found. Skipping file path example.")

        # 4. Optional: Detect objects using frame data (if image loaded successfully)
        if Path(EXAMPLE_FRAME_PATH).exists():
            logging.info(f"Loading frame data from {EXAMPLE_FRAME_PATH} for direct processing.")
            frame_bgr = cv2.imread(EXAMPLE_FRAME_PATH)
            if frame_bgr is not None:
                detection_output_data = detector.detect_objects(frame_data=frame_bgr)
                print("\n--- Detection Results (from data array) ---")
                print(f"Detected {len(detection_output_data['boxes'])} objects.")
                # (Add similar print loop as above if desired)
            else:
                logging.error(f"Failed to load frame data from {EXAMPLE_FRAME_PATH} using cv2.imread.")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
        print(f"\nError: {e}")
        print("Please ensure the model path ({MODEL_WEIGHTS_PATH}) and example frame path ({EXAMPLE_FRAME_PATH}) are correct.")
    except ImportError as e:
        logging.error(f"Import Error: {e}")
        print(f"\nImport Error: {e}. Have you installed all requirements from requirements.txt (including ultralytics and opencv-python)?")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)  # Log traceback
        print(f"\nAn unexpected error occurred: {e}")

    logging.info("Finished ObjectDetectorYOLO example usage.")
