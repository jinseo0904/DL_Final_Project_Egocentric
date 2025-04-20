import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ObjectDetectorYOLO:
    """
    Handles object detection and feature embedding using a YOLO model
    provided by the Ultralytics library.

    Designed for pipeline integration. Loads a YOLO model, processes input frames,
    performs inference to get detections (boxes, classes), and optionally extracts
    feature embeddings from specified intermediate layers using the `embed` argument.

    Attributes:
        model_path (Path): The path to the YOLO model weights file (.pt).
        device (str): The computing device ('cuda' or 'cpu') determined automatically.
        model (YOLO): The loaded Ultralytics YOLO model object.
        # Consider making embed_layers configurable if needed for different experiments
        default_embed_layers (list[int]): Default layers to extract features from.
    """

    def __init__(self, model_path: str | Path, embed_layers: list[int] | None = None):
        """
        Initializes the YOLO object detector and feature embedder.

        Args:
            model_path (str | Path): Path to the pre-trained YOLO model file
                                     (e.g., 'yolo12.pt' or 'yolov8n.pt').
                                     Ensure compatibility with the ultralytics library.
            embed_layers (list[int] | None): Optional list of layer indices to extract
                                             features from by default. If None, uses
                                             [10, 14, 17] based on user example.

        Raises:
            FileNotFoundError: If the model file does not exist.
            Exception: If any other error occurs during model loading.
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            logging.error(f"Model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        # Set default embedding layers based on user's YOLOv12 example if not provided
        self.default_embed_layers = embed_layers if embed_layers is not None else [10, 14, 17]
        logging.info(f"Default embedding layers set to: {self.default_embed_layers}")

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {self.device}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logging.info(f"Successfully loaded YOLO model from: {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            raise

    def detect_objects(
        self,
        frame_path: str | Path | None = None,
        frame_data: np.ndarray | None = None,
        extract_features: bool = False,
        custom_embed_layers: list[int] | None = None,
    ):
        """
        Detects objects and optionally extracts feature embeddings from a single video frame.

        Uses the `embed` argument in `model.predict()` for feature extraction based on
        the user-provided example.

        Args:
            frame_path (str | Path | None): Path to the input video frame image file.
            frame_data (np.ndarray | None): Input video frame as an OpenCV numpy array (BGR format).
            extract_features (bool): If True, extract feature representations using the specified layers.
            custom_embed_layers (list[int] | None): Override the default embedding layers for this call.

        Returns:
            dict: A dictionary containing the detection results:
                  - 'boxes' (list): List of detected bounding boxes [[x1, y1, x2, y2], ...].
                  - 'classes' (list): List of predicted class IDs [int, ...].
                  - 'features' (Any | None): Extracted feature embeddings (likely tensor(s)) if
                                            `extract_features` is True, otherwise None. Structure depends
                                            on the `results.embeddings` attribute from Ultralytics.
                  - 'class_names' (list): List of predicted class names [str, ...] (Bonus info).
                  - 'confidences' (list): List of detection confidences [float, ...] (Bonus info).

        Raises:
            FileNotFoundError, ValueError, TypeError, Exception: As documented previously.
            AttributeError: If `extract_features` is True but the `results` object lacks
                            the `.embeddings` attribute (e.g., unsupported library version).
        """
        source_description = ""
        source_input = None

        # --- Input Validation and Loading ---
        # (Same as previous version)
        if frame_path:
            if frame_data is not None:
                logging.error("Provide either frame_path or frame_data, not both.")
                raise ValueError("Provide either frame_path or frame_data, not both.")
            frame_path = Path(frame_path)
            if not frame_path.is_file():
                logging.error(f"Input frame file not found: {frame_path}")
                raise FileNotFoundError(f"Input frame file not found: {frame_path}")
            source_input = cv2.imread(str(frame_path))
            if source_input is None:
                logging.error(f"Failed to read frame image: {frame_path}")
                raise ValueError(f"Failed to read frame image: {frame_path}")
            source_description = f"path: {frame_path}"
        elif frame_data is not None:
            if not isinstance(frame_data, np.ndarray):
                logging.error("frame_data must be a numpy array.")
                raise TypeError("frame_data must be a numpy array.")
            source_input = frame_data
            source_description = "data array"
        else:
            logging.error("Either frame_path or frame_data must be provided.")
            raise ValueError("Either frame_path or frame_data must be provided.")

        logging.info(f"Processing frame from {source_description}.")

        # --- Prepare Prediction Arguments ---
        predict_args = {
            "source": source_input,
            "device": self.device,
            "verbose": False,  # Reduce console clutter
            # Add other args like conf, iou, imgsz here if needed
        }

        # Determine which layers to use for embedding
        layers_to_embed = custom_embed_layers if custom_embed_layers is not None else self.default_embed_layers

        # Add the 'embed' argument if feature extraction is requested
        if extract_features:
            logging.info(f"Feature extraction requested. Adding embed={layers_to_embed} to predict call.")
            predict_args["embed"] = layers_to_embed

        # --- Perform Inference ---
        try:
            results = self.model.predict(**predict_args)
        except Exception as e:
            # Catch potential errors if 'embed' is not supported by the library version
            logging.error(f"Error during YOLO prediction: {e}")
            logging.error(
                "This might indicate the 'embed' argument is not supported by your Ultralytics version or model."
            )
            raise

        # --- Process Results ---
        if not results:
            logging.warning(f"No results returned from YOLO model for source: {source_description}.")
            return {"boxes": [], "classes": [], "features": None, "class_names": [], "confidences": []}

        result = results[0]
        boxes = result.boxes
        names = result.names

        detected_boxes = boxes.xyxy.cpu().numpy().tolist() if boxes.xyxy is not None else []
        detected_classes = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else []
        detected_confidences = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
        detected_class_names = [names.get(int(cls_id), "Unknown") for cls_id in detected_classes]

        # --- Extract Features (Embeddings) ---
        features = None  # Default to None
        if extract_features:
            try:
                # Access the embeddings attribute based on the user-provided example
                features = result.embeddings
                if features is not None:
                    logging.info(f"Successfully extracted features (embeddings) from layers: {layers_to_embed}.")
                    # You might want to detach or move features to CPU here depending on subsequent use
                    # Example: features = features.detach().cpu() if isinstance(features, torch.Tensor) else features
                else:
                    logging.warning("Feature extraction requested, but result.embeddings is None.")
            except AttributeError:
                # Handle case where the library version doesn't have .embeddings
                logging.error("Failed to extract features: 'Results' object has no attribute 'embeddings'.")
                logging.error(
                    "Ensure your Ultralytics library version supports the 'embed' argument and '.embeddings' attribute."
                )
                # Optionally re-raise or just return None for features
                # raise # Uncomment to make this a fatal error

        logging.info(f"Detected {len(detected_boxes)} objects in frame from {source_description}.")

        # Return the structured results
        return {
            "boxes": detected_boxes,
            "classes": [int(c) for c in detected_classes],
            "features": features,  # Contains embeddings if extracted, else None
            "class_names": detected_class_names,
            "confidences": detected_confidences,
        }


# --- Example Usage Block ---
if __name__ == "__main__":
    logging.info("Starting ObjectDetectorYOLO example usage...")
    MODEL_WEIGHTS_PATH = "yolo12x.pt"  # <<<--- REPLACE WITH YOUR MODEL PATH (e.g., yolov12n.pt)
    EXAMPLE_FRAME_PATH = "0BDCZEGO-000001.jpg"

    try:
        logging.info(f"Attempting to load model: {MODEL_WEIGHTS_PATH}")
        # Initialize detector (using default embed layers [10, 14, 17] from your example)
        detector = ObjectDetectorYOLO(model_path=MODEL_WEIGHTS_PATH)
        logging.info("Object detector initialized successfully.")

        if Path(EXAMPLE_FRAME_PATH).exists():
            logging.info(f"Processing example frame from path: {EXAMPLE_FRAME_PATH}")

            # Example 1: Run detection *without* requesting features
            detection_output_no_features = detector.detect_objects(
                frame_path=EXAMPLE_FRAME_PATH, extract_features=False
            )
            print("\n--- Detection Results (No Features Requested) ---")
            num_objects = len(detection_output_no_features["boxes"])
            print(f"Detected {num_objects} objects.")
            for i in range(num_objects):
                box = [round(coord, 2) for coord in detection_output_no_features["boxes"][i]]
                class_id = detection_output_no_features["classes"][i]
                class_name = detection_output_no_features["class_names"][i]
                confidence = detection_output_no_features["confidences"][i]
                print(
                    f"  Object {i + 1}: Class='{class_name}' (ID: {class_id}), Confidence={confidence:.4f}, Box={box}"
                )
            print(f"Features: {detection_output_no_features['features']}")  # Should be None

            # Example 2: Run detection *requesting* features (using default layers [10, 14, 17])
            detection_output_with_features = detector.detect_objects(
                frame_path=EXAMPLE_FRAME_PATH, extract_features=True
            )
            print("\n--- Detection Results (Default Features Requested) ---")
            print(f"Detected {len(detection_output_with_features['boxes'])} objects.")
            # Check if features were extracted (depends on library support)
            if detection_output_with_features["features"] is not None:
                # The structure of 'features' depends on results.embeddings
                # It might be a tensor, or a list/tuple of tensors if multiple layers are requested.
                print("Features (Embeddings): Extracted (Structure depends on library output)")
            else:
                print(
                    f"Features: {detection_output_with_features['features']}"
                )  # Likely None if not supported or error occurred

        else:
            logging.warning(f"Example frame '{EXAMPLE_FRAME_PATH}' not found. Skipping examples.")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
        print(f"\nError: {e}")
    except ImportError as e:
        logging.error(f"Import Error: {e}")
        print(f"\nImport Error: {e}.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")

    logging.info("Finished ObjectDetectorYOLO example usage.")
