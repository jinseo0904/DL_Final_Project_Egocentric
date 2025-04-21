import pandas as pd
import os
import argparse
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Revised V2 Class Grouping Definition (N=19 classes) ---
# Mapping: Original Class ID (str 'cXXX') -> New Class Index (int 0-18)
# This mapping corresponds to the V2 proposal in the charades_class_grouping document.
REVISED_V2_CLASS_MAPPING = {
    # 0: Manage Clothing/Footwear
    "c000": 0,
    "c004": 0,
    "c148": 0,
    "c155": 0,  # Clothes
    **{f"c{i:03d}": 0 for i in range(53, 59)},  # Shoes
    # 1: Interact with Doors/Cabinets/Fridge
    "c006": 1,
    "c008": 1,
    "c112": 1,
    "c113": 1,  # Doors/Cabinets
    "c140": 1,
    "c141": 1,  # Doorknob
    "c142": 1,
    "c143": 1,  # Fridge
    "c007": 1,  # Fix door
    # 2: Interact with Furniture/Surfaces
    **{f"c{i:03d}": 2 for i in range(9, 12)},
    "c014": 2,  # Table
    "c081": 2,  # Shelf interaction
    # 3: Use Electronic Devices
    **{f"c{i:03d}": 3 for i in range(15, 19)},  # Phone/Camera
    **{f"c{i:03d}": 3 for i in range(46, 53)},  # Laptop
    "c087": 3,  # Take picture
    "c132": 3,  # Watch TV
    # 4: Interact with Containers
    **{f"c{i:03d}": 4 for i in range(20, 25)},  # Bag
    **{f"c{i:03d}": 4 for i in range(39, 46)},  # Box
    # 5: Interact with Reading Material
    **{f"c{i:03d}": 5 for i in range(25, 29)},  # Book interactions (close, hold, open, put)
    "c032": 5,  # Watch/Read book
    **{f"c{i:03d}": 5 for i in range(115, 118)},  # Paper/Notebook (hold, put, take)
    "c145": 5,  # Work on paper/notebook
    # c030, c031 (Take/Throw book) moved to General Object Manipulation
    # 6: Interact with Linens
    **{f"c{i:03d}": 6 for i in range(33, 37)},  # Towel interactions (hold, put, take, throw)
    "c072": 6,  # Snuggle blanket
    "c078": 6,  # Snuggle pillow
    # c070, c071, c073, c074 (Blanket hold, put, take, throw) moved to General
    # c076, c077, c079, c080 (Pillow hold, put, take, throw) moved to General
    # 7: Handle Food/Groceries
    **{f"c{i:03d}": 7 for i in range(61, 65)},  # Food interactions (hold, put, take, throw)
    **{f"c{i:03d}": 7 for i in range(67, 70)},  # Sandwich interactions (hold, put, take)
    "c130": 7,  # Put groceries
    # 8: Eat/Drink/Take Medicine
    "c065": 8,  # Eat sandwich
    "c106": 8,  # Drink
    "c128": 8,  # Hold medicine
    "c129": 8,  # Take medicine
    "c156": 8,  # Eat something
    # 9: Prepare Food/Cook
    "c066": 9,  # Make sandwich
    "c147": 9,  # Cooking
    # 10: Interact with Pictures
    "c083": 10,
    "c084": 10,  # Take/Hold picture
    "c088": 10,  # Watch picture
    # c086 (Put picture) moved to General
    # 11: Interact with Fixtures
    "c089": 11,
    "c090": 11,
    "c092": 11,  # Window (close, open, watch)
    "c093": 11,
    "c096": 11,  # Mirror (hold, watch)
    **{f"c{i:03d}": 11 for i in range(103, 106)},  # Light (fix, turn on, turn off)
    # 12: Cleaning / Tidying Actions
    "c005": 12,  # Wash clothes
    "c012": 12,
    "c013": 12,  # Tidy/Wash table
    "c037": 12,
    "c038": 12,  # Tidy/Wash towel
    "c075": 12,  # Tidy blanket
    "c082": 12,  # Tidy shelf
    "c091": 12,
    "c095": 12,  # Wash window/mirror
    "c102": 12,  # Use broom (tidying)
    "c111": 12,  # Wash cup
    "c114": 12,  # Tidy closet
    "c121": 12,  # Wash dish
    "c127": 12,  # Tidy floor
    "c136": 12,  # Fix vacuum
    "c139": 12,  # Wash hands
    # c137, c138 (Hold/Take vacuum) moved to General
    # 13: Handle Kitchenware/Tableware
    **{f"c{i:03d}": 13 for i in range(107, 111)},  # Cup/Glass/Bottle (hold, pour, put, take)
    **{f"c{i:03d}": 13 for i in range(118, 121)},  # Dish (hold, put, take)
    # 14: Movement Actions
    "c097": 14,  # Walk through doorway
    "c150": 14,  # Run
    # 15: Posture Change/Positioning
    "c010": 15,
    "c011": 15,  # Sit at table
    "c059": 15,  # Sit in chair (interaction moved to Furniture) -> Reassign c059 to Furniture (2)
    "c122": 15,  # Lie on sofa
    "c123": 15,  # Sit on sofa
    "c124": 15,
    "c125": 15,  # Lie/Sit on floor
    "c133": 15,  # Awaken in bed
    "c134": 15,
    "c135": 15,  # Lie/Sit in bed
    "c146": 15,  # Awaken somewhere
    "c151": 15,  # Sit down
    "c154": 15,  # Stand up
    # 16: Personal Grooming / Hygiene
    "c144": 16,  # Fix hair
    # 17: Communication/Expression
    "c019": 17,  # Talk on phone
    "c029": 17,  # Smile at book
    "c085": 17,  # Laugh at picture
    "c094": 17,  # Smile in mirror
    "c131": 17,  # Laugh at TV
    "c149": 17,  # Laugh
    "c152": 17,  # Smile
    "c153": 17,  # Sneeze
    # 18: General Object Manipulation
    "c001": 18,
    "c002": 18,
    "c003": 18,  # Clothes Put/Take/Throw
    "c017": 18,
    "c018": 18,  # Phone Put/Take
    "c022": 18,
    "c023": 18,
    "c024": 18,  # Bag Put/Take/Throw
    "c028": 18,
    "c030": 18,
    "c031": 18,  # Book Put/Take/Throw
    "c034": 18,
    "c035": 18,
    "c036": 18,  # Towel Put/Take/Throw
    "c042": 18,
    "c043": 18,
    "c044": 18,
    "c045": 18,  # Box Put/Take/From/Throw
    "c049": 18,
    "c050": 18,  # Laptop Put/Take
    "c054": 18,
    "c056": 18,  # Shoes Put/Take
    "c060": 18,  # Stand on chair
    "c062": 18,
    "c063": 18,
    "c064": 18,  # Food Put/Take/Throw
    "c068": 18,
    "c069": 18,  # Sandwich Put/Take
    "c070": 18,
    "c071": 18,
    "c073": 18,
    "c074": 18,  # Blanket Hold/Put/Take/Throw
    "c076": 18,
    "c077": 18,
    "c079": 18,
    "c080": 18,  # Pillow Hold/Put/Take/Throw
    "c086": 18,  # Picture Put
    "c098": 18,  # Holding a broom
    "c099": 18,
    "c100": 18,
    "c101": 18,  # Broom Put/Take/Throw
    "c109": 18,
    "c110": 18,  # Cup Put/Take
    "c116": 18,
    "c117": 18,  # Paper Put/Take
    "c119": 18,
    "c120": 18,  # Dish Put/Take
    "c126": 18,  # Throw on floor
    "c137": 18,
    "c138": 18,  # Hold/Take vacuum
}

# Define names for the new classes (index -> name)
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
NUM_NEW_CLASSES = len(REVISED_V2_CLASS_NAMES)
logging.info(f"Defined {NUM_NEW_CLASSES} revised classes (V2).")
# Sanity check the mapping covers expected range
assert NUM_NEW_CLASSES == 19, "Expected 19 classes"
assert max(REVISED_V2_CLASS_MAPPING.values()) == NUM_NEW_CLASSES - 1, "Max mapped index doesn't match class count"
# Verify all original classes are mapped (optional but good)
original_classes = {f"c{i:03d}" for i in range(157)}
mapped_classes = set(REVISED_V2_CLASS_MAPPING.keys())
unmapped = original_classes - mapped_classes
if unmapped:
    logging.warning(f"Unmapped original classes in V2: {sorted(list(unmapped))}")
else:
    logging.info("All original classes c000-c156 successfully mapped.")


def create_grouped_annotations(input_csv_path: str, output_txt_path: str, class_mapping: dict, video_identifier_col: str = "id", actions_col: str = "actions"):
    """
    Reads an original Charades-Ego CSV, applies class mapping, and writes
    a new annotation file suitable for MMAction2 VideoDataset (format: video_id label).

    Args:
        input_csv_path (str): Path to the input CSV file (e.g., CharadesEgo_v1_train.csv).
        output_txt_path (str): Path to write the output annotation file (e.g., train_revised_v2_grouped.txt).
        class_mapping (dict): Dictionary mapping original class IDs ('cXXX') to new indices (int).
        video_identifier_col (str): Name of the column containing the video ID in the CSV.
        actions_col (str): Name of the column containing the action annotations string.
        filter_ego (bool): If True, only include egocentric videos (ID ends with 'EGO').
    """
    logging.info(f"Processing annotations from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        logging.error(f"Input CSV file not found: {input_csv_path}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file {input_csv_path}: {e}")
        return

    output_lines = 0
    processed_videos = 0
    video_new_labels = defaultdict(set)  # Store unique new labels per video

    for _, row in df.iterrows():
        video_id = row[video_identifier_col]
        actions_str = row[actions_col]

        processed_videos += 1
        # Ensure actions_str is a string and not empty/NaN
        if not isinstance(actions_str, str) or pd.isna(actions_str) or not actions_str.strip():
            continue

        action_triplets = actions_str.strip().split(";")
        current_video_labels = set()  # Track new labels for *this* video

        for triplet in action_triplets:
            parts = triplet.strip().split()
            if len(parts) >= 1:
                original_class_id = parts[0]
                if original_class_id in class_mapping:
                    new_class_index = class_mapping[original_class_id]
                    current_video_labels.add(new_class_index)
                # else: # Log if an original class wasn't in our map
                # logging.warning(f"Original class ID '{original_class_id}' for video {video_id} not found in mapping.")

        # Add the unique new labels found for this video
        if current_video_labels:
            video_new_labels[video_id].update(current_video_labels)

    # Write the output file: video_id new_class_index (one line per unique new class per video)
    try:
        with open(output_txt_path, "w") as f_out:
            # Sort video IDs for consistent output order (optional)
            for video_id in sorted(video_new_labels.keys()):
                new_labels_set = video_new_labels[video_id]
                if not new_labels_set:
                    continue
                # Write one line per unique *new* label associated with this video
                for new_label in sorted(list(new_labels_set)):
                    # Assumes data_prefix points to where videos/clips named by ID are stored
                    f_out.write(f"{video_id} {new_label}\n")
                    output_lines += 1
    except IOError as e:
        logging.error(f"Error writing output file {output_txt_path}: {e}")
        return

    logging.info(f"Finished processing. Processed {processed_videos} relevant videos.")
    logging.info(f"Wrote {output_lines} lines (video_id label pairs) to {output_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create REVISED V2 grouped annotation files for Charades-Ego (N=19 classes).")
    parser.add_argument("-i", "--input_csv", help="Path to the original Charades-Ego CSV annotation file (e.g., CharadesEgo_v1_train.csv)")
    parser.add_argument("-o", "--output_txt", help="Path to save the new REVISED V2 grouped annotation list file (e.g., charadesego_train_revised_v2_grouped.txt)")

    args = parser.parse_args()

    create_grouped_annotations(
        input_csv_path=args.input_csv,
        output_txt_path=args.output_txt,
        class_mapping=REVISED_V2_CLASS_MAPPING,  # Use the final revised mapping
    )

    # Example: Save the new class names to a file
    new_classes_file = os.path.splitext(args.output_txt)[0] + "_classnames.txt"
    try:
        with open(new_classes_file, "w") as f:
            for idx, name in enumerate(REVISED_V2_CLASS_NAMES):
                f.write(f"{idx}: {name}\n")  # Write index and name
        logging.info(f"Saved revised V2 class names (N={NUM_NEW_CLASSES}) to {new_classes_file}")
    except IOError as e:
        logging.error(f"Error writing class names file: {e}")
