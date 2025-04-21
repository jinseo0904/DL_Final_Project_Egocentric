import os
import shutil
import csv

# --- Configuration ---
source_dir = "data/CharadesEgo_v1_rgb"
target_train_dir = "data/train"
target_val_dir = "data/val"
train_csv = "data/CharadesEgo/CharadesEgo_v1_train_only1st.csv"
val_csv = "data/CharadesEgo/CharadesEgo_v1_test_only1st.csv"  # Using test CSV for validation split
# --- End Configuration ---


def move_files(csv_path, target_dir, source_base_dir):
    """Reads a CSV, moves corresponding directories."""
    print(f"\nProcessing {csv_path} for target directory {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    moved_count = 0
    skipped_count = 0
    error_count = 0

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            if "id" not in header:
                print(f"Error: 'id' column not found in header of {csv_path}. Assuming first column.")
                id_index = 0
            else:
                id_index = header.index("id")

            for i, row in enumerate(reader):
                if not row:
                    continue
                try:
                    video_id = row[id_index].strip()
                    if not video_id:
                        print(f"Warning: Empty video ID found in row {i + 2} of {csv_path}")
                        skipped_count += 1
                        continue

                    source_path = os.path.join(source_base_dir, video_id)
                    target_path = os.path.join(target_dir, video_id)

                    if os.path.isdir(source_path):
                        if not os.path.exists(target_path):
                            try:
                                shutil.move(source_path, target_path)
                                moved_count += 1
                            except OSError as e:
                                print(f"Error moving {source_path} to {target_path}: {e}")
                                error_count += 1
                        else:
                            # print(f"Skipping: Target already exists - {target_path}")
                            skipped_count += 1
                    else:
                        # print(f"Skipping: Source directory not found - {source_path}")
                        skipped_count += 1
                except IndexError:
                    print(f"Warning: Malformed row {i + 2} in {csv_path}: {row}")
                    skipped_count += 1
                except Exception as e:
                    print(f"Unexpected error processing row {i + 2} ({video_id}) in {csv_path}: {e}")
                    error_count += 1

    except FileNotFoundError:
        print(f"Error: CSV file not found - {csv_path}")
        return 0, 0, 1  # Indicate error
    except Exception as e:
        print(f"An unexpected error occurred while reading {csv_path}: {e}")
        return moved_count, skipped_count, error_count + 1  # Indicate error

    print(f"Finished processing {csv_path}.")
    print(f"Moved: {moved_count}, Skipped (missing source or target exists): {skipped_count}, Errors: {error_count}")
    return moved_count, skipped_count, error_count


# --- Main Execution ---
print("Starting video directory organization...")
print(f"Source directory: {source_dir}")

if not os.path.isdir(source_dir):
    print(f"Error: Source directory '{source_dir}' not found. Aborting.")
else:
    # Move training files
    move_files(train_csv, target_train_dir, source_dir)

    # Move validation files (using the test CSV)
    move_files(val_csv, target_val_dir, source_dir)

    print("\nOrganization process complete.")
    # Optional: List remaining files in source_dir?
    try:
        remaining_items = os.listdir(source_dir)
        if remaining_items:
            print(f"\nNote: The following items still remain in '{source_dir}':")
            # for item in remaining_items[:10]: # Print first few
            #     print(f"- {item}")
            # if len(remaining_items) > 10:
            #     print(f"- ... and {len(remaining_items) - 10} more.")
            print(f"Total remaining items: {len(remaining_items)}")
        else:
            print(f"\nSource directory '{source_dir}' is now empty.")
    except Exception as e:
        print(f"Could not list remaining items in source directory: {e}")
