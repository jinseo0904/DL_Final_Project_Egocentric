#!/usr/bin/env python
import re
import yaml

# Read COCO classes
with open("CharadesEgo/COCO_classes.yml", "r") as f:
    coco_data = yaml.safe_load(f)

# Get COCO class names as a set for easy comparison
coco_classes = set(coco_data["names"].values())

# Read Charades classes
charades_classes = []
with open("CharadesEgo/Charades_v1_objectclasses.txt", "r") as f:
    for line in f:
        match = re.match(r"o\d+\s+(.*)", line.strip())
        if match:
            class_name = match.group(1)
            if class_name != "None":  # Skip the None class
                charades_classes.append(class_name)

# Find unique Charades classes (not in COCO)
unique_charades = []
for c in charades_classes:
    # Handle special cases with similar classes
    if c == "laptop" and "laptop" in coco_classes:
        continue
    if c == "chair" and "chair" in coco_classes:
        continue
    if c == "sofa/couch" and "couch" in coco_classes:
        continue
    if c == "table" and "dining table" in coco_classes:
        continue
    if c == "television" and "tv" in coco_classes:
        continue
    if c == "cup/glass/bottle" and ("cup" in coco_classes or "bottle" in coco_classes):
        continue
    if c == "sandwich" and "sandwich" in coco_classes:
        continue
    if c == "refrigerator" and "refrigerator" in coco_classes:
        continue
    if c == "book" and "book" in coco_classes:
        continue
    if c == "bed" and "bed" in coco_classes:
        continue
    if c == "phone/camera" and "cell phone" in coco_classes:
        continue

    # If the class doesn't exist in COCO classes, add it to our unique list
    if c not in coco_classes:
        unique_charades.append(c)

# Create a new combined dictionary
combined_data = coco_data.copy()
next_id = max(int(k) for k in coco_data["names"].keys()) + 1

# Add unique Charades classes
for c in unique_charades:
    combined_data["names"][next_id] = c
    next_id += 1

# Write the combined data to a new YAML file
with open("CharadesEgo/combined_classes.yml", "w") as f:
    yaml.dump(combined_data, f, default_flow_style=False, sort_keys=False)

print(f"Combined classes YAML created. Added {len(unique_charades)} unique classes from Charades dataset.")
print(f"Total classes: {len(combined_data['names'])}")
